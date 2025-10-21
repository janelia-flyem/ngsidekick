import ast
import string
from typing import List, Dict

import numpy as np
import pandas as pd

from ngsidekick.segmentprops.segmentprops import segment_properties_to_dataframe, segment_properties_json


def select_segment_properties(
    info: dict,
    subset: List[str],
    scalar_expressions: Dict[str, str] = {},
    tag_expressions: Dict[str, str] = {}
) -> pd.DataFrame:

    scalar_df, tags_df = segment_properties_to_dataframe(info, return_separate_tags=True)
    full_df = pd.concat((scalar_df, tags_df), axis=1)
    new_df = _select_segment_properties_from_dataframe(full_df, scalar_expressions, tag_expressions)
    new_tag_cols = new_df.select_dtypes(include=bool).columns

    subset_tags_df = tags_df[[c for c in subset if c in tags_df.columns]]
    subset_scalar_df = scalar_df[[c for c in subset if c in scalar_df.columns]]
    combined_df = pd.concat((new_df, subset_scalar_df, subset_tags_df), axis=1)
    return segment_properties_json(combined_df, tag_cols=[*new_tag_cols, *[c for c in tags_df.columns if c in subset]])


def _select_segment_properties_from_dataframe(
    full_df: pd.DataFrame,
    scalar_expressions: Dict[str, str] = {},
    tag_expressions: Dict[str, str] = {}
) -> pd.DataFrame:

    for col in full_df.columns.tolist():
        if full_df[col].dtype in ("category", "object", "string"):
            full_df[col] = full_df[col].astype('string').fillna('')
    
    new_df = full_df[[]].copy()
    for name, expr in scalar_expressions.items():
        if template_names := string_template_names(expr):
            # This is faster than using df.apply() with a lambda.
            new_df[name] = [
                expr.format(**dict(zip(template_names, row)), locals={}, globals={})
                for row in full_df[template_names].values
            ]
        else:
            rewritten = hoist_literals_to_columns(full_df, expr, prefix="__")
            new_df[name] = full_df.eval(
                rewritten,
                local_dict={},
                global_dict={},
                engine='python'
            )

    return new_df


def string_template_names(template: str) -> bool:
    try:
        return [name for (_, name, *_) in string.Formatter().parse(template) if name]
    except ValueError:
        return []


def hoist_literals_to_columns(df: pd.DataFrame, expr: str, prefix: str = "_") -> str:
    """
    Parse `expr`, replace literals with new dataframe columns (prefixed with `prefix`),
    broadcast those literal values to the columns, and return the rewritten expression.
    """
    class Hoister(ast.NodeTransformer):
        def __init__(self, df, prefix):
            self.df = df
            self.prefix = prefix
            self.pool = {}          # (type, value) -> colname  (dedupe identical literals)
            self.counter = 0

        def _fresh_name(self) -> str:
            # Find a column name not already in df
            while True:
                name = f"{self.prefix}L{self.counter}"
                self.counter += 1
                if name not in self.df.columns:
                    return name

        def _name_for(self, value):
            key = (type(value), value)
            if key not in self.pool:
                self.pool[key] = self._fresh_name()
            return self.pool[key]

        def visit_Constant(self, node: ast.Constant):
            # Hoist simple literals
            if isinstance(node.value, (str, int, float, bool, type(None))):
                name = self._name_for(node.value)
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
            return node

    tree = ast.parse(expr, mode="eval")
    hoister = Hoister(df, prefix)
    new_tree = hoister.visit(tree)
    ast.fix_missing_locations(new_tree)
    rewritten = ast.unparse(new_tree)

    # Create/broadcast columns for each hoisted literal
    for (typ, value), col in hoister.pool.items():
        if isinstance(value, str):
            # keep pandas StringDtype to avoid object/string mixing issues
            df[col] = pd.Series(pd.array([value] * len(df), dtype="string"), index=df.index)
        elif value is None:
            df[col] = pd.Series([pd.NA] * len(df), dtype="string", index=df.index)
        else:
            df[col] = value

    return rewritten
