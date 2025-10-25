import re
import string
from typing import List, Dict
from pandas.api.types import is_numeric_dtype

import numpy as np
import pandas as pd

from ngsidekick.segmentprops.segmentprops import segment_properties_to_dataframe, segment_properties_json


def select_segment_properties(
    info: dict,
    subset: List[str],
    scalar_expressions: Dict[str, str] = {},
    tag_expressions: Dict[str, str] = {}
) -> dict:

    scalar_df, tags_df = segment_properties_to_dataframe(info, return_separate_tags=True)
    full_df = pd.concat((scalar_df, tags_df), axis=1)
    new_df = _select_segment_properties_from_dataframe(full_df, {**scalar_expressions, **tag_expressions})
    new_tag_cols = new_df.select_dtypes(include=bool).columns

    if '_all' in subset:
        subset += [
            c for c in full_df.columns
            if c not in subset
        ]
        subset.remove('_all_tags')

    if '_all_tags' in subset:
        subset += [
            c for c in full_df.columns
            if c not in subset and not is_numeric_dtype(full_df[c].dtype)
        ]
        subset.remove('_all_tags')

    if '_default' in subset:
        counts = full_df.nunique()
        subset += [
            c for c in full_df.columns
            if c not in subset and counts[c] <= 1000
        ]
        subset.remove('_default')

    if '_default_tags' in subset:
        counts = full_df.nunique()
        subset += [
            c for c in full_df.columns
            if c not in subset and counts[c] <= 1000 and not is_numeric_dtype(full_df[c].dtype)
        ]
        subset.remove('_default_tags')

    if invalid_subset := set(subset) - (set(tags_df.columns) | set(scalar_df.columns)):
        raise ValueError(f"Invalid segment properties: {', '.join(invalid_subset)}")

    subset_tags_df = tags_df[[c for c in subset if c in tags_df.columns]]
    subset_scalar_df = scalar_df[[c for c in subset if c in scalar_df.columns]]
    combined_df = pd.concat((new_df, subset_scalar_df, subset_tags_df), axis=1)
    return segment_properties_json(combined_df, tag_cols=[*new_tag_cols, *[c for c in subset if c in tags_df.columns]])


def _select_segment_properties_from_dataframe(
    full_df: pd.DataFrame,
    expressions: Dict[str, str] = {},
) -> pd.DataFrame:

    for col in full_df.columns.tolist():
        if full_df[col].dtype in ("category", "object", "string"):
            full_df[col] = full_df[col].astype('string').fillna('')
    
    new_df = full_df[[]].copy()
    for name, expr in expressions.items():
        if template_names := string_template_names(expr):
            if invalid_template_names := set(template_names) - set(full_df.columns):
                raise ValueError(f"Invalid segment properties: {', '.join(invalid_template_names)}")
            # This is faster than using df.apply() with a lambda.
            new_df[name] = [
                expr.format(**dict(zip(template_names, row)), locals={}, globals={}).strip()
                for row in full_df[template_names].values
            ]
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', expr):
            # String appears not to be a template or an expression.
            # It must just be a column name.
            new_df[name] = full_df[expr]
        else:
            new_df[name] = full_df.eval(
                expr,
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
