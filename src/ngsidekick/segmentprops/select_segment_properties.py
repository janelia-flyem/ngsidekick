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
    """
    Given the JSON data from a neuroglancer segment properties 'info' file,
    select a subset of the properties and/or tags and return the reduced JSON info.

    You can also add *new* properties and/or tags by providing expressions
    (in pandas eval syntax) to be evaluated using the source data.

    In addition to expressions, you can provide a template string to be evaluated
    with Python string formatting.

    The variables available to the expressions are the columns extracted from the
    source data as produced via `segment_properties_to_dataframe()`.
    The tag list is parsed into separate columns according to their prefixes
    (assuming ':' delimits the prefix).

    The 'label' column is special: It is the one that will be displayed in the neuroglancer segment list.

    Args:
        info:
            The JSON data from a neuroglancer segment properties 'info' file.

        subset:
            The subset of properties or tag categories to select.
            Special values:
            - '_all': Select all properties and tags.
            - '_all_tags': Select all tag categories.
            - '_default': Select all properties and tags that have less than 1000 unique values.
            - '_default_tags': Select all tag categories that have less than 1000 unique values.

        scalar_expressions:
            Expressions to be evaluated using the source data to compute new properties.

        tag_expressions:
            Expressions to be evaluated using the source data to compute new tags.

    Returns:
        The reduced/enhanced JSON data from a neuroglancer segment properties 'info' file.
    

    Note:
        For the purpose of expression evaluation, we treat '~' and missing values (untagged) as "".

    Examples:

    .. code-block:: python

        new_info = select_segment_properties(info, subset=['superclass', 'type'])
        new_info = select_segment_properties(
            info,
            scalar_expressions={
                'size': 'length * width * height',
                'label': "{type} ({superclass})"
            },
        )
        new_info = select_segment_properties(
            info,
            tag_expressions={
                'is_left_hemilineage_01': 'hemilineage in ("01A", "01B") and soma_side == "L"'
            }
        )
    
    See also:
        The `ngsidekick-server <https://github.com/janelia-flyem/ngsidekick-server>`_
        project wraps this function in a web service to enable dynamic segment property
        selection directly in neuroglancer (via user-provided source URLs).
    """

    scalar_df, tags_df = segment_properties_to_dataframe(info, return_separate_tags=True)

    # It's possible for a scalar column to also be used in the tags.
    # We'll drop the scalar and just keep the one from the tags,
    # since the tag column takes less RAM.
    scalar_df = scalar_df[[c for c in scalar_df.columns if c not in tags_df.columns]]

    full_df = pd.concat((scalar_df, tags_df), axis=1)
    new_df = _select_segment_properties_from_dataframe(full_df, {**scalar_expressions, **tag_expressions})
    new_tag_cols = list(tag_expressions.keys())

    if '_all' in subset:
        subset += [
            c for c in full_df.columns
            if c not in subset
        ]
        subset.remove('_all')

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
    
    # In some datasets, the segment properties use '~' to indicate an empty field
    # instead of the empty string to avoid sorting empty types near the top of the ID list.
    # For the purposes of expression evaluation, we treat '~' and '' as None.
    full_df_with_empty = full_df.replace('~', '')

    new_df = full_df[[]].copy()
    for name, expr in expressions.items():
        if template_names := string_template_names(expr):
            # Only evaluate the expression for rows in which at least
            # one of the referenced template names is not None.
            # Other rows are not evaluated and get an empty string by default.
            # (Special case: For the 'label' column, it will be replaced with '~' later.)
            new_df[name] = ''
            if invalid_template_names := set(template_names) - set(full_df_with_empty.columns):
                raise ValueError(f"Invalid segment properties: {', '.join(invalid_template_names)}")
            valid_rows = (full_df_with_empty[template_names] != '').any(axis=1)

            # This list comprehension is faster than using df.apply() with a lambda.
            new_df.loc[valid_rows, name] = [
                expr.format(**dict(zip(template_names, row)), locals={}, globals={}).strip()
                for row in full_df_with_empty.loc[valid_rows, template_names].values
            ]
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', expr):
            # String appears not to be a template nor an expression;
            # it must just be a column name.
            new_df[name] = full_df[expr]
        else:
            new_df[name] = full_df_with_empty.eval(
                expr,
                local_dict={},
                global_dict={},
                engine='python'
            )

        # We append the newly computed column to the input dataframe so it
        # can be referenced in subsequent expressions if the user wants to.
        full_df_with_empty[name] = new_df[name]

    # It's annoying to have the empty labels sorted at the top of our ID list,
    # so we replace the empty string with '~' here to shove them to the bottom.
    if 'label' in new_df.columns:
        new_df['label'] = new_df['label'].replace('', '~')

    return new_df


def string_template_names(template: str) -> bool:
    try:
        return [name for (_, name, *_) in string.Formatter().parse(template) if name]
    except ValueError:
        return []
