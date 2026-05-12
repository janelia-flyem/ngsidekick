import logging
from dataclasses import dataclass
from itertools import chain
from typing import NamedTuple

from numba import njit
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PolylineGeometry(NamedTuple):
    """
    Bundle of arrays that describes a batch of polyline annotations after
    they've been unpacked from the user-supplied auxiliary table.

    - ``points``: (total_points, D) float32, all polyline vertices concatenated
      in main-df-row order. ``points[starts[i]:ends[i]]`` are the vertices of
      polyline ``i`` in traversal order.
    - ``starts``, ``ends``: (N,) int64 offsets into ``points``.
    """
    points: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


@dataclass
class TableHandle:
    """
    A wrapper for a pandas DataFrame that can be provided to transfer ownership
    of the DataFrame to ``write_precomputed_annotations()``, which will delete
    the handle's reference to the DataFrame as soon as possible to save RAM.

    Example:

    .. code-block:: python

        >>> handle = TableHandle(df)
        >>> del df  # Delete your own reference to the original data
        >>> write_precomputed_annotations(handle, 'xyz', 'point')
    """
    df: pd.DataFrame | None = None


def _geometry_cols(coord_names, annotation_type):
    """
    Determine the list of column groups that express
    the geometry of annotations of the given type.
    Point annotations have only one group,
    but other annotation types have two.
    
    Examples:
    
        >>> _geometry_cols([*'xyz'], 'point')
        [['x', 'y', 'z']]

        >>> _geometry_cols([*'xyz'], 'ellipsoid')
        [['x', 'y', 'z'], ['rx', 'ry', 'rz']]

        >>> _geometry_cols([*'xyz'], 'line')
        [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']]

        >>> _geometry_cols([*'xyz'], 'axis_aligned_bounding_box')
        [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']]
    """
    if annotation_type == 'point':
        return [[c for c in coord_names]]

    if annotation_type == 'ellipsoid':
        return [
            [c for c in coord_names],
            [f'r{c}' for c in coord_names]
        ]

    if annotation_type in ('line', 'axis_aligned_bounding_box'):
        return [
            [f'{c}a' for c in coord_names],
            [f'{c}b' for c in coord_names]
        ]

    if annotation_type == 'polyline':
        # Polyline geometry lives in an auxiliary table, not the main df.
        return []

    raise ValueError(f"Annotation type {annotation_type} not supported")


def _property_column_names(property_specs):
    """
    Return the dataframe column names that back the given property specs.
    For numeric/categorical/string properties this is the property id itself;
    for rgb/rgba properties the value comes from ``{p}_r``, ``{p}_g``, etc.
    """
    cols = []
    for spec in property_specs:
        p = spec['id']
        if spec['type'] == 'rgb':
            cols.extend([f'{p}_r', f'{p}_g', f'{p}_b'])
        elif spec['type'] == 'rgba':
            cols.extend([f'{p}_r', f'{p}_g', f'{p}_b', f'{p}_a'])
        else:
            cols.append(p)
    return cols


def _ann_required_cols(coord_space, annotation_type, property_specs):
    """
    Names of the columns in the main DataFrame that the geometry+property
    encoder will consume for the given annotation type. Used by writers
    that subset before exploding / iloc-ing.
    """
    geom = list(chain(*_geometry_cols(coord_space.names, annotation_type)))
    return geom + _property_column_names(property_specs)


def _drop_unused_columns(df, coord_space, annotation_type, property_specs, relationships):
    """
    Return a view of ``df`` containing only the columns this exporter
    actually consumes (geometry, properties, relationships). Logs a notice
    listing any dropped columns.
    """
    geom_cols = [*chain(*_geometry_cols(coord_space.names, annotation_type))]
    prop_cols = _property_column_names(property_specs)

    used = {*geom_cols, *prop_cols, *relationships}
    keep = [c for c in df.columns if c in used]
    drop = [c for c in df.columns if c not in used]
    if drop:
        logger.info(f"Ignoring {len(drop)} unused input column(s): {drop}")
    return df[keep]


@njit(inline='always')
def _unravel_index(flat_index, shape, out):
    """
    Allocation-free, single-index equivalent of ``numpy.unravel_index``.

    Decodes ``flat_index`` into a multi-D coordinate within an array of
    ``shape``, writing the result into the pre-allocated ``out`` buffer
    (which must have length ``len(shape)``). Like ``numpy.unravel_index``
    with its default ``order='C'``, the last axis varies fastest as
    ``flat_index`` increments.

    Equivalent to:

        out[:] = numpy.unravel_index(flat_index, shape)

    but without allocating a tuple of arrays for the result.
    """
    for d in range(len(shape) - 1, -1, -1):
        out[d] = flat_index % shape[d]
        flat_index = flat_index // shape[d]
