import os
import logging
from itertools import chain
from typing import NamedTuple

from numba import njit
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather  # noqa

from neuroglancer.coordinate_space import CoordinateSpace

from ._db import INPUT_VIEW

logger = logging.getLogger(__name__)

class PolylineGeometry(NamedTuple):
    """
    Bundle of arrays that describes a batch of polyline annotations after
    they've been unpacked from the user-supplied auxiliary table.

    - ``points``: (total_points, D) float32, all polyline vertices concatenated
      in main-df-row order. ``points[starts[i]:ends[i]]`` are the vertices of
      polyline ``i`` in traversal order.
    - ``starts``, ``ends``: (N,) int64 offsets into ``points``.
    - ``annotation_ids``: (N,) uint64 ID for each polyline. Streaming writers
      use this to look up a polyline's row position from its annotation_id at
      per-batch encode time (see :func:`_slice_polyline_geom`).
    """
    points: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    annotation_ids: np.ndarray


def _property_recsize(property_specs):
    """
    Padded per-record property size in bytes, matching the layout
    produced by :func:`._encode._geometry_prop_df`.
    """
    prop_size = 0
    for spec in property_specs:
        if spec['type'] == 'rgb':
            prop_size += 3
        elif spec['type'] == 'rgba':
            prop_size += 4
        else:
            prop_size += np.dtype(spec['type']).itemsize
    return prop_size + ((4 - (prop_size % 4)) % 4)


def _slice_polyline_geom(polyline_geom, rows):
    """
    Return a :class:`PolylineGeometry` that selects ``polyline_geom``'s
    rows in the order given by ``rows`` (an int array of row positions).
    The ``points`` array is shared with the original (zero-copy);
    ``starts``, ``ends``, and ``annotation_ids`` are gathered.

    Useful when a streaming writer has pulled a batch of annotation_ids
    out of DuckDB and needs to hand the encoder a polyline_geom whose
    starts/ends are aligned with that batch's row order.
    """
    return PolylineGeometry(
        points=polyline_geom.points,
        starts=polyline_geom.starts[rows],
        ends=polyline_geom.ends[rows],
        annotation_ids=polyline_geom.annotation_ids[rows],
    )


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


def _construct_coord_space(coord_space):
    """
    This function produces a CoordinateSpace object from any of our accepted
    formats as explained in the docs for write_precomputed_annotations().

    Returns:
        CoordinateSpace
    """
    if isinstance(coord_space, CoordinateSpace):
        return coord_space

    if isinstance(coord_space, str):
        if coord_space != coord_space.lower() or len(set(coord_space)) != len(coord_space):
            raise ValueError(f"Invalid coordinate space: {coord_space!r}.")
        return CoordinateSpace(
            names=list(coord_space),
            units=['nm']*len(coord_space),
            scales=[1]*len(coord_space),
        )

    if isinstance(coord_space, list):
        if not all(isinstance(c, str) and c == c.lower() for c in coord_space):
            raise ValueError(f"Invalid coordinate space: {coord_space!r}.")
        return CoordinateSpace(
            names=coord_space,
            units=['nm']*len(coord_space),
            scales=[1]*len(coord_space),
        )

    if isinstance(coord_space, dict):
        if 'names' not in coord_space:
            return CoordinateSpace(json=coord_space)

        if not (coord_space.keys() <= {'names', 'units', 'scales', 'coordinate_arrays'}):
            raise ValueError(f"Invalid coordinate space: {coord_space!r}.")

        default_coord_space = {
            'names': coord_space['names'],
            'units': ['nm']*len(coord_space['names']),
            'scales': [1]*len(coord_space['names']),
        }
        return CoordinateSpace(**(default_coord_space | coord_space))

    raise ValueError(f"Invalid coordinate space: {coord_space!r}.")


def _get_bounds(con, coord_space, annotation_type, *, polyline_geom=None):
    """
    Determine the upper and lower bounds of the annotation geometry by
    aggregating across :data:`INPUT_VIEW` via DuckDB. The same code path
    handles pandas-registered DataFrames and Feather-backed views.

    For polylines the bounds come from ``polyline_geom.points`` directly
    (the vertex coordinates live in memory, not in the DuckDB view).

    Raises ValueError if any geometry value is NaN or NULL, which would
    propagate to an invalid bound in the info file and obscure a data
    error.
    """
    if annotation_type == 'polyline':
        bounds = (
            polyline_geom.points.min(axis=0).astype(np.float64),
            polyline_geom.points.max(axis=0).astype(np.float64),
        )
    else:
        bounds = _bounds_via_sql(con, coord_space, annotation_type)

    if np.any(np.isnan(bounds[0])) or np.any(np.isnan(bounds[1])):
        raise ValueError(
            f"Bounds contain NaN values: lower={bounds[0]}, upper={bounds[1]}. "
            "Check your input data for missing or invalid coordinate values."
        )
    return bounds


def _bounds_via_sql(con, coord_space, annotation_type):
    """
    Compute per-axis (lower, upper) bounds via a single DuckDB
    aggregation, plus a count of NaN/NULL geometry values for validation.

    The aggregation expression depends on annotation type:

    - point: MIN/MAX of each axis column.
    - line / axis_aligned_bounding_box: LEAST(MIN(a), MIN(b)) and
      GREATEST(MAX(a), MAX(b)) per axis.
    - ellipsoid: MIN(center - radius) / MAX(center + radius) per axis.
    """
    geom_cols_groups = _geometry_cols(coord_space.names, annotation_type)

    if annotation_type == 'point':
        all_cols = list(geom_cols_groups[0])
        lo_exprs = [f"MIN({c})" for c in all_cols]
        hi_exprs = [f"MAX({c})" for c in all_cols]
    elif annotation_type in ('line', 'axis_aligned_bounding_box'):
        a_cols, b_cols = geom_cols_groups
        all_cols = list(a_cols) + list(b_cols)
        lo_exprs = [f"LEAST(MIN({a}), MIN({b}))" for a, b in zip(a_cols, b_cols)]
        hi_exprs = [f"GREATEST(MAX({a}), MAX({b}))" for a, b in zip(a_cols, b_cols)]
    elif annotation_type == 'ellipsoid':
        center_cols, radius_cols = geom_cols_groups
        all_cols = list(center_cols) + list(radius_cols)
        lo_exprs = [f"MIN({c} - {r})" for c, r in zip(center_cols, radius_cols)]
        hi_exprs = [f"MAX({c} + {r})" for c, r in zip(center_cols, radius_cols)]
    else:
        raise ValueError(f"Annotation type {annotation_type} not supported")

    nan_check = " OR ".join(f"isnan({c}) OR {c} IS NULL" for c in all_cols)
    select = ", ".join(
        lo_exprs + hi_exprs
        + [f"COUNT(*) FILTER (WHERE {nan_check})"]
    )
    row = con.execute(f"SELECT {select} FROM {INPUT_VIEW}").fetchone()

    rank = len(lo_exprs)
    lo = np.array(row[:rank], dtype=np.float64)
    hi = np.array(row[rank:2*rank], dtype=np.float64)
    nan_count = int(row[2*rank])

    if nan_count:
        raise ValueError(
            f"Geometry columns contain {nan_count} NaN/NULL value(s). "
            "Check your input for missing or invalid coordinate values."
        )
    return (lo, hi)


def _classify_input(df, annotation_type, polyline_points, properties, relationships):
    """
    Resolve the polymorphic first argument into one of ``(input_df,
    input_path)``, exactly one of which is non-None on return.

    - ``pd.DataFrame``: returned as ``(df, None)``.
    - ``str`` / ``os.PathLike``: returned as ``(None, path_str)`` so
      downstream code knows to use DuckDB's ``read_ipc`` rather than
      pandas registration.
    - ``None``: only valid for ``annotation_type='polyline'`` with no
      properties or relationships; we synthesize a column-less main
      table from ``polyline_points``'s unique annotation_ids.

    Also enforces the static invariants on ``polyline_points`` (required
    for polyline; forbidden otherwise; must be a pandas DataFrame).
    """
    if annotation_type == 'polyline':
        if polyline_points is None:
            raise ValueError("polyline_points must be provided for annotation_type='polyline'")
        if not isinstance(polyline_points, pd.DataFrame):
            raise TypeError("polyline_points must be a pandas DataFrame")
    elif polyline_points is not None:
        raise ValueError("polyline_points may only be provided for annotation_type='polyline'")

    if isinstance(df, pd.DataFrame):
        return df, None
    if isinstance(df, (str, os.PathLike)):
        return None, os.fspath(df)
    if df is None:
        if annotation_type != 'polyline':
            raise ValueError(
                "df=None is only valid for annotation_type='polyline' "
                "(used as a convenience when there are no properties or relationships)."
            )
        if properties:
            raise ValueError("Cannot pass properties=... when df is None.")
        if relationships:
            raise ValueError("Cannot pass relationships=... when df is None.")
        unique_ids = pd.unique(polyline_points['annotation_id'])
        return pd.DataFrame(index=pd.Index(unique_ids)), None
    raise TypeError(
        f"Expected a pandas DataFrame, Feather path, or None for df; "
        f"got {type(df).__name__}"
    )


def _resolve_polyline_geometry(input_df, input_path, annotation_type, polyline_points, coord_space):
    """
    Build the :class:`PolylineGeometry` for polyline writes and align it
    with the main table.

    For the in-memory (pandas) case we filter ``input_df`` in place;
    for the Feather case we instead return ``needs_polyline_filter=True``
    so the caller can call ``restrict_input_to_ids`` after registering
    the view in DuckDB (we can't filter the file itself).

    For non-polyline annotations this is a no-op.
    """
    if annotation_type != 'polyline':
        return None, input_df, False

    if input_df is not None:
        main_index = input_df.index
    else:
        # Read just the annotation_id column from the Feather file. For
        # 300M rows this is a few-seconds I/O pass that gives us the
        # ordered keys we need to align polyline_points against the
        # main table.
        ann_ids = (
            pa.feather.read_table(input_path, columns=['annotation_id'])
            .column('annotation_id')
            .to_numpy(zero_copy_only=False)
        )
        main_index = pd.Index(ann_ids)

    polyline_geom, valid_mask = _polyline_aux_to_arrays(
        polyline_points, main_index, coord_space.names
    )

    needs_polyline_filter = False
    if not valid_mask.all():
        if input_df is not None:
            input_df = input_df.loc[valid_mask].copy()
        else:
            needs_polyline_filter = True
    return polyline_geom, input_df, needs_polyline_filter


def _schema_sample(input_df, input_path):
    """
    Return a zero-row pandas DataFrame whose columns and dtypes (and
    categorical levels, when present) match the user's input. Used by
    :func:`annotation_property_specs` to infer property types without
    materializing the full Feather file.

    Pandas input passes through unchanged -- inspecting ``.columns`` and
    column dtypes on a full DataFrame is just as cheap as on a slice.
    """
    if input_df is not None:
        return input_df
    if input_path is None:
        # df=None polyline case without properties (validated upstream).
        return pd.DataFrame()
    return pa.feather.read_table(input_path).slice(0, 0).to_pandas()




def _polyline_aux_to_arrays(aux_df, main_index, coord_names):
    """
    Convert the user-supplied auxiliary polyline-points table into the flat
    numpy arrays the encoder and spatial kernel need.

    The aux table has one row per vertex with columns ``[*coord_names, 'annotation_id']``.
    Within each annotation, vertex order in the aux table defines polyline traversal order.

    Returns:
        polyline_geom:
            :class:`PolylineGeometry` whose ``points`` array is in stable-sorted
            annotation_id order (so each annotation's vertices are contiguous,
            preserving their input order within the group). ``starts``/``ends``
            are aligned with main-df row order, filtered to rows that have at
            least one vertex.
        valid_mask:
            (N,) bool, True for main-df rows with at least one vertex. Callers
            should ``df.loc[valid_mask]`` before downstream processing.
    """
    if 'annotation_id' not in aux_df.columns:
        raise ValueError("polyline_points must have an 'annotation_id' column.")
    missing = [c for c in coord_names if c not in aux_df.columns]
    if missing:
        raise ValueError(f"polyline_points is missing coordinate columns: {missing}")

    aux_df = aux_df.sort_values('annotation_id', kind='stable')
    aux_ids = aux_df['annotation_id'].to_numpy()

    if len(aux_ids) == 0:
        boundaries = np.array([0], dtype=np.int64)
    else:
        boundaries = np.concatenate((
            [0],
            np.flatnonzero(aux_ids[1:] != aux_ids[:-1]) + 1,
            [len(aux_ids)],
        )).astype(np.int64)

    unique_aux_ids = aux_ids[boundaries[:-1]]
    aux_slot_per_main = pd.Index(unique_aux_ids).get_indexer(main_index)
    valid_mask = aux_slot_per_main >= 0

    n_unused = int((~valid_mask).sum())
    if n_unused:
        logger.warning(
            f"{n_unused} of {len(main_index)} main-table annotations have no "
            f"vertices in polyline_points; those annotations will be dropped."
        )

    valid_slots = aux_slot_per_main[valid_mask]
    starts = boundaries[:-1][valid_slots]
    ends = boundaries[1:][valid_slots]
    valid_annotation_ids = np.asarray(main_index[valid_mask], dtype=np.uint64)

    points = aux_df[list(coord_names)].to_numpy(np.float32, copy=False)
    if np.isnan(points).any():
        raise ValueError("polyline_points contains NaN coordinate values.")

    geom = PolylineGeometry(
        points=points, starts=starts, ends=ends,
        annotation_ids=valid_annotation_ids,
    )
    return geom, valid_mask
