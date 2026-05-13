import os
import json
import logging
from itertools import chain
from typing import Literal

import pandas as pd
import numpy as np

from neuroglancer.coordinate_space import CoordinateSpace
from neuroglancer.viewer_state import AnnotationPropertySpec

from ..util import annotation_property_specs
from ._util import _drop_unused_columns, _geometry_cols, PolylineGeometry
from ._id import _write_annotations_by_id
from ._relationships import _write_annotations_by_relationships
from ._spatial import _compute_spatial_assignment, _write_annotations_by_spatial_chunk
from ._write_buffers import _build_ts_context, _default_max_threads

logger = logging.getLogger(__name__)


def write_precomputed_annotations(
    df: pd.DataFrame | None,
    coord_space: CoordinateSpace | str | list[str] | dict[str, list],
    annotation_type: Literal['point', 'line', 'ellipsoid', 'axis_aligned_bounding_box', 'polyline'],
    properties: list[str] | list[AnnotationPropertySpec] | dict[str, AnnotationPropertySpec] | list[dict] = (),
    relationships: list[str] = (),
    output_dir: str = 'annotations',
    write_sharded: bool = True,
    *,
    polyline_points: pd.DataFrame | None = None,
    write_by_id: bool = True,
    write_by_relationship: bool = True,
    write_by_spatial_chunk: bool = True,
    num_spatial_levels: int = 7,
    target_chunk_limit: int = 10_000,
    shuffle_before_assigning_spatial_levels: bool = True,
    max_threads: int | None = None,
    max_shards_per_transaction: int | None = None,
    tensorstore_context: dict | None = None,
    description: str = "",
):
    """
    Export the data from a pandas DataFrame into neuroglancer's precomputed annotations format
    as described in the `neuroglancer spec <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md>`_.

    A progress bar is shown when writing each portion of the export (annotation ID index, related ID indexes),
    but there may be a significant amount of preprocessing time that occurs before the actual writing begins.

    Note:

        Internally, the data will be copied during processing and again
        during writing, incurring significant RAM usage for large datasets.

    Args:
        df:
            pandas DataFrame.
            The index of the DataFrame is used as the annotation ID, so it must be unique.
            The required columns depend on the annotation_type and the coordinate space.
            For example, assuming ``coord_space.names == ['x', 'y', 'z']``,
            then provide the following columns:

            - For point annotations, provide ['x', 'y', 'z']
            - For line annotations or axis_aligned_bounding_box annotations,
              provide ['xa', 'ya', 'za', 'xb', 'yb', 'zb']
            - For ellipsoid annotations, provide ['x', 'y', 'z', 'rx', 'ry', 'rz']
              for the center point and radii.
            - For polyline annotations, do not provide x/y/z columns here.
              Instead, provide them in the ``polyline_points`` argument.
              If your polyline annotations have no properties or relationships,
              you may set df to None and pass only polyline_points.

            You may also provide additional columns to use as annotation properties, in which
            case their column names should be listed in the 'properties' argument. (See below.)

        coord_space:
            ``neuroglancer.coordinate_space.CoordinateSpace`` or equivalent.
            The coordinate space of the annotations.
            Among other things, this determines which input columns represent the annotation geometry.
            For convenience, we accept a couple different formats for the coordinate space,
            assuming a default scale of 1 nm if no scale/units are provided.

            Examples (all equivalent):

            .. code-block:: python

                >>> coord_space = "xyz"
                >>> coord_space = ['x', 'y', 'z']
                >>> coord_space = {"names": ['x', 'y', 'z']}
                >>> coord_space = {
                    "names": ['x', 'y', 'z'],
                    "units": ['nm', 'nm', 'nm'],
                    "scales": [1, 1, 1]
                }
                >>> coord_space = CoordinateSpace(
                ...     names=['x', 'y', 'z'],
                ...     scales=[1.0, 1.0, 1.0],
                ...     units=['nm', 'nm', 'nm']
                ... )

        annotation_type:
            Literal['point', 'line', 'ellipsoid', 'axis_aligned_bounding_box', 'polyline']
            The type of annotation to export. Note that the columns you provide in
            the DataFrame depend on the annotation type.

        properties:
            If your dataframe contains columns for annotation properties,
            list the names of those columns here.

            Categorical columns will be automatically converted to integers with associated
            enum labels.

            To provide an rgb or rgba property such as 'mycolor', provide separate columns
            in your dataframe named 'mycolor_r', 'mycolor_g', 'mycolor_b' (and 'mycolor_a'),
            and then include 'mycolor' in the properties list here.

            The full property spec for each property will be inferred from the column dtype,
            but if you want to explicitly override any property specs yourself, you can pass
            a list of AnnotationPropertySpec objects here instead of just listing column names.

            Property names must start with a lowercase letter and may contain only letters,
            numbers, and underscores.

        relationships:
            list[str]
            If your annotations have related segment IDs, such relationships can be provided
            in the columns of your DataFrame. Each relationship should be listed in a single column,
            whose values are lists of segment IDs.  In the special case where each annotation has
            exactly one related segment, the column may have dtype=np.uint64 instead of containing lists.

        output_dir:
            str
            The directory into which the exported annotations will be written.
            Subdirectories will be created for the "annotation ID index" and each
            "related object id index" as needed.

        write_sharded:
            bool
            Whether to write the output as sharded files.
            The sharded format is preferable for most use cases.
            Without sharding, every annotation results in a separate file in the annotation ID index.
            Similarly, every related ID results in a separate file in the related ID index.

        polyline_points:
            pandas DataFrame. Required when ``annotation_type='polyline'``;
            must be ``None`` otherwise.

            One row per polyline vertex, with one column per coordinate axis
            plus an ``'annotation_id'`` column indicating which polyline each
            vertex belongs to. For example, assuming ``coord_space.names == ['x', 'y', 'z']``,
            then provide the following columns: ['annotation_id', 'x', 'y', 'z'].
            (For a polyline with N vertices, its annotation_id should appear N times.)

            For each polyline, the point order in the emitted annotation will match
            the order in which they appear in this dataframe.

        write_by_id:
            bool
            Whether to write the annotations to the "Annotation ID Index".
            If False, skip writing.

        write_relationships:
            bool
            Whether to write the relationships to the "Related Object ID Index".
            If False, skip writing.

        write_by_spatial_chunk:
            bool
            Whether to write the spatial index.

        num_spatial_levels:
            int
            The maximum number of spatial index levels to write.
            If not all levels are needed (because all annotations fit within the first N levels),
            then the actual number of levels written will be less than this value.

        target_chunk_limit:
            int
            For the spatial index, this is how many annotations we aim to place in each
            chunk (regardless of the level).
            If there are more annotations than fit within the specified num_spacial_levels
            while (approximately) adhering to the target_chunk_limit at each level, then the
            extra annotations will be assigned to the last level.

            Note:
                Instead of specifying a valid limit here, you can disable subsampling in neuroglancer
                by setting this to the special value of 0.  In our implementation, this is only valid
                when num_spatial_levels=1.

        shuffle_before_assigning_spatial_levels:
            bool
            Whether to shuffle the annotations before assigning spatial levels.
            By default, we shuffle the annotations to avoid any bias in the spatial
            assignment, which is what the neuroglancer spec recommends.
            However, in some use-cases a bias may be desirable (e.g. deliberately
            preferring to show larger annotations when zoomed out).
            So if this is False, the annotations will be assigned to spatial levels in
            the order they appear in the input dataframe, with earlier annotations
            assigned to coarser spatial levels.

        max_threads:
            int or None
            Default cap on tensorstore's data-copy and file-I/O thread
            pools. Used to populate the ``data_copy_concurrency`` and
            ``file_io_concurrency`` keys of the tensorstore Context when
            ``tensorstore_context`` doesn't already specify them.
            Defaults to ``LSB_DJOB_NUMPROC`` on LSF clusters, otherwise
            ``multiprocessing.cpu_count()``.

        max_shards_per_transaction:
            int or None
            (Sharded mode only.) Caps the number of shards committed in a
            single tensorstore transaction. Tensorstore parallelizes the
            per-shard work (encode, compress, write) inside a transaction
            across its internal thread pool, so this knob trades RAM (more
            shards staged in memory at once) for throughput and effective
            CPU utilization (more parallel work available at commit).

            Defaults to ``max_threads`` so each transaction can saturate
            the available threads. Set higher for better throughput at
            extra RAM cost, or lower to reduce peak RAM.

        tensorstore_context:
            dict or None
            Optional JSON spec for the tensorstore
            `Context <https://google.github.io/tensorstore/context.html>`_
            used to open every kvstore in this run. Useful for tuning
            resource pool sizes (e.g. ``cache_pool.total_bytes_limit``)
            to cap peak RAM during sharded writes.

            Any key you supply is passed through verbatim;
            ``data_copy_concurrency`` and ``file_io_concurrency`` are
            filled in from ``max_threads`` only when you haven't
            already specified them.

        description:
            str
            A description of the annotation collection.
    """
    if write_by_spatial_chunk and num_spatial_levels == 0:
        raise ValueError(
            "If you want to write the spatial index, you must "
            "specify a non-zero value for num_spatial_levels."
        )

    # Resolve concurrency parameters once so all index writes share the
    # same batching and thread-cap behavior.
    if max_threads is None:
        max_threads = _default_max_threads()
    if max_shards_per_transaction is None:
        max_shards_per_transaction = max_threads

    ts_context = _build_ts_context(tensorstore_context, max_threads)

    if write_sharded:
        logger.info(
            f"Sharded writes will use up to {max_shards_per_transaction} "
            f"shards per transaction with up to {max_threads} tensorstore threads."
        )

    annotation_type = annotation_type.lower()
    coord_space = _construct_coord_space(coord_space)

    df, polyline_geom = _resolve_polyline_inputs(
        df, annotation_type, polyline_points, coord_space, properties, relationships
    )

    # Compute property specs, drop unused columns up front, then derive
    # bounds. Dropping unused columns first reduces RAM pressure during
    # bounds/encoding when the input df carries extra columns we don't
    # consume; and it makes ``df`` thereafter a shallow view that holds
    # only the data the writers need.
    property_specs = annotation_property_specs(df, properties)
    df = _drop_unused_columns(df, coord_space, annotation_type, property_specs, relationships)
    bounds = _get_bounds(df, coord_space, annotation_type, polyline_geom=polyline_geom)

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Each writer encodes its own bytes lazily from the native ``df`` (plus
    # polyline_geom for polyline). This trades the small cost of re-encoding
    # in each writer for a large RAM saving: nothing persists between
    # writers except the native-dtype DataFrame, whose per-row Python-object
    # overhead is zero.
    by_id_metadata = {}
    if write_by_id:
        by_id_metadata = _write_annotations_by_id(
            df, coord_space, annotation_type, property_specs, relationships, polyline_geom,
            output_dir, write_sharded, max_shards_per_transaction, ts_context,
        )

    by_rel_metadata = []
    if write_by_relationship:
        by_rel_metadata = _write_annotations_by_relationships(
            df, coord_space, annotation_type, property_specs, relationships, polyline_geom,
            output_dir, write_sharded, max_shards_per_transaction, ts_context,
        )

    spatial_metadata = []
    if write_by_spatial_chunk:
        spatial_metadata = _write_annotations_by_spatial_chunk(
            df.drop(columns=list(relationships)),
            coord_space, annotation_type, property_specs, polyline_geom,
            bounds, num_spatial_levels, target_chunk_limit,
            shuffle_before_assigning_spatial_levels,
            disable_subsampling=(target_chunk_limit == 0),
            output_dir=output_dir,
            write_sharded=write_sharded,
            max_shards_per_transaction=max_shards_per_transaction,
            ts_context=ts_context,
        )

    polyline_geom = None


    # Write the top-level 'info' file for the annotation output directory.
    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": coord_space.to_json(),
        "lower_bound": bounds[0].tolist(),
        "upper_bound": bounds[1].tolist(),
        "annotation_type": annotation_type,
        "properties": property_specs,
        "by_id": by_id_metadata,
        "relationships": by_rel_metadata,
        "spatial": spatial_metadata,
    }

    if description:
        info['description'] = description

    with open(f"{output_dir}/info", 'w') as f:
        json.dump(info, f)


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


def _get_bounds(df, coord_space, annotation_type, *, polyline_geom=None):
    """
    Inspect the geometry columns of the given dataframe (or, for polylines,
    the auxiliary points array) to determine the overall upper and lower
    bounds of the annotations.

    Also checks for the presence of NaN values in the geometry, and raises a
    ValueError if any are found.

    Returns:
        lower_bound, upper_bound
        (both numpy arrays of length 3)
    """
    bounds = None

    geometry_cols = _geometry_cols(coord_space.names, annotation_type)
    if not (required_cols := set(chain(*geometry_cols))) <= set(df.columns):
        raise ValueError(
            "Dataframe does not have all required geometry columns for the given coordinate space.\n"
            f"Required columns: {required_cols}"
        )

    if annotation_type == 'polyline':
        bounds = (
            polyline_geom.points.min(axis=0),
            polyline_geom.points.max(axis=0),
        )

    if annotation_type == 'point':
        points = df[geometry_cols[0]]
        bounds = (
            points.min(skipna=False).to_numpy(),
            points.max(skipna=False).to_numpy()
        )

    if annotation_type in ('line', 'axis_aligned_bounding_box'):
        points_a = df[geometry_cols[0]]
        points_b = df[geometry_cols[1]]
        bounds = (
            np.minimum(points_a.min(skipna=False).to_numpy(), points_b.min(skipna=False).to_numpy()),
            np.maximum(points_a.max(skipna=False).to_numpy(), points_b.max(skipna=False).to_numpy())
        )

    if annotation_type == 'ellipsoid':
        center = df[geometry_cols[0]].to_numpy()
        radii = df[geometry_cols[1]].to_numpy()
        bounds = np.asarray([
            (center - radii).min(axis=0),
            (center + radii).max(axis=0)
        ])

    if bounds is None:
        raise ValueError(f"Annotation type {annotation_type} not supported")

    # Check for NaN in bounds (which would produce invalid JSON in the info file)
    if np.any(np.isnan(bounds[0])) or np.any(np.isnan(bounds[1])):
        raise ValueError(
            f"Bounds contain NaN values: lower={bounds[0]}, upper={bounds[1]}. "
            "Check your input data for missing or invalid coordinate values."
        )

    return bounds


def _resolve_polyline_inputs(df, annotation_type, polyline_points, coord_space, properties, relationships):
    """
    Pre-processing for the polyline argument-handling conveniences.

    For ``annotation_type == 'polyline'``:
        - If the user passed the aux table as the first positional and omitted
          ``polyline_points``, swap them.
        - Validate that ``polyline_points`` is supplied.
        - Synthesize a column-less main df from the aux table's unique
          annotation IDs if ``df`` is None (the no-properties/no-relationships
          convenience path).
        - Build the flat point arrays the encoder and spatial kernel need.
        - Filter out main-df rows that have no vertices in the aux table.

    For other annotation types, validates that ``polyline_points`` is ``None``
    and returns ``(df, None)``.

    Returns:
        (df, polyline_geom) -- ``polyline_geom`` is a :class:`PolylineGeometry`
        for polyline annotations, ``None`` otherwise.
    """
    if annotation_type == 'polyline' and df is not None and polyline_points is None:
        polyline_points, df = df, None

    if annotation_type == 'polyline':
        if polyline_points is None:
            raise ValueError("polyline_points must be provided for annotation_type='polyline'")
    elif polyline_points is not None:
        raise ValueError("polyline_points may only be provided for annotation_type='polyline'")
    else:
        return df, None

    if not isinstance(polyline_points, pd.DataFrame):
        raise TypeError("polyline_points must be a pandas DataFrame")

    if df is None:
        if properties:
            raise ValueError("Cannot pass properties=... when the main table is None.")
        if relationships:
            raise ValueError("Cannot pass relationships=... when the main table is None.")
        unique_ids = pd.unique(polyline_points['annotation_id'])
        df = pd.DataFrame(index=pd.Index(unique_ids))

    polyline_geom, valid_mask = _polyline_aux_to_arrays(
        polyline_points, df.index, coord_space.names
    )
    if not valid_mask.all():
        df = df.loc[valid_mask].copy()

    return df, polyline_geom


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

    points = aux_df[list(coord_names)].to_numpy(np.float32, copy=False)
    if np.isnan(points).any():
        raise ValueError("polyline_points contains NaN coordinate values.")

    return PolylineGeometry(points=points, starts=starts, ends=ends), valid_mask
