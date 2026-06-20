import os
import json
import logging
from typing import Literal, Union

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.feather  # noqa

from neuroglancer.coordinate_space import CoordinateSpace
from neuroglancer.viewer_state import AnnotationPropertySpec

from ..util import annotation_property_specs
from ._db import INPUT_VIEW, open_connection, register_input, restrict_input_to_ids
from ._memory import log_memory
from ._util import _drop_unused_columns, _geometry_cols, PolylineGeometry
from ._id import _write_annotations_by_id
from ._relationships import _write_annotations_by_relationships
from ._spatial import _write_annotations_by_spatial_chunk
from ._write_buffers import _build_ts_context, _default_max_threads

logger = logging.getLogger(__name__)


def write_precomputed_annotations(
    df: Union[pd.DataFrame, str, os.PathLike, None],
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
    num_spatial_levels: int = 64,
    target_chunk_limit: int = 10_000,
    shuffle_spatial_ordering: bool = True,
    max_threads: int | None = None,
    max_shards_per_transaction: int | None = None,
    duckdb_memory_limit: str | None = None,
    duckdb_temp_directory: str | None = None,
    tensorstore_context: dict | None = None,
    description: str = "",
):
    """
    Export the data from a pandas DataFrame into neuroglancer's precomputed annotations format
    as described in the `neuroglancer spec <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md>`_.

    A progress bar is shown when writing each portion of the export (annotation ID index, related ID indexes),
    but there may be a significant amount of preprocessing time that occurs before the actual writing begins.

    Note:

        Internally the writers stream through the input in shard-aligned
        batches via DuckDB, so peak RAM during the encode+write phase is
        roughly one batch's worth of encoded bytes rather than the full
        dataset. To take maximum advantage of this for large inputs,
        pass the path to a Feather/Arrow IPC file in ``df`` rather than
        a fully-materialized pandas DataFrame.

    Args:
        df:
            The annotations table. Accepted forms:

            - **pandas DataFrame**: the DataFrame's index is used as the
              annotation ID and must be unique. Columns supply geometry,
              properties, and relationships per the rules below.
            - **Path-like (str or os.PathLike)**: a Feather/Arrow IPC file
              carrying the same columns as the DataFrame form. The
              annotation ID is taken from an explicit ``annotation_id``
              column if present; otherwise from a pandas-index column
              recorded in the file's schema metadata (so a file written
              via ``df.to_feather(path)`` just works); otherwise
              synthesized as ``0, 1, 2, ...``. The data is streamed from
              the file via DuckDB and never fully materialized in pandas
              memory.
            - **None**: only valid for ``annotation_type='polyline'`` when
              you have no properties or relationships -- the main table
              is synthesized from the unique annotation IDs in
              ``polyline_points``.

            Required geometry columns depend on the annotation_type and
            the coordinate space. For example, assuming
            ``coord_space.names == ['x', 'y', 'z']``:

            - For point annotations, provide ['x', 'y', 'z']
            - For line annotations or axis_aligned_bounding_box annotations,
              provide ['xa', 'ya', 'za', 'xb', 'yb', 'zb']
            - For ellipsoid annotations, provide ['x', 'y', 'z', 'rx', 'ry', 'rz']
              for the center point and radii.
            - For polyline annotations, do not provide x/y/z columns here.
              Instead, provide them in the ``polyline_points`` argument.

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
            must be ``None`` otherwise. (Feather input is not supported
            here -- the polyline aux table must be an in-memory pandas
            DataFrame.)

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
            The default allows up to 64 levels (at least 9e18 spatial subdivisions at the finest level),
            which far exceeds the max that any real dataset would need.

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

        shuffle_spatial_ordering:
            bool
            Whether to randomize the spatial assignment. When True (the
            default), two things happen randomly:

            (a) which level each annotation lands at is uniformly random,
                so coarse levels carry a uniform random sample of all
                annotations -- the neuroglancer spec recommendation.

            (b) the within-chunk order is also random, so that
                neuroglancer's prefix-based subsampling (it draws the
                first N annotations from a chunk's stored list) produces
                an unbiased sample at any zoom level.

            When False, both orderings use the input row order: earlier
            input rows go to coarser levels, and within each chunk
            annotations are stored in input row order. Set this False
            when you have deliberately ordered your input (e.g. by
            importance) and want neuroglancer to render the most
            important annotations first.

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

        duckdb_memory_limit:
            str or None
            Forwarded to DuckDB's ``memory_limit`` setting (e.g.
            ``'40GB'``) when opening the connection used for all
            shard-streamed writes. DuckDB will spill to its temp
            directory once its working set exceeds this value, which
            caps DuckDB's contribution to peak RAM at the cost of some
            extra I/O. Defaults to ``None``, which lets DuckDB pick its
            own limit (~80% of system RAM).

        duckdb_temp_directory:
            str or None
            Forwarded to DuckDB's ``temp_directory`` setting -- the
            location used for spill files when DuckDB's working set
            exceeds ``duckdb_memory_limit``. Defaults to ``None``,
            which leaves DuckDB on its own default of ``.tmp/`` under
            the process's current working directory. Set this to a
            fast local-scratch path (e.g. ``/scratch/...``) when
            running on a cluster node whose CWD is on a slow shared
            filesystem.

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
    if write_by_spatial_chunk and target_chunk_limit == 0 and num_spatial_levels != 1:
        raise ValueError(
            "target_chunk_limit=0 disables subsampling and is only valid "
            "with num_spatial_levels=1."
        )
    if num_spatial_levels > 64:
        raise ValueError(
            "num_spatial_levels must be less than or equal to 64."
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

    # Normalize the user's first argument into one of two shapes:
    # ``input_df`` (in-memory pandas) or ``input_path`` (Feather file
    # path streamed through DuckDB). For the polyline-only convenience
    # path (df=None), we synthesize a column-less input_df from the
    # aux table's unique annotation_ids below.
    input_df, input_path = _classify_input(df, annotation_type, polyline_points,
                                           properties, relationships)

    # Resolve polyline geometry. For the in-memory cases we may filter
    # ``input_df`` to drop orphan annotations; for the Feather path we
    # defer the equivalent filter to DuckDB (after the view is
    # registered) via ``restrict_input_to_ids``.
    polyline_geom, input_df, needs_polyline_filter = _resolve_polyline_geometry(
        input_df, input_path, annotation_type, polyline_points, coord_space,
    )

    # Property specs are inferred from a schema-only sample so we don't
    # have to load the full Feather file. For pandas input the "sample"
    # is just the full df (zero overhead -- only ``columns`` and dtype
    # metadata get inspected).
    schema_sample = _schema_sample(input_df, input_path)
    property_specs = annotation_property_specs(schema_sample, properties)
    if input_df is not None:
        # ``_drop_unused_columns`` is a pandas memory hygiene step; the
        # Feather path doesn't carry full data in pandas to begin with,
        # so this is a no-op there.
        input_df = _drop_unused_columns(input_df, coord_space, annotation_type, property_specs, relationships)

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # The by-id, by-rel, and by-spatial writers all use DuckDB-backed
    # streaming: they query one batch of annotations per tensorstore
    # transaction, encode that batch on the fly, and discard before the
    # next batch runs. Peak RAM during these phases is one batch's worth
    # of encoded bytes rather than the full encoded payload.
    by_id_metadata = {}
    by_rel_metadata = []
    spatial_metadata = []
    con = open_connection(
        memory_limit=duckdb_memory_limit,
        threads=max_threads,
        temp_directory=duckdb_temp_directory,
    )
    try:
        if input_df is not None:
            register_input(con, input_df)
        else:
            register_input(con, input_path)
        if needs_polyline_filter:
            restrict_input_to_ids(con, polyline_geom.annotation_ids)

        bounds = _get_bounds(con, coord_space, annotation_type, polyline_geom=polyline_geom)
        log_memory('after input registration + bounds')

        if write_by_id:
            by_id_metadata = _write_annotations_by_id(
                con, coord_space, annotation_type, property_specs, relationships, polyline_geom,
                output_dir, write_sharded, max_shards_per_transaction, ts_context,
            )
        if write_by_relationship:
            by_rel_metadata = _write_annotations_by_relationships(
                con, coord_space, annotation_type, property_specs, relationships, polyline_geom,
                output_dir, write_sharded, max_shards_per_transaction, ts_context,
            )
        if write_by_spatial_chunk:
            spatial_metadata = _write_annotations_by_spatial_chunk(
                con, input_df,
                coord_space, annotation_type, property_specs, polyline_geom,
                bounds, num_spatial_levels, target_chunk_limit,
                shuffle_spatial_ordering,
                disable_subsampling=(target_chunk_limit == 0),
                output_dir=output_dir,
                write_sharded=write_sharded,
                max_shards_per_transaction=max_shards_per_transaction,
                ts_context=ts_context,
            )
    finally:
        con.close()

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
