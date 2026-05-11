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
from ._util import _encode_uint64_series, _geometry_cols, PolylineGeometry, TableHandle
from ._id import _write_annotations_by_id
from ._relationships import _write_annotations_by_relationships, _encode_relationships
from ._spatial import _compute_spatial_assignment, _write_annotations_by_spatial_chunk
from ._write_buffers import _default_max_threads

logger = logging.getLogger(__name__)


def write_precomputed_annotations(
    df: pd.DataFrame | TableHandle | None,
    coord_space: CoordinateSpace | str | list[str] | dict[str, list],
    annotation_type: Literal['point', 'line', 'ellipsoid', 'axis_aligned_bounding_box', 'polyline'],
    properties: list[str] | list[AnnotationPropertySpec] | dict[str, AnnotationPropertySpec] | list[dict] = (),
    relationships: list[str] = (),
    output_dir: str = 'annotations',
    write_sharded: bool = True,
    *,
    polyline_points: pd.DataFrame | TableHandle | None = None,
    write_by_id: bool = True,
    write_by_relationship: bool = True,
    write_by_spatial_chunk: bool = True,
    num_spatial_levels: int = 7,
    target_chunk_limit: int = 10_000,
    shuffle_before_assigning_spatial_levels: bool = True,
    max_threads: int | None = None,
    max_shards_per_transaction: int | None = None,
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
        To save at least some RAM, you can wrap your dataframe in a TableHandle
        and then delete your own reference to the dataframe before calling this function.
        The TableHandle's reference will be deleted internally as soon as possible
        (after the data is transformed for writing, before this function returns).

    Args:
        df:
            DataFrame or TableHandle.
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

            If you provide a TableHandle, the handle's reference will be unset before this
            function returns, deleting your data if you didn't retain a reference to it yourself.
            (If you do retain a reference, it defeats the point of using a TableHandle in the first place.)

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
            DataFrame or TableHandle. Required when ``annotation_type='polyline'``;
            must be ``None`` otherwise.

            One row per polyline vertex, with one column per coordinate axis
            plus an ``'annotation_id'`` column indicating which polyline each
            vertex belongs to. For example, assuming ``coord_space.names == ['x', 'y', 'z']``,
            then provide the following columns: ['annotation_id', 'x', 'y', 'z'].
            (For a polyline with N vertices, its annotation_id should appear N times.)
            
            For each polyline, the point order in the emitted annotation will match
            the order in which they appear in this dataframe.

            As with ``df``, you may pass a ``TableHandle`` so the reference can
            be released as soon as the table has been consumed.

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
            Caps tensorstore's internal thread pool (data-copy and file-I/O
            concurrency) when writing. Defaults to ``LSB_DJOB_NUMPROC`` on
            LSF clusters, otherwise ``multiprocessing.cpu_count()``.

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

    if isinstance(df, TableHandle):
        # Take ownership of the dataframe.
        handle, df = df, df.df
        handle.df = None

    property_specs = annotation_property_specs(df, properties)
    bounds = _get_bounds(df, coord_space, annotation_type, polyline_geom=polyline_geom)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = _drop_unused_columns(df, coord_space, annotation_type, property_specs, relationships)

    # Construct a buffer for each annotation and additional buffers
    # for each annotation's relationships, stored in new columns of df.
    df = _encode_annotations(
        df,
        coord_space,
        annotation_type,
        property_specs,
        relationships,
        polyline_geom=polyline_geom,
    )

    # The property columns have been folded into ``ann_buf`` and aren't
    # needed by any downstream writer. Drop them now to release potentially
    # tens of GB of property-column arrays before by_id/by_rel/by_spatial.
    df = df.drop(columns=_property_column_names(property_specs))

    # Compute the spatial assignment up-front (while geometry is still
    # available), then drop the geometry columns. The much smaller
    # (rows, codes, levels) arrays carry through the by_id and by_rel
    # phases instead of the full geometry, releasing further GB of RAM.
    if write_by_spatial_chunk:
        spatial_assignment = _compute_spatial_assignment(
            df,
            coord_space,
            annotation_type,
            bounds,
            num_spatial_levels,
            target_chunk_limit,
            shuffle_before_assigning_spatial_levels,
            polyline_geom=polyline_geom,
        )
        geom_cols = [*chain(*_geometry_cols(coord_space.names, annotation_type))]
        df = df.drop(columns=geom_cols)
        # Polyline geometry was held in flat numpy arrays; release them
        # now that spatial assignment is done.
        polyline_geom = None

    by_id_metadata = {}
    if write_by_id:
        by_id_metadata = _write_annotations_by_id(
            df,
            output_dir,
            write_sharded,
            max_shards_per_transaction,
            max_threads,
        )

    # Done with rel_buf (only needed when writing by_id).
    df = df.drop(columns=['rel_buf'], errors='ignore')

    if write_by_relationship:
        df_handle_for_rel = TableHandle(df)

    if write_by_spatial_chunk:
        df_handle_for_spatial = TableHandle(df.drop(columns=list(relationships)))

    # Delete our reference to df.
    # The TableHandles own the data now.
    del df

    by_rel_metadata = []
    if write_by_relationship:
        by_rel_metadata = _write_annotations_by_relationships(
            df_handle_for_rel,
            relationships,
            output_dir,
            write_sharded,
            max_shards_per_transaction,
            max_threads,
        )

    spatial_metadata = []
    if write_by_spatial_chunk:
        spatial_metadata = _write_annotations_by_spatial_chunk(
            df_handle_for_spatial,
            spatial_assignment,
            disable_subsampling=(target_chunk_limit == 0),
            output_dir=output_dir,
            write_sharded=write_sharded,
            max_shards_per_transaction=max_shards_per_transaction,
            max_threads=max_threads,
        )
    
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


def _encode_annotations(df, coord_space, annotation_type, property_specs, relationships, *, polyline_geom=None):
    """
    Returns a (shallow) copy of the dataframe with additional columns containing
    buffers for each encoded annotation and its encoded id and encoded relationships.
    """
    df = df.copy(deep=False)

    logger.info("Encoding annotation IDs")
    df['id_buf'] = _encode_uint64_series(df.index)

    logger.info("Encoding annotation geometries and properties")
    df['ann_buf'] = _encode_geometries_and_properties(
        df, coord_space, annotation_type, property_specs,
        polyline_geom=polyline_geom,
    )

    logger.info("Encoding relationships")
    rel_bufs = _encode_relationships(df, relationships)
    if rel_bufs is not None:
        df['rel_buf'] = rel_bufs

    return df


def _encode_geometries_and_properties(df, coord_space, annotation_type, property_specs, *, polyline_geom=None):
    """
    For each annotation in the given dataframe, encode its geometry columns (e.g. x,y,z)
    and property columns into a buffer, plus any padding that was necessary to align the
    buffer to a 4-byte boundary, per the neuroglancer spec.

    (In the precomputed format, geometry and properties always appear together,
    regardless of whether they're being written to the "Annotation ID Index",
    the "Related Object ID Index" or the "Spatial Index".)

    Returns:
        pd.Series of dtype=object, containing one buffer for each annotation.
    """
    if annotation_type == 'polyline':
        return _encode_polyline_geometries_and_properties(df, property_specs, polyline_geom)

    geometry_cols = _geometry_cols(coord_space.names, annotation_type)
    geometry_prop_df = _geometry_prop_df(df, geometry_cols, property_specs)
    buf, recsize = _encode_geometry_prop_df(geometry_prop_df, geometry_cols, property_specs)
    del geometry_prop_df

    # extract bytes from the appropriate slice for each record
    encoded_annotations = [buf[i*recsize:(i+1)*recsize] for i in range(len(df))]
    ann_bufs = pd.Series(encoded_annotations, index=df.index)
    return ann_bufs


def _resolve_polyline_inputs(df, annotation_type, polyline_points, coord_space, properties, relationships):
    """
    Pre-processing for the polyline argument-handling conveniences.

    For ``annotation_type == 'polyline'``:
        - If the user passed the aux table as the first positional and omitted
          ``polyline_points``, swap them.
        - Validate that ``polyline_points`` is supplied.
        - Wrap the aux table in a :class:`TableHandle` so we can drop the
          reference as soon as we're done with it.
        - Synthesize a column-less main df from the aux table's unique
          annotation IDs if ``df`` is None (the no-properties/no-relationships
          convenience path).
        - Take ownership of ``df`` if it arrived as a TableHandle (so we can
          index into it below).
        - Build the flat point arrays the encoder and spatial kernel need,
          then release the aux table.
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

    if isinstance(polyline_points, TableHandle):
        aux_handle = polyline_points
    elif isinstance(polyline_points, pd.DataFrame):
        aux_handle = TableHandle(polyline_points)
    else:
        raise TypeError("polyline_points must be a pandas DataFrame or TableHandle")

    if df is None:
        if properties:
            raise ValueError("Cannot pass properties=... when the main table is None.")
        if relationships:
            raise ValueError("Cannot pass relationships=... when the main table is None.")
        unique_ids = pd.unique(aux_handle.df['annotation_id'])
        df = pd.DataFrame(index=pd.Index(unique_ids))
    elif isinstance(df, TableHandle):
        # The general TableHandle unwrap in write_precomputed_annotations runs
        # after this helper, but we need df.index here, so consume it now.
        handle, df = df, df.df
        handle.df = None

    polyline_geom, valid_mask = _polyline_aux_to_arrays(
        aux_handle.df, df.index, coord_space.names
    )
    aux_handle.df = None
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


def _encode_polyline_geometries_and_properties(df, property_specs, polyline_geom):
    """
    Encode each polyline's variable-length geometry plus its (fixed-size)
    property record. See the polyline branch of "Single annotation encoding"
    in the neuroglancer precomputed annotations spec.
    """
    points = polyline_geom.points
    starts = polyline_geom.starts
    ends = polyline_geom.ends
    D = points.shape[1]
    point_byte_size = 4 * D  # float32 per axis

    flat_bytes = points.astype(np.float32, copy=False).tobytes()
    counts = (ends - starts).astype(np.uint32)
    counts_buf = counts.tobytes()

    if property_specs:
        property_only_df = _geometry_prop_df(df, [], property_specs)
        prop_buf, prop_recsize = _encode_geometry_prop_df(property_only_df, [], property_specs)
        del property_only_df
    else:
        prop_buf = b''
        prop_recsize = 0

    encoded_annotations = [
        counts_buf[i*4:(i+1)*4]
        + flat_bytes[int(starts[i])*point_byte_size : int(ends[i])*point_byte_size]
        + prop_buf[i*prop_recsize:(i+1)*prop_recsize]
        for i in range(len(df))
    ]
    return pd.Series(encoded_annotations, index=df.index)


def _geometry_prop_df(df, geometry_cols, property_specs):
    """
    Select the subset of columns that specify the geometry and properties
    of the annotations, and append columns for padding that will ensure
    our encoded records have a width that is a multiple of 4 bytes.
    """
    # Note that the property specs are already sorted by
    # dtype in the order neuroglancer requires.
    # Order our columns accordingly before we encode them as records below.
    prop_cols = []
    for spec in property_specs:
        p = spec['id']
        if spec['type'] == 'rgb':
            prop_cols.extend([f'{p}_r', f'{p}_g', f'{p}_b'])
        elif spec['type'] == 'rgba':
            prop_cols.extend([f'{p}_r', f'{p}_g', f'{p}_b', f'{p}_a'])
        else:
            prop_cols.append(p)

    geometry_prop_df = df[[*chain(*geometry_cols), *prop_cols]].copy(deep=False)

    # Calculate padding as required by neuroglancer.
    property_widths = {}
    for spec in property_specs:
        if spec['type'] == 'rgb':
            property_widths[spec['id']] = 3
        elif spec['type'] == 'rgba':
            property_widths[spec['id']] = 4
        else:
            property_widths[spec['id']] = np.dtype(spec['type']).itemsize

    property_padding = (4 - (sum(property_widths.values()) % 4)) % 4
    for i in range(property_padding):
        geometry_prop_df[f'__padding_{i}__'] = np.uint8(0)

    return geometry_prop_df


def _encode_geometry_prop_df(geometry_prop_df, geometry_cols, property_specs):
    """
    Encode the geometry and properties of the annotations into a single buffer.

    Note: Replaces category columns of geometry_prop_df with their integer equivalents.
    """
    # Convert our column dtypes to match property specs.
    dtypes = {c: np.float32 for c in chain(*geometry_cols)}
    for spec in property_specs:
        p = spec['id']
        if spec['type'] in ('rgb', 'rgba'):
            dtypes.update({
                f'{p}_{channel}': np.uint8
                for channel in spec['type']
            })
        else:
            dtypes[p] = spec['type']

    if any(dt == np.int8 for dt in dtypes.values()):
        logger.warning(
            "Old versions of neuroglancer don't support int8 properties, "
            "so consider casting to uint8 or int16 if your annotations don't load."
        )

    # Convert category columns to their integer equivalents
    for spec in property_specs:
        p = spec['id']
        if spec['type'] in ('rgb', 'rgba'):
            continue
        if geometry_prop_df[p].dtype == 'category':
            geometry_prop_df[p] = geometry_prop_df[p].cat.codes
            dtypes[p] = spec['type']

    # Vectorized serialization
    records = geometry_prop_df.to_records(index=False, column_dtypes=dtypes)
    recsize = records.dtype.itemsize
    buf = records.tobytes()
    return buf, recsize
