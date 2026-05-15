import logging
import os
from itertools import chain
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import tensorstore as ts
from numba import njit
from numba.typed import List

from . import _write_buffers
from .compressed_morton import compressed_morton_code, compressed_morton_decode, _compressed_morton_code_no_alloc
from ._db import INPUT_VIEW
from ._encode import (
    PartitionedBuffer,
    _build_grouped_record_buffers,
    _encode_annotation_records,
    _encode_id_bytes,
)
from ._memory import log_memory
from ._shard_audit import ShardWriteAuditor
from ._shard_hash import shards_for_keys
from ._write_buffers import (
    _open_sharded_kvstore,
    _prepare_output_subdir,
    _sharded_metadata,
    _write_one_transaction,
)
from ._util import (
    _ann_required_cols,
    _geometry_cols,
    _property_recsize,
    _slice_polyline_geom,
    _unravel_index,
    PolylineGeometry,
)

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

GridSpec = NamedTuple("GridSpec", [('chunk_shapes', np.ndarray), ('grid_shapes', np.ndarray)])

# Default rows per batch for streaming the geometry through the
# grid-code kernels. 10M rows × 24 bytes (line/aabb geometry) is ~240
# MB per batch, which keeps Feather-input peak RAM well below the full
# materialization while staying large enough that per-batch numba/JIT
# overhead is negligible.
_SPATIAL_KERNEL_BATCH_SIZE = 10_000_000


def _compute_spatial_assignment(
    con,
    input_df,
    polyline_geom,
    coord_space,
    annotation_type,
    bounds,
    num_levels,
    target_chunk_limit,
    shuffle_before_assigning_spatial_levels,
    *,
    batch_size=_SPATIAL_KERNEL_BATCH_SIZE,
):
    """
    Compute the spatial assignment: for each (annotation, chunk) pair,
    a triple ``(row, chunk_code, level)``.

    The returned DataFrame is sorted by ``(level, chunk_code)`` so each
    chunk's contributions are contiguous; within a chunk the original
    input row order is preserved (a stable sort), which matters because
    neuroglancer subsamples a spatial chunk by taking a prefix of its
    annotation list.

    Geometry source dispatch:

    - polyline: ``polyline_geom`` is used directly (single full-data
      kernel call; not batched -- see the polyline kernel for why).
    - pandas input: ``input_df`` carries the geometry columns. We iterate
      ``df.iloc`` slices to feed the kernel in batches; iloc is a
      zero-copy view, so this gives a unified code path without
      duplicating memory.
    - Feather input (``input_df is None``): geometry batches stream out
      of DuckDB via an Arrow record-batch reader, so the full geometry
      never materializes in pandas memory.

    Args:
        con:
            DuckDB connection. Used to count rows and (for Feather input)
            stream geometry batches.
        input_df:
            pandas DataFrame with geometry columns, or ``None`` to stream
            from DuckDB. Ignored for polyline annotations.
        polyline_geom:
            :class:`PolylineGeometry` for polyline annotations, else ``None``.
        coord_space, annotation_type, bounds, num_levels, target_chunk_limit:
            See :func:`write_precomputed_annotations`.
        shuffle_before_assigning_spatial_levels:
            If True, level assignments are randomized across the row
            ordering (per the neuroglancer spec recommendation). The
            shuffle is tracked via a permutation array; input data is
            not modified.
        batch_size:
            Rows per kernel batch (non-polyline). Caps per-batch RAM
            without affecting the result.

    Returns:
        ``(assignment_df, gridspec)`` -- ``assignment_df`` is a pandas
        DataFrame with columns ``('level', 'chunk_code', 'row_pos')``
        sorted by ``(level, chunk_code)``; ``gridspec`` is the
        :class:`GridSpec` used to assign chunks.
    """
    if annotation_type == 'polyline':
        n_total = len(polyline_geom.starts)
    elif input_df is not None:
        n_total = len(input_df)
    else:
        n_total = int(con.execute(f"SELECT COUNT(*) FROM {INPUT_VIEW}").fetchone()[0])

    gridspec = _define_spatial_grids(bounds, coord_space, num_levels)
    level_counts = _compute_target_annotations_per_level(n_total, gridspec, target_chunk_limit)

    # Levels are assigned by *position* in a (possibly randomly permuted)
    # ordering of the input rows. Computing the permutation explicitly
    # lets us derive the per-row level array without mutating any input.
    if shuffle_before_assigning_spatial_levels:
        logger.info("Shuffling annotations before assigning spatial grid levels")
        perm = np.random.permutation(n_total)
    else:
        perm = np.arange(n_total)
    levels_by_perm_position = np.repeat(range(num_levels), level_counts.astype(int)).astype(np.uint64)
    per_row_levels = np.empty(n_total, dtype=np.uint64)
    per_row_levels[perm] = levels_by_perm_position
    del perm, levels_by_perm_position

    logger.info("Assigning spatial grid chunks...")
    if annotation_type == 'polyline':
        rows, codes = _compute_grid_codes_for_polylines(
            polyline_geom, bounds, gridspec, per_row_levels,
        )
    else:
        rows, codes = _compute_grid_codes_batched(
            annotation_type, input_df, con, coord_space, bounds, gridspec,
            per_row_levels, n_total, batch_size,
        )
    logger.info("Done assigning spatial grid chunks")

    # Stable-sort by (level, chunk_code) using ``np.lexsort`` and then
    # reorder ``rows``, ``codes``, and ``levels`` one at a time. This
    # avoids constructing an unsorted 3-column DataFrame just to sort it
    # (which transiently doubles memory during ``pd.sort_values``).
    levels = per_row_levels[rows]
    order = np.lexsort((codes, levels))
    rows = rows[order]
    codes = codes[order]
    levels = levels[order]
    del order

    assignment_df = pd.DataFrame({
        'level': levels,
        'chunk_code': codes,
        'row_pos': rows,
    })
    return assignment_df, gridspec


def _compute_grid_codes_batched(annotation_type, input_df, con, coord_space, bounds, gridspec,
                                per_row_levels, n_total, batch_size):
    """
    Run the per-annotation grid-code kernel over the input geometry in
    batches and concatenate the (rows, codes) outputs.

    Each kernel call returns ``rows`` as positions local to the batch
    (0..batch_len); we offset by ``batch_start`` to recover global row
    indices into the full input. The output of the per-batch concat is
    identical to a single full-data kernel call.
    """
    geometry_cols = _geometry_cols(coord_space.names, annotation_type)
    n_batches = max(1, (n_total + batch_size - 1) // batch_size)
    logger.info(f"Computing grid codes for {n_total} {annotation_type} annotations "
                f"in {n_batches} batch(es) of up to {batch_size:,} rows")

    all_rows = []
    all_codes = []
    with tqdm(total=int(n_total)) as pbar:
        for batch_start, batch_end, batch_df in _iter_geom_batches(
            input_df, con, coord_space, annotation_type, n_total, batch_size,
        ):
            batch_levels = per_row_levels[batch_start:batch_end]
            rows, codes = _dispatch_grid_code_kernel(
                annotation_type, batch_df, geometry_cols, bounds, gridspec, batch_levels,
            )
            if batch_start:
                rows = rows + np.uint32(batch_start)
            all_rows.append(rows)
            all_codes.append(codes)
            pbar.update(batch_end - batch_start)

    return np.concatenate(all_rows), np.concatenate(all_codes)


def _dispatch_grid_code_kernel(annotation_type, batch_df, geometry_cols, bounds, gridspec, batch_levels):
    """Pick the right grid-code kernel for ``annotation_type``."""
    match annotation_type:
        case 'point':
            return _compute_grid_codes_for_points(batch_df, geometry_cols, bounds, gridspec, batch_levels)
        case 'axis_aligned_bounding_box':
            return _compute_grid_codes_for_axis_aligned_bounding_boxes(batch_df, geometry_cols, bounds, gridspec, batch_levels)
        case 'ellipsoid':
            return _compute_grid_codes_for_ellipsoids(batch_df, geometry_cols, bounds, gridspec, batch_levels)
        case 'line':
            return _compute_grid_codes_for_lines(batch_df, geometry_cols, bounds, gridspec, batch_levels)
    raise NotImplementedError(f"Spatial indexing for {annotation_type} annotations is not implemented")


def _iter_geom_batches(input_df, con, coord_space, annotation_type, n_total, batch_size):
    """
    Yield ``(batch_start, batch_end, batch_df)`` tuples covering rows
    ``[0, n_total)`` of the input.

    - Pandas input: ``df.iloc`` slices (zero-copy views).
    - Feather input (``input_df is None``): geometry-only Arrow batches
      streamed from DuckDB via ``to_arrow_reader``, converted to pandas
      per batch.

    Both paths produce DataFrames whose columns are the annotation
    type's geometry columns (no annotation_id, no properties); the
    kernels never need more than that.
    """
    if input_df is not None:
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            yield start, end, input_df.iloc[start:end]
        return

    geom_cols = list(chain(*_geometry_cols(coord_space.names, annotation_type)))
    select_cols = ', '.join(geom_cols)
    # ``to_arrow_reader`` returns a RecordBatchReader; each Arrow batch
    # is then converted to pandas. The reader is held open until
    # iteration completes -- no other DuckDB queries should run on this
    # connection in the meantime.
    reader = con.execute(
        f"SELECT {select_cols} FROM {INPUT_VIEW}"
    ).to_arrow_reader(batch_size=batch_size)
    start = 0
    for arrow_batch in reader:
        df_batch = arrow_batch.to_pandas(zero_copy_only=False)
        end = start + len(df_batch)
        yield start, end, df_batch
        start = end


def _define_spatial_grids(bounds, coord_space, num_levels: int) -> GridSpec:
    """
    Compute suitable chunk shapes and grid shapes for each level 
    of the spatial index, following the guidelines from the spec[1]:

        > Typically the grid_shape for level 0 should be a vector of all 1
        > (with chunk_size equal to upper_bound - lower_bound), and each component
        > of chunk_size of each successively level should be either equal to, or half of,
        > the corresponding component of the prior level chunk_size, whichever results
        > in a more spatially isotropic chunk.

    [1]: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#spatial-index

    Args:
        bounds:
            np.ndarray, shape (2, D)
            lower and upper bounds of the union of all annotations

        coord_space:
            Needed to aim for roughly isotropic chunks in physical units.

        num_levels:
            The number of spatial index levels. Must be at least 1.

    Returns:
        GridSpec(chunk_shapes, grid_shapes)

        - chunk_shapes is the array (for N levels) of the size of each
          grid cell at the corresponding level, in coordinate units.
        - grid_shapes is the array (for N levels) of the number of grid cells
          along each dimension at the corresponding level.

        For instance, level 0 consists of a single chunk encompassing the entire
        volume occupied by the annotations, so its chunk_shape is the entire bounds
        (offset by the lower bound) and its grid_shape is [1,1,...].
    """
    # Level 0 chunk shape and grid shape -- just one chunk.
    bounds = np.asarray(bounds, np.float64)

    # We want roughly isotropic chunks in physical units, so we'll multiply
    # by the coordinate scales and then divide the scales out at the end.
    chunk_shape = (bounds[1] - bounds[0]) * coord_space.scales
    grid_shape = np.ones_like(chunk_shape, dtype=np.uint64)

    chunk_shapes = [chunk_shape]
    grid_shapes = [grid_shape]

    for level in range(1, num_levels):
        chunk_shape = chunk_shape.copy()
        grid_shape = grid_shape.copy()

        max_dim = np.argmax(chunk_shape)
        target_width = chunk_shape[max_dim] / 2

        for dim, dim_width in enumerate(chunk_shape):
            if dim == max_dim:
                # Always split across the widest dimension.
                chunk_shape[dim] = target_width
                grid_shape[dim] *= 2
            elif (dim_width / target_width) > 1.5:
                # Split across this dimension to make it more isotropic.
                chunk_shape[dim] = dim_width / 2
                grid_shape[dim] *= 2
            else:
                # Splitting would make it less isotropic,
                # so leave this dimension unsplit.
                chunk_shape[dim] = dim_width
                grid_shape[dim] *= 1

        chunk_shapes.append(chunk_shape)
        grid_shapes.append(grid_shape)

    # Convert from physical units back to coordinate units.
    chunk_shapes = np.array(chunk_shapes) / coord_space.scales
    chunk_shapes = chunk_shapes.astype(np.float32)

    grid_shapes = np.array(grid_shapes)
    grid_shapes = grid_shapes.astype(np.min_scalar_type(grid_shapes.max()))

    return GridSpec(chunk_shapes, grid_shapes)


def _axis_bits_c_order(grid_shapes):
    """
    Return ``ceil(log2(grid_shape))`` per axis, in C-order (slowest-varying
    first), for every level. Suitable for passing to ``_compressed_morton_code_no_alloc``
    together with C-order grid coordinates.
    """
    return np.ceil(np.log2(grid_shapes[:, ::-1])).astype(np.int8)


def _compute_target_annotations_per_level(num_annotations, gridspec, target_chunk_limit: int):
    """
    Compute the TOTAL number of annotations at each level of the spatial index.
    The target_chunk_limit is how many annotations we aim to place in each chunk
    (regardless of the level).
    
    Since the spatial annotations are not necessarily distributed uniformly in space,
    we will likely end up undershooting and overshooting the target for various
    chunks within a level.

    Furthermore, since the number of annotations passed in here is based on the
    table BEFORE duplicating annotations which span multiple chunks, the number
    of annotations at each level will eventually be more than what is returned here,
    after the appropriate duplications.

    Returns:
        np.ndarray, shape (num_levels,)
    """
    num_levels = len(gridspec.grid_shapes)
    chunk_counts_by_level = np.prod(gridspec.grid_shapes, axis=1)

    if target_chunk_limit != 0:
        annotation_counts = chunk_counts_by_level * target_chunk_limit
    else:
        # target_chunk_limit=0 → no subsampling, single-level (validated
        # at the entry point). All annotations land at level 0.
        annotation_counts = np.array([num_annotations])
    
    # Clamp to total number of annotations remaining after earlier levels
    for level in range(num_levels - 1):
        annotation_counts[level] = min(
            annotation_counts[level],
            num_annotations - sum(annotation_counts[:level])
        )

    # Last level gets all remaining annotations, if any.
    annotation_counts[-1] = num_annotations - sum(annotation_counts[:-1])

    return annotation_counts


def _compute_grid_codes_for_points(df, geometry_cols, bounds, gridspec, per_row_levels):
    coord_names = geometry_cols[0]
    chunk_shape_per_row = gridspec.chunk_shapes[per_row_levels]
    grid_shape_per_row = gridspec.grid_shapes[per_row_levels]
    grid_indices = (df[[*coord_names]] - bounds[0]) // chunk_shape_per_row

    # Make sure annotations at the exact upper bound get valid grid coordinates.
    grid_indices = np.minimum(grid_indices, grid_shape_per_row - 1)
    grid_indices = grid_indices.astype(gridspec.grid_shapes.dtype)

    # Switch to C order before computing compressed morton code.
    codes = compressed_morton_code(
        grid_indices.to_numpy()[:, ::-1],
        grid_shape_per_row[:, ::-1],
    )
    # Points always fall in exactly one chunk, so rows is simply [0..N).
    return np.arange(len(df), dtype=np.uint32), np.asarray(codes, dtype=np.uint64)


def _compute_grid_codes_for_axis_aligned_bounding_boxes(df, geometry_cols, bounds, gridspec, per_row_levels):
    boxes = df[[*geometry_cols[0], *geometry_cols[1]]].to_numpy().reshape(len(df), 2, -1)

    # Ensure start < end
    swap_mask = (boxes[:, 0, :] > boxes[:, 1, :])[:, None, :]
    swap_mask = np.concatenate([swap_mask, swap_mask], axis=1)
    boxes[swap_mask] = boxes[:, ::-1, :][swap_mask]

    return _box_grid_codes(
        boxes,
        per_row_levels,
        bounds[0],
        gridspec.chunk_shapes,
        _axis_bits_c_order(gridspec.grid_shapes),
    )


@njit
def _box_grid_codes(boxes, levels, grid_origin, chunk_shapes, axis_bits_per_level):
    D = boxes.shape[2]

    # Pre-allocate these and reuse them on each loop iteration
    # to avoid heap allocations in the loop.
    grid_span = np.zeros((2, D), dtype=np.uint64)
    grid_span_shape = np.empty(D, dtype=np.uint64)
    grid_index = np.empty(D, dtype=np.uint64)
    curr_axis_pos = np.empty(D, dtype=np.uint64)

    rows = List()
    codes = List()

    for row, (box, level) in enumerate(zip(boxes, levels)):
        chunk_shape = chunk_shapes[level]
        ab = axis_bits_per_level[level]

        # We'd prefer the following, but we're worried about little allocations,
        # so below we loop over the dimensions explicitly.
        ## grid_span[0] = np.floor((box[0] - grid_origin) / chunk_shape)
        ## grid_span[1] = np.ceil((box[1] - grid_origin) / chunk_shape)
        ## grid_span_cell_count = np.prod(grid_span[1] - grid_span[0])

        # Compute per-axis grid-cell range.
        grid_span_cell_count = np.uint64(1)
        for d in range(D):
            grid_span[0, d] = np.uint64(np.floor((box[0, d] - grid_origin[d]) / chunk_shape[d]))
            grid_span[1, d] = np.uint64(np.ceil((box[1, d] - grid_origin[d]) / chunk_shape[d]))
            grid_span_shape[d] = grid_span[1, d] - grid_span[0, d]
            grid_span_cell_count *= grid_span_shape[d]

        # Scan across all cells in the span.
        for flat_index in range(grid_span_cell_count):
            _unravel_index(flat_index, grid_span_shape, grid_index)
            grid_index[:] += grid_span[0]
            code = _compressed_morton_code_no_alloc(grid_index[::-1], ab, curr_axis_pos)
            rows.append(row)
            codes.append(code)

    # Return as arrays rather than reflecting into Python lists.
    rows = np.asarray(rows, dtype=np.uint32)
    codes = np.asarray(codes, dtype=np.uint64)
    return rows, codes


def _compute_grid_codes_for_ellipsoids(df, geometry_cols, bounds, gridspec, per_row_levels):
    centroids = df[geometry_cols[0]].to_numpy()
    radii = df[geometry_cols[1]].to_numpy()

    return _ellipsoid_grid_codes(
        centroids,
        radii,
        per_row_levels,
        bounds[0],
        gridspec.chunk_shapes,
        _axis_bits_c_order(gridspec.grid_shapes),
    )


@njit
def _ellipsoid_grid_codes(centroids, radii, levels, grid_origin, chunk_shapes, axis_bits_per_level):
    D = centroids.shape[1]

    # Pre-allocate these and reuse them on each loop iteration
    # to avoid heap allocations in the loop.
    grid_span = np.zeros((2, D), dtype=np.uint64)
    grid_span_shape = np.empty(D, dtype=np.uint64)
    grid_index = np.empty(D, dtype=np.uint64)
    curr_axis_pos = np.empty(D, dtype=np.uint64)

    rows = List()
    codes = List()
    for row, (centroid, radius, level) in enumerate(zip(centroids, radii, levels)):
        chunk_shape = chunk_shapes[level]
        ab = axis_bits_per_level[level]

        # We'd prefer the following, but we're worried about little allocations,
        # so below we loop over the dimensions explicitly.
        ## grid_span[0] = np.floor((centroid - radius - grid_origin) / chunk_shape)
        ## grid_span[1] = np.ceil((centroid + radius - grid_origin) / chunk_shape)
        ## grid_span_cell_count = np.prod(grid_span[1] - grid_span[0])

        grid_span_cell_count = np.uint64(1)
        for d in range(D):
            grid_span[0, d] = np.uint64(np.floor((centroid[d] - radius[d] - grid_origin[d]) / chunk_shape[d]))
            grid_span[1, d] = np.uint64(np.ceil((centroid[d] + radius[d] - grid_origin[d]) / chunk_shape[d]))
            grid_span_shape[d] = grid_span[1, d] - grid_span[0, d]
            grid_span_cell_count *= grid_span_shape[d]

        # Scan across all cells in the span.
        for flat_index in range(grid_span_cell_count):
            _unravel_index(flat_index, grid_span_shape, grid_index)
            grid_index[:] += grid_span[0]
            if _ellipsoid_chunk_overlap(centroid, radius, grid_origin, chunk_shape, grid_index):
                code = _compressed_morton_code_no_alloc(grid_index[::-1], ab, curr_axis_pos)
                rows.append(row)
                codes.append(code)

    # Return as arrays rather than reflecting into Python lists.
    rows = np.asarray(rows, dtype=np.uint32)
    codes = np.asarray(codes, dtype=np.uint64)
    return rows, codes


@njit
def _ellipsoid_chunk_overlap(center, radii, grid_origin, cell_shape, grid_index):
    """
    Ported from the C++ implementation[1] by jbms, except that we just return
    a boolean indicating whether the ellipsoid and cell have any overlap (True)
    or are completely disjoint (False).

    [1]: https://github.com/google/neuroglancer/pull/522#issuecomment-1940516294
    """
    rank = len(center)
    min_sum = 0.0

    for i in range(rank):
        cell_size = cell_shape[i]
        cell_start = grid_index[i] * cell_size + grid_origin[i]
        cell_end = cell_start + cell_size
        center_pos = center[i]
        
        start_dist = abs(cell_start - center_pos)
        end_dist = abs(cell_end - center_pos)
        
        if center_pos >= cell_start and center_pos <= cell_end:
            min_distance = 0.0
        else:
            min_distance = min(start_dist, end_dist)
        
        min_sum += min_distance**2 / radii[i]**2
    
    return min_sum <= 1.0


def _compute_grid_codes_for_lines(df, geometry_cols, bounds, gridspec, per_row_levels):
    endpoints = df[[*geometry_cols[0], *geometry_cols[1]]].to_numpy().reshape(len(df), 2, -1)

    # Ensure start < end
    swap_mask = (endpoints[:, 0, :] > endpoints[:, 1, :])[:, None, :]
    swap_mask = np.concatenate([swap_mask, swap_mask], axis=1)
    endpoints[swap_mask] = endpoints[:, ::-1, :][swap_mask]

    return _line_grid_codes(
        endpoints,
        per_row_levels,
        bounds[0],
        gridspec.chunk_shapes,
        _axis_bits_c_order(gridspec.grid_shapes),
    )


@njit
def _line_grid_codes(endpoints, levels, grid_origin, chunk_shapes, axis_bits_per_level):
    D = endpoints.shape[2]

    # Pre-allocate these and reuse them on each loop iteration
    # to avoid heap allocations in the loop.
    grid_span = np.zeros((2, D), dtype=np.uint64)
    grid_span_shape = np.empty(D, dtype=np.uint64)
    grid_index = np.empty(D, dtype=np.uint64)
    curr_axis_pos = np.empty(D, dtype=np.uint64)

    rows = List()
    codes = List()
    for row, ((point_a, point_b), level) in enumerate(zip(endpoints, levels)):
        chunk_shape = chunk_shapes[level]
        ab = axis_bits_per_level[level]

        # We'd prefer the following, but we're worried about little allocations,
        # so below we loop over the dimensions explicitly.
        ## grid_span[0] = np.floor((point_a - grid_origin) / chunk_shape)
        ## grid_span[1] = np.ceil((point_b - grid_origin) / chunk_shape)
        ## grid_span_cell_count = np.prod(grid_span[1] - grid_span[0])

        grid_span_cell_count = np.uint64(1)
        for d in range(D):
            grid_span[0, d] = np.uint64(np.floor((point_a[d] - grid_origin[d]) / chunk_shape[d]))
            grid_span[1, d] = np.uint64(np.ceil((point_b[d] - grid_origin[d]) / chunk_shape[d]))
            grid_span_shape[d] = grid_span[1, d] - grid_span[0, d]
            grid_span_cell_count *= grid_span_shape[d]

        # Scan across all cells in the span.
        for flat_index in range(grid_span_cell_count):
            _unravel_index(flat_index, grid_span_shape, grid_index)
            grid_index[:] += grid_span[0]
            if _line_chunk_overlap(point_a, point_b, grid_origin, chunk_shape, grid_index):
                code = _compressed_morton_code_no_alloc(grid_index[::-1], ab, curr_axis_pos)
                rows.append(row)
                codes.append(code)

    # Return as arrays rather than reflecting into Python lists.
    rows = np.asarray(rows, dtype=np.uint32)
    codes = np.asarray(codes, dtype=np.uint64)
    return rows, codes


@njit
def _line_chunk_overlap(point_a, point_b, grid_origin, cell_shape, grid_index):
    """
    Ported from the C++ implementation[1] by jbms.
    Returns True if the line intersects the cell, False otherwise.

    [1]: https://github.com/google/neuroglancer/pull/522#issuecomment-1940516294
    """
    rank = len(point_a)
    min_t = 0.0
    max_t = 1.0
    
    for i in range(rank):
        a = point_a[i]
        b = point_b[i]
        line_lower = min(a, b)
        line_upper = max(a, b)
        box_lower = grid_origin[i] + cell_shape[i] * grid_index[i]
        box_upper = box_lower + cell_shape[i]
        
        line_range = line_upper - line_lower
        
        if box_lower > line_lower:
            if line_range == 0.0:
                # Line is a point, check if it's outside the box
                if line_lower < box_lower:
                    return False
            else:
                t = (box_lower - line_lower) / line_range
                if t > 1:
                    return False
                min_t = max(min_t, t)
        
        if box_upper < line_upper:
            if line_range == 0.0:
                # Line is a point, check if it's outside the box
                if line_lower > box_upper:
                    return False
            else:
                t = (box_upper - line_lower) / line_range
                if t < 0:
                    return False
                max_t = min(max_t, t)
    
    return max_t >= min_t


def _compute_grid_codes_for_polylines(polyline_geom, bounds, gridspec, per_row_levels):
    """
    Wrapper around the @njit polyline-grid-codes kernel.

    Args:
        polyline_geom:
            :class:`PolylineGeometry`. ``points[starts[i]:ends[i]]`` gives
            the vertices of polyline ``i`` in traversal order.
        bounds, gridspec, per_row_levels:
            See callers in :func:`_compute_spatial_assignment`.

    Note:
        Unlike the other geometry types, this kernel is called once
        over the full ``polyline_geom`` rather than batched. There's
        nothing to gain from batching here: the polyline aux table
        only ever ships as an in-memory pandas DataFrame (Feather aux
        is not supported), so the vertex array is already fully
        resident before this kernel runs. If aux ever moves to a
        streamed source, this would be the place to add batching --
        the kernel itself is row-iterative and would batch the same
        way as ``_box_grid_codes`` and friends, with the caller
        responsible for slicing ``starts``/``ends``/``annotation_ids``
        and offsetting the returned ``row`` values.
    """
    logger.info(f"Computing grid codes for {len(polyline_geom.starts)} polylines")
    return _polyline_grid_codes(
        polyline_geom.points,
        polyline_geom.starts,
        polyline_geom.ends,
        per_row_levels,
        bounds[0],
        gridspec.chunk_shapes,
        _axis_bits_c_order(gridspec.grid_shapes),
    )


@njit
def _polyline_grid_codes(points, starts_per_row, ends_per_row, levels, grid_origin, chunk_shapes, axis_bits_per_level):
    D = points.shape[1]

    # Pre-allocate these and reuse them on each loop iteration
    # to avoid heap allocations in the loop.
    grid_span = np.zeros((2, D), dtype=np.uint64)
    grid_span_shape = np.empty(D, dtype=np.uint64)
    grid_index = np.empty(D, dtype=np.uint64)
    curr_axis_pos = np.empty(D, dtype=np.uint64)
    poly_bbox = np.empty((2, D), dtype=np.float32)

    rows = List()
    codes = List()
    for row, (start, end, level) in enumerate(zip(starts_per_row, ends_per_row, levels)):
        chunk_shape = chunk_shapes[level]
        ab = axis_bits_per_level[level]

        poly_points = points[start:end]

        # Compute bounding box of the current polyline.
        # Since min(axis=0) doesn't work in numba, we have to loop explicitly.
        for d in range(D):
            poly_bbox[0, d] = poly_points[:, d].min()
            poly_bbox[1, d] = poly_points[:, d].max()

        grid_span_cell_count = np.uint64(1)
        for d in range(D):
            grid_span[0, d] = np.uint64(np.floor((poly_bbox[0, d] - grid_origin[d]) / chunk_shape[d]))
            grid_span[1, d] = np.uint64(np.ceil((poly_bbox[1, d] - grid_origin[d]) / chunk_shape[d]))
            grid_span_shape[d] = grid_span[1, d] - grid_span[0, d]
            grid_span_cell_count *= grid_span_shape[d]

        # Scan across all cells in the bounding-box span.
        for flat_index in range(grid_span_cell_count):
            _unravel_index(flat_index, grid_span_shape, grid_index)
            grid_index[:] += grid_span[0]

            if len(poly_points) <= 1:
                # Single-vertex polylines have no segments;
                # This must be the one grid cell that contains the vertex.
                overlaps = True
            else:
                # Check all segments in the polyline for overlap.
                overlaps = False
                for i in range(len(poly_points) - 1):
                    a = poly_points[i]
                    b = poly_points[i+1]
                    if _line_chunk_overlap(a, b, grid_origin, chunk_shape, grid_index):
                        overlaps = True
                        break

            if overlaps:
                code = _compressed_morton_code_no_alloc(grid_index[::-1], ab, curr_axis_pos)
                rows.append(row)
                codes.append(code)

    rows = np.asarray(rows, dtype=np.uint32)
    codes = np.asarray(codes, dtype=np.uint64)
    return rows, codes


SPATIAL_ASSIGNMENTS_TABLE = '_by_spatial_assignments'


def _write_annotations_by_spatial_chunk(con, input_df, coord_space, annotation_type, property_specs, polyline_geom,
                                        bounds, num_spatial_levels, target_chunk_limit,
                                        shuffle_before_assigning_spatial_levels,
                                        disable_subsampling, output_dir, write_sharded,
                                        max_shards_per_transaction, ts_context):
    """
    Write the spatial index using DuckDB-backed streaming.

    The numba geometry kernels run over the input to produce a
    ``(level, chunk_code, annotation_id)`` assignment, which is then
    materialized as a DuckDB table. For each spatial level we open a
    sharded kvstore and stream per-batch transactions: query the
    relevant chunks' rows, encode, write, release.

    Args:
        con:
            DuckDB connection with the input registered as
            :data:`INPUT_VIEW`.
        input_df:
            pandas DataFrame, or ``None`` to read geometry from
            ``INPUT_VIEW`` via DuckDB. Ignored for polyline annotations,
            which read geometry from ``polyline_geom``.
        coord_space, annotation_type, property_specs, polyline_geom:
            See :func:`write_precomputed_annotations`.
        bounds, num_spatial_levels, target_chunk_limit,
        shuffle_before_assigning_spatial_levels:
            See :func:`_compute_spatial_assignment` /
            :func:`write_precomputed_annotations`.
        disable_subsampling:
            Whether to disable subsampling by setting "limit" to 1 in
            the info file.
        output_dir, write_sharded, max_shards_per_transaction, ts_context:
            See :func:`._write_buffers._write_buffers`.

    Returns:
        JSON metadata to write into the 'spatial' key of the info file.
    """
    # 1. Run the numba kernels and stable-sort to produce the assignment.
    assignment_df, gridspec = _compute_spatial_assignment(
        con, input_df, polyline_geom, coord_space, annotation_type, bounds,
        num_spatial_levels, target_chunk_limit,
        shuffle_before_assigning_spatial_levels,
    )

    # 2. Convert (positional row -> annotation_id) and register the
    #    assignment as a DuckDB table. The ``seq`` column preserves the
    #    stable-sorted order within each (level, chunk_code) group, so
    #    queries that ``ORDER BY chunk_code, seq`` reproduce the
    #    subsampling-friendly per-chunk row order downstream code expects.
    if annotation_type == 'polyline':
        ann_ids = np.asarray(polyline_geom.annotation_ids, dtype=np.uint64)
    elif input_df is not None:
        ann_ids = input_df.index.to_numpy(np.uint64, copy=False)
    else:
        ann_ids = (
            con.execute(f"SELECT annotation_id FROM {INPUT_VIEW}")
            .to_arrow_table()
            .column('annotation_id')
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
    sorted_rows = assignment_df['row_pos'].to_numpy()
    n_rows = len(assignment_df)

    arrow_assignment = pa.table({
        'seq': np.arange(n_rows, dtype=np.int64),
        'level': assignment_df['level'].to_numpy(np.uint8, copy=False),
        'chunk_code': assignment_df['chunk_code'].to_numpy(np.uint64, copy=False),
        'annotation_id': ann_ids[sorted_rows],
    })
    del assignment_df, sorted_rows

    con.execute(f"DROP TABLE IF EXISTS {SPATIAL_ASSIGNMENTS_TABLE}")
    con.register('_by_spatial_assignments_arrow', arrow_assignment)
    try:
        con.execute(f"CREATE TABLE {SPATIAL_ASSIGNMENTS_TABLE} AS SELECT * FROM _by_spatial_assignments_arrow")
    finally:
        con.unregister('_by_spatial_assignments_arrow')
    del arrow_assignment

    try:
        # 3. For each level, write that level's portion as one sharded
        #    (or unsharded) kvstore subdirectory.
        metadata = []
        for level_idx in range(num_spatial_levels):
            n_level_rows = con.execute(
                f"SELECT COUNT(*) FROM {SPATIAL_ASSIGNMENTS_TABLE} WHERE level = ?",
                [level_idx],
            ).fetchone()[0]
            if n_level_rows == 0:
                continue

            level_metadata = _write_one_spatial_level(
                con, level_idx, gridspec,
                coord_space, annotation_type, property_specs, polyline_geom,
                output_dir, write_sharded, max_shards_per_transaction, ts_context,
                disable_subsampling,
            )
            metadata.append(level_metadata)
    finally:
        con.execute(f"DROP TABLE IF EXISTS {SPATIAL_ASSIGNMENTS_TABLE}")

    return metadata


def _write_one_spatial_level(con, level, gridspec,
                              coord_space, annotation_type, property_specs, polyline_geom,
                              output_dir, write_sharded, max_shards_per_transaction, ts_context,
                              disable_subsampling):
    """
    Write a single spatial level's subdirectory (``by_spatial_level_<level>``).
    """
    subdir = f"by_spatial_level_{level}"
    logger.info(f"Preparing 'by_spatial_level_{level}' index")

    if not write_sharded:
        return _write_one_spatial_level_unsharded(
            con, level, gridspec,
            coord_space, annotation_type, property_specs, polyline_geom,
            output_dir, subdir, ts_context, disable_subsampling,
        )

    polyline_id_lookup = (
        pd.Index(polyline_geom.annotation_ids) if polyline_geom is not None else None
    )

    # 1. Distinct chunk codes for this level (= the output keys).
    distinct_chunks = con.execute(f"""
        SELECT DISTINCT chunk_code FROM {SPATIAL_ASSIGNMENTS_TABLE}
        WHERE level = ?
        ORDER BY chunk_code
    """, [level]).to_arrow_table().column('chunk_code').to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    n_chunks = len(distinct_chunks)

    # 2. Choose shard spec from a payload-size estimate.
    total_bytes = _estimate_total_bytes_for_spatial_level(
        con, level, n_chunks, coord_space, annotation_type, property_specs, polyline_geom,
    )
    shard_spec = _write_buffers._choose_output_spec(
        total_count=int(n_chunks),
        total_bytes=int(total_bytes),
        max_key=int(distinct_chunks.max()),
        hashtype='murmurhash3_x86_128',
        gzip_compress=True,
    )

    # 3. Compute shards for each distinct chunk_code, store as DuckDB table.
    shards = shards_for_keys(distinct_chunks, shard_spec)
    pairs = pa.table({
        'chunk_code': distinct_chunks,
        'shard_id': shards.astype(np.uint64, copy=False),
    })
    del distinct_chunks, shards

    shard_assignments_table = f'_by_spatial_shards_level_{level}'
    working_data_table = f'_by_spatial_level_{level}_working_data'
    batch_shards_view = f'_by_spatial_batch_shards_level_{level}'

    con.execute(f"DROP TABLE IF EXISTS {shard_assignments_table}")
    con.register('_by_spatial_shards_arrow', pairs)
    try:
        con.execute(f"CREATE TABLE {shard_assignments_table} AS SELECT * FROM _by_spatial_shards_arrow")
    finally:
        con.unregister('_by_spatial_shards_arrow')
    del pairs

    try:
        _prepare_output_subdir(output_dir, subdir)
        kvstore = _open_sharded_kvstore(output_dir, subdir, shard_spec, ts_context)

        # 4. Materialize a *narrow* per-level working table containing
        #    only ``(shard_id, chunk_code, seq, annotation_id)`` -- no
        #    geometry or property columns. Sorted by shard_id so
        #    per-row-group min/max statistics are tight on shard_id;
        #    a WHERE-IN(shard_id) filter then only reads the row groups
        #    that actually contain the batch's shards.
        #
        #    The per-batch query (below) JOINs this narrow table back
        #    against INPUT_VIEW to pick up the geometry and property
        #    columns from the Feather mmap. Compared to embedding all
        #    columns in the working table, this trades:
        #      - lower memory (the working table is ~40% smaller for
        #        line geometry, much smaller for property-heavy data),
        #      - against one extra JOIN per batch, with the build side
        #        being the small filtered ~22M-row batch and the probe
        #        side being INPUT_VIEW (Feather mmap; pages already
        #        warm in the page cache from earlier phases).
        #
        #    The alternative -- per-batch 3-way JOIN against the full
        #    SPATIAL_ASSIGNMENTS_TABLE -- would re-scan SPATIAL_ASSIGNMENTS
        #    (311M rows) every batch. Materializing the narrow table
        #    once collapses that to one pass over SPATIAL_ASSIGNMENTS at
        #    materialization time.
        #
        #    Note that some annotations appear in the assignments table
        #    multiple times -- once per chunk they intersect -- so the
        #    per-batch JOIN may duplicate rows from INPUT_VIEW. The
        #    annotation order (preserved by ``seq``) determines per-chunk
        #    subsampling prefix order in neuroglancer when the user
        #    chose not to shuffle.
        con.execute(f"DROP TABLE IF EXISTS {working_data_table}")
        log_memory(f'level {level} pre-materialize')
        con.execute(f"""
            CREATE TABLE {working_data_table} AS
            SELECT
                chunk_to_shard.shard_id,
                ann_to_chunk.chunk_code,
                ann_to_chunk.seq,
                ann_to_chunk.annotation_id
            FROM {SPATIAL_ASSIGNMENTS_TABLE} ann_to_chunk
            JOIN {shard_assignments_table} chunk_to_shard
                ON chunk_to_shard.chunk_code = ann_to_chunk.chunk_code
            WHERE ann_to_chunk.level = ?
            ORDER BY chunk_to_shard.shard_id, ann_to_chunk.chunk_code, ann_to_chunk.seq
        """, [level])
        log_memory(f'level {level} post-materialize')

        # 5. Iterate occupied shards in batches.
        occupied_shards = con.execute(f"""
            SELECT DISTINCT shard_id FROM {shard_assignments_table}
            ORDER BY shard_id
        """).to_arrow_table().column('shard_id').to_numpy(zero_copy_only=False)

        batch_size = int(max_shards_per_transaction)
        n_transactions = (len(occupied_shards) + batch_size - 1) // batch_size
        logger.info(f"Writing annotations to '{subdir}' index "
                    f"({n_transactions} transactions over "
                    f"{len(occupied_shards)} occupied shards "
                    f"(of {1 << shard_spec.shard_bits} possible))")

        needed_cols = _ann_required_cols(coord_space, annotation_type, property_specs)
        select_cols = ', '.join(f'ann_table.{c}' for c in (['annotation_id'] + needed_cols))
        auditor = ShardWriteAuditor(os.path.join(output_dir, subdir), subdir)

        n_level_rows = con.execute(
            f"SELECT COUNT(*) FROM {working_data_table}",
        ).fetchone()[0]
        with tqdm(total=int(n_level_rows)) as pbar:
            for batch_idx, chunk_start in enumerate(range(0, len(occupied_shards), batch_size)):
                batch_shards = occupied_shards[chunk_start:chunk_start + batch_size]
                con.register(batch_shards_view, pa.table({'shard_id': batch_shards}))
                try:
                    # Filter the narrow working table to this batch's
                    # shards (zone-map-pruned WHERE on shard_id), then
                    # JOIN to INPUT_VIEW to pick up geometry/property
                    # columns. The build side of the hash join is the
                    # small filtered working-table result; the probe
                    # side is INPUT_VIEW.
                    df_batch = con.execute(f"""
                        SELECT
                            {select_cols},
                            work.chunk_code AS _chunk_code
                        FROM {working_data_table} work
                        JOIN {INPUT_VIEW} ann_table
                            ON ann_table.annotation_id = work.annotation_id
                        WHERE work.shard_id IN (SELECT shard_id FROM {batch_shards_view})
                        ORDER BY work.chunk_code, work.seq
                    """).df()
                finally:
                    con.unregister(batch_shards_view)

                if len(df_batch) == 0:
                    log_memory(f'level {level} post-batch {batch_idx + 1}/{n_transactions} (empty)')
                    continue

                batch_polyline_geom = None
                if polyline_geom is not None:
                    rows = polyline_id_lookup.get_indexer(df_batch['annotation_id'].to_numpy())
                    batch_polyline_geom = _slice_polyline_geom(polyline_geom, rows)

                buffers, batch_chunks = _build_grouped_record_buffers(
                    df_batch, '_chunk_code', coord_space, annotation_type, property_specs,
                    polyline_geom=batch_polyline_geom,
                )
                _write_one_transaction(kvstore, batch_chunks, buffers)
                auditor.record_batch(batch_shards)
                pbar.update(len(df_batch))
                del df_batch, buffers, batch_chunks, batch_polyline_geom
                log_memory(f'level {level} post-batch {batch_idx + 1}/{n_transactions}')

        # 6. Build level metadata.
        level_metadata = _sharded_metadata(subdir, shard_spec)
        level_metadata['chunk_size'] = gridspec.chunk_shapes[level].tolist()
        level_metadata['grid_shape'] = gridspec.grid_shapes[level].tolist()
        level_metadata['limit'] = _compute_subsampling_limit(con, level, disable_subsampling)
        return level_metadata
    finally:
        con.execute(f"DROP TABLE IF EXISTS {working_data_table}")
        con.execute(f"DROP TABLE IF EXISTS {shard_assignments_table}")
        log_memory(f'level {level} done')


def _write_one_spatial_level_unsharded(con, level, gridspec,
                                       coord_space, annotation_type, property_specs, polyline_geom,
                                       output_dir, subdir, ts_context, disable_subsampling):
    """
    Unsharded variant: one file per chunk, named by underscore-joined grid
    coordinates (e.g. ``'0_3_2'``). The file contents follow the same
    ``<count><records><ids>`` layout as the sharded case.
    """
    _prepare_output_subdir(output_dir, subdir)

    needed_cols = _ann_required_cols(coord_space, annotation_type, property_specs)
    select_cols = ', '.join(f'v.{c}' for c in (['annotation_id'] + needed_cols))

    df_full = con.execute(f"""
        SELECT {select_cols}, a.chunk_code AS _chunk_code
        FROM {SPATIAL_ASSIGNMENTS_TABLE} a
        JOIN {INPUT_VIEW} v ON v.annotation_id = a.annotation_id
        WHERE a.level = ?
        ORDER BY a.chunk_code, a.seq
    """, [level]).df()

    output_dir = os.path.abspath(output_dir)
    kvstore = ts.KvStore.open(f"file://{output_dir}/{subdir}/", context=ts_context).result()

    level_metadata = {
        "key": subdir,
        "chunk_size": gridspec.chunk_shapes[level].tolist(),
        "grid_shape": gridspec.grid_shapes[level].tolist(),
        "limit": _compute_subsampling_limit(con, level, disable_subsampling),
    }
    if len(df_full) == 0:
        return level_metadata

    sliced_polyline_geom = None
    if polyline_geom is not None:
        polyline_id_lookup = pd.Index(polyline_geom.annotation_ids)
        rows = polyline_id_lookup.get_indexer(df_full['annotation_id'].to_numpy())
        sliced_polyline_geom = _slice_polyline_geom(polyline_geom, rows)

    buffers, unique_chunks = _build_grouped_record_buffers(
        df_full, '_chunk_code', coord_space, annotation_type, property_specs,
        polyline_geom=sliced_polyline_geom,
    )

    # For unsharded the key is the chunk's grid coordinate joined with '_'.
    grid_coords = compressed_morton_decode(unique_chunks, gridspec.grid_shapes[level])
    string_keys = list(map('_'.join, grid_coords.astype(str)))

    logger.info(f"Writing annotations to '{subdir}' index "
                f"({len(unique_chunks)} chunks, unsharded)")
    with tqdm(total=len(unique_chunks)) as pbar, ts.Transaction() as txn:
        txn_kv = kvstore.with_transaction(txn)
        for i, key in enumerate(string_keys):
            txn_kv[key] = b''.join(pb.slice_for_partition(i) for pb in buffers)
            pbar.update(1)

    return level_metadata


def _compute_subsampling_limit(con, level, disable_subsampling):
    """
    Return the ``limit`` value to record in this level's spatial-index
    metadata. Normally this is the max per-chunk annotation count at
    this level.

    The spec defines ``limit`` as the per-cell annotation target used
    during index *construction* (each annotation emitted at this level
    with probability ``min(1, limit / maxCount(level))``). But the
    renderer also consults ``limit`` at *display* time as a denominator:
    ``drawFraction = min(1, desiredCount / limit)``, where the chunk's
    stored list is then truncated to ``count * drawFraction``
    (see ``annotation/base.ts`` and ``annotation/renderlayer.ts`` in
    the neuroglancer source).

    So when the caller asks to disable subsampling
    (``target_chunk_limit == 0``) we emit ``limit=1``: that saturates
    ``drawFraction`` at 1 and the renderer draws every annotation in
    each chunk. As explained[1] by jbms:

        > Neuroglancer "subsamples" by showing only a prefix of the list of
        > annotations according to the spacing setting.  If you set "limit" to 1 in
        > the info file, you won't get subsampling by default.

    [1]: https://github.com/google/neuroglancer/issues/227#issuecomment-651944575
    """
    if disable_subsampling:
        return 1
    max_count = con.execute(f"""
        SELECT MAX(c) FROM (
            SELECT COUNT(*) AS c
            FROM {SPATIAL_ASSIGNMENTS_TABLE}
            WHERE level = ?
            GROUP BY chunk_code
        )
    """, [level]).fetchone()[0]
    return int(max_count or 1)


def _estimate_total_bytes_for_spatial_level(con, level, n_chunks, coord_space, annotation_type, property_specs,
                                            polyline_geom):
    """
    Rough estimate of one spatial level's payload bytes: 8 bytes per
    chunk for the count header plus ``(ann_recsize + 8) * total_rows``
    for the encoded records and annotation-id buffers.

    For polylines the per-record size varies; we use the average
    per-polyline record size as the stand-in.
    """
    if n_chunks == 0:
        return 0

    n_total_rows = con.execute(
        f"SELECT COUNT(*) FROM {SPATIAL_ASSIGNMENTS_TABLE} WHERE level = ?",
        [level],
    ).fetchone()[0]

    if annotation_type == 'polyline':
        if polyline_geom is None or len(polyline_geom.starts) == 0:
            ann_recsize = 0
        else:
            n_polylines = len(polyline_geom.starts)
            avg_vertex_bytes = int(polyline_geom.points.nbytes) // n_polylines
            ann_recsize = 4 + avg_vertex_bytes + _property_recsize(property_specs)
        return 8 * n_chunks + (ann_recsize + 8) * int(n_total_rows)

    probe_df = (
        con.execute(f"SELECT * FROM {INPUT_VIEW} LIMIT 0")
        .df()
        .set_index('annotation_id')
    )
    ann_pb = _encode_annotation_records(
        probe_df, coord_space, annotation_type, property_specs, polyline_geom=None,
    )
    ann_recsize = int(ann_pb.layout) if isinstance(ann_pb.layout, (int, np.integer)) else 0
    return 8 * n_chunks + (ann_recsize + 8) * int(n_total_rows)
