import logging
from itertools import chain
from typing import NamedTuple

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List

from .compressed_morton import compressed_morton_code, compressed_morton_decode, _compressed_morton_code_no_alloc
from ._write_buffers import _write_buffers
from ._util import _encode_uint64_series, _geometry_cols, _unravel_index, TableHandle

logger = logging.getLogger(__name__)

GridSpec = NamedTuple("GridSpec", [('chunk_shapes', np.ndarray), ('grid_shapes', np.ndarray)])

SpatialAssignment = NamedTuple("SpatialAssignment", [
    # Each entry of (rows, codes, levels) is a single (annotation, chunk)
    # pairing. ``rows`` may contain duplicates (multi-chunk annotations).
    ('rows', np.ndarray),       # uint32; positional index into the input df
    ('codes', np.ndarray),      # uint64; chunk_code at the assigned level
    ('levels', np.ndarray),     # uint64; level at which this entry lives
    ('gridspec', GridSpec),
])


def _compute_spatial_assignment(
    df,
    coord_space,
    annotation_type,
    bounds,
    num_levels,
    target_chunk_limit,
    shuffle_before_assigning_spatial_levels,
    *,
    polyline_geom=None,
):
    """
    Compute a :class:`SpatialAssignment` for ``df`` without modifying it
    or building a duplicated dataframe.

    Splitting this step out from the actual writing lets the caller
    drop the geometry columns from ``df`` as soon as this returns,
    holding only the much smaller ``(rows, codes, levels)`` arrays
    until the spatial index is finally written.

    Args:
        df:
            DataFrame containing the geometry columns appropriate for
            ``annotation_type``. Other columns are ignored. Row order is
            preserved.

        coord_space, annotation_type, bounds, num_levels, target_chunk_limit:
            See :func:`write_precomputed_annotations`.

        shuffle_before_assigning_spatial_levels:
            If True, level assignments are randomized across df rows
            (per the neuroglancer spec recommendation). The shuffle is
            tracked via a permutation array; ``df`` itself is not
            modified.

    Returns:
        SpatialAssignment.
    """
    geometry_cols = _geometry_cols(coord_space.names, annotation_type)
    gridspec = _define_spatial_grids(bounds, coord_space, num_levels)
    level_counts = _compute_target_annotations_per_level(len(df), gridspec, target_chunk_limit)

    # Levels are assigned by *position* in a (possibly randomly permuted)
    # ordering of df. Computing the permutation explicitly lets us derive
    # the per-row level array without mutating df.
    if shuffle_before_assigning_spatial_levels:
        logger.info("Shuffling annotations before assigning spatial grid levels")
        perm = np.random.permutation(len(df))
    else:
        perm = np.arange(len(df))
    levels_by_perm_position = np.repeat(range(num_levels), level_counts.astype(int)).astype(np.uint64)
    per_row_levels = np.empty(len(df), dtype=np.uint64)
    per_row_levels[perm] = levels_by_perm_position
    del perm, levels_by_perm_position

    logger.info("Assigning spatial grid chunks...")
    match annotation_type:
        case 'point':
            rows, codes = _compute_grid_codes_for_points(df, geometry_cols, bounds, gridspec, per_row_levels)
        case 'axis_aligned_bounding_box':
            rows, codes = _compute_grid_codes_for_axis_aligned_bounding_boxes(df, geometry_cols, bounds, gridspec, per_row_levels)
        case 'ellipsoid':
            rows, codes = _compute_grid_codes_for_ellipsoids(df, geometry_cols, bounds, gridspec, per_row_levels)
        case 'line':
            rows, codes = _compute_grid_codes_for_lines(df, geometry_cols, bounds, gridspec, per_row_levels)
        case 'polyline':
            rows, codes = _compute_grid_codes_for_polylines(
                polyline_geom, bounds, gridspec, per_row_levels,
            )
        case _:
            raise NotImplementedError(f"Spatial indexing for {annotation_type} annotations is not implemented")
    logger.info("Done assigning spatial grid chunks")

    return SpatialAssignment(
        rows=rows,
        codes=codes,
        levels=per_row_levels[rows],
        gridspec=gridspec,
    )


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
        assert num_levels == 1, \
            "The special target_chunk_limit of 0 is only permitted when num_spatial_levels=1"
        assert chunk_counts_by_level.tolist() == [1]
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

    logger.info(f"Computing grid codes for {len(df)} boxes")
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

    logger.info(f"Computing grid codes for {len(df)} ellipsoids")
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

    logger.info(f"Computing grid codes for {len(df)} lines")
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


def _write_annotations_by_spatial_chunk(df_handle, spatial_assignment, disable_subsampling, output_dir, write_sharded, max_shards_per_transaction, max_threads):
    """
    Write the spatial index, given a precomputed :class:`SpatialAssignment`
    and a dataframe holding the encoded annotations.

    Args:
        df_handle:
            TableHandle wrapping a DataFrame with ``id_buf`` and
            ``ann_buf`` columns. The handle's reference will be unset
            before this function returns. The dataframe is indexed
            positionally by ``spatial_assignment.rows``.

        spatial_assignment:
            Output of :func:`_compute_spatial_assignment`. Holds the
            (row, chunk_code, level) tuples that determine which
            annotations land in which spatial cells; supplied here in
            preference to recomputing it because the caller can drop the
            input geometry columns once the assignment is known.

        disable_subsampling:
            Whether to disable subsampling by setting "limit" to 1 in
            the info file. (See inline comments.)

        output_dir:
            Directory to write the annotations to.
            Subdirectories for each level of the spatial index will be
            created in output_dir, named ``by_spatial_level_<level>``.

        write_sharded:
            Whether to write the annotations in sharded format.

        max_shards_per_transaction, max_threads:
            See :func:`._write_buffers._write_buffers`.

    Returns:
        JSON metadata to write into the 'spatial' key of the info file.
    """
    rows = spatial_assignment.rows
    codes = spatial_assignment.codes
    levels = spatial_assignment.levels
    gridspec = spatial_assignment.gridspec

    # Build the per-(level, chunk_code, annotation) frame by gathering
    # id_buf and ann_buf at the assignment's row positions. Numpy fancy
    # indexing on the underlying object arrays just produces parallel
    # arrays of pointers to the same bytes objects — no buffer data is
    # copied — so the cost is one pointer per (annotation, chunk) pair.
    logger.info(f"Concatenating annotations by spatial chunk")
    df = pd.DataFrame({
        'level': levels,
        'chunk_code': codes,
        'id_buf': df_handle.df['id_buf'].values[rows],
        'ann_buf': df_handle.df['ann_buf'].values[rows],
    })
    df_handle.df = None
    bufs_by_grid = (
        df
        .groupby(['level', 'chunk_code'], sort=False)
        .agg({'id_buf': ['count', b''.join], 'ann_buf': b''.join})
    )
    del df

    bufs_by_grid.columns = ['count', 'id_buf', 'ann_buf']
    bufs_by_grid['count_buf'] = _encode_uint64_series(bufs_by_grid['count'])

    bufs_by_grid = bufs_by_grid.reset_index()

    metadata = []
    for level, level_bufs in bufs_by_grid.groupby('level'):
        logger.info(f"Writing annotations to 'by_spatial_level_{level}' index")

        if write_sharded:
            # Sharded key is the compressed morton code of the grid coordinate.
            level_bufs.index = level_bufs['chunk_code']
        else:
            # Unsharded key is string of the grid coordinate, e.g. '0_0_0'
            grid_coords = compressed_morton_decode(
                level_bufs['chunk_code'].to_numpy(),
                gridspec.grid_shapes[level]
            )
            level_bufs.index = list(map('_'.join, grid_coords.astype(str)))

        level_metadata = _write_buffers(
            # _write_buffers concatenates these columns row-wise inline; no
            # need to materialize a precomputed combined-buffer column.
            level_bufs[['count_buf', 'ann_buf', 'id_buf']],
            output_dir,
            f"by_spatial_level_{level}",
            write_sharded,
            max_shards_per_transaction,
            max_threads,
        )
        level_metadata['chunk_size'] = gridspec.chunk_shapes[level].tolist()
        level_metadata['grid_shape'] = gridspec.grid_shapes[level].tolist()

        if disable_subsampling:
            # To be honest, I don't completely understand why this
            # disables subsampling, but according to jbms[1]:
            #
            #   > Neuroglancer "subsamples" by showing only a prefix of the list of
            #   > annotations according to the spacing setting.  If you set "limit" to 1 in
            #   > the info file, you won't get subsampling by default.
            #
            # [1]: https://github.com/google/neuroglancer/issues/227#issuecomment-651944575
            level_metadata['limit'] = 1
        else:
            level_metadata['limit'] = int(level_bufs['count'].max())
        metadata.append(level_metadata)

    return metadata
