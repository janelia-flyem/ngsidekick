"""
Tests for the spatial-index assignment helpers in
``ngsidekick.annotations.precomputed._spatial``.
"""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from ngsidekick.annotations.precomputed._spatial import (
    _compute_grid_codes_for_axis_aligned_bounding_boxes,
    _compute_grid_codes_for_ellipsoids,
    _compute_grid_codes_for_lines,
    _compute_grid_codes_for_points,
    _compute_grid_codes_for_polylines,
    GridSpec,
)
from ngsidekick.annotations.precomputed._util import PolylineGeometry


def _single_level_gridspec(grid_shape=(4, 4, 4), bounds_upper=1.0):
    """A trivial single-level cubic grid in [0, bounds_upper]^len(grid_shape)."""
    grid_shapes = np.array([grid_shape], dtype=np.uint64)
    chunk_shapes = np.array(
        [[bounds_upper / s for s in grid_shape]],
        dtype=np.float64,
    )
    return GridSpec(chunk_shapes=chunk_shapes, grid_shapes=grid_shapes)


def test_2d_lines_span_multiple_chunks():
    """
    The kernels are documented to be agnostic to coordinate-space
    dimensionality. Verify with a 2-D coord space that line annotations
    spanning multiple cells produce one entry per cell.
    """
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4))
    geometry_cols = [['xa', 'ya'], ['xb', 'yb']]

    df = pd.DataFrame({
        'xa': [0.1, 0.1, 0.1],
        'ya': [0.1, 0.1, 0.1],
        'xb': [0.2, 0.6, 0.9],   # spans 1, 3, and 4 cells along x
        'yb': [0.2, 0.1, 0.1],
    })
    per_row_levels = np.zeros(len(df), dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_lines(df, geometry_cols, bounds, gridspec, per_row_levels)
    counts = Counter(rows.tolist())
    assert counts == {0: 1, 1: 3, 2: 4}, counts


def test_2d_points_get_correct_chunk_codes():
    """Points in a 2-D coord space hash to a single chunk each."""
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4))

    df = pd.DataFrame({'x': [0.1, 0.6, 0.9], 'y': [0.1, 0.5, 0.9]})
    per_row_levels = np.zeros(len(df), dtype=np.uint64)
    rows, codes = _compute_grid_codes_for_points(df, [['x', 'y']], bounds, gridspec, per_row_levels)
    assert rows.tolist() == [0, 1, 2]
    # All chunk codes distinct (each point is in its own cell).
    assert len(set(codes.tolist())) == 3


def test_2d_boxes_span_multiple_chunks():
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4))
    df = pd.DataFrame({
        'xa': [0.05, 0.05],
        'ya': [0.05, 0.05],
        'xb': [0.20, 0.30],
        'yb': [0.20, 0.30],
    })
    per_row_levels = np.zeros(len(df), dtype=np.uint64)
    rows, codes = _compute_grid_codes_for_axis_aligned_bounding_boxes(df, [['xa', 'ya'], ['xb', 'yb']], bounds, gridspec, per_row_levels)
    counts = Counter(rows.tolist())
    # Box 0 fits in 1 cell. Box 1 spans 2x2 cells = 4.
    assert counts == {0: 1, 1: 4}, counts


def test_4d_lines_round_trip_via_public_api():
    """
    End-to-end check via write_precomputed_annotations that a 4-D coord
    space produces a valid spatial index. Catches regressions in any
    stage that hardcodes a 3-D assumption (geometry encoder, spatial
    kernels, gridspec construction).
    """
    import json
    import tempfile
    from neuroglancer.coordinate_space import CoordinateSpace
    from ngsidekick.annotations.precomputed import write_precomputed_annotations

    n = 100
    rng = np.random.default_rng(0)
    ids = rng.choice(2**40, size=n, replace=False).astype(np.uint64)
    df = pd.DataFrame({
        'xa': rng.normal(0, 5, n), 'ya': rng.normal(0, 5, n),
        'za': rng.normal(0, 5, n), 'ta': rng.normal(0, 5, n),
        'xb': rng.normal(0, 5, n), 'yb': rng.normal(0, 5, n),
        'zb': rng.normal(0, 5, n), 'tb': rng.normal(0, 5, n),
    }, index=pd.Index(ids))

    cs = CoordinateSpace(names=['x', 'y', 'z', 't'], units=['nm']*4, scales=[1, 1, 1, 1])
    with tempfile.TemporaryDirectory() as tmpdir:
        write_precomputed_annotations(
            df, cs, annotation_type='line',
            output_dir=f"{tmpdir}/l4",
            write_sharded=True, write_by_relationship=False,
            num_spatial_levels=3, target_chunk_limit=10,
        )
        info = json.loads(open(f"{tmpdir}/l4/info").read())
        assert list(info['dimensions'].keys()) == ['x', 'y', 'z', 't']
        # Each level's grid_shape should have 4 entries (one per dimension).
        for level_meta in info['spatial']:
            assert len(level_meta['grid_shape']) == 4
            assert len(level_meta['chunk_size']) == 4


def test_lines_spanning_multiple_chunks_are_duplicated():
    """
    Annotations whose geometry crosses chunk boundaries must produce one
    output entry per chunk they span. Regression test for a bug where the
    old wrapper used ``df.loc[df.index[rows], 'chunk_code'] = codes``,
    which silently kept only the last code per duplicate row label.
    """
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4, 4))
    geometry_cols = [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']]

    # Three lines along x: short (1 chunk), medium (3 chunks), long (4 chunks).
    df = pd.DataFrame({
        'xa': [0.1, 0.1, 0.1],
        'ya': [0.1, 0.1, 0.1],
        'za': [0.1, 0.1, 0.1],
        'xb': [0.2, 0.6, 0.9],
        'yb': [0.2, 0.1, 0.1],
        'zb': [0.2, 0.1, 0.1],
    }, index=[100, 200, 300])
    per_row_levels = np.zeros(len(df), dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_lines(df, geometry_cols, bounds, gridspec, per_row_levels)
    counts = Counter(rows.tolist())
    assert counts == {0: 1, 1: 3, 2: 4}, (
        f"Expected one output row per chunk spanned, got: {dict(counts)}"
    )
    # And the codes within each row must be unique (distinct chunks).
    for r in {0, 1, 2}:
        per_row_codes = codes[rows == r]
        assert len(set(per_row_codes.tolist())) == len(per_row_codes), (
            f"chunk_code duplicates for row {r}"
        )


def test_boxes_spanning_multiple_chunks_are_duplicated():
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4, 4))
    geometry_cols = [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']]

    # A 1-chunk box, and a 2x2x1 box (grid span ⌊0.05/0.25⌋..⌈0.30/0.25⌉ = 0..2,
    # so x and y each cover 2 chunk indices; z covers 1).
    df = pd.DataFrame({
        'xa': [0.05, 0.05],
        'ya': [0.05, 0.05],
        'za': [0.05, 0.05],
        'xb': [0.20, 0.30],
        'yb': [0.20, 0.30],
        'zb': [0.20, 0.20],
    }, index=[10, 20])
    per_row_levels = np.zeros(len(df), dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_axis_aligned_bounding_boxes(df, geometry_cols, bounds, gridspec, per_row_levels)
    counts = Counter(rows.tolist())
    assert counts == {0: 1, 1: 4}, counts


def test_ellipsoids_spanning_multiple_chunks_are_duplicated():
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4, 4))
    geometry_cols = [['x', 'y', 'z'], ['rx', 'ry', 'rz']]

    # Tiny ellipsoid (within a chunk) and one that overlaps several.
    df = pd.DataFrame({
        'x':  [0.125, 0.5],
        'y':  [0.125, 0.5],
        'z':  [0.125, 0.5],
        'rx': [0.05, 0.3],
        'ry': [0.05, 0.3],
        'rz': [0.05, 0.3],
    }, index=[10, 20])
    per_row_levels = np.zeros(len(df), dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_ellipsoids(df, geometry_cols, bounds, gridspec, per_row_levels)
    counts = Counter(rows.tolist())
    assert counts[0] == 1
    assert counts[1] > 1, (
        f"A 0.6-diameter ellipsoid centred mid-grid should overlap multiple "
        f"0.25-wide chunks, but produced {counts[1]} entries"
    )


def test_short_annotation_inside_one_chunk_produces_single_entry():
    """Round-trip sanity: a single-chunk annotation must still emit exactly one entry."""
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4, 4))
    geometry_cols = [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']]

    df = pd.DataFrame({
        'xa': [0.1], 'ya': [0.1], 'za': [0.1],
        'xb': [0.15], 'yb': [0.15], 'zb': [0.15],
    }, index=[42])
    per_row_levels = np.zeros(len(df), dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_lines(df, geometry_cols, bounds, gridspec, per_row_levels)
    assert rows.tolist() == [0]
    assert len(codes) == 1


def test_polylines_spanning_multiple_chunks_are_duplicated():
    """
    A polyline that crosses chunk boundaries must produce one output entry
    per chunk it overlaps. Mirrors the line analogue.
    """
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4, 4))

    # Two polylines:
    #  poly 0: stays in 1 chunk (3 close vertices in [0.1, 0.15])
    #  poly 1: a 4-vertex zigzag along x covering ~4 cells
    points = np.array([
        # poly 0 -- 3 vertices in one chunk
        [0.10, 0.10, 0.10],
        [0.12, 0.12, 0.12],
        [0.14, 0.14, 0.14],
        # poly 1 -- 4 vertices spanning 4 chunks along x
        [0.10, 0.10, 0.10],
        [0.40, 0.10, 0.10],
        [0.60, 0.10, 0.10],
        [0.90, 0.10, 0.10],
    ], dtype=np.float32)
    starts = np.array([0, 3], dtype=np.int64)
    ends = np.array([3, 7], dtype=np.int64)
    annotation_ids = np.array([0, 1], dtype=np.uint64)
    per_row_levels = np.zeros(2, dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_polylines(
        PolylineGeometry(points, starts, ends, annotation_ids),
        bounds, gridspec, per_row_levels,
    )
    counts = Counter(rows.tolist())
    assert counts[0] == 1
    assert counts[1] == 4, counts


def test_2d_polylines_span_multiple_chunks():
    """Polyline kernel must be agnostic to coordinate-space dimensionality."""
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4))

    points = np.array([
        # poly 0: 1 chunk
        [0.10, 0.10],
        [0.15, 0.15],
        # poly 1: zigzag across x covering chunks 0,1,2,3
        [0.10, 0.10],
        [0.40, 0.10],
        [0.90, 0.10],
    ], dtype=np.float32)
    starts = np.array([0, 2], dtype=np.int64)
    ends = np.array([2, 5], dtype=np.int64)
    annotation_ids = np.array([0, 1], dtype=np.uint64)
    per_row_levels = np.zeros(2, dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_polylines(
        PolylineGeometry(points, starts, ends, annotation_ids),
        bounds, gridspec, per_row_levels,
    )
    counts = Counter(rows.tolist())
    assert counts[0] == 1
    assert counts[1] == 4, counts


def test_polyline_with_single_point_emits_one_chunk():
    """A 1-vertex polyline is degenerate but spec-permitted; emit its containing chunk."""
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4, 4))

    points = np.array([[0.10, 0.10, 0.10]], dtype=np.float32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([1], dtype=np.int64)
    annotation_ids = np.array([0], dtype=np.uint64)
    per_row_levels = np.zeros(1, dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_polylines(
        PolylineGeometry(points, starts, ends, annotation_ids),
        bounds, gridspec, per_row_levels,
    )
    assert rows.tolist() == [0]
    assert len(codes) == 1


def _cells(codes, grid_shape):
    """
    Decode chunk codes (as emitted by the grid-code kernels) into the
    set of grid cell coordinates (in the same axis order as grid_shape).
    """
    from ngsidekick.annotations.precomputed.compressed_morton import compressed_morton_decode
    grid_shape = np.asarray(grid_shape, dtype=np.uint64)
    coords = compressed_morton_decode(np.asarray(codes, np.uint64), grid_shape[::-1])[..., ::-1]
    return set(map(tuple, coords.reshape(-1, len(grid_shape)).tolist()))


def _line_cells(point_a, point_b, grid_shape, kernel='line'):
    """Cells reported for a single line (or 2-vertex polyline) in [0, 1]^D."""
    D = len(grid_shape)
    bounds = np.array([[0.0] * D, [1.0] * D])
    gridspec = _single_level_gridspec(tuple(grid_shape))
    per_row_levels = np.zeros(1, dtype=np.uint64)
    if kernel == 'line':
        names = [f'c{d}' for d in range(D)]
        geometry_cols = [[f'{c}a' for c in names], [f'{c}b' for c in names]]
        df = pd.DataFrame([[*point_a, *point_b]], columns=[*geometry_cols[0], *geometry_cols[1]])
        rows, codes = _compute_grid_codes_for_lines(df, geometry_cols, bounds, gridspec, per_row_levels)
    else:
        points = np.array([point_a, point_b], dtype=np.float32)
        geom = PolylineGeometry(points, np.array([0]), np.array([2]), np.array([0], dtype=np.uint64))
        rows, codes = _compute_grid_codes_for_polylines(geom, bounds, gridspec, per_row_levels)
    return _cells(codes, grid_shape)


@pytest.mark.parametrize('kernel', ['line', 'polyline'])
def test_line_direction_matters(kernel):
    """
    A line's chunks depend on its direction, not just its bounding box.
    Regression test for a bug in which every line was treated as the main
    diagonal of its bounding box, so an anti-diagonal line was assigned
    to the wrong chunks.

    In a 4x4 grid over [0,1]^2 (cells of width 0.25), the line from
    (0.1, 0.9) to (0.9, 0.3) passes through these cells (in order):
    """
    expected = {(0, 3), (1, 3), (1, 2), (2, 2), (2, 1), (3, 1)}
    assert _line_cells([0.1, 0.9], [0.9, 0.3], (4, 4), kernel) == expected

    # The reversed line covers the same cells.
    assert _line_cells([0.9, 0.3], [0.1, 0.9], (4, 4), kernel) == expected

    # The main-diagonal line with the same bounding box covers different cells.
    assert _line_cells([0.1, 0.3], [0.9, 0.9], (4, 4), kernel) == {
        (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)
    }


@pytest.mark.parametrize('kernel', ['line', 'polyline'])
def test_line_cells_mirror_symmetry(kernel):
    """
    Mirroring a line along any axis must mirror the set of cells it occupies.
    """
    rng = np.random.default_rng(0)
    grid_shape = np.array([8, 4, 2])
    for _ in range(200):
        a, b = rng.uniform(0, 1, (2, 3)).astype(np.float32)
        cells = _line_cells(a, b, grid_shape, kernel)
        for axis in range(3):
            # Mirror around 0.5 (which is exact in float32 for these grids'
            # cell boundaries, so no new boundary-touching cases arise).
            ma, mb = a.copy(), b.copy()
            ma[axis] = 1 - ma[axis]
            mb[axis] = 1 - mb[axis]
            mirrored = {
                tuple(int(grid_shape[d] - 1 - c[d]) if d == axis else c[d] for d in range(3))
                for c in cells
            }
            assert _line_cells(ma, mb, grid_shape, kernel) == mirrored


def test_line_cells_contain_sampled_points():
    """
    Every cell containing a point on the line must be reported,
    and each reported cell must lie close to the line.
    """
    rng = np.random.default_rng(1)
    grid_shape = np.array([8, 8, 4])
    cell_shape = 1.0 / grid_shape
    t = np.linspace(0, 1, 2001)[:, None]
    for _ in range(200):
        a, b = rng.uniform(0, 1, (2, 3))
        cells = _line_cells(a, b, grid_shape)

        samples = a + t * (b - a)
        sampled_cells = set(map(tuple, np.minimum(samples // cell_shape, grid_shape - 1).astype(int).tolist()))
        assert sampled_cells <= cells

        # Cells which weren't hit by any sample can only be cells that the line
        # barely clips, so they must be adjacent to a sampled cell.
        for c in cells - sampled_cells:
            assert any(max(abs(np.subtract(c, s))) <= 1 for s in sampled_cells), c


def test_flat_geometry_on_chunk_boundaries():
    """
    Geometry with zero extent along some axis must still be assigned to a chunk,
    even if it lies exactly on a chunk boundary (or on the grid's lower/upper bound).
    Regression test for a bug in which such annotations were assigned to no
    chunks at all (and thus silently omitted from the spatial index).
    """
    # 4x4x4 grid over [0,1]^3 (cell width 0.25)
    grid_shape = (4, 4, 4)

    # Line in the plane z=0 (the lower bound) and line in the plane x=0.5 (a cell boundary).
    assert _line_cells([0.1, 0.1, 0.0], [0.9, 0.1, 0.0], grid_shape) == {(i, 0, 0) for i in range(4)}
    assert _line_cells([0.5, 0.1, 0.1], [0.5, 0.9, 0.1], grid_shape) == {(2, j, 0) for j in range(4)}

    # Line in the plane z=1 (the upper bound) goes in the last cell.
    assert _line_cells([0.1, 0.1, 1.0], [0.3, 0.1, 1.0], grid_shape) == {(0, 0, 3), (1, 0, 3)}

    # Same for polylines (both multi-vertex and single-vertex).
    assert _line_cells([0.1, 0.1, 0.0], [0.9, 0.1, 0.0], grid_shape, 'polyline') == {(i, 0, 0) for i in range(4)}
    bounds = np.array([[0.0] * 3, [1.0] * 3])
    gridspec = _single_level_gridspec(grid_shape)
    geom = PolylineGeometry(
        np.array([[0.5, 0.5, 1.0]], dtype=np.float32), np.array([0]), np.array([1]), np.array([0], dtype=np.uint64)
    )
    rows, codes = _compute_grid_codes_for_polylines(geom, bounds, gridspec, np.zeros(1, dtype=np.uint64))
    assert _cells(codes, grid_shape) == {(2, 2, 3)}

    # Flat box (zero extent in z, on a cell boundary).
    df = pd.DataFrame({'xa': [0.1], 'ya': [0.1], 'za': [0.5], 'xb': [0.3], 'yb': [0.2], 'zb': [0.5]})
    rows, codes = _compute_grid_codes_for_axis_aligned_bounding_boxes(
        df, [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']], bounds, gridspec, np.zeros(1, dtype=np.uint64)
    )
    assert _cells(codes, grid_shape) == {(0, 0, 2), (1, 0, 2)}

    # Ellipsoid with zero radius in z, centered on a cell boundary.
    df = pd.DataFrame({'x': [0.1], 'y': [0.1], 'z': [0.5], 'rx': [0.05], 'ry': [0.05], 'rz': [0.0]})
    rows, codes = _compute_grid_codes_for_ellipsoids(
        df, [['x', 'y', 'z'], ['rx', 'ry', 'rz']], bounds, gridspec, np.zeros(1, dtype=np.uint64)
    )
    assert _cells(codes, grid_shape) == {(0, 0, 2)}


def _spatial_index_ids(out):
    """Return the set of annotation IDs found anywhere in the (sharded) spatial index."""
    import json
    import tensorstore as ts

    info = json.loads((out / 'info').read_text())
    ids = set()
    for level in info['spatial']:
        kv = ts.KvStore.open({
            'driver': 'neuroglancer_uint64_sharded',
            'metadata': level['sharding'],
            'base': f"file://{out}/{level['key']}/",
        }).result()
        for key in kv.list().result():
            value = kv[key]
            count = int(np.frombuffer(value[:8], '<u8')[0])
            ids |= set(np.frombuffer(value[len(value) - 8 * count:], '<u8').tolist())
    return ids


@pytest.mark.parametrize('annotation_type', ['point', 'line', 'axis_aligned_bounding_box', 'ellipsoid', 'polyline'])
def test_flat_dataset(annotation_type, tmp_path):
    """
    A dataset with zero extent along one axis (e.g. 2D annotations embedded
    in a 3D coordinate space) can be written, and every annotation appears
    in the spatial index. Also covers ellipsoids with a zero radius.
    Regression test for crashes (division by zero) in that case.
    """
    from ngsidekick.annotations.precomputed import write_precomputed_annotations

    n = 200
    rng = np.random.default_rng(0)
    xy = lambda: rng.uniform(0, 100, n).astype(np.float32)  # noqa: E731
    zero = np.zeros(n, dtype=np.float32)
    polyline_points = None
    if annotation_type == 'point':
        df = pd.DataFrame({'x': xy(), 'y': xy(), 'z': zero})
    elif annotation_type in ('line', 'axis_aligned_bounding_box'):
        df = pd.DataFrame({'xa': xy(), 'ya': xy(), 'za': zero, 'xb': xy(), 'yb': xy(), 'zb': zero})
    elif annotation_type == 'ellipsoid':
        df = pd.DataFrame({'x': xy(), 'y': xy(), 'z': zero, 'rx': zero + 1, 'ry': zero + 1, 'rz': zero})
    else:
        df = None
        polyline_points = pd.DataFrame({
            'annotation_id': np.repeat(np.arange(n), 3),
            'x': rng.uniform(0, 100, 3 * n), 'y': rng.uniform(0, 100, 3 * n), 'z': 0.0,
        })

    out = tmp_path / annotation_type
    write_precomputed_annotations(
        df, 'xyz', annotation_type, output_dir=out, polyline_points=polyline_points,
        num_spatial_levels=4, target_chunk_limit=20,
    )
    assert _spatial_index_ids(out) == set(range(n))
