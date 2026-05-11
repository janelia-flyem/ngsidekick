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
    per_row_levels = np.zeros(2, dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_polylines(
        PolylineGeometry(points, starts, ends), bounds, gridspec, per_row_levels
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
    per_row_levels = np.zeros(2, dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_polylines(
        PolylineGeometry(points, starts, ends), bounds, gridspec, per_row_levels
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
    per_row_levels = np.zeros(1, dtype=np.uint64)

    rows, codes = _compute_grid_codes_for_polylines(
        PolylineGeometry(points, starts, ends), bounds, gridspec, per_row_levels
    )
    assert rows.tolist() == [0]
    assert len(codes) == 1
