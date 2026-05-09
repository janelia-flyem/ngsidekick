"""
Tests for the spatial-index assignment helpers in
``ngsidekick.annotations.precomputed._spatial``.
"""
import numpy as np
import pandas as pd
import pytest

from ngsidekick.annotations.precomputed._spatial import (
    _assign_spatial_chunks_for_axis_aligned_bounding_boxes,
    _assign_spatial_chunks_for_ellipsoids,
    _assign_spatial_chunks_for_lines,
    _box_grid_codes,
    _ellipsoid_grid_codes,
    _line_grid_codes,
    GridSpec,
)


def _single_level_gridspec(grid_shape=(4, 4, 4), bounds_upper=1.0):
    """A trivial single-level cubic grid in [0, bounds_upper]^3."""
    grid_shapes = np.array([grid_shape], dtype=np.uint64)
    chunk_shapes = np.array(
        [[bounds_upper / s for s in grid_shape]],
        dtype=np.float64,
    )
    return GridSpec(chunk_shapes=chunk_shapes, grid_shapes=grid_shapes)


def test_lines_spanning_multiple_chunks_are_duplicated():
    """
    Annotations whose geometry crosses chunk boundaries must produce one
    output row per chunk they span. Regression test for a bug where the
    wrapper used ``df.loc[df.index[rows], 'chunk_code'] = codes``, which
    silently kept only the last code per duplicate row label.
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
        'level': [0, 0, 0],
        'id_buf': [b'A', b'B', b'C'],
        'ann_buf': [b'a', b'b', b'c'],
    }, index=[100, 200, 300])

    result = _assign_spatial_chunks_for_lines(df, geometry_cols, bounds, gridspec)

    counts = result['id_buf'].value_counts().to_dict()
    assert counts == {b'A': 1, b'B': 3, b'C': 4}, (
        f"Expected one output row per chunk spanned, got: {counts}"
    )
    assert result['chunk_code'].nunique() >= 4
    # And no chunk_code duplicates within the same id (they should be in
    # distinct chunks).
    for id_, group in result.groupby('id_buf', sort=False):
        assert group['chunk_code'].is_unique, f"chunk_code duplicates for id {id_!r}"


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
        'level': [0, 0],
        'id_buf': [b'A', b'B'],
        'ann_buf': [b'a', b'b'],
    }, index=[10, 20])

    result = _assign_spatial_chunks_for_axis_aligned_bounding_boxes(df, geometry_cols, bounds, gridspec)
    counts = result['id_buf'].value_counts().to_dict()
    assert counts == {b'A': 1, b'B': 4}, counts


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
        'level': [0, 0],
        'id_buf': [b'A', b'B'],
        'ann_buf': [b'a', b'b'],
    }, index=[10, 20])

    result = _assign_spatial_chunks_for_ellipsoids(df, geometry_cols, bounds, gridspec)
    counts = result['id_buf'].value_counts().to_dict()
    assert counts[b'A'] == 1
    assert counts[b'B'] > 1, (
        f"A 0.6-diameter ellipsoid centred mid-grid should overlap multiple "
        f"0.25-wide chunks, but produced {counts[b'B']} rows"
    )


def test_short_annotation_inside_one_chunk_produces_single_row():
    """Round-trip sanity: a single-chunk annotation must still emit exactly one row."""
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    gridspec = _single_level_gridspec((4, 4, 4))
    geometry_cols = [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']]

    df = pd.DataFrame({
        'xa': [0.1], 'ya': [0.1], 'za': [0.1],
        'xb': [0.15], 'yb': [0.15], 'zb': [0.15],
        'level': [0],
        'id_buf': [b'A'],
        'ann_buf': [b'a'],
    }, index=[42])

    result = _assign_spatial_chunks_for_lines(df, geometry_cols, bounds, gridspec)
    assert len(result) == 1
    assert result['id_buf'].iloc[0] == b'A'
