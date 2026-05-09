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
