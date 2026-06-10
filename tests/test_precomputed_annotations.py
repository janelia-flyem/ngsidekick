import pytest
import numpy as np
import pandas as pd
from bokeh.palettes import Category10

from neuroglancer.coordinate_space import CoordinateSpace
from ngsidekick.annotations.precomputed import write_precomputed_annotations


@pytest.fixture(scope='session')
def test_output_dir(tmp_path_factory):
    """
    Create a common temporary directory for all tests to write to.
    This is session-scoped, so all tests share the same temp directory.
    """
    return tmp_path_factory.mktemp('test_annotations')


def _point_testdata(
    num_clusters=100,
    num_points_per_cluster=1_000,
    cluster_spacing=[10, 100, 1000],
    point_spacing=[1, 10, 100]
):
    centers = np.random.normal(0, cluster_spacing, (num_clusters, 3))
    points = []
    for center in centers:
        p = np.random.normal(center, point_spacing, (num_points_per_cluster, 3))
        points.append(p)
    points = np.concatenate(points, axis=0)
    points_df = pd.DataFrame(points, columns=[*'xyz'])
    points_df['cluster_id'] = np.repeat(np.arange(num_clusters), num_points_per_cluster).astype(np.uint32)

    colormap = pd.Series(Category10[10])

    def hex_to_rgb(h):
        return [int(c, base=16) for c in (h[1:3], h[3:5], h[5:7])]

    rgb = colormap.loc[points_df['cluster_id'] % 10].map(hex_to_rgb).tolist()
    points_df[['cluster_color_r', 'cluster_color_g', 'cluster_color_b']] = rgb
    points_df['cluster_color_a'] = 255

    return points_df

@pytest.fixture
def point_testdata():
    return _point_testdata()

@pytest.fixture
def pointpair_testdata(
    num_clusters=100,
    num_points_per_cluster=1_000,
    cluster_spacing=[10, 100, 1000],
    point_spacing=[1, 10, 100],
    line_sigma=[.1, 1, 10]
):
    midpoints = _point_testdata(
        num_clusters=num_clusters,
        num_points_per_cluster=num_points_per_cluster,
        cluster_spacing=cluster_spacing,
        point_spacing=point_spacing
    )
    deltas = np.random.normal(0, line_sigma, (len(midpoints), 3))
    starts = midpoints[[*'xyz']] - deltas
    ends = midpoints[[*'xyz']] + deltas
    pairs = pd.DataFrame(
        np.concatenate([starts, ends], axis=1),
        columns=['xa', 'ya', 'za', 'xb', 'yb', 'zb']
    )
    return pd.concat((pairs, midpoints[[c for c in midpoints.columns if c not in 'xyz']]), axis=1)


@pytest.fixture
def ellipsoid_testdata(
    num_clusters=100,
    num_points_per_cluster=1_000,
    cluster_spacing=[10, 100, 1000],
    point_spacing=[1, 10, 100],
    radius_sigma=[0.2, 2, 20],
    radius_mean=[1, 10, 100]
):
    ellipsoid_centers = _point_testdata(
        num_clusters=num_clusters,
        num_points_per_cluster=num_points_per_cluster,
        cluster_spacing=cluster_spacing,
        point_spacing=point_spacing
    )
    # Create radii for each ellipsoid (one per row, not per cluster)
    num_ellipsoids = len(ellipsoid_centers)
    radii = np.abs(radius_mean + np.random.normal(0, radius_sigma, (num_ellipsoids, 3)))
    ellipsoid_centers[['rx', 'ry', 'rz']] = radii
    return ellipsoid_centers


def test_point_annotations(point_testdata, test_output_dir):
    cs = CoordinateSpace(names=[*'xyz'], units=['m', 'm', 'm'], scales=[100, 10, 1])
    write_precomputed_annotations(
        point_testdata,
        cs,
        'point',
        ['cluster_color'],
        ['cluster_id'],
        output_dir=test_output_dir / 'test-point-annotations',
        write_by_spatial_chunk=True,
        num_spatial_levels=6,
        target_chunk_limit=10_000
    )


def test_point_annotations_from_feather(point_testdata, test_output_dir, tmp_path):
    """End-to-end Feather input path: write to .feather, then pass the
    path to write_precomputed_annotations and verify the output exists."""
    import pyarrow.feather as feather

    feather_path = tmp_path / 'points.feather'
    df = point_testdata.copy()
    # Feather has no index column; surface annotation_id explicitly.
    df.insert(0, 'annotation_id', df.index.to_numpy(np.uint64))
    feather.write_feather(df, feather_path)

    cs = CoordinateSpace(names=[*'xyz'], units=['m', 'm', 'm'], scales=[100, 10, 1])
    out_dir = test_output_dir / 'test-point-annotations-feather'
    write_precomputed_annotations(
        str(feather_path),
        cs,
        'point',
        ['cluster_color'],
        ['cluster_id'],
        output_dir=out_dir,
        write_by_spatial_chunk=True,
        num_spatial_levels=4,
        target_chunk_limit=10_000,
    )
    assert (out_dir / 'info').exists()
    assert (out_dir / 'by_id').exists()


def _write_and_run(point_df, cs, tmp_path, out_dir):
    """Helper: write a df to feather, run write_precomputed_annotations, return out_dir."""
    import pyarrow.feather as feather
    feather_path = tmp_path / 'pts.feather'
    feather.write_feather(point_df, feather_path)
    write_precomputed_annotations(
        str(feather_path), cs, 'point',
        output_dir=out_dir,
        write_by_spatial_chunk=True,
        num_spatial_levels=2,
        target_chunk_limit=10_000,
    )
    return out_dir


def test_feather_input_with_explicit_annotation_id(point_testdata, test_output_dir, tmp_path):
    """File has an explicit ``annotation_id`` column: used as-is."""
    cs = CoordinateSpace(names=[*'xyz'], units=['m']*3, scales=[100, 10, 1])
    df = point_testdata[[*'xyz']].copy()
    df['annotation_id'] = np.arange(len(df), dtype=np.uint64)
    out = _write_and_run(df, cs, tmp_path, test_output_dir / 'feather-explicit-ann-id')
    assert (out / 'info').exists()


def test_feather_input_with_named_index(point_testdata, test_output_dir, tmp_path):
    """File was written by pandas with a named index: column is renamed
    to annotation_id via zero-copy PyArrow rename."""
    cs = CoordinateSpace(names=[*'xyz'], units=['m']*3, scales=[100, 10, 1])
    df = point_testdata[[*'xyz']].copy()
    df.index = pd.Index(np.arange(len(df), dtype=np.uint64) + 1000, name='my_id')
    out = _write_and_run(df, cs, tmp_path, test_output_dir / 'feather-named-index')
    assert (out / 'info').exists()


def test_feather_input_with_default_rangeindex(point_testdata, test_output_dir, tmp_path):
    """File was written from a default RangeIndex: no real id column;
    annotation_id is synthesized via ROW_NUMBER()."""
    cs = CoordinateSpace(names=[*'xyz'], units=['m']*3, scales=[100, 10, 1])
    df = point_testdata[[*'xyz']].copy()  # default RangeIndex
    out = _write_and_run(df, cs, tmp_path, test_output_dir / 'feather-rangeindex')
    assert (out / 'info').exists()


def test_feather_input_with_categorical_property(test_output_dir, tmp_path):
    """Feather files store pandas categoricals as Arrow dictionary-encoded
    columns. DuckDB registers those as plain VARCHAR, so by the time the
    encoder sees the column it's object-dtype strings, not pandas
    Categorical. The encoder must recover the codes from the spec's
    ``enum_labels``."""
    import pyarrow.feather as feather

    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        'x': rng.random(n).astype(np.float32),
        'y': rng.random(n).astype(np.float32),
        'z': rng.random(n).astype(np.float32),
        # Includes a label with bracket characters that look like a
        # number-coercion failure when surfaced in error messages,
        # matching the field-report case.
        'kind': pd.Categorical(rng.choice(['a', 'b', '<unspecified>'], n)),
    })
    feather_path = tmp_path / 'cat.feather'
    feather.write_feather(df, feather_path)

    cs = CoordinateSpace(names=[*'xyz'], units=['nm']*3, scales=[1, 1, 1])
    out_dir = test_output_dir / 'feather-categorical'
    write_precomputed_annotations(
        str(feather_path), cs, 'point',
        properties=['kind'],
        output_dir=out_dir,
        num_spatial_levels=2,
        target_chunk_limit=50,
    )
    assert (out_dir / 'info').exists()


@pytest.mark.parametrize('annotation_type,geom_cols', [
    ('point', ['x', 'y', 'z']),
    ('line', ['xa', 'ya', 'za', 'xb', 'yb', 'zb']),
    ('axis_aligned_bounding_box', ['xa', 'ya', 'za', 'xb', 'yb', 'zb']),
    ('ellipsoid', ['x', 'y', 'z', 'rx', 'ry', 'rz']),
])
def test_integer_geometry_columns(annotation_type, geom_cols, test_output_dir, tmp_path):
    """Geometry columns stored as int32 must round-trip through bounds,
    spatial assignment, and encoding. The DuckDB ``isnan()`` check, the
    numba grid-code kernels, and the float32 encoder cast all need to
    cope with integer-dtype input."""
    import pyarrow.feather as feather

    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({c: rng.integers(0, 1000, n).astype(np.int32) for c in geom_cols})
    # For line/aabb the geometry expects ``a < b``; nudge so each axis's
    # second point is strictly greater than the first.
    if annotation_type in ('line', 'axis_aligned_bounding_box'):
        for axis in 'xyz':
            df[f'{axis}b'] = df[f'{axis}a'] + 10

    feather_path = tmp_path / f'int-{annotation_type}.feather'
    feather.write_feather(df, feather_path)

    cs = CoordinateSpace(names=[*'xyz'], units=['nm']*3, scales=[1, 1, 1])
    out_dir = test_output_dir / f'int-geom-{annotation_type}'
    write_precomputed_annotations(
        str(feather_path), cs, annotation_type,
        output_dir=out_dir,
        num_spatial_levels=2,
        target_chunk_limit=20,
    )
    assert (out_dir / 'info').exists()


def test_spatial_kernel_batched_matches_single_batch(point_testdata, test_output_dir, tmp_path, monkeypatch):
    """The batched spatial kernel must produce identical (level,
    chunk_code, row_pos) assignments to a single full-data kernel call,
    so downstream output is bit-for-bit equivalent."""
    import pyarrow.feather as feather
    from ngsidekick.annotations.precomputed import _spatial

    feather_path = tmp_path / 'points.feather'
    df = point_testdata.copy()
    df.insert(0, 'annotation_id', df.index.to_numpy(np.uint64))
    feather.write_feather(df, feather_path)

    cs = CoordinateSpace(names=[*'xyz'], units=['m', 'm', 'm'], scales=[100, 10, 1])
    seed = 12345

    def _run(batch_size, subdir):
        monkeypatch.setattr(_spatial, '_SPATIAL_KERNEL_BATCH_SIZE', batch_size)
        np.random.seed(seed)
        out_dir = test_output_dir / subdir
        write_precomputed_annotations(
            str(feather_path),
            cs,
            'point',
            output_dir=out_dir,
            write_by_id=False,
            write_by_relationship=False,
            write_by_spatial_chunk=True,
            num_spatial_levels=4,
            target_chunk_limit=10_000,
        )
        return out_dir

    full_dir = _run(batch_size=10_000_000, subdir='test-spatial-single-batch')
    batched_dir = _run(batch_size=7_000, subdir='test-spatial-multi-batch')

    # The spatial subdirectories should be byte-equivalent across runs.
    import os
    level_dirs = sorted(p.name for p in full_dir.iterdir() if p.name.startswith('by_spatial_level_'))
    assert level_dirs, "Expected at least one spatial level directory"
    for level_dir in level_dirs:
        for fname in os.listdir(full_dir / level_dir):
            full_bytes = (full_dir / level_dir / fname).read_bytes()
            batched_bytes = (batched_dir / level_dir / fname).read_bytes()
            assert full_bytes == batched_bytes, f"{level_dir}/{fname} differs"


def test_line_annotations(pointpair_testdata, test_output_dir):
    cs = CoordinateSpace(names=[*'xyz'], units=['m', 'm', 'm'], scales=[100, 10, 1])
    write_precomputed_annotations(
        pointpair_testdata,
        cs,
        'line',
        ['cluster_color'],
        ['cluster_id'],
        output_dir=test_output_dir / 'test-line-annotations',
        write_by_spatial_chunk=True,
        num_spatial_levels=6,
        target_chunk_limit=10
    )


def test_list_typed_relationship(test_output_dir):
    """
    A list-typed relationship column (one annotation referencing multiple
    related segments) must:
      - Enumerate the distinct segment ids via UNNEST.
      - Drop within-annotation duplicate ids.
      - Skip annotations whose list is empty or NULL.
    """
    import struct, json
    import tensorstore as ts

    # 3 annotations + the by-rel index 'nearby_mito':
    #   ann 10 -> [100, 200]
    #   ann 20 -> [100, 100, 300]  (the duplicate 100 must be deduped)
    #   ann 30 -> []                 (empty list contributes nothing)
    df = pd.DataFrame({
        'xa': [0.0, 10.0, 20.0], 'ya': [0.0, 0.0, 0.0], 'za': [0.0, 0.0, 0.0],
        'xb': [5.0, 15.0, 25.0], 'yb': [5.0, 5.0, 5.0], 'zb': [0.0, 0.0, 0.0],
        'nearby_mito': [[100, 200], [100, 100, 300], []],
    }, index=pd.Index([10, 20, 30], dtype=np.uint64))

    cs = CoordinateSpace(names=[*'xyz'], units=['nm']*3, scales=[1, 1, 1])
    out = test_output_dir / 'test-list-rel'
    write_precomputed_annotations(
        df, cs, 'line',
        relationships=['nearby_mito'],
        output_dir=out,
        write_sharded=True,
        write_by_id=False, write_by_spatial_chunk=False,
    )

    info = json.loads((out / 'info').read_text())
    kv = ts.KvStore.open({
        'driver': 'neuroglancer_uint64_sharded',
        'metadata': info['relationships'][0]['sharding'],
        'base': f'file://{out}/by_rel_nearby_mito',
    }).result()

    def read_segment(seg):
        raw = bytes(kv.read(int(seg).to_bytes(8, 'big')).result().value)
        count = struct.unpack('<Q', raw[:8])[0]
        rec_size = 24
        ids = struct.unpack(f'<{count}Q', raw[8 + count*rec_size:])
        return count, sorted(ids)

    assert read_segment(100) == (2, [10, 20])  # deduped
    assert read_segment(200) == (1, [10])
    assert read_segment(300) == (1, [20])
    # Empty list (annotation 30) contributes no segment.
    assert kv.read(int(0).to_bytes(8, 'big')).result().state == 'missing'


def test_box_annotations(pointpair_testdata, test_output_dir):
    cs = dict(names=[*'xyz'], units=['m', 'm', 'm'], scales=[100, 10, 1])
    write_precomputed_annotations(
        pointpair_testdata,
        cs,
        'axis_aligned_bounding_box',
        ['cluster_color'],
        ['cluster_id'],
        output_dir=test_output_dir / 'test-box-annotations',
        write_by_spatial_chunk=True,
        num_spatial_levels=6,
        target_chunk_limit=10
    )


@pytest.fixture
def polyline_testdata(
    num_clusters=20,
    num_polylines_per_cluster=50,
    cluster_spacing=[10, 100, 1000],
    point_spacing=[1, 10, 100],
    points_per_polyline_range=(3, 7),
    step_sigma=[0.1, 1, 10],
):
    """
    Returns ``(main_df, aux_df)``:

    - ``main_df`` is indexed by annotation_id with property + relationship columns
      (matching the conventions used by the other annotation types in this module).
    - ``aux_df`` has one row per polyline vertex with columns ``['x','y','z','annotation_id']``.

    Each polyline's vertex count is drawn uniformly from
    ``[points_per_polyline_range[0], points_per_polyline_range[1]]`` (inclusive).
    """
    centers = np.random.normal(0, cluster_spacing, (num_clusters, 3))
    n_polys = num_clusters * num_polylines_per_cluster

    annotation_ids = np.arange(n_polys, dtype=np.uint64)
    cluster_ids = np.repeat(np.arange(num_clusters), num_polylines_per_cluster).astype(np.uint32)

    polyline_starts = np.empty((n_polys, 3), dtype=np.float64)
    for i, center in enumerate(centers):
        polyline_starts[i*num_polylines_per_cluster:(i+1)*num_polylines_per_cluster] = (
            np.random.normal(center, point_spacing, (num_polylines_per_cluster, 3))
        )

    lo, hi = points_per_polyline_range
    per_polyline_counts = np.random.randint(lo, hi + 1, size=n_polys)

    aux_rows = []
    for poly_i, start, n_points in zip(annotation_ids, polyline_starts, per_polyline_counts):
        steps = np.cumsum(np.random.normal(0, step_sigma, (n_points, 3)), axis=0)
        verts = start + steps
        for v in verts:
            aux_rows.append((v[0], v[1], v[2], int(poly_i)))
    aux_df = pd.DataFrame(aux_rows, columns=['x', 'y', 'z', 'annotation_id'])

    colormap = pd.Series(Category10[10])

    def hex_to_rgb(h):
        return [int(c, base=16) for c in (h[1:3], h[3:5], h[5:7])]

    rgb = colormap.loc[cluster_ids % 10].map(hex_to_rgb).tolist()
    main_df = pd.DataFrame(index=pd.Index(annotation_ids))
    main_df[['cluster_color_r', 'cluster_color_g', 'cluster_color_b']] = rgb
    main_df['cluster_color_a'] = 255
    main_df['cluster_id'] = cluster_ids
    return main_df, aux_df


def test_polyline_annotations(polyline_testdata, test_output_dir):
    main_df, aux_df = polyline_testdata
    cs = CoordinateSpace(names=[*'xyz'], units=['m', 'm', 'm'], scales=[100, 10, 1])
    write_precomputed_annotations(
        main_df,
        cs,
        'polyline',
        ['cluster_color'],
        ['cluster_id'],
        output_dir=test_output_dir / 'test-polyline-annotations',
        polyline_points=aux_df,
        write_by_spatial_chunk=True,
        num_spatial_levels=4,
        target_chunk_limit=10,
    )


def test_polyline_annotations_aux_only(test_output_dir):
    """df=None convenience: main table is synthesized from polyline_points."""
    aux_df = pd.DataFrame({
        'x': [0.1, 0.2, 0.3, 0.5, 0.9],
        'y': [0.1, 0.2, 0.3, 0.5, 0.5],
        'z': [0.1, 0.2, 0.3, 0.5, 0.5],
        'annotation_id': [10, 10, 10, 20, 20],
    })
    cs = CoordinateSpace(names=[*'xyz'], units=['nm']*3, scales=[1, 1, 1])
    write_precomputed_annotations(
        None,
        cs,
        'polyline',
        polyline_points=aux_df,
        output_dir=test_output_dir / 'test-polyline-aux-only',
        write_by_spatial_chunk=True,
        num_spatial_levels=2,
        target_chunk_limit=1,
    )


def test_ellipsoid_annotations(ellipsoid_testdata, test_output_dir):
    cs = CoordinateSpace(names=[*'xyz'], units=['m', 'm', 'm'], scales=[100, 10, 1])
    write_precomputed_annotations(
        ellipsoid_testdata,
        cs,
        'ellipsoid',
        ['cluster_color'],
        ['cluster_id'],
        output_dir=test_output_dir / 'test-ellipsoid-annotations',
        write_by_spatial_chunk=True,
        num_spatial_levels=6,
        target_chunk_limit=10
    )

@pytest.mark.manual
def test_inspect_test_results(
    test_output_dir,
    point_testdata,
    pointpair_testdata,
    ellipsoid_testdata,
    polyline_testdata,
):
    """
    Inspect the exported annotations from this test suite in neuroglancer.
    This test is marked as 'manual' and skipped by default since it requires manual inspection.
    
    - Produce a neuroglancer link populated with the exported
      annotations, and print it to the console.
    - Launch cors_webserver.py to host the temporary directory
    - Wait for the user to interrupt.
    
    To run this test: pytest -s -m manual tests/test_precomputed_annotations.py
    """
    import json
    import ngsidekick as ngsk
    
    # First, ensure all test data is written
    print("\nWriting test annotations to", test_output_dir)
    test_point_annotations(point_testdata, test_output_dir)
    test_line_annotations(pointpair_testdata, test_output_dir)
    test_box_annotations(pointpair_testdata, test_output_dir)
    test_ellipsoid_annotations(ellipsoid_testdata, test_output_dir)
    test_polyline_annotations(polyline_testdata, test_output_dir)
    
    # Start CORS webserver in background
    port = 9010
    bind_addr = "127.0.0.1"
    
    print(f"\nStarting CORS webserver on http://{bind_addr}:{port}")
    print(f"Serving directory: {test_output_dir}")
    
    server_process, NGSK_SERVER_ADDRESS, log_file = ngsk.serve_directory(
        test_output_dir,
        port=port,
        bind=bind_addr,
        background=True
    )
    
    # Give server time to start
    import time
    time.sleep(1)
    
    ng_state = {
        "dimensions": {"x": [1, "m"], "y": [1, "m"], "z": [1, "m"]},
        "position": [0, 0, 0],
        "crossSectionScale": 10,
        "projectionScale": 10_000,
        "showSlices": False,
        "layers": [
            {
                "type": "annotation",
                "source": f"precomputed://{NGSK_SERVER_ADDRESS}/test-point-annotations",
                "shader": """\nvoid main() {\n  setColor(prop_cluster_color());\n}\n""",
                "name": "points",
                "annotations": []
            },
            {
                "type": "annotation",
                "source": f"precomputed://{NGSK_SERVER_ADDRESS}/test-line-annotations",
                "shader": """\nvoid main() {\n  setColor(prop_cluster_color());\n}\n""",
                "name": "lines",
                "annotations": []
            },
            {
                "type": "annotation",
                "source": f"precomputed://{NGSK_SERVER_ADDRESS}/test-box-annotations",
                "shader": """\nvoid main() {\n  setColor(prop_cluster_color());\n}\n""",
                "name": "boxes",
                "annotations": []
            },
            {
                "type": "annotation",
                "source": f"precomputed://{NGSK_SERVER_ADDRESS}/test-ellipsoid-annotations",
                "shader": """\nvoid main() {\n  setColor(prop_cluster_color());\n}\n""",
                "name": "ellipsoids",
                "annotations": []
            },
            {
                "type": "annotation",
                "source": f"precomputed://{NGSK_SERVER_ADDRESS}/test-polyline-annotations",
                "shader": """\nvoid main() {\n  setColor(prop_cluster_color());\n}\n""",
                "name": "polylines",
                "annotations": []
            }
        ],
        "layout": "4panel"
    }
    
    # Encode state as JSON fragment
    import urllib.parse
    state_json = json.dumps(ng_state)
    encoded_state = urllib.parse.quote(state_json)
    
    neuroglancer_url = f"https://neuroglancer-demo.appspot.com/#!{encoded_state}"
    
    print("\n" + "="*80)
    print("Neuroglancer URL:")
    print(neuroglancer_url)
    print("="*80)
    print("\nPress Ctrl+C to stop the server and exit...")
    print()
    
    try:
        # Wait for user interrupt
        server_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server_process.terminate()
        server_process.wait(timeout=5)


if __name__ == "__main__":
    pytest.main(['-s', '-m', 'manual', 'test_precomputed_annotations.py'])
