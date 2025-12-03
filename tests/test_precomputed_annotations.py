import pytest
import weakref
import numpy as np
import pandas as pd
from bokeh.palettes import Category10

from neuroglancer.coordinate_space import CoordinateSpace
from ngsidekick.annotations.precomputed import write_precomputed_annotations, TableHandle


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

def test_early_data_deletion(point_testdata, test_output_dir):
    """
    If we wrap the data in a TableHandle and delete our own reference to the data,
    then the original reference should be invalid by the time the annotations are written.
    We can verify this with a weakref.
    """
    point_testdata = point_testdata.sample(100).copy()

    ref = weakref.ref(point_testdata)
    lines_handle = TableHandle(point_testdata)
    del point_testdata

    cs = CoordinateSpace(names=[*'xyz'], units=['m', 'm', 'm'], scales=[100, 10, 1])
    write_precomputed_annotations(
        lines_handle,
        cs,
        'point',
        output_dir=test_output_dir / 'test-tablehandle-deletion',
        write_by_spatial_chunk=True,
        num_spatial_levels=6,
        target_chunk_limit=10_000
    )
    
    assert lines_handle.df is None
    assert ref() is None


@pytest.mark.manual
def test_inspect_test_results(test_output_dir, point_testdata, pointpair_testdata, ellipsoid_testdata):
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
    from ngsidekick import serve_directory
    
    # First, ensure all test data is written
    print("\nWriting test annotations to", test_output_dir)
    test_point_annotations(point_testdata, test_output_dir)
    test_line_annotations(pointpair_testdata, test_output_dir)
    test_box_annotations(pointpair_testdata, test_output_dir)
    test_ellipsoid_annotations(ellipsoid_testdata, test_output_dir)
    
    # Start CORS webserver in background
    port = 9000
    bind_addr = "127.0.0.1"
    
    print(f"\nStarting CORS webserver on http://{bind_addr}:{port}")
    print(f"Serving directory: {test_output_dir}")
    
    server_process = serve_directory(
        test_output_dir,
        port=port,
        bind=bind_addr,
        background=True
    )
    
    # Give server time to start
    import time
    time.sleep(1)
    
    # Construct neuroglancer state with all annotation layers
    base_url = f"http://{bind_addr}:{port}"
    
    ng_state = {
        "dimensions": {"x": [1, "m"], "y": [1, "m"], "z": [1, "m"]},
        "position": [0, 0, 0],
        "crossSectionScale": 10,
        "projectionScale": 10_000,
        "showSlices": False,
        "layers": [
            {
                "type": "annotation",
                "source": f"precomputed://{base_url}/test-point-annotations",
                "shader": """\nvoid main() {\n  setColor(prop_cluster_color());\n}\n""",
                "name": "points",
                "annotations": []
            },
            {
                "type": "annotation",
                "source": f"precomputed://{base_url}/test-line-annotations",
                "shader": """\nvoid main() {\n  setColor(prop_cluster_color());\n}\n""",
                "name": "lines",
                "annotations": []
            },
            {
                "type": "annotation",
                "source": f"precomputed://{base_url}/test-box-annotations",
                "shader": """\nvoid main() {\n  setColor(prop_cluster_color());\n}\n""",
                "name": "boxes",
                "annotations": []
            },
            {
                "type": "annotation",
                "source": f"precomputed://{base_url}/test-ellipsoid-annotations",
                "shader": """\nvoid main() {\n  setColor(prop_cluster_color());\n}\n""",
                "name": "ellipsoids",
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
    pytest.main(['-s', 'test_precomputed_annotations.py'])
