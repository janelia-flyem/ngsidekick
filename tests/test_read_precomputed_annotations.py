import json

import numpy as np
import pandas as pd
import pytest

from neuroglancer.coordinate_space import CoordinateSpace
from ngsidekick.annotations.precomputed import (
    write_precomputed_annotations,
    read_precomputed_annotations,
)

CS = CoordinateSpace(names=[*'xyz'], units=['nm'] * 3, scales=[8, 8, 8])

GEOMETRY_COLS = {
    'point': [*'xyz'],
    'line': ['xa', 'ya', 'za', 'xb', 'yb', 'zb'],
    'axis_aligned_bounding_box': ['xa', 'ya', 'za', 'xb', 'yb', 'zb'],
    'ellipsoid': ['x', 'y', 'z', 'rx', 'ry', 'rz'],
    'polyline': [],
}

PROPERTIES = ['score', 'level', 'tag', 'kind', 'color', 'fill']
RELATIONSHIPS = ['pre', 'partners']


def _annotation_ids(n, rng):
    """Unique uint64 IDs, including some that don't fit in int64."""
    ids = rng.choice(2**62, n, replace=False).astype(np.uint64)
    ids[:3] += np.uint64(2**63)
    return ids


def _testdata(annotation_type, n=500, seed=0):
    """
    Construct a DataFrame (and polyline_points, for polylines) with
    every kind of property and relationship.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=pd.Index(_annotation_ids(n, rng), name='annotation_id'))
    for c in GEOMETRY_COLS[annotation_type]:
        lo = 1 if c.startswith('r') else 0
        df[c] = rng.uniform(lo, 1000, n).astype(np.float32)

    df['score'] = rng.random(n).astype(np.float32)
    df['level'] = rng.integers(-1000, 1000, n).astype(np.int16)
    df['tag'] = rng.integers(0, 256, n).astype(np.uint8)
    df['kind'] = pd.Categorical.from_codes(rng.integers(0, 3, n), ['alpha', 'beta', 'gamma'])
    for c in 'rgb':
        df[f'color_{c}'] = rng.integers(0, 256, n).astype(np.uint8)
    for c in 'rgba':
        df[f'fill_{c}'] = rng.integers(0, 256, n).astype(np.uint8)

    df['pre'] = rng.integers(1, 2**63, n).astype(np.uint64)
    df['partners'] = [
        rng.integers(1, 50, k).astype(np.uint64)
        for k in rng.integers(0, 4, n)
    ]

    polyline_points = None
    if annotation_type == 'polyline':
        counts = rng.integers(1, 6, n)
        polyline_points = pd.DataFrame({
            'annotation_id': np.repeat(df.index.to_numpy(), counts),
            **{c: rng.uniform(0, 1000, counts.sum()).astype(np.float32) for c in 'xyz'},
        })
    return df, polyline_points


def _write(df, annotation_type, out, polyline_points=None, **kwargs):
    kwargs = {
        'properties': PROPERTIES,
        'relationships': RELATIONSHIPS,
        'num_spatial_levels': 3,
        'target_chunk_limit': 100,
        **kwargs,
    }
    write_precomputed_annotations(
        df, CS, annotation_type, output_dir=out, polyline_points=polyline_points, **kwargs,
    )


def _assert_df_equal(actual, expected):
    """
    Compare annotation tables, including list-valued relationship columns.
    """
    assert list(actual.columns) == list(expected.columns)
    assert (actual.index == expected.index).all()
    assert actual.index.dtype == np.uint64
    assert actual.index.name == 'annotation_id'
    for c in expected.columns:
        a, e = actual[c], expected[c]
        if e.dtype == object:
            assert a.dtype == object, c
            assert all(np.array_equal(x, y) for x, y in zip(a, e)), c
            assert all(x.dtype == np.uint64 for x in a), c
        else:
            pd.testing.assert_series_equal(a, e, check_names=False, obj=c)


def _assert_points_equal(actual, expected):
    """Compare polyline_points tables (order of rows matters only within each annotation)."""
    expected = expected.sort_values('annotation_id', kind='stable').reset_index(drop=True)
    assert list(actual.columns) == ['annotation_id', *CS.names]
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected[['annotation_id', *CS.names]],
        check_dtype=False,
    )
    assert actual['annotation_id'].dtype == np.uint64


@pytest.mark.parametrize('sharded', [True, False])
@pytest.mark.parametrize('annotation_type', list(GEOMETRY_COLS))
def test_roundtrip(annotation_type, sharded, tmp_path):
    df, polyline_points = _testdata(annotation_type)
    _write(df, annotation_type, tmp_path, polyline_points, write_sharded=sharded)

    a = read_precomputed_annotations(tmp_path)

    assert a.annotation_type == annotation_type
    assert a.coord_space.to_json() == CS.to_json()
    assert a.relationships == RELATIONSHIPS
    assert [p['id'] for p in a.properties] == [p['id'] for p in a.info['properties']]
    assert a.info == json.loads((tmp_path / 'info').read_text())

    expected = df.sort_index()
    _assert_df_equal(a.df, expected)
    assert a.df['kind'].cat.categories.tolist() == ['alpha', 'beta', 'gamma']

    if annotation_type == 'polyline':
        _assert_points_equal(a.polyline_points, polyline_points)
    else:
        assert a.polyline_points is None


@pytest.mark.parametrize('sharded', [True, False])
def test_read_ids(sharded, tmp_path):
    df, _ = _testdata('point')
    _write(df, 'point', tmp_path, write_sharded=sharded)

    ids = df.index[[7, 3, 400, 3]]
    a = read_precomputed_annotations(tmp_path, ids)
    _assert_df_equal(a.df, df.loc[ids])

    # Also accept a plain list of Python ints.
    a = read_precomputed_annotations(tmp_path, [int(i) for i in ids])
    _assert_df_equal(a.df, df.loc[ids])

    a = read_precomputed_annotations(tmp_path, [])
    assert len(a.df) == 0
    assert list(a.df.columns) == list(df.columns)

    missing = int(df.index.max()) + 1
    with pytest.raises(KeyError):
        read_precomputed_annotations(tmp_path, [ids[0], missing])


def test_read_ids_polyline(tmp_path):
    df, polyline_points = _testdata('polyline')
    _write(df, 'polyline', tmp_path, polyline_points)

    ids = df.index[[9, 2, 5]]
    a = read_precomputed_annotations(tmp_path, ids)
    _assert_df_equal(a.df, df.loc[ids])

    # Vertices are listed in the order of the requested ids.
    expected_points = pd.concat([
        polyline_points.query('annotation_id == @i') for i in ids
    ])
    pd.testing.assert_frame_equal(
        a.polyline_points.reset_index(drop=True),
        expected_points[['annotation_id', *CS.names]].reset_index(drop=True),
    )


def test_relationship_formats(tmp_path):
    """
    Relationships with exactly one related ID per annotation are returned as uint64,
    and other relationships (even if they're only 0-or-1) are returned as arrays.
    """
    df = pd.DataFrame({
        'x': [1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0], 'z': [1.0, 2.0, 3.0],
        'one': np.array([10, 20, 30], dtype=np.uint64),
        'one_as_lists': [[10], [20], [30]],
        'at_most_one': [[10], [], [30]],
        'none': [[], [], []],
        'many': [[1, 2], [3], [4, 5, 6]],
    }, index=pd.Index([1, 2, 3], dtype=np.uint64)).astype({c: np.float32 for c in 'xyz'})
    rels = ['one', 'one_as_lists', 'at_most_one', 'none', 'many']
    write_precomputed_annotations(df, CS, 'point', relationships=rels, output_dir=tmp_path)

    a = read_precomputed_annotations(tmp_path)
    assert a.df['one'].dtype == np.uint64
    assert a.df['one_as_lists'].dtype == np.uint64
    assert a.df['one_as_lists'].tolist() == [10, 20, 30]
    for c in ['at_most_one', 'none', 'many']:
        assert a.df[c].dtype == object
        assert [x.tolist() for x in a.df[c]] == df[c].tolist()


@pytest.mark.parametrize('annotation_type', ['line', 'polyline'])
def test_rewrite(annotation_type, tmp_path):
    """The result can be passed directly back to write_precomputed_annotations()."""
    df, polyline_points = _testdata(annotation_type)
    _write(df, annotation_type, tmp_path / 'a', polyline_points)

    a = read_precomputed_annotations(tmp_path / 'a')
    write_precomputed_annotations(
        *a[:5], output_dir=tmp_path / 'b', polyline_points=a.polyline_points
    )
    b = read_precomputed_annotations(tmp_path / 'b')

    _assert_df_equal(b.df, a.df)
    assert b.properties == a.properties
    if annotation_type == 'polyline':
        pd.testing.assert_frame_equal(b.polyline_points, a.polyline_points)


def test_property_order_in_info(tmp_path):
    """
    Per the spec, properties are encoded grouped by alignment (4-byte, then 2-byte,
    then 1-byte), regardless of the order they're listed in the info file.
    """
    df, _ = _testdata('point')
    _write(df, 'point', tmp_path)

    info_path = tmp_path / 'info'
    info = json.loads(info_path.read_text())
    props = {p['id']: p for p in info['properties']}

    # Reverse the order in which the alignment classes are listed,
    # but keep the relative order within each class.
    info['properties'] = [props[p] for p in ['tag', 'kind', 'color', 'fill', 'level', 'score']]
    info_path.write_text(json.dumps(info))

    a = read_precomputed_annotations(tmp_path)
    assert [p['id'] for p in a.properties] == ['tag', 'kind', 'color', 'fill', 'level', 'score']
    expected = df.sort_index()
    _assert_df_equal(a.df[expected.columns], expected)


def test_matches_neuroglancer_reader(tmp_path):
    """Cross-check against neuroglancer's own reader."""
    from neuroglancer.read_precomputed_annotations import AnnotationReader

    df, _ = _testdata('line', n=50)
    _write(df, 'line', tmp_path)
    a = read_precomputed_annotations(tmp_path)

    reader = AnnotationReader(f'file://{tmp_path}')
    for annotation_id, row in a.df.iterrows():
        ann = reader.by_id[int(annotation_id)]
        assert np.array_equal(ann.point_a, row[['xa', 'ya', 'za']].to_numpy(np.float32))
        assert np.array_equal(ann.point_b, row[['xb', 'yb', 'zb']].to_numpy(np.float32))
        assert [int(s) for s in ann.segments[0]] == [row['pre']]
        assert [int(s) for s in ann.segments[1]] == row['partners'].tolist()

        # neuroglancer's reader returns properties in info-file order,
        # with enums as their raw stored values.
        props = dict(zip([p.id for p in reader.properties], ann.props))
        assert props['score'] == row['score']
        assert props['level'] == row['level']
        assert ['alpha', 'beta', 'gamma'][props['kind']] == row['kind']
        assert list(props['color']) == [row['color_r'], row['color_g'], row['color_b']]


def test_reads_neuroglancer_writer_output(tmp_path):
    """
    Read data written by neuroglancer's own (unsharded) writer,
    including an enum property whose enum_values aren't 0..N-1.
    """
    from neuroglancer.write_annotations import AnnotationWriter
    from neuroglancer.viewer_state import AnnotationPropertySpec

    writer = AnnotationWriter(
        CS, 'point',
        relationships=['seg'],
        properties=[
            AnnotationPropertySpec(id='kind', type='uint8', enum_values=[5, 7, 9], enum_labels=['a', 'b', 'c']),
            AnnotationPropertySpec(id='score', type='float32'),
        ],
    )
    writer.add_point([1, 2, 3], id=11, kind=7, score=0.5, seg=[100, 200])
    writer.add_point([4, 5, 6], id=22, kind=9, score=1.5, seg=[])
    writer.add_point([7, 8, 9], id=33, kind=5, score=2.5, seg=[300])
    writer.write(tmp_path)

    a = read_precomputed_annotations(tmp_path)
    assert a.df.index.tolist() == [11, 22, 33]
    assert a.df[['x', 'y', 'z']].to_numpy().tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert a.df['score'].tolist() == [0.5, 1.5, 2.5]
    assert a.df['kind'].tolist() == ['b', 'c', 'a']
    assert [x.tolist() for x in a.df['seg']] == [[100, 200], [], [300]]

    # The returned spec describes the categorical codes, not the original stored values.
    kind_spec = next(p for p in a.properties if p['id'] == 'kind')
    assert kind_spec['enum_values'] == [0, 1, 2]
    assert kind_spec['enum_labels'] == ['a', 'b', 'c']
    assert next(p for p in a.info['properties'] if p['id'] == 'kind')['enum_values'] == [5, 7, 9]


def test_enum_with_unlisted_values(tmp_path, caplog):
    """If a stored enum value isn't listed in enum_values, return the raw values."""
    from neuroglancer.write_annotations import AnnotationWriter
    from neuroglancer.viewer_state import AnnotationPropertySpec

    writer = AnnotationWriter(
        CS, 'point',
        properties=[AnnotationPropertySpec(id='kind', type='uint8', enum_values=[5, 7], enum_labels=['a', 'b'])],
    )
    writer.add_point([1, 2, 3], id=1, kind=7)
    writer.add_point([4, 5, 6], id=2, kind=8)
    writer.write(tmp_path)

    a = read_precomputed_annotations(tmp_path)
    assert a.df['kind'].dtype == np.uint8
    assert a.df['kind'].tolist() == [7, 8]
    assert a.properties == a.info['properties']
    assert "aren't listed in its enum_values" in caplog.text


def test_mismatched_info(tmp_path):
    """A clear error is raised if the records don't match the info file."""
    df, _ = _testdata('point')
    _write(df, 'point', tmp_path)

    info_path = tmp_path / 'info'
    info = json.loads(info_path.read_text())
    info['properties'] = info['properties'][1:]
    info_path.write_text(json.dumps(info))

    with pytest.raises(ValueError, match='unexpected size'):
        read_precomputed_annotations(tmp_path)


def test_not_an_annotation_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_precomputed_annotations(tmp_path)

    (tmp_path / 'info').write_text(json.dumps({'@type': 'neuroglancer_multiscale_volume'}))
    with pytest.raises(ValueError, match='Not a precomputed annotations info file'):
        read_precomputed_annotations(tmp_path)
