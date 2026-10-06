import os
import json
import logging
import posixpath
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import pandas as pd
import tensorstore as ts
from neuroglancer.coordinate_space import CoordinateSpace

from ._decode import _decode_by_id_values
from ._read_shards import _iter_shards
from ._util import _geometry_cols

logger = logging.getLogger(__name__)


class PrecomputedAnnotations(NamedTuple):
    """
    The result of :func:`read_precomputed_annotations`.

    The first five fields match the first five arguments of
    :func:`write_precomputed_annotations`, so a dataset can be
    re-written like this:

    .. code-block:: python

        a = read_precomputed_annotations('path/to/annotations')
        write_precomputed_annotations(
            *a[:5],
            output_dir='path/to/copy',
            polyline_points=a.polyline_points,
        )
    """

    df: pd.DataFrame
    """
    One row per annotation, indexed by ``annotation_id`` (uint64).
    The columns follow the same conventions as the input to
    :func:`write_precomputed_annotations`.
    """

    coord_space: CoordinateSpace
    """The coordinate space of the annotations."""

    annotation_type: str
    """One of 'point', 'line', 'axis_aligned_bounding_box', 'ellipsoid', 'polyline'."""

    properties: list[dict]
    """
    The property specs, in the order listed in the info file.
    For enum properties, these specs describe the categorical
    columns of ``df`` (see :func:`read_precomputed_annotations`).
    """

    relationships: list[str]
    """The relationship names, i.e. the names of the relationship columns in ``df``."""

    polyline_points: pd.DataFrame | None
    """
    For polyline annotations, one row per vertex with columns
    ``['annotation_id', *coord_space.names]``, with each polyline's
    vertices listed in order.  Otherwise None.
    """

    info: dict
    """The complete (unmodified) contents of the info file."""


def read_precomputed_annotations(
    path: str | os.PathLike,
    ids: Iterable[int] | None = None,
    *,
    tensorstore_context: dict | None = None,
) -> PrecomputedAnnotations:
    """
    Read annotations stored in neuroglancer's precomputed annotations format
    as described in the `neuroglancer spec <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md>`_.

    The annotations are read from the "annotation ID index" (``by_id``),
    which is the only index that contains all annotations along with their
    complete relationships.

    The result is returned in the same form accepted by
    :func:`write_precomputed_annotations`:

    - Geometry columns are named according to the coordinate space,
      e.g. ``['x', 'y', 'z']`` for points or ``['xa', 'ya', 'za', 'xb', 'yb', 'zb']`` for lines.
    - Numeric properties are returned with their stored dtype.
    - Enum properties are returned as pandas Categorical columns whose
      categories are the ``enum_labels``. (If the stored values can't be
      interpreted that way -- e.g. because a stored value isn't listed in
      ``enum_values`` -- the raw numeric values are returned instead, and
      a warning is logged.)
    - Color properties are returned as separate uint8 columns per channel,
      e.g. ``mycolor_r``, ``mycolor_g``, ``mycolor_b`` (and ``mycolor_a``).
    - Relationships are returned as one column per relationship. If every
      annotation has exactly one related ID for a given relationship, the
      column has dtype uint64. Otherwise, it has dtype object and each
      value is a uint64 array of related IDs.
    - For polylines, the vertices are returned in a separate DataFrame
      (``polyline_points``).

    Args:
        path:
            The annotation directory (which contains the ``info`` file).
            A local path, or a URL supported by tensorstore (e.g. ``gs://bucket/path``).

        ids:
            Optional. If provided, read only these annotation IDs, and return them
            in the order given. Raises ``KeyError`` if any of them don't exist.
            Otherwise, read all annotations and return them sorted by ID.

        tensorstore_context:
            Optional JSON spec for the tensorstore
            `Context <https://google.github.io/tensorstore/context.html>`_
            used to open the kvstores.

    Returns:
        :class:`PrecomputedAnnotations`
    """
    context = ts.Context(tensorstore_context or {})
    base_spec = _base_kvstore_spec(path)
    base_kvstore = ts.KvStore.open(base_spec, context=context).result()
    info = _read_info(base_kvstore, path)

    annotation_type = info['annotation_type'].lower()
    coord_space = CoordinateSpace(json=info['dimensions'])
    property_specs = info.get('properties', [])
    relationships = [r['id'] for r in info.get('relationships', [])]

    by_id_metadata = info.get('by_id')
    if not by_id_metadata:
        raise ValueError(f"The info file for {path} doesn't list a 'by_id' index")
    by_id_spec = _child_kvstore_spec(base_spec, by_id_metadata['key'])

    decode_args = (annotation_type, coord_space.rank, property_specs, len(relationships))
    if ids is not None:
        keys, chunks = _read_by_id_keys(by_id_spec, by_id_metadata, ids, context, decode_args)
    elif 'sharding' in by_id_metadata:
        keys, chunks = _read_all_by_id_sharded(by_id_spec, by_id_metadata['sharding'], context, decode_args)
    else:
        keys, chunks = _read_all_by_id_unsharded(by_id_spec, context, decode_args)

    # When reading everything, return the annotations sorted by ID.
    # (The decoded chunks are concatenated and permuted column-by-column while
    # constructing the DataFrame, to avoid making a full copy of the decoded records.)
    order = None
    if ids is None:
        order = np.argsort(keys, kind='stable')
        keys = keys[order]

    df, property_specs = _records_to_dataframe(
        keys, chunks, order, coord_space, annotation_type, property_specs, relationships
    )

    polyline_points = None
    if annotation_type == 'polyline':
        counts, points = _permute_ragged(
            _concat_field(chunks, lambda c: c.polyline_counts),
            _concat_field(chunks, lambda c: c.polyline_points),
            order,
        )
        polyline_points = pd.DataFrame(points, columns=coord_space.names)
        polyline_points.insert(0, 'annotation_id', np.repeat(keys, counts))

    return PrecomputedAnnotations(
        df, coord_space, annotation_type, property_specs, relationships, polyline_points, info
    )


def _base_kvstore_spec(path):
    """
    Construct a tensorstore KvStore.Spec for the annotation directory.
    Local paths are converted to file:// URLs.
    """
    path = os.fspath(path)
    if '://' not in path:
        path = 'file://' + os.path.abspath(path)
    spec = ts.KvStore.Spec(path)
    if spec.path and not spec.path.endswith('/'):
        spec.path += '/'
    return spec


def _child_kvstore_spec(base_spec, key):
    """
    Construct the KvStore.Spec for an index subdirectory.
    Per the spec, the key may be a relative path containing '..' components.
    """
    spec = base_spec.copy()
    spec.path = posixpath.normpath(posixpath.join(base_spec.path, key)) + '/'
    return spec


def _read_info(base_kvstore, path):
    result = base_kvstore.read('info').result()
    if result.state != 'value':
        raise FileNotFoundError(f"No info file found in {path}")
    info = json.loads(result.value)
    if info.get('@type') != 'neuroglancer_annotations_v1':
        raise ValueError(f"Not a precomputed annotations info file (@type={info.get('@type')!r}): {path}")
    return info


def _read_all_by_id_sharded(by_id_spec, sharding, context, decode_args):
    """
    Read every annotation in a sharded by-id index,
    by parsing each shard file in its entirety.
    """
    kvstore = ts.KvStore.open(by_id_spec, context=context).result()
    all_keys = []
    chunks = []
    for keys, values in _iter_shards(kvstore, sharding):
        all_keys.append(keys)
        chunks.append(_decode_by_id_values(values, *decode_args))
        del values

    if not chunks:
        return np.zeros(0, dtype=np.uint64), [_decode_by_id_values([], *decode_args)]
    return np.concatenate(all_keys), chunks


def _read_all_by_id_unsharded(by_id_spec, context, decode_args):
    """
    Read every annotation in an unsharded by-id index, in which each
    annotation is stored in a file named by its (decimal) ID.
    """
    kvstore = ts.KvStore.open(by_id_spec, context=context).result()
    names = [k.decode() for k in kvstore.list().result()]
    keys = np.array([int(n) for n in names if n.isdigit()], dtype=np.uint64)
    if len(keys) < len(names):
        logger.warning(f"Ignoring {len(names) - len(keys)} non-numeric file(s) in the by_id directory")
    values = _batch_read(kvstore, [str(k).encode() for k in keys.tolist()], keys)
    return keys, [_decode_by_id_values(values, *decode_args)]


def _read_by_id_keys(by_id_spec, by_id_metadata, ids, context, decode_args):
    """
    Read the given annotation IDs (in the given order) from the by-id index.
    """
    keys = _as_uint64_ids(ids)

    if 'sharding' in by_id_metadata:
        kvstore = ts.KvStore.open({
            'driver': 'neuroglancer_uint64_sharded',
            'metadata': by_id_metadata['sharding'],
            'base': by_id_spec,
        }, context=context).result()
        encoded_keys = [k.to_bytes(8, 'big') for k in keys.tolist()]
    else:
        kvstore = ts.KvStore.open(by_id_spec, context=context).result()
        encoded_keys = [str(k).encode() for k in keys.tolist()]

    values = _batch_read(kvstore, encoded_keys, keys)
    return keys, [_decode_by_id_values(values, *decode_args)]


def _as_uint64_ids(ids):
    """
    Convert a collection of annotation IDs to a uint64 array,
    without any loss of precision.

    (We must be careful: for example, np.asarray() on a list of Python ints
    which includes values >= 2**63 silently produces a float64 array.)
    """
    if isinstance(ids, (np.ndarray, pd.Index, pd.Series)):
        arr = np.asarray(ids)
        if arr.ndim != 1:
            raise ValueError("ids must be a one-dimensional collection of annotation IDs")
        if arr.dtype.kind == 'u':
            return arr.astype(np.uint64, copy=False)
        if arr.dtype.kind == 'i':
            if (arr < 0).any():
                raise ValueError("Annotation IDs must be non-negative")
            return arr.astype(np.uint64)
        if arr.dtype.kind != 'O':
            raise TypeError(f"Annotation IDs must be integers, not {arr.dtype}")
        ids = arr.tolist()

    ids = list(ids)
    if not all(isinstance(i, (int, np.integer)) for i in ids):
        raise TypeError("Annotation IDs must be integers")
    try:
        return np.fromiter((int(i) for i in ids), dtype=np.uint64, count=len(ids))
    except OverflowError:
        raise ValueError("Annotation IDs must be in the range [0, 2**64)") from None


def _batch_read(kvstore, encoded_keys, keys, batch_size=100_000):
    """
    Read the given keys from the kvstore using batched reads.
    Raises KeyError if any key is missing.
    """
    values = []
    missing = []
    for start in range(0, len(encoded_keys), batch_size):
        with ts.Batch() as batch:
            futures = [kvstore.read(k, batch=batch) for k in encoded_keys[start:start + batch_size]]
        for i, f in enumerate(futures, start):
            result = f.result()
            if result.state != 'value':
                missing.append(int(keys[i]))
            values.append(result.value)
    if missing:
        raise KeyError(
            f"{len(missing)} annotation ID(s) not found, e.g. {missing[:10]}"
        )
    return values


def _concat_field(chunks, get):
    """
    Concatenate one field of the decoded :class:`ByIdRecords` chunks,
    e.g. ``_concat_field(chunks, lambda c: c.records['geometry'][:, 0])``
    """
    if len(chunks) == 1:
        return get(chunks[0])
    return np.concatenate([get(c) for c in chunks])


def _permute_ragged(counts, flat_values, order):
    """
    Reorder the rows of a ragged array (flat values plus per-row counts).
    If ``order`` is None, the inputs are returned unchanged.

    Returns:
        (counts, flat_values)
    """
    if order is None:
        return counts, flat_values
    if (counts == 1).all():
        # Common case (e.g. relationships with exactly one related ID per annotation):
        # skip the (relatively expensive) general-purpose gather index.
        return counts[order], flat_values[order]
    return counts[order], flat_values[_ragged_gather_index(counts, order)]


def _ragged_gather_index(counts, order):
    """
    Given a ragged array (flat values plus per-row ``counts``),
    return the index into the flat values that reorders its rows
    according to ``order``.
    """
    counts = counts.astype(np.int64)
    starts = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    new_counts = counts[order]
    new_starts = np.zeros(len(order), dtype=np.int64)
    np.cumsum(new_counts[:-1], out=new_starts[1:])
    # For each element of the output, its position within its row
    # plus the start of that row in the input.
    return (
        np.arange(new_counts.sum(), dtype=np.int64)
        + np.repeat(starts[order] - new_starts, new_counts)
    )


def _records_to_dataframe(keys, chunks, order, coord_space, annotation_type, property_specs, relationships):
    """
    Construct the annotation DataFrame from a list of decoded :class:`ByIdRecords` chunks.

    If ``order`` is not None, the (concatenated) records are permuted accordingly.
    (``keys`` is assumed to be concatenated and permuted already.)

    Returns:
        (df, property_specs), where property_specs has been adjusted
        to match the returned columns (see :func:`_decode_enum`).
    """
    def column(field, index=None):
        # Produce a contiguous (and permuted, if necessary) copy of
        # a record field (or one element of a vector-valued field).
        if index is None:
            values = _concat_field(chunks, lambda c: c.records[field])
        else:
            values = _concat_field(chunks, lambda c: c.records[field][:, index])
        if order is None:
            return np.ascontiguousarray(values)
        return values[order]

    columns = {}
    geometry_names = [c for group in _geometry_cols(coord_space.names, annotation_type) for c in group]
    for i, name in enumerate(geometry_names):
        columns[name] = column('geometry', i)

    output_specs = []
    for spec in property_specs:
        p = spec['id']
        if spec['type'] in ('rgb', 'rgba'):
            for i, channel in enumerate(spec['type']):
                columns[f'{p}_{channel}'] = column(p, i)
            output_specs.append(spec)
        elif 'enum_values' in spec:
            columns[p], spec = _decode_enum(column(p), spec)
            output_specs.append(spec)
        else:
            columns[p] = column(p)
            output_specs.append(spec)

    for r, rel in enumerate(relationships):
        counts, flat_ids = _permute_ragged(
            _concat_field(chunks, lambda c, r=r: c.relationship_counts[:, r]),
            _concat_field(chunks, lambda c, r=r: c.relationship_ids[r]),
            order,
        )
        columns[rel] = _relationship_column(counts, flat_ids)

    index = pd.Index(keys, dtype=np.uint64, name='annotation_id')
    return pd.DataFrame(columns, index=index, copy=False), output_specs


def _decode_enum(values, spec):
    """
    Convert the stored values of an enum property to a pandas Categorical
    whose categories are the property's ``enum_labels``.

    The returned spec is updated to describe the Categorical's codes,
    i.e. ``enum_values`` becomes ``[0, 1, ..., N-1]`` (which is what
    :func:`write_precomputed_annotations` writes for categorical columns).

    If the values can't be represented that way (non-unique labels, or
    stored values that don't appear in ``enum_values``), the raw values
    and original spec are returned, with a warning.
    """
    p = spec['id']
    enum_values = np.asarray(spec['enum_values'], dtype=values.dtype)
    enum_labels = spec['enum_labels']

    if len(set(enum_labels)) != len(enum_labels):
        logger.warning(f"Property {p!r} has duplicate enum labels; returning its raw values instead of a Categorical.")
        return values, spec

    # Map each stored value to the index of its label.
    sort_order = np.argsort(enum_values, kind='stable')
    sorted_values = enum_values[sort_order]
    pos = np.searchsorted(sorted_values, values)
    pos_clipped = np.minimum(pos, len(sorted_values) - 1)
    found = (pos < len(sorted_values)) & (sorted_values[pos_clipped] == values)
    if not found.all():
        unknown = np.unique(values[~found])
        logger.warning(
            f"Property {p!r} contains values which aren't listed in its enum_values "
            f"(e.g. {unknown[:10].tolist()}); returning its raw values instead of a Categorical."
        )
        return values, spec

    codes = sort_order[pos_clipped]
    categorical = pd.Categorical.from_codes(codes, categories=enum_labels)
    spec = {**spec, 'enum_values': list(range(len(enum_labels)))}
    return categorical, spec


def _relationship_column(counts, flat_ids):
    """
    Construct a relationship column. If every annotation has exactly
    one related ID, return a uint64 array. Otherwise, return an object
    array of uint64 arrays (one per annotation).
    """
    if (counts == 1).all():
        return flat_ids

    # This is relatively expensive (a separate small array per annotation),
    # but we only do it if the relationship isn't strictly one-to-one.
    # (We fill the object array element-by-element because assigning a
    # list of equal-length arrays to column[:] would be broadcast as 2D.)
    splits = np.cumsum(counts[:-1], dtype=np.int64)
    column = np.empty(len(counts), dtype=object)
    for i, ids in enumerate(np.split(flat_ids, splits)):
        column[i] = ids
    return column
