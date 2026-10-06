"""
Decoding of the binary annotation records described in the
`neuroglancer spec <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md>`_.

This is the read-side counterpart of :mod:`._encode`. Where possible we
decode whole batches of records at once: the values for a batch of keys
are concatenated into a single flat byte buffer (plus an offsets array),
which a numba kernel splits into fixed-width "geometry + properties"
records, polyline vertices, and per-relationship related-ID arrays.
"""
from typing import NamedTuple

import numpy as np
from numba import njit

from ..util import _PROPERTY_DTYPES

# Per the spec, properties are encoded in groups ordered by their
# alignment: all 4-byte properties, then all 2-byte properties, then
# all 1-byte properties (including rgb and rgba). Within each group
# they appear in the order listed in the info file.
_PROPERTY_ALIGNMENT = {
    'uint32': 4, 'int32': 4, 'float32': 4,
    'uint16': 2, 'int16': 2,
    'uint8': 1, 'int8': 1, 'rgb': 1, 'rgba': 1,
}


def _num_geometry_vectors(annotation_type):
    """
    Number of rank-length float32 vectors in the fixed-width geometry
    portion of a record. (Polylines have a variable-width geometry, which
    isn't part of the fixed-width record.)
    """
    if annotation_type == 'point':
        return 1
    if annotation_type in ('line', 'axis_aligned_bounding_box', 'ellipsoid'):
        return 2
    if annotation_type == 'polyline':
        return 0
    raise ValueError(f"Annotation type {annotation_type} not supported")


def _property_encoding_order(property_specs):
    """
    Return the property specs in the order in which they are encoded
    within each record, which is not necessarily the order in which they
    are listed in the info file. (See note on ``_PROPERTY_ALIGNMENT``.)
    """
    for spec in property_specs:
        if spec['type'] not in _PROPERTY_ALIGNMENT:
            raise ValueError(f"Unsupported property type: {spec['type']!r} (property {spec['id']!r})")
    return sorted(property_specs, key=lambda spec: -_PROPERTY_ALIGNMENT[spec['type']])


def _record_dtype(annotation_type, rank, property_specs):
    """
    Construct the numpy structured dtype for the fixed-width portion
    of each annotation record: the geometry (except for polylines),
    followed by the properties and padding.

    The geometry is stored in a single field named ``'geometry'``
    (shape ``(num_vectors * rank,)``), and each property is stored in
    a field named after its id.
    """
    fields = []
    k = _num_geometry_vectors(annotation_type)
    if k:
        fields.append(('geometry', '<f4', (k * rank,)))

    prop_size = 0
    for spec in _property_encoding_order(property_specs):
        dtype_entry, _ = _PROPERTY_DTYPES[spec['type']]
        fields.append((spec['id'], *dtype_entry))
        prop_size += np.dtype([fields[-1]]).itemsize

    padding = (4 - (prop_size % 4)) % 4
    if padding:
        fields.append(('__padding__', '|u1', (padding,)))

    return np.dtype(fields)


class ByIdRecords(NamedTuple):
    """
    The decoded contents of a batch of values from the annotation ID index.

    - ``records``: (N,) structured array with dtype from :func:`_record_dtype`.
    - ``polyline_counts``: (N,) uint32 vertex count per annotation (polylines only, otherwise None).
    - ``polyline_points``: (total_points, rank) float32 vertices, concatenated in record order
      (polylines only, otherwise None).
    - ``relationship_counts``: (N, R) uint32 number of related IDs for each annotation and relationship.
    - ``relationship_ids``: list of R uint64 arrays, each holding the concatenated
      related IDs for one relationship, in record order.
    """
    records: np.ndarray
    polyline_counts: np.ndarray | None
    polyline_points: np.ndarray | None
    relationship_counts: np.ndarray
    relationship_ids: list[np.ndarray]


def _concat_values(values):
    """
    Concatenate a list of bytes-like values into a flat uint8 buffer,
    and return it along with a (N+1,) int64 array of offsets.
    """
    lengths = np.fromiter(map(len, values), dtype=np.int64, count=len(values))
    offsets = np.zeros(len(values) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    buf = np.frombuffer(b''.join(values), dtype=np.uint8)
    return buf, offsets


def _decode_by_id_values(values, annotation_type, rank, property_specs, num_relationships):
    """
    Decode a batch of values from the annotation ID index.

    Args:
        values:
            list of bytes-like values, one per annotation.
        annotation_type, rank, property_specs:
            From the info file. (``property_specs`` in info-file order.)
        num_relationships:
            Number of relationships listed in the info file.

    Returns:
        :class:`ByIdRecords`
    """
    buf, offsets = _concat_values(values)
    return _decode_by_id_buffer(buf, offsets, annotation_type, rank, property_specs, num_relationships)


def _decode_by_id_buffer(buf, offsets, annotation_type, rank, property_specs, num_relationships):
    """
    Same as :func:`_decode_by_id_values`, but for values which have
    already been concatenated into a flat buffer with offsets.
    """
    dtype = _record_dtype(annotation_type, rank, property_specs)
    is_polyline = (annotation_type == 'polyline')
    n = len(offsets) - 1

    point_counts, rel_counts, bad_index = _scan_by_id_records(
        buf, offsets, dtype.itemsize, rank, is_polyline, num_relationships
    )
    if bad_index >= 0:
        s, e = offsets[bad_index], offsets[bad_index + 1]
        raise ValueError(
            f"Annotation record #{bad_index} has an unexpected size ({e - s} bytes) "
            f"given the annotation type ({annotation_type!r}), properties ({len(property_specs)}), "
            f"and relationships ({num_relationships}) listed in the info file. "
            "(Does the info file list all of the relationships that were encoded in the annotation ID index?)"
        )

    point_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(point_counts, out=point_offsets[1:])

    rel_offsets = np.zeros((num_relationships, n + 1), dtype=np.int64)
    np.cumsum(rel_counts.T, axis=1, out=rel_offsets[:, 1:])

    fixed = np.empty((n, dtype.itemsize), dtype=np.uint8)
    points = np.empty(point_offsets[-1] * rank * 4, dtype=np.uint8)
    rel_bufs = np.empty(int(rel_offsets[:, -1].sum()) * 8, dtype=np.uint8)
    rel_bases = np.zeros(num_relationships, dtype=np.int64)
    if num_relationships:
        np.cumsum(rel_offsets[:-1, -1], out=rel_bases[1:])

    _split_by_id_records(
        buf, offsets, dtype.itemsize, rank, is_polyline,
        point_offsets, rel_offsets, rel_bases,
        fixed, points, rel_bufs
    )

    rel_ids_all = rel_bufs.view('<u8')
    relationship_ids = [
        rel_ids_all[rel_bases[r]:rel_bases[r] + rel_offsets[r, -1]]
        for r in range(num_relationships)
    ]

    if dtype.itemsize == 0:
        # Polylines without properties have an empty fixed-width record,
        # and numpy can't view() a buffer as a zero-sized dtype.
        records = np.zeros(n, dtype=dtype)
    else:
        records = fixed.view(dtype).reshape(n)

    if not is_polyline:
        return ByIdRecords(records, None, None, rel_counts, relationship_ids)

    polyline_points = points.view('<f4').reshape(-1, rank)
    return ByIdRecords(records, point_counts.astype(np.uint32), polyline_points, rel_counts, relationship_ids)


@njit(cache=True, inline='always')
def _read_u32(buf, pos):
    return (
        np.uint32(buf[pos])
        | (np.uint32(buf[pos + 1]) << np.uint32(8))
        | (np.uint32(buf[pos + 2]) << np.uint32(16))
        | (np.uint32(buf[pos + 3]) << np.uint32(24))
    )


@njit(cache=True)
def _scan_by_id_records(buf, offsets, fixed_size, rank, is_polyline, num_relationships):
    """
    First pass over a batch of by-id records: determine the number of
    polyline vertices and related IDs in each record, and verify that
    each record's size is consistent with its contents.

    For polylines, ``fixed_size`` is the size of the property portion
    of the record (which follows the variable-width geometry).

    Returns:
        (point_counts, rel_counts, bad_index), where bad_index is the
        index of the first malformed record, or -1 if all records are OK.
    """
    n = len(offsets) - 1
    point_counts = np.zeros(n, dtype=np.int64)
    rel_counts = np.zeros((n, num_relationships), dtype=np.uint32)
    for i in range(n):
        pos = offsets[i]
        end = offsets[i + 1]
        if is_polyline:
            if pos + 4 > end:
                return point_counts, rel_counts, i
            num_points = _read_u32(buf, pos)
            point_counts[i] = num_points
            pos += 4 + np.int64(num_points) * rank * 4
        pos += fixed_size
        for r in range(num_relationships):
            if pos + 4 > end:
                return point_counts, rel_counts, i
            count = _read_u32(buf, pos)
            rel_counts[i, r] = count
            pos += 4 + np.int64(count) * 8
        if pos != end:
            return point_counts, rel_counts, i
    return point_counts, rel_counts, -1


@njit(cache=True)
def _split_by_id_records(buf, offsets, fixed_size, rank, is_polyline,
                         point_offsets, rel_offsets, rel_bases,
                         fixed_out, points_out, rel_out):
    """
    Second pass over a batch of by-id records (already validated by
    :func:`_scan_by_id_records`): copy each record's pieces into the
    pre-allocated output buffers.

    - ``fixed_out``: (N, fixed_size) uint8
    - ``points_out``: flat uint8 buffer for all polyline vertices
    - ``rel_out``: flat uint8 buffer for all related IDs, with each
      relationship's IDs stored contiguously starting at element
      ``rel_bases[r]`` (in units of uint64).
    """
    n = len(offsets) - 1
    point_size = rank * 4
    num_relationships = rel_offsets.shape[0]
    for i in range(n):
        pos = offsets[i]
        if is_polyline:
            pos += 4
            nbytes = (point_offsets[i + 1] - point_offsets[i]) * point_size
            dst = point_offsets[i] * point_size
            points_out[dst:dst + nbytes] = buf[pos:pos + nbytes]
            pos += nbytes

        fixed_out[i, :] = buf[pos:pos + fixed_size]
        pos += fixed_size

        for r in range(num_relationships):
            pos += 4
            nbytes = (rel_offsets[r, i + 1] - rel_offsets[r, i]) * 8
            dst = (rel_bases[r] + rel_offsets[r, i]) * 8
            rel_out[dst:dst + nbytes] = buf[pos:pos + nbytes]
            pos += nbytes
