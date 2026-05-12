import logging
from dataclasses import dataclass
from itertools import chain

import numpy as np
import pandas as pd

from ._util import _geometry_cols

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartitionedBuffer:
    """
    One "column" of a per-key write: a flat byte buffer plus a layout
    describing how to slice out each row's contribution.

    - ``buf`` is a flat ``bytes`` (or bytes-like) object containing every
      row's contribution concatenated.
    - ``layout`` is either:

      - an ``int`` recsize: every row's contribution is exactly that many
        bytes, so row ``i`` is ``buf[i*recsize:(i+1)*recsize]``; or
      - a length-``(N+1)`` int64 array of byte offsets: row ``i`` is
        ``buf[layout[i]:layout[i+1]]``.

    A ``_write_buffers`` call takes a list of ``PartitionedBuffer`` instances;
    the value written for key ``i`` is the concatenation of each buffer's
    :meth:`slice_for_partition` result.
    """
    buf: bytes
    layout: object  # int | np.ndarray

    def slice_for_partition(self, i):
        """
        Return the bytes for row ``i``.

        - If ``self.layout`` is an int, the buffer is fixed-width: row i is
          ``self.buf[i*layout:(i+1)*layout]``.
        - If ``self.layout`` is a (N+1,) int64 array of byte offsets, row i is
          ``self.buf[layout[i]:layout[i+1]]``.
        """
        layout = self.layout
        if isinstance(layout, (int, np.integer)):
            return self.buf[i*layout:(i+1)*layout]
        return self.buf[int(layout[i]):int(layout[i+1])]

    def total_bytes(self, n_rows):
        """Total size in bytes covering ``n_rows`` rows."""
        layout = self.layout
        if isinstance(layout, (int, np.integer)):
            return int(layout) * n_rows
        # Offset array of length n_rows+1; total = last offset.
        return int(layout[n_rows])


def _encode_id_bytes(annotation_ids):
    """
    Encode annotation IDs as ``<id:uint64le>`` records. Returns a
    :class:`PartitionedBuffer` whose layout is the fixed 8-byte recsize.
    Permute ``annotation_ids`` externally if a non-row-order encoding is required.
    """
    encoded = np.asarray(annotation_ids).astype('<u8', copy=False).tobytes()
    return PartitionedBuffer(encoded, 8)


def _encode_annotation_records(df, coord_space, annotation_type, property_specs, polyline_geom=None):
    """
    Serialize each annotation's geometry+property record into a single flat byte
    buffer, in ``df`` row order. Returns a :class:`PartitionedBuffer` whose
    layout is either:

    - an ``int`` ``recsize`` (every annotation's record is exactly this many
      bytes — true for ``point``, ``line``, ``ellipsoid``,
      ``axis_aligned_bounding_box``); or
    - a ``(N+1,)`` int64 array of byte offsets (variable-width records —
      true for polyline).

    Callers that need a permuted order (e.g. by-relationship sorted, by-chunk
    sorted) should pass an already-permuted ``df`` and ``polyline_geom``;
    encoding is always done in ``df`` row order.
    """
    if annotation_type == 'polyline':
        return _encode_polyline_records(df, property_specs, polyline_geom)

    geometry_cols = _geometry_cols(coord_space.names, annotation_type)
    geometry_prop_df = _geometry_prop_df(df, geometry_cols, property_specs)
    buf, recsize = _encode_geometry_prop_df(geometry_prop_df, geometry_cols, property_specs)
    return PartitionedBuffer(buf, recsize)


def _encode_polyline_records(df, property_specs, polyline_geom):
    """
    Polyline branch of :func:`_encode_annotation_records`. Each annotation's
    record is ``<count:uint32le><N x D float32le points><property record>``,
    so size varies per row. ``polyline_geom.starts`` / ``ends`` must be
    aligned with ``df`` row order; permute them externally before calling.
    """
    points = polyline_geom.points
    starts = polyline_geom.starts
    ends = polyline_geom.ends
    D = points.shape[1]
    point_byte_size = 4 * D
    flat_points = points.astype(np.float32, copy=False).tobytes()

    n = len(df)
    counts = (ends - starts).astype(np.uint32)
    counts_buf = counts.tobytes()

    if property_specs:
        property_only_df = _geometry_prop_df(df, [], property_specs)
        prop_buf, prop_recsize = _encode_geometry_prop_df(property_only_df, [], property_specs)
        del property_only_df
    else:
        prop_buf = b''
        prop_recsize = 0

    # Per-record byte size = 4 (count) + n_points * D * 4 + prop_recsize.
    rec_sizes = 4 + counts.astype(np.int64) * point_byte_size + prop_recsize
    offsets = np.concatenate(([0], np.cumsum(rec_sizes))).astype(np.int64)

    out = bytearray(int(offsets[-1]))
    for i in range(n):
        s = int(offsets[i])
        out[s:s+4] = counts_buf[i*4:(i+1)*4]
        pts_start = s + 4
        src_s = int(starts[i]) * point_byte_size
        src_e = int(ends[i]) * point_byte_size
        out[pts_start:pts_start + (src_e - src_s)] = flat_points[src_s:src_e]
        if prop_recsize:
            prop_off = pts_start + (src_e - src_s)
            out[prop_off:prop_off + prop_recsize] = prop_buf[i*prop_recsize:(i+1)*prop_recsize]

    return PartitionedBuffer(bytes(out), offsets)


def _geometry_prop_df(df, geometry_cols, property_specs):
    """
    Select the subset of columns that specify the geometry and properties
    of the annotations, and append columns for padding that will ensure
    our encoded records have a width that is a multiple of 4 bytes.
    """
    # Note that the property specs are already sorted by
    # dtype in the order neuroglancer requires.
    # Order our columns accordingly before we encode them as records below.
    prop_cols = []
    for spec in property_specs:
        p = spec['id']
        if spec['type'] == 'rgb':
            prop_cols.extend([f'{p}_r', f'{p}_g', f'{p}_b'])
        elif spec['type'] == 'rgba':
            prop_cols.extend([f'{p}_r', f'{p}_g', f'{p}_b', f'{p}_a'])
        else:
            prop_cols.append(p)

    geometry_prop_df = df[[*chain(*geometry_cols), *prop_cols]].copy(deep=False)

    # Calculate padding as required by neuroglancer.
    property_widths = {}
    for spec in property_specs:
        if spec['type'] == 'rgb':
            property_widths[spec['id']] = 3
        elif spec['type'] == 'rgba':
            property_widths[spec['id']] = 4
        else:
            property_widths[spec['id']] = np.dtype(spec['type']).itemsize

    property_padding = (4 - (sum(property_widths.values()) % 4)) % 4
    for i in range(property_padding):
        geometry_prop_df[f'__padding_{i}__'] = np.uint8(0)

    return geometry_prop_df


def _encode_geometry_prop_df(geometry_prop_df, geometry_cols, property_specs):
    """
    Encode the geometry and properties of the annotations into a single buffer.

    Note: Replaces category columns of geometry_prop_df with their integer equivalents.
    """
    # Convert our column dtypes to match property specs.
    dtypes = {c: np.float32 for c in chain(*geometry_cols)}
    for spec in property_specs:
        p = spec['id']
        if spec['type'] in ('rgb', 'rgba'):
            dtypes.update({
                f'{p}_{channel}': np.uint8
                for channel in spec['type']
            })
        else:
            dtypes[p] = spec['type']

    if any(dt == np.int8 for dt in dtypes.values()):
        logger.warning(
            "Old versions of neuroglancer don't support int8 properties, "
            "so consider casting to uint8 or int16 if your annotations don't load."
        )

    # Convert category columns to their integer equivalents
    for spec in property_specs:
        p = spec['id']
        if spec['type'] in ('rgb', 'rgba'):
            continue
        if geometry_prop_df[p].dtype == 'category':
            geometry_prop_df[p] = geometry_prop_df[p].cat.codes
            dtypes[p] = spec['type']

    # Vectorized serialization
    records = geometry_prop_df.to_records(index=False, column_dtypes=dtypes)
    recsize = records.dtype.itemsize
    buf = records.tobytes()
    return buf, recsize


def _encode_relationship_records(df, relationships):
    """
    Serialize each annotation's relationship buffer into a flat byte buffer,
    in row order. Returns a :class:`PartitionedBuffer` with the same layout
    convention as :func:`_encode_annotation_records`, or ``None`` if there
    are no relationships.

    The on-disk layout per annotation is, per the spec, ``<count:uint32le>
    <id_1:uint64le>...<id_count:uint64le>`` repeated once per declared
    relationship. When every relationship column is scalar uint64 the
    result is fixed-width; if any column contains a list-of-uint64 then
    the result is variable-width with explicit offsets.
    """
    if not relationships:
        return None

    per_rel = [_encode_one_relationship(df[r]) for r in relationships]

    if all(isinstance(pb.layout, (int, np.integer)) for pb in per_rel):
        total_recsize = sum(int(pb.layout) for pb in per_rel)
        n = len(df)
        combined = np.empty((n, total_recsize), dtype=np.uint8)
        offset = 0
        for pb in per_rel:
            recsize = int(pb.layout)
            combined[:, offset:offset+recsize] = np.frombuffer(pb.buf, dtype=np.uint8).reshape(n, recsize)
            offset += recsize
        return PartitionedBuffer(combined.tobytes(), total_recsize)

    # Variable-width: at least one relationship is a list column.
    n = len(df)
    per_row_sizes = np.zeros(n, dtype=np.int64)
    for pb in per_rel:
        if isinstance(pb.layout, (int, np.integer)):
            per_row_sizes += int(pb.layout)
        else:
            per_row_sizes += (pb.layout[1:] - pb.layout[:-1])
    offsets = np.concatenate(([0], np.cumsum(per_row_sizes))).astype(np.int64)

    out = bytearray(int(offsets[-1]))
    for i in range(n):
        cursor = int(offsets[i])
        for pb in per_rel:
            if isinstance(pb.layout, (int, np.integer)):
                w = int(pb.layout)
                out[cursor:cursor + w] = pb.buf[i*w:(i+1)*w]
                cursor += w
            else:
                src_s = int(pb.layout[i])
                src_e = int(pb.layout[i+1])
                w = src_e - src_s
                out[cursor:cursor + w] = pb.buf[src_s:src_e]
                cursor += w
    return PartitionedBuffer(bytes(out), offsets)


def _encode_one_relationship(s):
    """
    Encode a single relationship column. Returns a :class:`PartitionedBuffer`
    with the same layout convention as :func:`_encode_relationship_records`.

    For a scalar-uint64 column, every annotation has exactly one related ID
    so each record is a fixed 12 bytes (``<count=1:uint32><id:uint64>``).
    For an object column of lists, the per-row record is variable-width.
    """
    if pd.api.types.is_integer_dtype(s):
        records = (
            pd.DataFrame({'count': np.uint32(1), 'id': s.to_numpy()})
            .astype({'count': np.uint32, 'id': np.uint64}, copy=False)
            .to_records(index=False)
        )
        return PartitionedBuffer(records.tobytes(), 12)

    assert s.dtype == object
    counts = s.map(len).to_numpy(np.uint32)
    counts_buf = counts.tobytes()
    ids_buf = np.concatenate(s.to_list(), dtype=np.uint64).tobytes()

    per_row_sizes = 4 + counts.astype(np.int64) * 8
    offsets = np.concatenate(([0], np.cumsum(per_row_sizes))).astype(np.int64)

    out = bytearray(int(offsets[-1]))
    id_cursor = 0
    for i in range(len(counts)):
        s_off = int(offsets[i])
        out[s_off:s_off+4] = counts_buf[i*4:(i+1)*4]
        ids_bytes = int(counts[i]) * 8
        out[s_off+4:s_off+4+ids_bytes] = ids_buf[id_cursor:id_cursor+ids_bytes]
        id_cursor += ids_bytes
    return PartitionedBuffer(bytes(out), offsets)
