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

    - ``buf`` is a flat 1-D ``np.uint8`` ndarray containing every row's
      contribution concatenated. For convenience, the constructor also
      accepts a ``bytes`` or ``bytearray`` value and wraps it in a uint8
      view without copying the underlying memory.
    - ``layout`` is either:

      - an ``int`` recsize: every row's contribution is exactly that many
        bytes, so row ``i`` is ``buf[i*recsize:(i+1)*recsize]``; or
      - a length-``(N+1)`` int64 array of byte offsets: row ``i`` is
        ``buf[layout[i]:layout[i+1]]``.

    A ``_write_buffers`` call takes a list of ``PartitionedBuffer`` instances;
    the value written for key ``i`` is the concatenation of each buffer's
    :meth:`slice_for_partition` result.

    Holding the data as a uint8 ndarray (rather than ``bytes``) lets the
    encoders return ``records.view(np.uint8)`` directly without paying for
    a separate full-size ``.tobytes()`` copy; the per-row ``.tobytes()`` in
    :meth:`slice_for_partition` is small (recsize bytes) and freed
    immediately after each tensorstore write.
    """
    buf: np.ndarray
    layout: object  # int | np.ndarray

    def __post_init__(self):
        if not isinstance(self.buf, np.ndarray):
            object.__setattr__(self, 'buf', np.frombuffer(self.buf, dtype=np.uint8))

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
            return self.buf[i*layout:(i+1)*layout].tobytes()
        return self.buf[int(layout[i]):int(layout[i+1])].tobytes()

    def total_bytes(self, n_rows):
        """Total size in bytes covering ``n_rows`` rows."""
        layout = self.layout
        if isinstance(layout, (int, np.integer)):
            return int(layout) * n_rows
        # Offset array of length n_rows+1; total = last offset.
        return int(layout[n_rows])


def _records_to_uint8(df, dtypes, batch_size=10_000_000):
    """
    Vectorized DataFrame serialization to a flat 1-D ``uint8`` ndarray,
    suitable for use as a :class:`PartitionedBuffer`'s ``buf``.

    Encodes ``batch_size`` rows at a time. The pre-allocated output is the
    only large allocation that survives the call -- each batch's transient
    structured records ndarray is freed before the next batch runs --
    capping peak transient memory at roughly ``len(df) * recsize +
    batch_size * recsize`` bytes (instead of the ``2 * len(df) * recsize``
    we'd pay if we built one full-size records ndarray then called
    ``.tobytes()`` on it).

    Returns:
        ``(out, recsize)`` where ``out`` is a 1-D ``np.uint8`` ndarray of
        length ``len(df) * recsize``, and ``recsize`` is the per-record
        size in bytes (derived from ``dtypes``).
    """
    # Determine recsize from an empty-rows sample, so we can pre-allocate.
    sample = df.iloc[:0].to_records(index=False, column_dtypes=dtypes)
    recsize = sample.dtype.itemsize
    n = len(df)
    out = np.empty(n * recsize, dtype=np.uint8)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        records = df.iloc[s:e].to_records(index=False, column_dtypes=dtypes)
        out[s * recsize : e * recsize] = records.view(np.uint8).reshape(-1)
    return out, recsize


def _encode_id_bytes(annotation_ids):
    """
    Encode annotation IDs as ``<id:uint64le>`` records. Returns a
    :class:`PartitionedBuffer` whose layout is the fixed 8-byte recsize.
    Permute ``annotation_ids`` externally if a non-row-order encoding is required.
    """
    ids = np.asarray(annotation_ids).astype('<u8', copy=False)
    return PartitionedBuffer(ids.view(np.uint8).reshape(-1), 8)


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
    # Flatten then view: a 2-D ``(N, D)`` array can't be view'd directly to
    # uint8 if its last axis isn't byte-contiguous; ravel-then-view avoids
    # that constraint.
    flat_points = points.astype(np.float32, copy=False).ravel().view(np.uint8)

    n = len(df)
    counts = (ends - starts).astype(np.uint32)
    counts_uint8 = counts.view(np.uint8).reshape(-1)  # 4 bytes per count

    if property_specs:
        property_only_df = _geometry_prop_df(df, [], property_specs)
        prop_buf, prop_recsize = _encode_geometry_prop_df(property_only_df, [], property_specs)
        del property_only_df
    else:
        prop_buf = np.empty(0, dtype=np.uint8)
        prop_recsize = 0

    # Per-record byte size = 4 (count) + n_points * D * 4 + prop_recsize.
    rec_sizes = 4 + counts.astype(np.int64) * point_byte_size + prop_recsize
    offsets = np.concatenate(([0], np.cumsum(rec_sizes))).astype(np.int64)

    out = np.empty(int(offsets[-1]), dtype=np.uint8)
    for i in range(n):
        s = int(offsets[i])
        out[s:s+4] = counts_uint8[i*4:(i+1)*4]
        pts_start = s + 4
        src_s = int(starts[i]) * point_byte_size
        src_e = int(ends[i]) * point_byte_size
        out[pts_start:pts_start + (src_e - src_s)] = flat_points[src_s:src_e]
        if prop_recsize:
            prop_off = pts_start + (src_e - src_s)
            out[prop_off:prop_off + prop_recsize] = prop_buf[i*prop_recsize:(i+1)*prop_recsize]

    return PartitionedBuffer(out, offsets)


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

    # Convert enum columns to their integer code equivalents.
    #
    # A categorical column may arrive here in one of two shapes
    # depending on the upstream path:
    #
    #   - dtype='category' -- the pandas Categorical survived (e.g. for
    #     direct pandas input registered with DuckDB, where DuckDB
    #     recognizes the column as ENUM). Use .cat.codes.
    #
    #   - dtype=object (strings) -- the categorical-ness was stripped on
    #     the way through DuckDB. This is what happens for Feather input:
    #     Arrow dictionary-encoded columns get registered as VARCHAR, and
    #     queries return them as object dtype. We recover the codes by
    #     reconstructing a Categorical against ``enum_labels`` from the
    #     spec (which was captured before streaming).
    for spec in property_specs:
        p = spec['id']
        if spec['type'] in ('rgb', 'rgba'):
            continue
        col_series = geometry_prop_df[p]
        if col_series.dtype == 'category':
            geometry_prop_df[p] = col_series.cat.codes
        elif 'enum_labels' in spec:
            cat = pd.Categorical(col_series, categories=spec['enum_labels'], ordered=False)
            if (cat.codes == -1).any():
                unknown = sorted({v for v in col_series[cat.codes == -1].unique() if v is not None})
                raise ValueError(
                    f"Column {p!r}: values not present in enum_labels {spec['enum_labels']}: "
                    f"{unknown}"
                )
            geometry_prop_df[p] = cat.codes

    return _records_to_uint8(geometry_prop_df, dtypes)


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
            combined[:, offset:offset+recsize] = pb.buf.reshape(n, recsize)
            offset += recsize
        return PartitionedBuffer(combined.reshape(-1), total_recsize)

    # Variable-width: at least one relationship is a list column.
    n = len(df)
    per_row_sizes = np.zeros(n, dtype=np.int64)
    for pb in per_rel:
        if isinstance(pb.layout, (int, np.integer)):
            per_row_sizes += int(pb.layout)
        else:
            per_row_sizes += (pb.layout[1:] - pb.layout[:-1])
    offsets = np.concatenate(([0], np.cumsum(per_row_sizes))).astype(np.int64)

    out = np.empty(int(offsets[-1]), dtype=np.uint8)
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
    return PartitionedBuffer(out, offsets)


def _encode_one_relationship(s):
    """
    Encode a single relationship column. Returns a :class:`PartitionedBuffer`
    with the same layout convention as :func:`_encode_relationship_records`.

    For a scalar-uint64 column, every annotation has exactly one related ID
    so each record is a fixed 12 bytes (``<count=1:uint32><id:uint64>``).
    For an object column of lists, the per-row record is variable-width.
    """
    if pd.api.types.is_integer_dtype(s):
        df_count_id = (
            pd.DataFrame({'count': np.uint32(1), 'id': s.to_numpy()})
            .astype({'count': np.uint32, 'id': np.uint64}, copy=False)
        )
        buf, recsize = _records_to_uint8(df_count_id, {'count': np.uint32, 'id': np.uint64})
        return PartitionedBuffer(buf, recsize)

    assert s.dtype == object
    counts = s.map(len).to_numpy(np.uint32)
    counts_uint8 = counts.view(np.uint8).reshape(-1)  # 4 bytes per count
    ids_uint8 = np.concatenate(s.to_list(), dtype=np.uint64).view(np.uint8).reshape(-1)

    per_row_sizes = 4 + counts.astype(np.int64) * 8
    offsets = np.concatenate(([0], np.cumsum(per_row_sizes))).astype(np.int64)

    out = np.empty(int(offsets[-1]), dtype=np.uint8)
    id_cursor = 0
    for i in range(len(counts)):
        s_off = int(offsets[i])
        out[s_off:s_off+4] = counts_uint8[i*4:(i+1)*4]
        ids_bytes = int(counts[i]) * 8
        out[s_off+4:s_off+4+ids_bytes] = ids_uint8[id_cursor:id_cursor+ids_bytes]
        id_cursor += ids_bytes
    return PartitionedBuffer(out, offsets)


def _build_grouped_record_buffers(df_batch, group_col, coord_space, annotation_type, property_specs,
                                  polyline_geom=None):
    """
    Given a batch DataFrame containing the rows for one tensorstore
    transaction (already sorted by ``group_col``, then by annotation_id
    within each group), build the three :class:`PartitionedBuffer`
    instances that together encode each group's by-rel or by-spatial
    record::

        <count:uint64le><ann_record_1>..<ann_record_count>
        <ann_id_1:uint64le>..<ann_id_count:uint64le>

    Args:
        df_batch:
            DataFrame with one row per (annotation, group) pairing.
            Must have an ``annotation_id`` column and a ``group_col``
            column; everything else is geometry+property data for the
            encoder.
        group_col:
            Name of the column that identifies the output group
            (e.g. ``'_segment_id'`` for by-rel, ``'_chunk_code'`` for
            by-spatial).
        polyline_geom:
            For polyline annotations, the per-batch
            :class:`PolylineGeometry` whose ``starts``/``ends`` are
            aligned with ``df_batch`` row order. Pass ``None`` for
            other annotation types.

    Returns:
        ``(buffers, unique_groups)``: a list of three
        :class:`PartitionedBuffer` (count_buf, ann_buf, id_buf) and the
        uint64 array of distinct group ids actually present in the
        batch, in the same order as the buffers.
    """
    group_ids = df_batch[group_col].to_numpy(np.uint64, copy=False)
    df_batch = df_batch.drop(columns=group_col).set_index('annotation_id')

    # Run-boundaries inside group_ids (one run per group).
    if len(group_ids) == 0:
        boundaries = np.array([0], dtype=np.int64)
    else:
        boundaries = np.concatenate((
            [0],
            np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1,
            [len(group_ids)],
        )).astype(np.int64)
    unique_groups = group_ids[boundaries[:-1]]
    counts = (boundaries[1:] - boundaries[:-1]).astype(np.uint64)

    # Encode ann and id buffers, in group (then annotation_id) order.
    ann_pb = _encode_annotation_records(
        df_batch, coord_space, annotation_type, property_specs, polyline_geom=polyline_geom,
    )
    id_pb = _encode_id_bytes(df_batch.index)

    # Per-group byte ranges into the flat ann/id buffers.
    if isinstance(ann_pb.layout, (int, np.integer)):
        ann_offsets = (boundaries * int(ann_pb.layout)).astype(np.int64)
    else:
        ann_offsets = ann_pb.layout[boundaries].astype(np.int64, copy=False)
    id_offsets = (boundaries * int(id_pb.layout)).astype(np.int64)

    buffers = [
        PartitionedBuffer(counts.astype('<u8', copy=False).tobytes(), 8),
        PartitionedBuffer(ann_pb.buf, ann_offsets),
        PartitionedBuffer(id_pb.buf, id_offsets),
    ]
    return buffers, unique_groups
