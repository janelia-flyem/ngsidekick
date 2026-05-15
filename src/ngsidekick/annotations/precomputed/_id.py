import logging
import os

import numpy as np
import pandas as pd
import tensorstore as ts
from tqdm.auto import tqdm

from . import _write_buffers
from ._db import INPUT_VIEW
from ._encode import _encode_annotation_records, _encode_relationship_records
from ._memory import log_memory
from ._shard_hash import compute_shard_assignments_in_db
from ._util import _property_recsize, _slice_polyline_geom
from ._write_buffers import (
    _open_sharded_kvstore,
    _prepare_output_subdir,
    _sharded_metadata,
    _write_one_transaction,
)

# Row count per encoding batch in unsharded mode. The unsharded format
# is typically used only for small datasets, but we still encode in
# chunks so the per-encode transient is bounded.
_UNSHARDED_BATCH_SIZE = 100_000

logger = logging.getLogger(__name__)

SHARD_ASSIGNMENTS_TABLE = 'by_id_shard_assignments'


def _write_annotations_by_id(con, coord_space, annotation_type, property_specs, relationships, polyline_geom,
                              output_dir, write_sharded, max_shards_per_transaction, ts_context):
    """
    Write the annotations to the "Annotation ID Index" in shard-aligned
    batches.

    Shard assignments for all annotation IDs are computed via DuckDB
    streaming (see :func:`compute_shard_assignments_in_db`), so the full
    keyspace never lives in Python at once. Then for each batch of
    adjacent shards we query its annotations, encode them, and commit a
    single tensorstore transaction. Peak Python RAM during by-id is one
    batch's worth of encoded bytes plus a small DuckDB working set.

    Args:
        con:
            DuckDB connection with the input registered as
            :data:`INPUT_VIEW`.
        coord_space, annotation_type, property_specs, relationships, polyline_geom:
            See :func:`write_precomputed_annotations`.
        output_dir, write_sharded, max_shards_per_transaction, ts_context:
            See :func:`._write_buffers._write_buffers`.

    Returns:
        JSON metadata for the 'by_id' key in the top-level 'info' file.
    """
    if not write_sharded:
        return _write_annotations_by_id_unsharded(
            con, coord_space, annotation_type, property_specs, relationships,
            polyline_geom, output_dir, ts_context,
        )

    # For polylines we need to look up each batch's rows in the
    # in-memory polyline_geom (whose starts/ends are indexed by row
    # position, not annotation_id). Build the lookup once.
    polyline_id_lookup = (
        pd.Index(polyline_geom.annotation_ids) if polyline_geom is not None else None
    )

    logger.info("Preparing 'by_id' index")

    # Compute total count and max key for shard sizing. Both are O(1) queries.
    n, max_key = con.execute(
        f"SELECT COUNT(*), MAX(annotation_id) FROM {INPUT_VIEW}"
    ).fetchone()
    if n == 0:
        # Nothing to write; emit an empty sharded output.
        _prepare_output_subdir(output_dir, "by_id")
        return {"key": "by_id"}

    # Choose shard spec from a size estimate (a few percent error is fine;
    # _choose_output_spec uses these inputs only via log2-shaped thresholds).
    total_bytes = _estimate_total_bytes_for_by_id(
        con, n, coord_space, annotation_type, property_specs, relationships,
        polyline_geom,
    )
    shard_spec = _write_buffers._choose_output_spec(
        total_count=int(n),
        total_bytes=int(total_bytes),
        max_key=int(max_key),
        hashtype='murmurhash3_x86_128',
        gzip_compress=True,
    )

    # Compute (annotation_id, shard_id) pairs into a DuckDB table, in
    # streaming chunks so we never hold all the shard IDs in Python.
    logger.info("Computing by-id shard assignments")
    compute_shard_assignments_in_db(
        con, INPUT_VIEW, SHARD_ASSIGNMENTS_TABLE, shard_spec,
        key_col='annotation_id',
    )

    _prepare_output_subdir(output_dir, "by_id")
    kvstore = _open_sharded_kvstore(output_dir, "by_id", shard_spec, ts_context)

    # Find the distinct occupied shard IDs (the keyspace may not fill
    # ``2 ** shard_bits`` cleanly — clumpy with ``identity_hash``, and
    # even with murmurhash3 we may have empty tail shards). Then chunk
    # those distinct shard IDs into groups of ``max_shards_per_transaction``;
    # each group becomes one tensorstore transaction. This is the
    # contract ``max_shards_per_transaction`` is meant to enforce
    # ("at most N distinct shards staged in memory per transaction").
    batch_size = int(max_shards_per_transaction)
    occupied_shards = con.execute(f"""
        SELECT DISTINCT shard_id FROM {SHARD_ASSIGNMENTS_TABLE}
        ORDER BY shard_id
    """).to_arrow_table().column('shard_id').to_numpy(zero_copy_only=False)

    n_transactions = (len(occupied_shards) + batch_size - 1) // batch_size
    logger.info(f"Writing annotations to 'by_id' index "
                f"({n_transactions} transactions over "
                f"{len(occupied_shards)} occupied shards "
                f"(of {1 << shard_spec.shard_bits} possible))")
    log_memory('by_id pre-write-loop')
    with tqdm(total=int(n)) as pbar:
        for batch_idx, chunk_start in enumerate(range(0, len(occupied_shards), batch_size)):
            chunk_shards = occupied_shards[chunk_start:chunk_start + batch_size]
            df_batch = _query_rows_for_shards(con, chunk_shards)

            batch_polyline_geom = None
            if polyline_geom is not None:
                rows = polyline_id_lookup.get_indexer(df_batch.index.to_numpy())
                batch_polyline_geom = _slice_polyline_geom(polyline_geom, rows)

            ann_pb = _encode_annotation_records(
                df_batch, coord_space, annotation_type, property_specs,
                polyline_geom=batch_polyline_geom,
            )
            rel_pb = _encode_relationship_records(df_batch, relationships)
            buffers = [ann_pb] + ([rel_pb] if rel_pb is not None else [])

            batch_keys = df_batch.index.to_numpy(np.uint64, copy=False)
            _write_one_transaction(kvstore, batch_keys, buffers)
            pbar.update(len(df_batch))

            # Release this batch before the next one runs.
            del df_batch, ann_pb, rel_pb, buffers, batch_keys, batch_polyline_geom
            log_memory(f'by_id post-batch {batch_idx + 1}/{n_transactions}')

    # Drop the temporary assignments table now that we're done with it.
    con.execute(f"DROP TABLE {SHARD_ASSIGNMENTS_TABLE}")
    log_memory('by_id done')
    return _sharded_metadata("by_id", shard_spec)


def _query_rows_for_shards(con, shard_ids):
    """
    Fetch the annotations whose by-id shard is in ``shard_ids`` (a
    (potentially non-contiguous) collection of uint64 shard IDs), sorted
    by shard_id and then annotation_id.

    The shard IDs are passed as a small DuckDB-registered Arrow table so
    the SQL can ``JOIN`` rather than build a giant ``IN`` literal -- but
    since ``shard_ids`` is at most ``max_shards_per_transaction`` (typically
    a few dozen), the JOIN is tiny.
    """
    import pyarrow as pa
    shard_table = pa.table({'shard_id': np.asarray(shard_ids, dtype=np.uint64)})
    con.register('_by_id_chunk_shards', shard_table)
    try:
        result_df = con.execute(f"""
            SELECT v.*
            FROM _by_id_chunk_shards c
            JOIN {SHARD_ASSIGNMENTS_TABLE} s ON s.shard_id = c.shard_id
            JOIN {INPUT_VIEW} v USING (annotation_id)
            ORDER BY s.shard_id, v.annotation_id
        """).df()
    finally:
        con.unregister('_by_id_chunk_shards')
    return result_df.set_index('annotation_id')


def _estimate_total_bytes_for_by_id(con, n, coord_space, annotation_type, property_specs, relationships,
                                    polyline_geom):
    """
    Rough upper-bound estimate of the by-id payload size, used only by
    :func:`_choose_output_spec`'s log2-shaped heuristic.

    Encodes an empty zero-row probe through the existing encoder helpers
    to discover the per-record byte sizes for geometry+properties and
    relationships, then multiplies by ``n``. For polylines, where the
    geometry record is variable-width, the vertex payload total is
    derived directly from ``polyline_geom.points.nbytes``.
    """
    if n == 0:
        return 0

    probe_df = (
        con.execute(f"SELECT * FROM {INPUT_VIEW} LIMIT 0")
        .df()
        .set_index('annotation_id')
    )
    rel_pb = _encode_relationship_records(probe_df, relationships)
    rel_recsize = 0
    if rel_pb is not None and isinstance(rel_pb.layout, (int, np.integer)):
        rel_recsize = int(rel_pb.layout)

    if annotation_type == 'polyline':
        ann_total = _polyline_total_bytes(property_specs, polyline_geom)
        return ann_total + n * rel_recsize

    ann_pb = _encode_annotation_records(
        probe_df, coord_space, annotation_type, property_specs, polyline_geom=None,
    )
    ann_recsize = int(ann_pb.layout) if isinstance(ann_pb.layout, (int, np.integer)) else 0
    return n * (ann_recsize + rel_recsize)


def _polyline_total_bytes(property_specs, polyline_geom):
    """
    Total annotation-record bytes for a by-id polyline write: 4 bytes
    per annotation for the vertex count, plus all vertex bytes (each
    vertex is stored exactly once at by-id), plus the per-record
    property payload (padded to a 4-byte boundary as the encoder does).
    """
    if polyline_geom is None:
        return 0
    n = len(polyline_geom.starts)
    prop_recsize = _property_recsize(property_specs)
    return 4 * n + int(polyline_geom.points.nbytes) + prop_recsize * n


def _write_annotations_by_id_unsharded(con, coord_space, annotation_type, property_specs, relationships,
                                       polyline_geom, output_dir, ts_context):
    """
    Unsharded variant of :func:`_write_annotations_by_id`: one file per
    annotation, with the decimal annotation_id as the filename.

    Encoding is done in fixed-size row chunks so the encoder transient
    is bounded; the chunks are committed inside a single tensorstore
    transaction to match the original unsharded semantics. Unsharded
    output is generally suitable only for small datasets; for large data
    use ``write_sharded=True``.
    """
    n = con.execute(f"SELECT COUNT(*) FROM {INPUT_VIEW}").fetchone()[0]
    _prepare_output_subdir(output_dir, "by_id")

    output_dir = os.path.abspath(output_dir)
    kvstore = ts.KvStore.open(f"file://{output_dir}/by_id/", context=ts_context).result()

    if n == 0:
        return {"key": "by_id"}

    polyline_id_lookup = (
        pd.Index(polyline_geom.annotation_ids) if polyline_geom is not None else None
    )

    logger.info(f"Writing annotations to 'by_id' index ({n} rows, unsharded)")
    with tqdm(total=int(n)) as pbar, ts.Transaction() as txn:
        txn_kv = kvstore.with_transaction(txn)
        for offset in range(0, int(n), _UNSHARDED_BATCH_SIZE):
            df_chunk = con.execute(f"""
                SELECT * FROM {INPUT_VIEW}
                LIMIT {_UNSHARDED_BATCH_SIZE} OFFSET {offset}
            """).df().set_index('annotation_id')

            batch_polyline_geom = None
            if polyline_geom is not None:
                rows = polyline_id_lookup.get_indexer(df_chunk.index.to_numpy())
                batch_polyline_geom = _slice_polyline_geom(polyline_geom, rows)

            ann_pb = _encode_annotation_records(
                df_chunk, coord_space, annotation_type, property_specs,
                polyline_geom=batch_polyline_geom,
            )
            rel_pb = _encode_relationship_records(df_chunk, relationships)
            buffers = [ann_pb] + ([rel_pb] if rel_pb is not None else [])

            chunk_keys = df_chunk.index.to_numpy()
            for i, key in enumerate(chunk_keys):
                txn_kv[str(int(key))] = b''.join(pb.slice_for_partition(i) for pb in buffers)

            pbar.update(len(df_chunk))
            del df_chunk, ann_pb, rel_pb, buffers, chunk_keys, batch_polyline_geom

    return {"key": "by_id"}
