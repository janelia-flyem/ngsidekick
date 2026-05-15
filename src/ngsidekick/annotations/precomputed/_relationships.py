import logging

import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm.auto import tqdm

from . import _write_buffers
from ._db import INPUT_VIEW
from ._encode import (
    PartitionedBuffer,
    _build_grouped_record_buffers,
    _encode_annotation_records,
    _encode_id_bytes,
)
from ._memory import log_memory
from ._shard_hash import shards_for_keys
from ._util import _ann_required_cols, _property_recsize, _slice_polyline_geom
from ._write_buffers import (
    _open_sharded_kvstore,
    _prepare_output_subdir,
    _sharded_metadata,
    _write_one_transaction,
)

logger = logging.getLogger(__name__)


def _write_annotations_by_relationships(con, coord_space, annotation_type, property_specs, relationships, polyline_geom,
                                        output_dir, write_sharded, max_shards_per_transaction, ts_context):
    """
    Write the annotations to a "Related Object ID Index" for each
    relationship in ``relationships``.

    See :func:`_write_annotations_by_relationship` for the per-relationship
    streaming pipeline.

    Args:
        con:
            DuckDB connection with the input registered as
            :data:`INPUT_VIEW`.
        coord_space, annotation_type, property_specs, relationships, polyline_geom:
            See :func:`write_precomputed_annotations`.
        output_dir, write_sharded, max_shards_per_transaction, ts_context:
            See :func:`._write_buffers._write_buffers`.

    Returns:
        List of JSON metadata dicts (one per relationship) for the
        'relationships' key in the top-level 'info' file.
    """
    by_rel_metadata = []
    for relationship in relationships:
        metadata = _write_annotations_by_relationship(
            con, coord_space, annotation_type, property_specs, relationship, polyline_geom,
            output_dir, write_sharded, max_shards_per_transaction, ts_context,
        )
        by_rel_metadata.append(metadata)
    return by_rel_metadata


def _write_annotations_by_relationship(con, coord_space, annotation_type, property_specs, relationship,
                                       polyline_geom,
                                       output_dir, write_sharded, max_shards_per_transaction, ts_context):
    """
    Write the annotations to a "Related Object ID Index" for a single
    relationship.

    For each unique related-object id, the output value is::

        <count:uint64le><ann_record_1>...<ann_record_count><ann_id_1:uint64le>...<ann_id_count:uint64le>

    where ``count`` is the number of annotations that reference this
    related id, and each ``ann_record_k`` is the encoded geometry+property
    record of that annotation (per the spec, relationship lists are
    *omitted* from records in this index).

    Streaming pipeline:
      1. Enumerate the distinct related-object ids via SQL. These are the
         output keys.
      2. Compute a shard for each key; store the (key, shard_id)
         assignments as a DuckDB table.
      3. Group the distinct occupied shards into chunks of
         ``max_shards_per_transaction``. Each chunk is one tensorstore
         transaction.
      4. For each chunk: query the annotations whose relationship value
         falls in the chunk's shard set; encode the geometry+property
         records and annotation-id buffer once; build per-segment offset
         layouts; write the transaction.

    Only scalar uint64 relationship columns are supported in this
    streaming pipeline; list-typed relationships raise
    NotImplementedError.
    """
    if not write_sharded:
        return _write_annotations_by_relationship_unsharded(
            con, coord_space, annotation_type, property_specs, relationship, polyline_geom,
            output_dir, ts_context,
        )

    shard_assignments_table = f'_by_rel_shard_assignments__{relationship}'
    chunk_segments_view = f'_by_rel_chunk_segments__{relationship}'
    pairs_view = f'_by_rel_pairs__{relationship}'

    logger.info(f"Preparing 'by_rel_{relationship}' index")

    # 0. Build the per-relationship (annotation_id, segment_id) pairs
    #    source. Scalar columns become a lightweight view; list columns
    #    are materialized as a table (so the UNNEST + DISTINCT isn't
    #    repeated across every per-chunk query). Downstream queries see
    #    a uniform flat pairs source regardless.
    pairs_kind = _create_pairs_source(con, relationship, pairs_view)

    try:
        return _write_annotations_by_relationship_sharded(
            con, coord_space, annotation_type, property_specs, relationship, polyline_geom,
            pairs_view, shard_assignments_table, chunk_segments_view,
            output_dir, max_shards_per_transaction, ts_context,
        )
    finally:
        _drop_pairs_source(con, pairs_view, pairs_kind)


def _write_annotations_by_relationship_sharded(con, coord_space, annotation_type, property_specs, relationship,
                                                polyline_geom,
                                                pairs_view, shard_assignments_table, chunk_segments_view,
                                                output_dir, max_shards_per_transaction, ts_context):
    """Sharded by-rel write. Uses the pre-built pairs view as its row source."""
    polyline_id_lookup = (
        pd.Index(polyline_geom.annotation_ids) if polyline_geom is not None else None
    )
    # 1. Distinct related-object ids (= the output keys for this relationship).
    distinct_ids_table = con.execute(f"""
        SELECT DISTINCT segment_id FROM {pairs_view} ORDER BY segment_id
    """).to_arrow_table()
    segment_ids = distinct_ids_table.column('segment_id').to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    n_segments = len(segment_ids)
    if n_segments == 0:
        # Nothing to write. Emit an empty index.
        _prepare_output_subdir(output_dir, f"by_rel_{relationship}")
        return {"key": f"by_rel_{relationship}", "id": relationship}

    # 2. Choose shard spec from a payload-size estimate.
    total_bytes = _estimate_total_bytes_for_by_rel(
        con, n_segments, pairs_view, coord_space, annotation_type, property_specs, polyline_geom,
    )
    shard_spec = _write_buffers._choose_output_spec(
        total_count=int(n_segments),
        total_bytes=int(total_bytes),
        max_key=int(segment_ids.max()),
        hashtype='murmurhash3_x86_128',
        gzip_compress=True,
    )

    # 3. Compute shards for each distinct segment id, store as DuckDB table.
    shards = shards_for_keys(segment_ids, shard_spec)
    pairs = pa.table({
        'segment_id': segment_ids,
        'shard_id': shards.astype(np.uint64, copy=False),
    })
    del segment_ids, shards
    con.execute(f"DROP TABLE IF EXISTS {shard_assignments_table}")
    con.register('_by_rel_pairs', pairs)
    try:
        con.execute(f"CREATE TABLE {shard_assignments_table} AS SELECT * FROM _by_rel_pairs")
    finally:
        con.unregister('_by_rel_pairs')
    del pairs

    # 4. Open kvstore and iterate occupied shards in batches.
    _prepare_output_subdir(output_dir, f"by_rel_{relationship}")
    kvstore = _open_sharded_kvstore(output_dir, f"by_rel_{relationship}", shard_spec, ts_context)

    occupied_shards = con.execute(f"""
        SELECT DISTINCT shard_id FROM {shard_assignments_table}
        ORDER BY shard_id
    """).to_arrow_table().column('shard_id').to_numpy(zero_copy_only=False)

    batch_size = int(max_shards_per_transaction)
    n_transactions = (len(occupied_shards) + batch_size - 1) // batch_size
    logger.info(f"Writing annotations to 'by_rel_{relationship}' index "
                f"({n_transactions} transactions over "
                f"{len(occupied_shards)} occupied shards "
                f"(of {1 << shard_spec.shard_bits} possible))")

    needed_cols = _ann_required_cols(coord_space, annotation_type, property_specs)
    select_cols = ', '.join(f'v.{c}' for c in (['annotation_id'] + needed_cols))

    log_memory(f'by_rel_{relationship} pre-write-loop')
    with tqdm(total=int(n_segments)) as pbar:
        for batch_idx, chunk_start in enumerate(range(0, len(occupied_shards), batch_size)):
            chunk_shards = occupied_shards[chunk_start:chunk_start + batch_size]

            # The set of segment ids in this batch; we'll need it both for
            # the JOIN that fetches the relevant annotations and for the
            # final per-segment count/offsets.
            con.register(chunk_segments_view, pa.table({'shard_id': chunk_shards}))
            try:
                batch_segments = con.execute(f"""
                    SELECT s.segment_id, s.shard_id
                    FROM {chunk_segments_view} c
                    JOIN {shard_assignments_table} s ON s.shard_id = c.shard_id
                    ORDER BY s.shard_id, s.segment_id
                """).to_arrow_table()
                segs_in_chunk = batch_segments.column('segment_id').to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
                del batch_segments

                df_batch = con.execute(f"""
                    SELECT {select_cols}, p.segment_id AS _segment_id
                    FROM {chunk_segments_view} c
                    JOIN {shard_assignments_table} s ON s.shard_id = c.shard_id
                    JOIN {pairs_view} p ON p.segment_id = s.segment_id
                    JOIN {INPUT_VIEW} v ON v.annotation_id = p.annotation_id
                    ORDER BY _segment_id, v.annotation_id
                """).df()
            finally:
                con.unregister(chunk_segments_view)

            if len(df_batch) == 0:
                pbar.update(len(segs_in_chunk))
                continue

            batch_polyline_geom = None
            if polyline_geom is not None:
                rows = polyline_id_lookup.get_indexer(df_batch['annotation_id'].to_numpy())
                batch_polyline_geom = _slice_polyline_geom(polyline_geom, rows)

            buffers, chunk_segs_with_data = _build_grouped_record_buffers(
                df_batch, '_segment_id', coord_space, annotation_type, property_specs,
                polyline_geom=batch_polyline_geom,
            )
            _write_one_transaction(kvstore, chunk_segs_with_data, buffers)
            pbar.update(len(segs_in_chunk))
            del df_batch, buffers, chunk_segs_with_data, segs_in_chunk, batch_polyline_geom
            log_memory(f'by_rel_{relationship} post-batch {batch_idx + 1}/{n_transactions}')

    con.execute(f"DROP TABLE {shard_assignments_table}")
    log_memory(f'by_rel_{relationship} done')
    metadata = _sharded_metadata(f"by_rel_{relationship}", shard_spec)
    metadata['id'] = relationship
    return metadata




def _create_pairs_source(con, relationship, name):
    """
    Build a (annotation_id, segment_id) DuckDB source named ``name`` that
    flattens one relationship column into a uniform pairs source
    consumed by the rest of the by-rel pipeline.

    For a scalar integer column, this is a lightweight view -- the
    per-chunk JOIN can push its segment-id filter through cleanly and
    re-running the WHERE on each chunk is negligible.

    For a list column it's a materialized table -- the UNNEST + DISTINCT
    inside a view would be re-executed for every per-chunk query, since
    DuckDB's planner can't generally push a chunk-segment filter through
    DISTINCT. Materializing once costs ~16 bytes per (annotation, segment)
    pair in DuckDB storage but saves ~N_transactions×UNNEST work.

    The segment_id is cast to ``UBIGINT`` so downstream consumers can
    rely on a fixed type regardless of what DuckDB inferred from the
    source data.

    Returns:
        ``'view'`` or ``'table'``, so callers know which ``DROP`` flavor
        to use at cleanup.
    """
    duck_type = con.execute(
        f"SELECT typeof({relationship}) FROM {INPUT_VIEW} LIMIT 1"
    ).fetchone()
    type_str = str(duck_type[0]).upper() if duck_type is not None else ''
    is_list = type_str.endswith('[]') or type_str.startswith('LIST')

    con.execute(f"DROP VIEW IF EXISTS {name}")
    con.execute(f"DROP TABLE IF EXISTS {name}")

    if is_list:
        con.execute(f"""
            CREATE TABLE {name} AS
            SELECT DISTINCT
                annotation_id,
                CAST(UNNEST({relationship}) AS UBIGINT) AS segment_id
            FROM {INPUT_VIEW}
            WHERE {relationship} IS NOT NULL
        """)
        return 'table'
    else:
        con.execute(f"""
            CREATE VIEW {name} AS
            SELECT
                annotation_id,
                CAST({relationship} AS UBIGINT) AS segment_id
            FROM {INPUT_VIEW}
            WHERE {relationship} IS NOT NULL
        """)
        return 'view'


def _drop_pairs_source(con, name, kind):
    """Drop a pairs source previously created by :func:`_create_pairs_source`."""
    if kind == 'table':
        con.execute(f"DROP TABLE IF EXISTS {name}")
    else:
        con.execute(f"DROP VIEW IF EXISTS {name}")


def _estimate_total_bytes_for_by_rel(con, n_segments, pairs_view, coord_space, annotation_type, property_specs,
                                     polyline_geom):
    """
    Rough upper-bound estimate of the by-rel payload bytes for one
    relationship: 8 bytes per segment for the count header plus
    ``(ann_recsize + 8) * total_pairs`` for the encoded records and
    annotation-id buffers. Used only by ``_choose_output_spec``.

    Polylines have variable per-record size, so the per-record byte
    cost is estimated as the by-id polyline payload divided by the
    polyline count (i.e. the average per-annotation record size).
    """
    if n_segments == 0:
        return 0

    n_total_rows = con.execute(f"SELECT COUNT(*) FROM {pairs_view}").fetchone()[0]

    if annotation_type == 'polyline':
        ann_recsize = _avg_polyline_recsize(polyline_geom, property_specs)
        return 8 * n_segments + (ann_recsize + 8) * int(n_total_rows)

    probe_df = (
        con.execute(f"SELECT * FROM {INPUT_VIEW} LIMIT 0")
        .df()
        .set_index('annotation_id')
    )
    ann_pb = _encode_annotation_records(
        probe_df, coord_space, annotation_type, property_specs, polyline_geom=None,
    )
    ann_recsize = int(ann_pb.layout) if isinstance(ann_pb.layout, (int, np.integer)) else 0
    return 8 * n_segments + (ann_recsize + 8) * int(n_total_rows)


def _avg_polyline_recsize(polyline_geom, property_specs):
    """
    Average per-annotation record size for polylines: 4-byte count
    header plus the per-polyline vertex bytes plus the property record
    (padded). Used for shard-size heuristics where a single recsize
    must stand in for the encoder's variable-width records.
    """
    if polyline_geom is None or len(polyline_geom.starts) == 0:
        return 0
    n = len(polyline_geom.starts)
    avg_vertex_bytes = int(polyline_geom.points.nbytes) // n
    return 4 + avg_vertex_bytes + _property_recsize(property_specs)


def _write_annotations_by_relationship_unsharded(con, coord_space, annotation_type, property_specs, relationship,
                                                 polyline_geom, output_dir, ts_context):
    """
    Unsharded variant: one file per related-object id, with the decimal
    id as the filename. The file contents follow the same
    ``<count><records><ids>`` layout as the sharded case.
    """
    import os
    import tensorstore as ts

    pairs_view = f'_by_rel_pairs__{relationship}'
    pairs_kind = _create_pairs_source(con, relationship, pairs_view)
    try:
        logger.info(f"Preparing 'by_rel_{relationship}' index (unsharded)")
        _prepare_output_subdir(output_dir, f"by_rel_{relationship}")

        needed_cols = _ann_required_cols(coord_space, annotation_type, property_specs)
        select_cols = ', '.join(f'v.{c}' for c in (['annotation_id'] + needed_cols))

        # For unsharded we read everything at once, sorted by segment id.
        df_full = con.execute(f"""
            SELECT {select_cols}, p.segment_id AS _segment_id
            FROM {pairs_view} p
            JOIN {INPUT_VIEW} v ON v.annotation_id = p.annotation_id
            ORDER BY _segment_id, v.annotation_id
        """).df()

        output_dir = os.path.abspath(output_dir)
        kvstore = ts.KvStore.open(
            f"file://{output_dir}/by_rel_{relationship}/", context=ts_context,
        ).result()

        if len(df_full) == 0:
            return {"key": f"by_rel_{relationship}", "id": relationship}

        sliced_polyline_geom = None
        if polyline_geom is not None:
            polyline_id_lookup = pd.Index(polyline_geom.annotation_ids)
            rows = polyline_id_lookup.get_indexer(df_full['annotation_id'].to_numpy())
            sliced_polyline_geom = _slice_polyline_geom(polyline_geom, rows)

        buffers, unique_segs = _build_grouped_record_buffers(
            df_full, '_segment_id', coord_space, annotation_type, property_specs,
            polyline_geom=sliced_polyline_geom,
        )

        logger.info(f"Writing annotations to 'by_rel_{relationship}' index "
                    f"({len(unique_segs)} segments, unsharded)")
        with tqdm(total=len(unique_segs)) as pbar, ts.Transaction() as txn:
            txn_kv = kvstore.with_transaction(txn)
            for i, seg in enumerate(unique_segs):
                txn_kv[str(int(seg))] = b''.join(pb.slice_for_partition(i) for pb in buffers)
                pbar.update(1)

        return {"key": f"by_rel_{relationship}", "id": relationship}
    finally:
        _drop_pairs_source(con, pairs_view, pairs_kind)
