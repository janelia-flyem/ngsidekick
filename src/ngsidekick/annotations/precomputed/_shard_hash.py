"""
Pure-Python re-implementation of tensorstore's neuroglancer_uint64_sharded
key-to-shard mapping.

The sharded format groups uint64 chunk IDs into shards via a hash function,
as specified by the sharding spec. Tensorstore performs this mapping
internally, but does not expose it in its Python API. We replicate it here
so we can pre-compute which shard each key belongs to and group writes into
shard-aligned transactions, dramatically reducing peak RAM during sharded
writes.

References:
- https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md
- https://github.com/google/tensorstore/tree/master/tensorstore/kvstore/neuroglancer_uint64_sharded/{murmurhash3,uint64_sharded}.cc
"""
import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def _mix32(h):
    h ^= h >> np.uint32(16)
    h = np.uint32(h * np.uint32(0x85ebca6b))
    h ^= h >> np.uint32(13)
    h = np.uint32(h * np.uint32(0xc2b2ae35))
    h ^= h >> np.uint32(16)
    return h


@njit(cache=True, inline='always')
def _murmurhash3_x86_128_low64(input_val):
    """
    MurmurHash3_x86_128 specialized to an 8-byte input with a zero seed,
    returning the low 64 bits of the 128-bit hash (i.e. ``h[1] << 32 | h[0]``).

    Bit-exact port of
    ``tensorstore::neuroglancer_uint64_sharded::MurmurHash3_x86_128Hash64Bits``.
    """
    c1 = np.uint32(0x239b961b)
    c2 = np.uint32(0xab0e9789)
    c3 = np.uint32(0x38b34ae5)

    low = np.uint32(input_val & np.uint64(0xFFFFFFFF))
    high = np.uint32((input_val >> np.uint64(32)) & np.uint64(0xFFFFFFFF))

    k2 = np.uint32(high * c2)
    k2 = np.uint32((k2 << np.uint32(16)) | (k2 >> np.uint32(16)))
    k2 = np.uint32(k2 * c3)

    k1 = np.uint32(low * c1)
    k1 = np.uint32((k1 << np.uint32(15)) | (k1 >> np.uint32(17)))
    k1 = np.uint32(k1 * c2)

    # Seed = (0,0,0,0); h2 ^= k2, h1 ^= k1, h3 = h4 = 0.
    # Tensorstore stores h1..h4 as uint64 throughout, with a single uint32
    # truncation when entering the mix function.
    h1 = np.uint64(k1)
    h2 = np.uint64(k2)
    h3 = np.uint64(0)
    h4 = np.uint64(0)

    eight = np.uint64(8)
    h1 ^= eight
    h2 ^= eight
    h3 ^= eight
    h4 ^= eight

    h1 = h1 + h2
    h1 = h1 + h3
    h1 = h1 + h4
    h2 = h2 + h1
    h3 = h3 + h1
    h4 = h4 + h1

    # Mix takes uint32: tensorstore truncates each h to 32 bits at this call.
    mask32 = np.uint64(0xFFFFFFFF)
    h1 = np.uint64(_mix32(np.uint32(h1 & mask32)))
    h2 = np.uint64(_mix32(np.uint32(h2 & mask32)))
    h3 = np.uint64(_mix32(np.uint32(h3 & mask32)))
    h4 = np.uint64(_mix32(np.uint32(h4 & mask32)))

    h1 = h1 + h2
    h1 = h1 + h3
    h1 = h1 + h4
    h2 = h2 + h1
    h3 = h3 + h1
    h4 = h4 + h1

    out0 = h1 & mask32
    out1 = h2 & mask32
    return (out1 << np.uint64(32)) | out0


# Hash function IDs used internally to avoid passing strings into numba.
_HASH_IDENTITY = 0
_HASH_MURMURHASH3 = 1


@njit(cache=True)
def _shards_for_keys_jit(keys, hash_id, preshift_bits, minishard_bits, shard_bits):
    n = keys.shape[0]
    out = np.empty(n, dtype=np.uint64)

    if minishard_bits >= 64:
        minishard_shift = np.uint64(64)
    else:
        minishard_shift = np.uint64(minishard_bits)

    total_bits = minishard_bits + shard_bits
    if total_bits >= 64:
        combined_mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    else:
        combined_mask = (np.uint64(1) << np.uint64(total_bits)) - np.uint64(1)

    if shard_bits >= 64:
        shard_mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    else:
        shard_mask = (np.uint64(1) << np.uint64(shard_bits)) - np.uint64(1)

    preshift = np.uint64(preshift_bits) if preshift_bits < 64 else np.uint64(64)

    for i in range(n):
        key = keys[i]
        if preshift_bits >= 64:
            shifted = np.uint64(0)
        else:
            shifted = key >> preshift

        if hash_id == 0:
            hash_output = shifted
        else:
            hash_output = _murmurhash3_x86_128_low64(shifted)

        shard_and_minishard = hash_output & combined_mask
        if minishard_bits >= 64:
            out[i] = np.uint64(0)
        else:
            out[i] = (shard_and_minishard >> minishard_shift) & shard_mask

    return out


def compute_shard_assignments_in_db(
    con,
    src_view,
    dest_table,
    shard_spec,
    key_col='annotation_id',
):
    """
    Compute the shard number for every key in ``src_view`` and store the
    result as a DuckDB table ``dest_table`` with columns
    ``(key_col, shard_id)``, both uint64.

    Args:
        con:
            DuckDB connection.
        src_view:
            Name of the view/table containing ``key_col`` (and other
            columns, which are ignored).
        dest_table:
            Name of the table to create (or replace) with the
            assignments. Any prior table of the same name is dropped.
        shard_spec:
            ``ShardSpec`` whose hash + bit fields determine the
            assignments.
        key_col:
            Column name to read from ``src_view`` and write to
            ``dest_table``.
    """
    import pyarrow as pa

    # Materialize keys + shards in one shot, then ingest into DuckDB. For
    # 312M annotations this is ~5 GB transient (2.5 GB keys + 2.5 GB
    # shards), which is one-shot and recoverable. A truly streaming
    # variant is hard to express cleanly here -- DuckDB's connection
    # cursors don't share registered views, and using a streaming
    # ``RecordBatchReader`` + ``CREATE TABLE AS SELECT`` on the same
    # connection deadlocks. The dominant RAM win for the by-id phase is
    # in per-batch encoding (one batch's worth of encoded buffers at a
    # time) rather than here, so we accept the one-shot transient.
    keys_arrow = con.execute(f"SELECT {key_col} FROM {src_view}").to_arrow_table()
    keys = keys_arrow.column(key_col).to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    del keys_arrow
    shards = shards_for_keys(keys, shard_spec)
    pairs = pa.table({key_col: keys, 'shard_id': shards.astype(np.uint64, copy=False)})
    del keys, shards

    con.execute(f"DROP TABLE IF EXISTS {dest_table}")
    con.register('_shard_pairs', pairs)
    try:
        # ORDER BY shard_id so DuckDB lays the resulting table out in
        # shard-sorted order. Per-row-group min/max statistics on
        # ``shard_id`` then become tight, and downstream per-batch
        # queries that filter by shard_id (e.g. the by-id writer's
        # ``WHERE shard_id IN (...)`` JOIN) can zone-map-prune almost
        # all row groups instead of doing a full-table scan every
        # batch.
        con.execute(f"""
            CREATE TABLE {dest_table} AS
            SELECT * FROM _shard_pairs ORDER BY shard_id
        """)
    finally:
        con.unregister('_shard_pairs')
    del pairs


def shards_for_keys(keys, shard_spec):
    """
    Compute the shard number for each uint64 key under ``shard_spec``.

    Mirrors tensorstore's
    ``GetSplitShardInfo(GetChunkShardInfo(spec, key)).shard``.

    Args:
        keys:
            1-D array-like of uint64 chunk IDs.
        shard_spec:
            A ``ShardSpec`` (see ``_write_buffers.py``).

    Returns:
        np.ndarray of dtype uint64 with the shard number for each key.
    """
    keys = np.ascontiguousarray(keys, dtype=np.uint64)

    if shard_spec.hash == "identity_hash" or shard_spec.hash == "identity":
        hash_id = _HASH_IDENTITY
    elif shard_spec.hash == "murmurhash3_x86_128":
        hash_id = _HASH_MURMURHASH3
    else:
        raise ValueError(f"Unknown hash function: {shard_spec.hash!r}")

    return _shards_for_keys_jit(
        keys,
        hash_id,
        int(shard_spec.preshift_bits),
        int(shard_spec.minishard_bits),
        int(shard_spec.shard_bits),
    )
