import os
import logging
import shutil
import multiprocessing
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorstore as ts

from ._encode import PartitionedBuffer
from ._shard_hash import shards_for_keys

logger = logging.getLogger(__name__)


def _default_max_threads():
    """
    Process-level default for the number of CPU threads we'll let
    tensorstore use during sharded writes.

    On LSF clusters, ``LSB_DJOB_NUMPROC`` reflects the actual job slot
    count and is preferred over ``multiprocessing.cpu_count()`` (which
    would otherwise see all hardware cores even on a smaller slot).
    """
    if (n := os.environ.get('LSB_DJOB_NUMPROC')):
        return int(n)
    return multiprocessing.cpu_count()


def _build_ts_context(user_spec, max_threads):
    """
    Build the :class:`tensorstore.Context` used by every write in this run.

    ``user_spec`` is the optional JSON spec dict the caller passed to
    :func:`write_precomputed_annotations`. We copy it and fill in
    ``data_copy_concurrency`` / ``file_io_concurrency`` with limits of
    ``max_threads`` only if those keys are missing; any setting the
    caller did include is preserved verbatim.
    """
    spec = dict(user_spec or {})
    spec.setdefault("data_copy_concurrency", {"limit": max_threads})
    spec.setdefault("file_io_concurrency", {"limit": max_threads})
    return ts.Context(spec)


def _write_buffers(keys, buffers: list[PartitionedBuffer], output_dir, subdir, write_sharded, max_shards_per_transaction, ts_context):
    """
    Write per-key byte values to a tensorstore kvstore (all data at once).

    Used by writers that have already encoded the full per-key output into
    PartitionedBuffers. For streaming writes where each transaction's
    buffers are encoded on the fly, see :func:`_write_buffers_streaming`.

    Args:
        keys:
            pd.Index or numpy array of length N, will be cast to uint64.
            For sharded writes, each key is encoded big-endian as the kvstore key;
            for unsharded writes, each key is converted to its decimal string
            representation and used as a filename.

        buffers:
            list of :class:`PartitionedBuffer`. The value written for key ``keys[i]`` is
            the concatenation of each part's slice for row ``i``.

        output_dir:
            str. The directory into which the data will be written.

        subdir:
            str. Subdirectory of ``output_dir`` to (re-)create and populate.

        write_sharded:
            bool. If True, use the neuroglancer_uint64_sharded driver;
            otherwise, write one file per key.

        max_shards_per_transaction:
            int. Caps shards committed per transaction (sharded mode only).

        ts_context:
            :class:`tensorstore.Context` to use for opening the kvstore.
            Pre-built once at the top of
            :func:`write_precomputed_annotations` and threaded through.

    Returns:
        JSON metadata dict for the written subdir.
    """
    _prepare_output_subdir(output_dir, subdir)

    keys = keys.astype(np.uint64, copy=False)

    n_rows = len(keys)
    for pb in buffers:
        if not isinstance(pb.layout, (int, np.integer)):
            assert len(pb.layout) == n_rows + 1, \
                f"offset-layout buffer has length {len(pb.layout)}, expected {n_rows+1}"

    if write_sharded:
        return _write_buffers_sharded(keys, buffers, output_dir, subdir, max_shards_per_transaction, ts_context)
    return _write_buffers_unsharded(keys, buffers, output_dir, subdir, ts_context)


def _prepare_output_subdir(output_dir, subdir):
    """Wipe any previous data at ``output_dir/subdir``."""
    if os.path.exists(f"{output_dir}/{subdir}"):
        shutil.rmtree(f"{output_dir}/{subdir}")


def _open_sharded_kvstore(output_dir, subdir, shard_spec, ts_context):
    """
    Open a sharded tensorstore kvstore at ``output_dir/subdir`` with the
    given ``shard_spec``. The caller is responsible for choosing the
    shard spec (see :func:`_choose_output_spec`) and then writing
    transactions to the returned kvstore (see
    :func:`_write_one_transaction`).
    """
    output_dir = os.path.abspath(output_dir)
    spec = {
        "driver": "neuroglancer_uint64_sharded",
        "metadata": shard_spec.to_json(),
        "base": f"file://{output_dir}/{subdir}",
    }
    return ts.KvStore.open(spec, context=ts_context).result()


def _write_one_transaction(kvstore, batch_keys, batch_buffers):
    """
    Commit one tensorstore transaction containing ``len(batch_keys)``
    key/value pairs.

    ``batch_keys`` is a uint64 ndarray; ``batch_buffers`` is a list of
    :class:`PartitionedBuffer` (each describing one "column" of the
    per-key value, with length matching ``batch_keys``). The value
    written for key ``batch_keys[i]`` is the concatenation of each
    buffer's slice for row ``i``.
    """
    with ts.Transaction() as txn:
        txn_kv = kvstore.with_transaction(txn)
        n = len(batch_keys)
        if len(batch_buffers) == 1:
            pb = batch_buffers[0]
            for i in range(n):
                key = int(batch_keys[i]).to_bytes(8, 'big')
                txn_kv[key] = pb.slice_for_partition(i)
        else:
            for i in range(n):
                key = int(batch_keys[i]).to_bytes(8, 'big')
                txn_kv[key] = b''.join(pb.slice_for_partition(i) for pb in batch_buffers)


def _sharded_metadata(subdir, shard_spec):
    """Build the JSON metadata block for a sharded index subdir."""
    return {"key": subdir, "sharding": shard_spec.to_json()}


def _write_buffers_unsharded(keys, buffers: list[PartitionedBuffer], output_dir, subdir, ts_context):
    """
    Write the buffers to the appropriate subdirectory of output_dir,
    in unsharded format, i.e. one file per item.

    The keys are converted to strings (after going through ``np.asarray``)
    for use as filenames within ``subdir``.
    """
    output_dir = os.path.abspath(output_dir)

    n_rows = len(keys)
    string_keys = [str(k) for k in np.asarray(keys)]

    # Using tensorstore here is mostly a matter of taste, but it makes it
    # straightforward to add alternative storage backends later.
    kvstore = ts.KvStore.open(f"file://{output_dir}/{subdir}/", context=ts_context).result()

    with ts.Transaction() as txn:
        txn_kv = kvstore.with_transaction(txn)
        if len(buffers) == 1:
            pb = buffers[0]
            for i, segment_key in tqdm(enumerate(string_keys), total=n_rows):
                txn_kv[segment_key] = pb.slice_for_partition(i)
        else:
            for i, segment_key in tqdm(enumerate(string_keys), total=n_rows):
                txn_kv[segment_key] = b''.join(pb.slice_for_partition(i) for pb in buffers)

    return {"key": subdir}


def _write_buffers_sharded(keys, buffers: list[PartitionedBuffer], output_dir, subdir, max_shards_per_transaction, ts_context):
    """
    Write the buffers using the neuroglancer_uint64_sharded driver. ``keys``
    is converted to a numpy uint64 array; each key is encoded big-endian
    inline at write time.
    """
    output_dir = os.path.abspath(output_dir)
    n_rows = len(keys)
    keys_arr = np.asarray(keys) if n_rows else np.zeros(0, dtype=np.uint64)
    assert keys_arr.dtype == np.uint64, \
        f"keys must be uint64, got {keys_arr.dtype}"

    total_bytes = sum(pb.total_bytes(n_rows) for pb in buffers)

    shard_spec = _choose_output_spec(
        total_count=n_rows,
        total_bytes=total_bytes,
        max_key=int(keys_arr.max()) if n_rows else (2**64 - 1),
        hashtype='murmurhash3_x86_128',
        gzip_compress=True,
    )
    spec = {
        "driver": "neuroglancer_uint64_sharded",
        "metadata": shard_spec.to_json(),
        "base": f"file://{output_dir}/{subdir}",
    }
    kvstore = ts.KvStore.open(spec, context=ts_context).result()

    # Pre-compute which shard each key will land in (replicating tensorstore's
    # hash) so we can group writes into shard-aligned batches and commit each
    # batch as its own transaction. This caps peak RAM (tensorstore buffers
    # every staged shard until commit) while still letting tensorstore
    # parallelize the per-shard commit work across its thread pool.
    shard_assignments = shards_for_keys(pd.Index(keys_arr), shard_spec)
    batches = shard_assignments // np.uint64(max_shards_per_transaction)

    # Sort rows by batch and find run-boundaries. We index into ``keys`` and
    # ``buffers`` via ``sort_order`` rather than physically permuting the part
    # buffers, which avoids materializing potentially-huge sorted copies.
    sort_order = np.argsort(batches, kind='stable')
    if n_rows == 0:
        boundaries = np.array([0], dtype=np.int64)
    else:
        sorted_batches = batches[sort_order]
        boundaries = np.concatenate((
            [0],
            np.flatnonzero(sorted_batches[1:] != sorted_batches[:-1]) + 1,
            [n_rows],
        ))
        del sorted_batches
    del shard_assignments, batches

    # Tensorstore's neuroglancer_uint64_sharded driver requires keys to be
    # the bigendian-uint64 encoding of the chunk ID
    # (https://github.com/google/neuroglancer/pull/522#issuecomment-1923137085).
    # We encode each one inline with int.to_bytes() rather than materializing
    # all N keys upfront: a parallel array of N small bytes objects costs
    # ~50 B of Python-object overhead per item (~15 GB at 300M items),
    # whereas the inline encoding allocates one transient 8-byte bytes object
    # per write that tensorstore copies and immediately frees.
    with tqdm(total=n_rows) as pbar:
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            with ts.Transaction() as txn:
                txn_kv = kvstore.with_transaction(txn)
                if len(buffers) == 1:
                    pb = buffers[0]
                    for i in range(s, e):
                        orig = int(sort_order[i])
                        key = int(keys_arr[orig]).to_bytes(8, 'big')
                        txn_kv[key] = pb.slice_for_partition(orig)
                else:
                    for i in range(s, e):
                        orig = int(sort_order[i])
                        key = int(keys_arr[orig]).to_bytes(8, 'big')
                        txn_kv[key] = b''.join(
                            pb.slice_for_partition(orig)
                            for pb in buffers
                        )
            pbar.update(e - s)

    return {"key": subdir, "sharding": shard_spec.to_json()}


@dataclass
class ShardSpec:
    """
    Copied from Forrest Collman's PR:
    https://github.com/google/neuroglancer/pull/522
    """
    type: str
    hash: Literal["murmurhash3_x86_128", "identity_hash"]
    preshift_bits: int
    shard_bits: int
    minishard_bits: int
    data_encoding: Literal["raw", "gzip"]
    minishard_index_encoding: Literal["raw", "gzip"]

    def to_json(self):
        return {
            "@type": self.type,
            "hash": self.hash,
            "preshift_bits": self.preshift_bits,
            "shard_bits": self.shard_bits,
            "minishard_bits": self.minishard_bits,
            "data_encoding": str(self.data_encoding),
            "minishard_index_encoding": str(self.minishard_index_encoding),
        }


def _choose_output_spec(
    total_count,
    total_bytes,
    max_key=2**64 - 1,
    hashtype: Literal["murmurhash3_x86_128", "identity_hash"] = "murmurhash3_x86_128",
    gzip_compress=True,
):
    """
    Adapted from Forrest Collman's PR:
    https://github.com/google/neuroglancer/pull/522

    Deviations from the original:
        - Accepts ``max_key`` (the largest chunk ID that will be written) so
          that ``preshift_bits`` is capped at the actual keyspace headroom.
          Without this cap, a small dense keyspace (e.g. spatial-index
          chunk codes that span only 9 bits) would have its low bits
          shifted away and every key would hash to the same shard, defeating
          the chosen ``shard_bits``.
    """
    MINISHARD_TARGET_COUNT = 1000
    SHARD_TARGET_SIZE = 50000000

    if hashtype not in ["murmurhash3_x86_128", "identity_hash"]:
        raise ValueError(
            f"Invalid hashtype {hashtype}."
            "Must be one of 'murmurhash3_x86_128' "
            "or 'identity_hash'"
        )

    total_minishard_bits = 0
    while (total_count >> total_minishard_bits) > MINISHARD_TARGET_COUNT:
        total_minishard_bits += 1

    shard_bits = 0
    while (total_bytes >> shard_bits) > SHARD_TARGET_SIZE:
        shard_bits += 1

    minishard_bits = total_minishard_bits - min(total_minishard_bits, shard_bits)

    # The original heuristic strips ~log2(MINISHARD_TARGET_COUNT) low bits
    # so that runs of adjacent chunk IDs co-locate in one minishard. That
    # only makes sense if the keyspace has enough bits to spare: after
    # preshifting we still need at least 2**(shard_bits + minishard_bits)
    # distinct shifted values, otherwise every key collapses to the same
    # shard. ``keyspace_cap`` enforces that.
    target_preshift = 0
    while MINISHARD_TARGET_COUNT >> target_preshift:
        target_preshift += 1
    keyspace_cap = max(0, int(max_key).bit_length() - shard_bits - minishard_bits)
    preshift_bits = min(target_preshift, keyspace_cap)

    data_encoding: Literal["raw", "gzip"] = "raw"
    minishard_index_encoding: Literal["raw", "gzip"] = "raw"

    if gzip_compress:
        data_encoding = "gzip"
        minishard_index_encoding = "gzip"

    return ShardSpec(
        type="neuroglancer_uint64_sharded_v1",
        hash=hashtype,
        preshift_bits=preshift_bits,
        shard_bits=shard_bits,
        minishard_bits=minishard_bits,
        data_encoding=data_encoding,
        minishard_index_encoding=minishard_index_encoding,
    )
