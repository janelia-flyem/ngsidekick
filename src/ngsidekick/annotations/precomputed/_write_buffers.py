import os
import logging
import shutil
import multiprocessing
from dataclasses import dataclass
from typing import Literal

import numpy as np
from tqdm.auto import tqdm
import tensorstore as ts

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


def _write_buffers(buf_series, output_dir, subdir, write_sharded, max_shards_per_transaction, max_threads):
    """
    Write the buffers to the appropriate subdirectory of output_dir,
    in sharded or unsharded format.

    Args:
        buf_series:
            pd.Series of dtype=object, whose values are buffers (bytes objects).
            The index of the series provides the keys under which each item is stored.

        output_dir:
            str
            The directory into which the exported annotations will be written.

        subdir:
            str
            The subdirectory into which the buffers will be written.
            If it already exists, it will be deleted before we (re)create it.

        write_sharded:
            bool
            If True, write the buffers in sharded format.
            If False, write one file per item.

        max_shards_per_transaction:
            int
            (Sharded mode only.) Caps the number of shards committed in a
            single tensorstore transaction, controlling the RAM/throughput
            tradeoff.

        max_threads:
            int
            Caps tensorstore's internal data-copy and file-I/O thread pool
            via the ``ts.Context`` constructed when opening the kvstore.

    Returns:
        JSON metadata for the written data, including the key (subdir)
        and sharding spec if applicable.
    """
    if os.path.exists(f"{output_dir}/{subdir}"):
        shutil.rmtree(f"{output_dir}/{subdir}")

    if write_sharded:
        return _write_buffers_sharded(buf_series, output_dir, subdir, max_shards_per_transaction, max_threads)
    else:
        return _write_buffers_unsharded(buf_series, output_dir, subdir, max_threads)


def _write_buffers_unsharded(buf_series, output_dir, subdir, max_threads):
    """
    Write the buffers to the appropriate subdirectory of output_dir,
    in unsharded format, i.e. one file per item.

    The index of buf_series is used as the key for each item, after being
    converted to a string (as decimal values in the case of integer keys).

    Returns:
        JSON metadata, always {"key": subdir}
    """
    output_dir = os.path.abspath(output_dir)

    # In the unsharded format, the keys are just strings (e.g. decimal IDs).
    string_keys = buf_series.index.astype(str)
    buf_series = buf_series.set_axis(string_keys)

    # Since we're writing unsharded files, we could have just used
    # standard Python open() and write() here for each key.
    # Using tensorstore here is mostly just a matter of taste, but it will
    # become useful if we ever support alternative storage backends such as gcs.
    context = ts.Context({
        "data_copy_concurrency": {"limit": max_threads},
        "file_io_concurrency": {"limit": max_threads},
    })
    kvstore = ts.KvStore.open(f"file://{output_dir}/{subdir}/", context=context).result()

    # Using a transaction here is not necessary, at least for plain files.
    # I'm not sure if it helps or hurts, but it probably doesn't matter much
    # for small datasets, which is presumably what we're dealing with if the
    # user has chosen the unsharded format.
    with ts.Transaction() as txn:
        for segment_key, buf in tqdm(buf_series.items(), total=len(buf_series)):
            kvstore.with_transaction(txn)[segment_key] = buf

    metadata = {"key": subdir}
    return metadata


def _write_buffers_sharded(buf_series, output_dir, subdir, max_shards_per_transaction, max_threads):
    """
    Write the buffers to the appropriate subdirectory of output_dir,
    in sharded format.

    The index of buf_series is used as the key for each item,
    after being encoded as a bigendian uint64.

    Args:
        buf_series, output_dir, subdir:
            See :func:`_write_buffers`.

        max_shards_per_transaction:
            int
            Caps how many shards' worth of data may be staged in a single
            tensorstore transaction. Within a single transaction tensorstore
            parallelizes encoding, compression, and writing across its
            internal thread pool — so a transaction containing many shards
            saturates the available cores at commit, while a transaction
            containing one shard leaves them idle. The tradeoff is RAM:
            tensorstore holds every staged shard's data in memory until
            commit.

        max_threads:
            int
            Caps tensorstore's internal data-copy and file-I/O thread pool
            via the ``ts.Context`` constructed when opening the kvstore.

    Returns:
        JSON metadata, including the output "key" (subdir) and sharding spec.
    """
    output_dir = os.path.abspath(output_dir)

    shard_spec = _choose_output_spec(
        total_count=len(buf_series),
        total_bytes=buf_series.map(len).sum(),  # fixme, might be slow
        max_key=int(buf_series.index.max()),
        hashtype='murmurhash3_x86_128',
        gzip_compress=True
    )
    spec = {
        "driver": "neuroglancer_uint64_sharded",
        "metadata": shard_spec.to_json(),
        "base": f"file://{output_dir}/{subdir}",
    }
    context = ts.Context({
        "data_copy_concurrency": {"limit": max_threads},
        "file_io_concurrency": {"limit": max_threads},
    })
    kvstore = ts.KvStore.open(spec, context=context).result()

    # Pre-compute which shard each key will land in (replicating tensorstore's
    # hash) so we can group writes into shard-aligned batches and commit each
    # batch as its own transaction. This caps peak RAM (tensorstore buffers
    # every staged shard until commit) while still letting tensorstore
    # parallelize the per-shard commit work across its thread pool: each
    # transaction owns up to ``max_shards_per_transaction`` distinct shards.
    shard_assignments = shards_for_keys(buf_series.index, shard_spec)

    # Bucket adjacent shard numbers into batches. Shard numbers occupy
    # [0, 2**shard_bits), so integer-dividing by max_shards_per_transaction
    # yields up to ``max_shards_per_transaction`` distinct shards per batch.
    # When ``max_shards_per_transaction`` >= 2**shard_bits all data lands in
    # one batch (matching the prior single-transaction behavior).
    batches = shard_assignments // np.uint64(max_shards_per_transaction)

    # Tensorstore's neuroglancer_uint64_sharded driver requires keys to be
    # the bigendian-uint64 encoding of the chunk ID
    # (https://github.com/google/neuroglancer/pull/522#issuecomment-1923137085).
    with tqdm(total=len(buf_series)) as pbar:
        for _batch, group in buf_series.groupby(batches, sort=False):
            with ts.Transaction() as txn:
                txn_kv = kvstore.with_transaction(txn)
                for key, buf in group.items():
                    txn_kv[int(key).to_bytes(8, 'big')] = buf
            pbar.update(len(group))

    metadata = {
        "key": subdir,
        "sharding": shard_spec.to_json()
    }
    return metadata


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
    import tensorstore as ts
    MINISHARD_TARGET_COUNT = 1000
    SHARD_TARGET_SIZE = 50000000

    # if total_count == 1:
    #     return None
    # if ts is None:
    #     return None

    # test if hashtype is valid
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
