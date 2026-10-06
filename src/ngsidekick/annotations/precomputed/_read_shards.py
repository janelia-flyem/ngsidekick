"""
Direct parsing of neuroglancer_uint64_sharded_v1 shard files, as described in the
`sharded format spec <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md>`_.

Tensorstore's ``neuroglancer_uint64_sharded`` kvstore driver can list
and read individual keys, but reading every key of a large index that
way costs ~15 µs of Python overhead per key. When we want *all* of the
keys anyway, it's much faster to read each shard file in one shot and
decode its shard index and minishard indexes ourselves with numpy.

Note:
    The obsolete variant of the format (separate ``.index`` and ``.data``
    files per shard) is not supported.
"""
import zlib
from collections import deque

import numpy as np

# wbits value that tells zlib to expect a gzip header.
_GZIP_WBITS = 16 + zlib.MAX_WBITS


def _decode(data, encoding):
    if encoding == 'gzip':
        return zlib.decompress(data, _GZIP_WBITS)
    if encoding in ('raw', None):
        return data
    raise ValueError(f"Unsupported sharded encoding: {encoding!r}")


def _shard_filename(shard, shard_bits):
    """
    Shard files are named by their lowercase hex shard number,
    zero-padded to ``ceil(shard_bits/4)`` digits.
    """
    width = (shard_bits + 3) // 4
    return f"{shard:0{width}x}.shard"


def _parse_shard(shard_bytes, sharding):
    """
    Parse a complete shard file.

    Args:
        shard_bytes:
            The full contents of the shard file.
        sharding:
            The sharding spec JSON (from the info file).

    Returns:
        (keys, values), where ``keys`` is a uint64 array of the chunk IDs
        in the shard (in storage order) and ``values`` is a list of the
        corresponding decoded values (bytes).
    """
    minishard_index_encoding = sharding.get('minishard_index_encoding', 'raw')
    data_encoding = sharding.get('data_encoding', 'raw')

    num_minishards = 1 << sharding['minishard_bits']
    shard_index_end = 16 * num_minishards
    if len(shard_bytes) < shard_index_end:
        raise ValueError(
            f"Shard file is too small ({len(shard_bytes)} bytes) to contain "
            f"a shard index for {num_minishards} minishards"
        )

    shard_index = np.frombuffer(shard_bytes, '<u8', count=2 * num_minishards).reshape(-1, 2)
    shard_index = shard_index.astype(np.int64) + shard_index_end

    all_keys = []
    all_starts = []
    all_sizes = []
    for start, end in shard_index.tolist():
        if start == end:
            continue
        minishard_index = _decode(shard_bytes[start:end], minishard_index_encoding)
        minishard_index = np.frombuffer(minishard_index, '<u8').reshape(3, -1)

        # Chunk IDs are delta-encoded.
        keys = np.cumsum(minishard_index[0], dtype=np.uint64)

        # Each chunk's start offset is delta-encoded relative
        # to the *end* of the previous chunk.
        sizes = minishard_index[2].astype(np.int64)
        starts = np.cumsum(minishard_index[1].astype(np.int64))
        starts[1:] += np.cumsum(sizes[:-1])
        starts += shard_index_end

        all_keys.append(keys)
        all_starts.append(starts)
        all_sizes.append(sizes)

    if not all_keys:
        return np.zeros(0, np.uint64), []

    keys = np.concatenate(all_keys)
    starts = np.concatenate(all_starts)
    sizes = np.concatenate(all_sizes)
    if (starts + sizes > len(shard_bytes)).any():
        raise ValueError("Shard file is truncated: minishard index refers to data beyond the end of the file.")

    view = memoryview(shard_bytes)
    if data_encoding == 'gzip':
        values = [
            zlib.decompress(view[s:s + z], _GZIP_WBITS)
            for s, z in zip(starts.tolist(), sizes.tolist())
        ]
    else:
        _decode(b'', data_encoding)  # Validate encoding name.
        values = [
            view[s:s + z]
            for s, z in zip(starts.tolist(), sizes.tolist())
        ]
    return keys, values


def _iter_shards(kvstore, sharding, prefetch=4):
    """
    Iterate over all shards in a sharded index, yielding ``(keys, values)``
    for each shard that exists (see :func:`_parse_shard`).

    Rather than listing the directory (which isn't possible for some
    storage backends, e.g. plain HTTP), we attempt to read every possible
    shard number. Shards which were never written (because no keys hashed
    to them) are simply skipped.

    Args:
        kvstore:
            A tensorstore KvStore for the index directory (not the sharded driver).
        sharding:
            The sharding spec JSON (from the info file).
        prefetch:
            How many shard reads to keep in flight while the current
            shard is being parsed.
    """
    shard_bits = sharding['shard_bits']
    shard_names = (_shard_filename(shard, shard_bits) for shard in range(1 << shard_bits))

    pending = deque()
    for name in shard_names:
        pending.append(kvstore.read(name))
        if len(pending) > prefetch:
            yield from _parse_if_present(pending.popleft().result(), sharding)
    while pending:
        yield from _parse_if_present(pending.popleft().result(), sharding)


def _parse_if_present(read_result, sharding):
    if read_result.state == 'value':
        yield _parse_shard(read_result.value, sharding)
