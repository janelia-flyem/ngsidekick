"""
Tests for our re-implementation of tensorstore's neuroglancer_uint64_sharded
key-to-shard mapping, and for the per-shard batched write path that depends
on it.

The shard-prediction logic in ``_shard_hash.py`` exists to let us split
a single huge tensorstore transaction into per-shard transactions, which
cuts peak RAM by an order of magnitude on large writes. If our prediction
ever diverges from tensorstore's actual placement, sharded writes could
silently land in the wrong file (or, more likely, just lose the RAM
benefit of being shard-aligned). These tests are the safety net for that.
"""
import os
import json
import glob

import numpy as np
import pandas as pd
import pytest
import tensorstore as ts

from neuroglancer.coordinate_space import CoordinateSpace

from ngsidekick.annotations.precomputed import write_precomputed_annotations
from ngsidekick.annotations.precomputed import _write_buffers as wb
from ngsidekick.annotations.precomputed._shard_hash import (
    shards_for_keys,
    _murmurhash3_x86_128_low64,
)
from ngsidekick.annotations.precomputed._write_buffers import ShardSpec


def _be_key(uint64_value):
    """
    Encode a single uint64 as 8 bigendian bytes — the only key form accepted
    by the tensorstore neuroglancer_uint64_sharded driver.
    """
    return np.array([uint64_value], dtype=np.uint64).astype('>u8').tobytes()


def _shard_spec(hash, preshift_bits, shard_bits, minishard_bits,
                data_encoding="raw", minishard_index_encoding="raw"):
    return ShardSpec(
        type="neuroglancer_uint64_sharded_v1",
        hash=hash,
        preshift_bits=preshift_bits,
        shard_bits=shard_bits,
        minishard_bits=minishard_bits,
        data_encoding=data_encoding,
        minishard_index_encoding=minishard_index_encoding,
    )


def _shard_files(base_dir):
    """Return the set of shard numbers for which a .shard file exists."""
    return {
        int(os.path.basename(f).split('.')[0], 16)
        for f in glob.glob(os.path.join(str(base_dir), "*.shard"))
    }


# Golden vectors copied verbatim from tensorstore's
# tensorstore/kvstore/neuroglancer_uint64_sharded/murmurhash3_test.cc
# (the rows where the test starts with h[0..3] = 0).
@pytest.mark.parametrize("input_val,expected_h0,expected_h1", [
    (0,  0xe028ae41, 0x4772b084),
    (1,  0x16d4ce9a, 0xe8bd67d6),
    (42, 0x5119f47a, 0xc20b94f9),
])
def test_murmurhash3_golden_vectors(input_val, expected_h0, expected_h1):
    expected = (expected_h1 << 32) | expected_h0
    got = int(_murmurhash3_x86_128_low64(np.uint64(input_val)))
    assert got == expected


@pytest.mark.parametrize("spec", [
    # A spread of murmur and identity configurations, including
    # edge cases (shard_bits == 0, big preshift, etc).
    _shard_spec("murmurhash3_x86_128", preshift_bits=0,  shard_bits=4, minishard_bits=3),
    _shard_spec("murmurhash3_x86_128", preshift_bits=2,  shard_bits=5, minishard_bits=4),
    _shard_spec("murmurhash3_x86_128", preshift_bits=10, shard_bits=0, minishard_bits=4),
    _shard_spec("identity",            preshift_bits=0,  shard_bits=4, minishard_bits=3),
    _shard_spec("identity",            preshift_bits=8,  shard_bits=3, minishard_bits=2),
])
def test_shards_for_keys_per_key_matches_tensorstore(spec, tmp_path):
    """
    Per-key strict check: write each key into its own (fresh) sharded
    kvstore and verify that the shard file tensorstore created has the
    shard number we predicted. This is the strongest possible verification
    short of parsing the shard files ourselves.
    """
    rng = np.random.default_rng(2026)
    keys = rng.integers(0, 2**63, size=20, dtype=np.uint64)
    predicted = shards_for_keys(keys, spec)

    for k, pred_shard in zip(keys, predicted):
        kvdir = tmp_path / f"k{int(k)}"
        kv = ts.KvStore.open({
            "driver": "neuroglancer_uint64_sharded",
            "metadata": spec.to_json(),
            "base": f"file://{kvdir}",
        }).result()
        with ts.Transaction() as txn:
            kv.with_transaction(txn)[_be_key(int(k))] = b"v"

        actual = _shard_files(kvdir)
        assert actual == {int(pred_shard)}, (
            f"key={int(k)} ({int(k):#x}): predicted shard {int(pred_shard)}, "
            f"tensorstore wrote to {actual}"
        )


@pytest.mark.parametrize("spec", [
    _shard_spec("murmurhash3_x86_128", preshift_bits=0, shard_bits=6, minishard_bits=4),
    _shard_spec("murmurhash3_x86_128", preshift_bits=4, shard_bits=5, minishard_bits=3),
    _shard_spec("identity",            preshift_bits=4, shard_bits=4, minishard_bits=3),
])
def test_shards_for_keys_set_matches_tensorstore(spec, tmp_path):
    """
    Bulk check: write many keys to a single kvstore in one transaction and
    confirm the set of shards we predicted matches the set of .shard files
    tensorstore actually produced.
    """
    rng = np.random.default_rng(7)
    keys = rng.integers(0, 2**63, size=5_000, dtype=np.uint64)
    predicted = {int(s) for s in shards_for_keys(keys, spec)}

    kv = ts.KvStore.open({
        "driver": "neuroglancer_uint64_sharded",
        "metadata": spec.to_json(),
        "base": f"file://{tmp_path}",
    }).result()
    with ts.Transaction() as txn:
        for k in keys:
            kv.with_transaction(txn)[_be_key(int(k))] = b"x"

    assert _shard_files(tmp_path) == predicted


def test_choose_output_spec_caps_preshift_for_small_keyspace():
    """
    Regression test for the bug where ``_choose_output_spec`` always
    returned ``preshift_bits=10``, which collapses every key in a
    sub-1024 keyspace (e.g. a small spatial-index level) to chunk_id 0
    after preshifting — producing one shard regardless of the chosen
    ``shard_bits``.

    For a keyspace that needs all of its bits to fill the requested
    shard layout, ``preshift_bits`` must be 0.
    """
    from ngsidekick.annotations.precomputed._write_buffers import _choose_output_spec

    # Mirrors by_spatial_level_6 from the bug report: 192 chunks, ~14 GB
    # of compressed data, max chunk code = 511 (9 bits, dense).
    spec = _choose_output_spec(
        total_count=192,
        total_bytes=14_000_000_000,
        max_key=511,
    )
    assert spec.shard_bits == 9, "test premise: heuristic should still pick shard_bits=9"
    assert spec.preshift_bits == 0, (
        f"preshift_bits={spec.preshift_bits} would shift every 9-bit key to 0 "
        f"and collapse all keys into a single shard"
    )

    # And the canonical wide-keyspace case still gets the historical preshift
    # (so we know the fix is targeted, not blanket).
    spec = _choose_output_spec(
        total_count=300_000_000,
        total_bytes=30_000_000_000,
        max_key=2**40,
    )
    assert spec.preshift_bits == 10


def test_choose_output_spec_default_max_key_is_unrestrictive():
    """
    With no ``max_key`` supplied, ``_choose_output_spec`` should fall back
    to the original heuristic (preshift_bits=10 for any non-trivial spec)
    so this remains a drop-in replacement for callers that haven't been
    updated yet.
    """
    from ngsidekick.annotations.precomputed._write_buffers import _choose_output_spec
    spec = _choose_output_spec(
        total_count=192,
        total_bytes=14_000_000_000,
    )
    assert spec.preshift_bits == 10


def test_dense_small_keyspace_actually_multishards(tmp_path):
    """
    End-to-end check: writing 192 buffers under dense Morton-code-style
    keys [0, 192) with a multi-shard spec must produce more than one
    .shard file (the symptom of the original bug was exactly one).
    Also verifies the shard files line up with what shards_for_keys
    predicts for the new spec.
    """
    from ngsidekick.annotations.precomputed._write_buffers import PartitionedBuffer, _write_buffers_sharded

    n = 192
    ids = np.arange(n, dtype=np.uint64)
    rng = np.random.default_rng(0)
    # Each value is large enough that the 50 MB/shard target picks
    # shard_bits > 0. A 1 MB buffer per item gives ~192 MB total.
    recsize = 1_000_000
    buf = bytes(rng.integers(0, 256, size=n * recsize, dtype=np.uint8))

    metadata = _write_buffers_sharded(
        ids, [PartitionedBuffer(buf, recsize)],
        str(tmp_path), "sub",
        max_shards_per_transaction=8,
        max_threads=2,
    )
    sharding = metadata['sharding']
    assert sharding['shard_bits'] >= 1, "test premise: spec must request multiple shards"
    # The fix limits preshift_bits to the keyspace headroom so that not
    # every key collapses to chunk_id 0. (Concretely, with 8-bit keys and
    # shard_bits=2 we expect preshift to be at most 8-2 = 6.)
    keyspace_bits = (n - 1).bit_length()
    assert sharding['preshift_bits'] <= keyspace_bits - sharding['shard_bits'] - sharding['minishard_bits']

    spec = _shard_spec(
        hash=sharding['hash'],
        preshift_bits=sharding['preshift_bits'],
        shard_bits=sharding['shard_bits'],
        minishard_bits=sharding['minishard_bits'],
        data_encoding=sharding['data_encoding'],
        minishard_index_encoding=sharding['minishard_index_encoding'],
    )
    sub_dir = tmp_path / "sub"
    actual = _shard_files(sub_dir)
    predicted = {int(s) for s in shards_for_keys(ids, spec)}
    assert actual == predicted
    assert len(actual) > 1, (
        "fix in effect: a dense small keyspace with shard_bits>0 should "
        "produce multiple shard files"
    )


@pytest.mark.parametrize("max_shards_per_transaction", [1, 4, 32, 1024])
def test_max_shards_per_transaction_roundtrip(tmp_path, max_shards_per_transaction):
    """
    Regardless of how aggressively we batch shards into transactions, every
    written value must be readable, and the produced shard files must match
    the predicted set. Covers both the per-shard end of the spectrum
    (max=1) and the single-transaction end (max>=num_shards).
    """
    from ngsidekick.annotations.precomputed._write_buffers import PartitionedBuffer, _write_buffers_sharded

    rng = np.random.default_rng(11)
    n = 5_000
    ids = rng.choice(np.arange(1, 10_000_000), size=n, replace=False).astype(np.uint64)
    bufs = [bytes(rng.integers(0, 256, size=rng.integers(20, 100), dtype=np.uint8)) for _ in range(n)]
    truth = dict(zip(ids.tolist(), bufs))

    # Pack the variable-length bufs into a flat buffer + offsets so we can
    # use the new PartitionedBuffer-based ``_write_buffers_sharded`` signature.
    sizes = np.array([len(b) for b in bufs], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    flat_buf = b''.join(bufs)

    # Force a non-trivial multi-shard layout so the batching parameter is
    # actually exercised at all values of max_shards_per_transaction.
    orig_choose = wb._choose_output_spec
    def force_multi_shard(*args, **kwargs):
        spec = orig_choose(*args, **kwargs)
        spec.shard_bits = 4   # 16 shards
        spec.minishard_bits = 2
        return spec
    wb._choose_output_spec = force_multi_shard
    try:
        metadata = _write_buffers_sharded(
            ids, [PartitionedBuffer(flat_buf, offsets)],
            str(tmp_path), "sub",
            max_shards_per_transaction,
            max_threads=2,
        )
    finally:
        wb._choose_output_spec = orig_choose

    sharding = metadata['sharding']
    spec = _shard_spec(
        hash=sharding['hash'],
        preshift_bits=sharding['preshift_bits'],
        shard_bits=sharding['shard_bits'],
        minishard_bits=sharding['minishard_bits'],
        data_encoding=sharding['data_encoding'],
        minishard_index_encoding=sharding['minishard_index_encoding'],
    )

    sub_dir = tmp_path / "sub"
    predicted = {int(s) for s in shards_for_keys(ids, spec)}
    assert _shard_files(sub_dir) == predicted

    # Round-trip every value via tensorstore.
    kv = ts.KvStore.open({
        "driver": "neuroglancer_uint64_sharded",
        "metadata": sharding,
        "base": f"file://{sub_dir}",
    }).result()
    for k, v_truth in truth.items():
        rr = kv.read(_be_key(int(k))).result()
        assert rr.state == 'value' and bytes(rr.value) == v_truth, f"id {k} bad"


def test_write_precomputed_annotations_shard_placement(tmp_path, monkeypatch):
    """
    End-to-end: write point annotations via the public
    write_precomputed_annotations() API with a controlled set of
    annotation IDs, then verify (a) that the by_id shard files line up
    with what shards_for_keys() predicts for those IDs, and (b) that
    each annotation reads back from the shard file we predicted.

    To force a multi-shard layout (and therefore actually exercise the
    per-shard transaction loop) without ballooning the test to hundreds
    of MB, we shrink _choose_output_spec's shard target via monkeypatch.
    """
    # Force a multi-shard sharding spec for this test.
    orig_choose = wb._choose_output_spec
    def force_multi_shard(*args, **kwargs):
        spec = orig_choose(*args, **kwargs)
        spec.shard_bits = 4   # 16 shards
        spec.minishard_bits = 2
        return spec
    monkeypatch.setattr(wb, '_choose_output_spec', force_multi_shard)

    n = 5_000
    rng = np.random.default_rng(123)
    # Use explicit annotation IDs so we can predict exactly which shards
    # they should land in. The IDs are the chunk_ids of the by_id index.
    ids = rng.choice(np.arange(1, 10_000_000), size=n, replace=False).astype(np.uint64)
    df = pd.DataFrame({
        'x': rng.normal(0, 100, n),
        'y': rng.normal(0, 100, n),
        'z': rng.normal(0, 100, n),
    }, index=pd.Index(ids, name='id'))

    output_dir = tmp_path / "annotations"
    write_precomputed_annotations(
        df,
        coord_space=CoordinateSpace(
            names=[*'xyz'], units=['nm']*3, scales=[1, 1, 1]
        ),
        annotation_type='point',
        output_dir=str(output_dir),
        write_sharded=True,
        write_by_relationship=False,
        write_by_spatial_chunk=False,
    )

    # Recover the resolved sharding spec from the info file.
    info = json.loads((output_dir / "info").read_text())
    sharding = info['by_id']['sharding']
    spec = _shard_spec(
        hash=sharding['hash'],
        preshift_bits=sharding['preshift_bits'],
        shard_bits=sharding['shard_bits'],
        minishard_bits=sharding['minishard_bits'],
        data_encoding=sharding['data_encoding'],
        minishard_index_encoding=sharding['minishard_index_encoding'],
    )
    assert spec.shard_bits == 4, (
        "Expected the monkeypatch to give us 16 shards, "
        "otherwise this test isn't actually verifying multi-shard placement."
    )

    # (a) Set of shard files matches the set of predicted shards.
    by_id_dir = output_dir / "by_id"
    predicted_shards = {int(s) for s in shards_for_keys(ids, spec)}
    assert _shard_files(by_id_dir) == predicted_shards

    # (b) Spot-check: a sample of annotations reads back, and the
    #     specific shard file we predicted for each one exists.
    kv = ts.KvStore.open({
        "driver": "neuroglancer_uint64_sharded",
        "metadata": sharding,
        "base": f"file://{by_id_dir}",
    }).result()
    sample_idx = rng.choice(n, size=50, replace=False)
    sample_ids = ids[sample_idx]
    sample_predicted = shards_for_keys(sample_ids, spec)
    width = max(1, (spec.shard_bits + 3) // 4)
    for k, pred in zip(sample_ids, sample_predicted):
        rr = kv.read(_be_key(int(k))).result()
        assert rr.state == 'value', f"id {int(k)} not readable from kvstore"
        assert (by_id_dir / f"{int(pred):0{width}x}.shard").exists()
