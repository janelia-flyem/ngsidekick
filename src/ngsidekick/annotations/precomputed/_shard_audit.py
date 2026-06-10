"""
Optional shard-file audit for the precomputed-annotations writer.

When the environment variable ``NGSK_DEBUG_SHARD_FILES=1`` is set, the
by-id, by-rel, and by-spatial sharded writers verify after each
transaction that the on-disk shard-file set matches the cumulative set
of shard IDs they intended to write.

Why this matters: each writer batches transactions by the shard IDs it
*predicts* will be touched (via the same shard-hash function tensorstore
uses internally). If the prediction is wrong -- a bug in our shard
computation, a hash-spec mismatch, or anything else -- tensorstore
ends up staging extra shards inside the transaction, blowing past the
RAM budget the writer's per-batch sizing assumed. A mismatch warning
here is an early signal that the per-transaction RAM accounting is off.

Activation:

    NGSK_DEBUG_SHARD_FILES=1 python my_export.py 2>&1 | tee job.log
    grep '\\[SHARD_AUDIT\\]' job.log

A warning is logged on any mismatch; the writer otherwise proceeds.
No-op when the env var is unset.
"""
import logging
import os
import re

logger = logging.getLogger(__name__)


# Tensorstore shard files look like ``<lowercase-hex>.shard`` -- the
# zero-pad width depends on the kvstore's shard-bits configuration
# (e.g. ``000.shard`` through ``3ff.shard`` for shard_bits=10). We don't
# need to predict the pad width here; we compare integer values.
_SHARD_FILE_RE = re.compile(r'^([0-9a-fA-F]+)\.shard$')


def _enabled() -> bool:
    return bool(os.environ.get('NGSK_DEBUG_SHARD_FILES'))


def _list_shard_ids(directory):
    """
    Return the set of integer shard IDs from any ``<hex>.shard`` files
    in ``directory``. Returns an empty set if the directory doesn't
    exist or can't be read.
    """
    out = set()
    try:
        entries = os.listdir(directory)
    except OSError:
        return out
    for name in entries:
        match = _SHARD_FILE_RE.match(name)
        if match:
            out.add(int(match.group(1), 16))
    return out


class ShardWriteAuditor:
    """
    Per-writer auditor: tracks the cumulative expected shard-id set
    across transactions, and after each :meth:`record_batch` call
    compares it to the actual on-disk shard-file set.

    Logs a single ``[SHARD_AUDIT] ... mismatch ...`` warning on each
    mismatch; the writer otherwise proceeds. No-op when the
    ``NGSK_DEBUG_SHARD_FILES`` env var is unset, so creating an auditor
    is essentially free in normal runs.

    Args:
        output_directory:
            Absolute path to the kvstore subdirectory for this writer
            (e.g. ``/out/by_id``, ``/out/by_rel_body_pre``,
            ``/out/by_spatial_level_6``).
        label:
            Short tag used in warning lines so different writers can be
            told apart (e.g. ``'by_id'``).
    """

    def __init__(self, output_directory: str, label: str):
        self.enabled = _enabled()
        self.output_directory = output_directory
        self.label = label
        self.expected: set[int] = set()

    def record_batch(self, batch_shard_ids) -> None:
        """
        Call this once *after* committing the tensorstore transaction
        that wrote ``batch_shard_ids``. The auditor adds them to its
        cumulative-expected set and compares against the actual files
        on disk.
        """
        if not self.enabled:
            return
        # Defensive int conversion: callers typically pass a numpy
        # uint64 slice.
        new_shards = {int(s) for s in batch_shard_ids}
        self.expected |= new_shards
        actual = _list_shard_ids(self.output_directory)
        extra = actual - self.expected
        missing = self.expected - actual
        if extra or missing:
            logger.warning(
                f"[SHARD_AUDIT] {self.label} mismatch: "
                f"extra (unexpected on disk)={sorted(extra)}, "
                f"missing (expected but absent)={sorted(missing)}"
            )
