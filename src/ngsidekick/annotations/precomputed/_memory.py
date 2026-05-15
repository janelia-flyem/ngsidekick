"""
Optional Linux memory instrumentation for the precomputed-annotations writer.

When the environment variable ``NGSK_DEBUG_MEMORY=1`` is set,
:func:`log_memory` emits a single info-level log line per call with the
current process's memory breakdown read from ``/proc/self/status`` and
``/proc/self/smaps_rollup``. Useful for narrowing down what kind of
memory is growing during a long-running write:

- ``RssAnon``  -- anonymous mappings (Python/DuckDB/tensorstore heap).
- ``RssFile``  -- file-backed mmap pages resident in the process
  (Feather mmap, DuckDB temp-spill if it uses mmap, etc.).
- ``RssShmem`` -- shared memory regions.
- ``Pss``      -- proportional set size (shared pages weighted by share
  count). Closer to "real" memory cost.
- ``VmRSS``    -- total resident set, the headline number.
- ``VmPeak``   -- highest VmSize observed since process start.

When the env var is unset (the default), :func:`log_memory` is a no-op
and adds no measurable overhead.

Linux-only. On other platforms calls are silently skipped.
"""
import logging
import os

logger = logging.getLogger(__name__)


# /proc/self/status keys to surface. Sizes are in kB in the file; we
# convert to bytes for uniformity with /proc/self/smaps_rollup.
_STATUS_KEYS = ('VmPeak', 'VmSize', 'VmRSS', 'RssAnon', 'RssFile', 'RssShmem')
_SMAPS_KEYS = ('Pss',)


def _enabled() -> bool:
    return bool(os.environ.get('NGSK_DEBUG_MEMORY'))


def _read_kv(path, wanted):
    """
    Parse a /proc/self/{status,smaps_rollup}-style key:value file and
    return a dict of bytes-valued entries for the requested keys.
    """
    out = {}
    try:
        with open(path) as f:
            for line in f:
                key, _, rest = line.partition(':')
                if key in wanted:
                    parts = rest.strip().split()
                    if parts and parts[-1] == 'kB':
                        out[key] = int(parts[0]) * 1024
                    elif parts:
                        out[key] = int(parts[0])
    except (OSError, ValueError):
        pass
    return out


def _gather():
    """Return a dict of memory metrics, or empty dict on non-Linux/unsupported."""
    out = {}
    out.update(_read_kv('/proc/self/status', _STATUS_KEYS))
    out.update(_read_kv('/proc/self/smaps_rollup', _SMAPS_KEYS))
    return out


def log_memory(label: str) -> None:
    """
    Emit a single info log line summarizing the current process's memory
    breakdown, prefixed with ``[MEMORY] {label}``. No-op unless the
    environment variable ``NGSK_DEBUG_MEMORY=1`` is set.

    Lines are grep-friendly: ``grep '\\[MEMORY\\]' job.log`` extracts the
    full memory trace.
    """
    if not _enabled():
        return
    metrics = _gather()
    if not metrics:
        return
    parts = ' '.join(f'{k}={metrics[k] / 1024**3:.2f}GB' for k in metrics)
    logger.info(f"[MEMORY] {label} {parts}")
