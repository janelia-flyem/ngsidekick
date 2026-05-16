"""
Optional cross-platform memory instrumentation for the
precomputed-annotations writer.

When the environment variable ``NGSK_DEBUG_MEMORY=1`` is set,
:func:`log_memory` emits a single info-level log line per call with the
current process's memory metrics:

- ``MaxRSS``   -- peak RSS since process start. Cross-platform (Linux
  and macOS), via POSIX ``resource.getrusage``. Monotonic, so a spike
  between two log markers is captured even if its instant value isn't.
  The delta of ``MaxRSS`` across consecutive markers is "how much new
  peak this phase claimed".

On Linux (where ``/proc/self/{status,smaps_rollup}`` exist) we also
include a per-region breakdown, useful for narrowing down what *kind*
of memory is growing during a long-running write:

- ``RssAnon``  -- anonymous mappings (Python/DuckDB/tensorstore heap).
- ``RssFile``  -- file-backed mmap pages resident in the process
  (Feather mmap, DuckDB temp-spill if it uses mmap, etc.).
- ``RssShmem`` -- shared memory regions.
- ``Pss``      -- proportional set size (shared pages weighted by share
  count). Closer to "real" memory cost.
- ``VmRSS``    -- total current resident set.
- ``VmPeak``   -- peak VmSize since process start (peak virtual address
  space, not peak RSS; ``MaxRSS`` is the RSS-specific high-water mark).

When the env var is unset (the default), :func:`log_memory` is a no-op
and adds no measurable overhead.
"""
import logging
import os
import resource
import sys

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


def _peak_rss_bytes():
    """
    Return peak RSS since process start, in bytes.

    Uses POSIX ``resource.getrusage(RUSAGE_SELF).ru_maxrss``. The units
    of that field differ by platform: Linux reports kilobytes, macOS
    reports bytes. We normalize to bytes here.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return rss
    return rss * 1024


def _gather():
    """
    Return a dict of memory metrics in bytes.

    ``MaxRSS`` (peak RSS since process start) is always included since
    it's portable. On Linux we additionally include the per-region
    breakdown from ``/proc``.
    """
    out = {'MaxRSS': _peak_rss_bytes()}
    out.update(_read_kv('/proc/self/status', _STATUS_KEYS))
    out.update(_read_kv('/proc/self/smaps_rollup', _SMAPS_KEYS))
    return out


def log_memory(label: str) -> None:
    """
    Emit a single info log line summarizing the current process's memory
    breakdown, prefixed with ``[MEMORY] {label}``. No-op unless the
    environment variable ``NGSK_DEBUG_MEMORY=1`` is set.

    Lines are grep-friendly: ``grep '\\[MEMORY\\]' job.log`` extracts the
    full memory trace. The ``MaxRSS`` field is monotonic, so for each
    phase you can read off the peak-so-far at its boundary; the delta
    between consecutive ``MaxRSS`` values is the high-water-mark
    contribution from the phase in between.
    """
    if not _enabled():
        return
    metrics = _gather()
    if not metrics:
        return
    parts = ' '.join(f'{k}={metrics[k] / 1024**3:.2f}GB' for k in metrics)
    logger.info(f"[MEMORY] {label} {parts}")
