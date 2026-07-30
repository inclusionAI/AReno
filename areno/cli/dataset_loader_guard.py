"""Resource-guarded execution of custom dataset loaders.

This module wraps user-provided ``--dataset-loader-fn`` callbacks with:

* **Timeout** – raises :class:`DatasetLoaderTimeout` when the loader exceeds
  ``loader_timeout_s`` seconds (Unix only, uses ``SIGALRM`` / ``setitimer``).
* **Record cap** – truncates the returned dataset to ``max_loader_records``
  rows when the loader returns more than requested.  Falls back to
  ``itertools.islice`` for non-sliceable iterables.
* **Diagnostics** – measures wall-clock duration and peak process memory so
  the caller can surface them through logs or the run-end summary.

The guard is a *pure wrapper*: it calls the loader exactly once, preserves the
original traceback on user exceptions, and applies limits only when the
corresponding parameter is non-zero.  When both limits are zero (the default)
the loader still runs through the guard for diagnostics, but no timeout or
truncation is enforced.
"""

from __future__ import annotations

import itertools
import logging
import platform
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional imports (resource is Unix-only)
# ---------------------------------------------------------------------------

try:
    import resource as _resource
except ImportError:
    _resource = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DatasetLoaderTimeout(TimeoutError):
    """Raised when a dataset loader exceeds the configured wall-clock budget."""


# ---------------------------------------------------------------------------
# Diagnostic result
# ---------------------------------------------------------------------------

@dataclass
class LoaderDiagnostics:
    """Metrics collected during a guarded loader invocation."""

    duration_s: float = 0.0
    mem_before_kb: int = 0
    mem_after_kb: int = 0
    record_count: int = 0
    truncated: bool = False
    original_record_count: int = 0
    error: str | None = None

    @property
    def mem_delta_kb(self) -> int:
        """Memory delta in KB (after - before)."""

        return self.mem_after_kb - self.mem_before_kb

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for structured output."""

        return {
            "duration_s": round(self.duration_s, 3),
            "record_count": self.record_count,
            "mem_delta_kb": self.mem_delta_kb,
            "truncated": self.truncated,
            "original_record_count": self.original_record_count,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# SIGALRM handler
# ---------------------------------------------------------------------------

def _raise_timeout(signum: int, frame: Any) -> None:  # noqa: ARG001
    """SIGALRM handler that raises :class:`DatasetLoaderTimeout`."""

    raise DatasetLoaderTimeout("Dataset loader exceeded the configured timeout")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_loader_with_limits(
    loader_fn: Callable[..., Any],
    *args: Any,
    timeout_s: float = 0.0,
    max_records: int = 0,
    **kwargs: Any,
) -> tuple[Any, LoaderDiagnostics]:
    """Execute *loader_fn* with optional timeout and record-cap.

    Parameters
    ----------
    loader_fn:
        The user-provided loader callable (already resolved by
        ``_load_dataset_loader_fn``).
    *args, **kwargs:
        Forwarded to *loader_fn*.
    timeout_s:
        Wall-clock budget in seconds.  ``0`` disables the timeout.
        On Unix this uses ``setitimer`` (sub-second precision); on platforms
        without ``SIGALRM`` the timeout is silently skipped and a warning
        is logged.
    max_records:
        Maximum number of records to keep.  ``0`` disables the cap.
        When the loader returns a sequence longer than *max_records* the
        result is truncated to the first *max_records* elements.  For
        non-sliceable iterables, ``itertools.islice`` is used as fallback.

    Returns
    -------
    (dataset, diagnostics)
        The (possibly truncated) dataset and a :class:`LoaderDiagnostics`
        with timing, memory, and count information.

    Raises
    ------
    DatasetLoaderTimeout
        When the loader exceeds *timeout_s*.
    Exception
        Any exception raised by the loader itself is re-raised unchanged.
    """

    diag = LoaderDiagnostics()
    start = time.perf_counter()
    diag.mem_before_kb = _peak_rss_kb()

    # ------------------------------------------------------------------
    # Install timer-based timeout (Unix only, sub-second via setitimer).
    # ------------------------------------------------------------------
    use_timer = False
    old_handler = None
    old_itimer = None
    if timeout_s > 0:
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
            # Save existing itimer so we can restore it.
            if hasattr(signal, "setitimer"):
                try:
                    old_itimer = signal.setitimer(signal.ITIMER_REAL, timeout_s)
                except (OSError, ValueError):
                    # Fallback to alarm if setitimer fails.
                    signal.alarm(max(int(timeout_s), 1))
            else:
                signal.alarm(max(int(timeout_s), 1))
            use_timer = True
        else:
            logger.warning(
                "loader-timeout-s=%.1f requested but SIGALRM is unavailable "
                "on this platform; timeout will not be enforced.",
                timeout_s,
            )

    # ------------------------------------------------------------------
    # Run the loader.
    # ------------------------------------------------------------------
    try:
        dataset = loader_fn(*args, **kwargs)
    except DatasetLoaderTimeout:
        diag.duration_s = time.perf_counter() - start
        diag.mem_after_kb = _peak_rss_kb()
        diag.error = "timeout"
        raise
    except Exception as exc:
        diag.duration_s = time.perf_counter() - start
        diag.mem_after_kb = _peak_rss_kb()
        diag.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if use_timer:
            # Cancel timer and restore original handler + itimer.
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, 0)
            else:
                signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler if old_handler is not None else signal.SIG_DFL)
            if old_itimer and old_itimer[0] > 0 and hasattr(signal, "setitimer"):
                try:
                    signal.setitimer(signal.ITIMER_REAL, old_itimer[0], old_itimer[1])
                except (OSError, ValueError):
                    pass

    # ------------------------------------------------------------------
    # Record count cap.
    # ------------------------------------------------------------------
    if max_records > 0:
        count = _safe_len(dataset)
        diag.original_record_count = count
        if count > max_records:
            dataset = _truncate(dataset, max_records)
            diag.truncated = True
            logger.info(
                "dataset loader returned %d records, truncated to %d "
                "(max-loader-records=%d)",
                count, max_records, max_records,
            )
        elif count == 0:
            # Could be a generator or non-sized iterable — try islice.
            try:
                materialised = list(itertools.islice(dataset, max_records + 1))
            except TypeError:
                materialised = []
            if len(materialised) > max_records:
                dataset = materialised[:max_records]
                diag.truncated = True
                diag.original_record_count = len(materialised)
                logger.info(
                    "dataset loader returned >=%d records (iterable), "
                    "truncated to %d (max-loader-records=%d)",
                    len(materialised), max_records, max_records,
                )
            else:
                dataset = materialised
                diag.original_record_count = len(materialised)

    diag.duration_s = time.perf_counter() - start
    diag.mem_after_kb = _peak_rss_kb()
    diag.record_count = _safe_len(dataset)

    logger.info(
        "dataset loader finished: duration=%.3fs records=%d mem_delta=%dKB "
        "truncated=%s",
        diag.duration_s, diag.record_count, diag.mem_delta_kb, diag.truncated,
    )

    return dataset, diag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_rss_kb() -> int:
    """Return peak RSS of the current process in KB (0 if unavailable).

    On Linux ``ru_maxrss`` is in KB; on macOS it is in bytes, so we
    convert.  Returns 0 on platforms without ``resource``.
    """

    if _resource is None:
        return 0
    try:
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KB.
        if platform.system() == "Darwin":
            return rss // 1024
        return rss
    except (OSError, AttributeError):
        return 0


def _safe_len(obj: Any) -> int:
    """Return ``len(obj)`` or 0 if not measurable."""

    try:
        return len(obj)
    except TypeError:
        return 0


def _truncate(obj: Any, n: int) -> Any:
    """Truncate *obj* to first *n* elements.

    Tries slicing first; falls back to ``itertools.islice`` for
    non-sliceable iterables.
    """

    try:
        return obj[:n]
    except TypeError:
        return list(itertools.islice(obj, n))
