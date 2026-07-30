"""Resource-guarded execution of custom dataset loaders.

This module wraps user-provided ``--dataset-loader-fn`` callbacks with:

* **Timeout** – raises :class:`DatasetLoaderTimeout` when the loader exceeds
  ``loader_timeout_s`` seconds (Unix only, uses ``SIGALRM``).
* **Record cap** – truncates the returned dataset to ``max_loader_records``
  rows when the loader returns more than requested.
* **Diagnostics** – measures wall-clock duration and peak process memory so
  the caller can surface them through logs or the run-end summary.

The guard is a *pure wrapper*: it calls the loader exactly once, preserves the
original traceback on user exceptions, and applies limits only when the
corresponding parameter is non-zero.  When both limits are zero (the default)
the loader runs with no overhead beyond timing.
"""

from __future__ import annotations

import logging
import resource
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


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
        On Unix this uses ``SIGALRM``; on platforms without ``SIGALRM`` the
        timeout is silently skipped and a warning is logged.
    max_records:
        Maximum number of records to keep.  ``0`` disables the cap.
        When the loader returns a sequence longer than *max_records* the
        result is truncated to the first *max_records* elements.

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
    # Install SIGALRM-based timeout (Unix only).
    # ------------------------------------------------------------------
    use_alarm = False
    if timeout_s > 0:
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
            # Round up to whole seconds; SIGALRM has 1-second granularity.
            signal.alarm(max(int(timeout_s), 1))
            use_alarm = True
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
        if use_alarm:
            signal.alarm(0)  # Cancel any pending alarm.
            signal.signal(signal.SIGALRM, old_handler)

    # ------------------------------------------------------------------
    # Record count cap.
    # ------------------------------------------------------------------
    if max_records > 0:
        count = _safe_len(dataset)
        diag.original_record_count = count
        if count > max_records:
            dataset = dataset[:max_records]
            diag.truncated = True
            logger.info(
                "dataset loader returned %d records, truncated to %d "
                "(max-loader-records=%d)",
                count, max_records, max_records,
            )

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
    """Return peak RSS of the current process in KB (0 if unavailable)."""

    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, AttributeError):
        return 0


def _safe_len(obj: Any) -> int:
    """Return ``len(obj)`` or 0 if not measurable."""

    try:
        return len(obj)
    except TypeError:
        return 0