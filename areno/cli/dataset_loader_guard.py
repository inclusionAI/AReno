"""Resource-guarded execution of custom dataset loaders.

This module wraps user-provided ``--dataset-loader-fn`` callbacks with:

* **Timeout** – raises :class:`DatasetLoaderTimeout` when the loader exceeds
  ``loader_timeout_s`` seconds (Unix-only main thread, uses ``SIGALRM`` /
  ``setitimer``).
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
import json
import logging
import platform
import signal
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
# Timeout context manager
# ---------------------------------------------------------------------------


@contextmanager
def _timeout_context(timeout_s: float) -> Generator[None, None, None]:
    """Install and safely remove a SIGALRM-based timeout.

    This context manager only has an effect when:

    * ``timeout_s`` is positive;
    * the platform provides ``SIGALRM``; and
    * the calling thread is the main thread (``signal.signal`` only works
      there).

    The original signal handler and any previously scheduled ``ITIMER_REAL``
    timer are restored when the context exits, even if an exception is raised.
    """

    if timeout_s <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        logger.warning(
            "loader-timeout-s=%.1f requested but SIGALRM is unavailable "
            "on this platform; timeout will not be enforced.",
            timeout_s,
        )
        yield
        return

    if threading.current_thread() is not threading.main_thread():
        logger.warning(
            "loader-timeout-s=%.1f requested but the current thread is not "
            "the main thread; SIGALRM-based timeouts only work in the main "
            "thread. Timeout will not be enforced.",
            timeout_s,
        )
        yield
        return

    use_setitimer = hasattr(signal, "setitimer")
    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    old_itimer = None
    try:
        if use_setitimer:
            try:
                old_itimer = signal.setitimer(signal.ITIMER_REAL, timeout_s)
            except (OSError, ValueError):
                # Fallback to whole-second alarm if setitimer fails.
                signal.alarm(max(int(timeout_s), 1))
                use_setitimer = False
        else:
            signal.alarm(max(int(timeout_s), 1))
        yield
    finally:
        # Cancel any pending timer first, then restore the handler, then
        # restore the previous itimer.  Keeping cancellation and restoration
        # tightly coupled avoids leaving a stale alarm or wrong handler.
        try:
            if use_setitimer:
                signal.setitimer(signal.ITIMER_REAL, 0)
            else:
                signal.alarm(0)
        finally:
            signal.signal(signal.SIGALRM, old_handler)
            if old_itimer is not None and use_setitimer:
                try:
                    signal.setitimer(signal.ITIMER_REAL, old_itimer[0], old_itimer[1])
                except (OSError, ValueError):
                    pass


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
        without ``SIGALRM`` or when called from a non-main thread the timeout
        is skipped and a warning is logged.
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

    with _timeout_context(timeout_s):
        try:
            dataset = loader_fn(*args, **kwargs)
        except DatasetLoaderTimeout:
            diag.duration_s = time.perf_counter() - start
            diag.mem_after_kb = _peak_rss_kb()
            diag.error = "timeout"
            logger.warning(
                "dataset loader timed out after %.3fs (timeout_s=%.1f)",
                diag.duration_s,
                timeout_s,
            )
            raise
        except Exception as exc:
            diag.duration_s = time.perf_counter() - start
            diag.mem_after_kb = _peak_rss_kb()
            diag.error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "dataset loader failed after %.3fs: %s",
                diag.duration_s,
                diag.error,
            )
            raise

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
                "dataset loader returned %d records, truncated to %d (max-loader-records=%d)",
                count,
                max_records,
                max_records,
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
                    "dataset loader returned >=%d records (iterable), truncated to %d (max-loader-records=%d)",
                    len(materialised),
                    max_records,
                    max_records,
                )
            else:
                dataset = materialised
                diag.original_record_count = len(materialised)

    diag.duration_s = time.perf_counter() - start
    diag.mem_after_kb = _peak_rss_kb()
    diag.record_count = _safe_len(dataset)

    logger.info(
        "dataset loader finished: duration=%.3fs records=%d mem_delta=%dKB truncated=%s",
        diag.duration_s,
        diag.record_count,
        diag.mem_delta_kb,
        diag.truncated,
    )

    return dataset, diag


def write_loader_diagnostics(metrics_log_dir: str | None, diag: LoaderDiagnostics) -> None:
    """Persist loader diagnostics to the metrics log directory and log them.

    The diagnostics are always emitted as a structured INFO log.  When
    ``metrics_log_dir`` is set, a JSON file ``areno_loader_diagnostics.json``
    is written so operators and dashboards can retrieve them without parsing
    stdout.
    """

    payload = diag.to_dict()
    logger.info("dataset loader diagnostics: %s", json.dumps(payload, ensure_ascii=False))
    if not metrics_log_dir:
        return
    path = Path(metrics_log_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "areno_loader_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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
