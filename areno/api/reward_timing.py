"""Slow reward hook and sample identification.

Wraps a user reward function with per-hook and per-sample timing, flags
configurable outliers, and enforces an optional per-sample timeout. The
wrapper produces human-readable log lines and structured ``RewardTimingReport``
records without logging full prompts or completions.

When disabled (the default), the wrapper is a transparent passthrough with
near-zero overhead -- the original callable is called directly and no timing
data is collected.
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from areno.api.rewards import RewardRecord

logger = logging.getLogger(__name__)

# Sentinel returned when a reward sample times out. Using NaN rather than 0.0
# avoids skewing group-relative advantage computations that assume a valid
# scalar; downstream advantage code should filter or replace NaN before use.
TIMEOUT_REWARD = float("nan")


@dataclass(slots=True)
class RewardSampleTiming:
    """Timing result for a single reward function invocation.

    ``sample_id`` is a short, opaque identifier (e.g. ``"p2_s5"``) that
    distinguishes samples without exposing prompt or completion text.
    """

    hook_name: str
    sample_id: str
    elapsed_s: float
    timed_out: bool = False


@dataclass(slots=True)
class RewardTimingReport:
    """Aggregated timing report for one reward-scoring batch.

    The report is produced after every batch of reward evaluations when
    timing is enabled. It carries per-sample timings, summary statistics,
    flagged outliers, and any timeouts -- all without prompts or answers.
    """

    hook_name: str
    step: int
    sample_timings: list[RewardSampleTiming] = field(default_factory=list)
    total_elapsed_s: float = 0.0
    mean_elapsed_s: float = 0.0
    max_elapsed_s: float = 0.0
    p95_elapsed_s: float = 0.0
    outliers: list[RewardSampleTiming] = field(default_factory=list)
    timeouts: list[RewardSampleTiming] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for structured output."""

        return {
            "hook_name": self.hook_name,
            "step": self.step,
            "num_samples": len(self.sample_timings),
            "total_elapsed_s": round(self.total_elapsed_s, 6),
            "mean_elapsed_s": round(self.mean_elapsed_s, 6),
            "max_elapsed_s": round(self.max_elapsed_s, 6),
            "p95_elapsed_s": round(self.p95_elapsed_s, 6),
            "outliers": [
                {"sample_id": s.sample_id, "elapsed_s": round(s.elapsed_s, 6), "timed_out": s.timed_out}
                for s in self.outliers
            ],
            "timeouts": [
                {"sample_id": s.sample_id, "elapsed_s": round(s.elapsed_s, 6)} for s in self.timeouts
            ],
        }

    def format_human(self) -> str:
        """Return a single-line, human-readable summary for console logs."""

        parts = [
            f"reward_timing hook={self.hook_name} step={self.step}",
            f"n={len(self.sample_timings)}",
            f"total={self.total_elapsed_s:.4f}s",
            f"mean={self.mean_elapsed_s:.4f}s",
            f"max={self.max_elapsed_s:.4f}s",
            f"p95={self.p95_elapsed_s:.4f}s",
        ]
        if self.outliers:
            ids = ",".join(s.sample_id for s in self.outliers)
            parts.append(f"slow_samples=[{ids}]")
        if self.timeouts:
            ids = ",".join(s.sample_id for s in self.timeouts)
            parts.append(f"timeouts=[{ids}]")
        return " ".join(parts)


@dataclass(slots=True)
class RewardTimingConfig:
    """Configuration for reward timing instrumentation.

    All fields default to values that preserve current (untimed) behavior.
    Set ``enabled=True`` to activate timing collection and reporting.

    Attributes:
        enabled: Master switch. When ``False`` the wrapper is a transparent
            passthrough and no timing data is collected.
        slow_threshold_s: Samples taking longer than this are flagged as
            outliers in the report. Set to ``None`` or ``0`` to disable
            outlier flagging.
        timeout_s: Per-sample wall-clock timeout. Samples exceeding this
            receive ``TIMEOUT_REWARD`` and are recorded in the report's
            ``timeouts`` list. Set to ``None`` to disable timeout enforcement.
            Only effective on POSIX (uses ``SIGALRM``).
        hook_name: Name used in log lines and reports to identify the reward
            hook. Defaults to ``"reward_fn"``.
    """

    enabled: bool = False
    slow_threshold_s: float | None = None
    timeout_s: float | None = None
    hook_name: str = "reward_fn"

    def validate(self) -> None:
        """Raise ``ValueError`` if the configuration is internally inconsistent."""

        if self.slow_threshold_s is not None and self.slow_threshold_s <= 0:
            raise ValueError("slow_threshold_s must be positive or None")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive or None")
        if (
            self.timeout_s is not None
            and self.slow_threshold_s is not None
            and self.timeout_s < self.slow_threshold_s
        ):
            raise ValueError("timeout_s must be >= slow_threshold_s when both are set")


class TimedRewardFn:
    """Wrap a reward callable with optional timing, outlier flagging, and timeout.

    When ``config.enabled`` is ``False`` (the default), calling this wrapper
    is equivalent to calling the original function -- no timing data is
    collected and no overhead is added beyond one ``if`` check.

    When enabled, each invocation is timed. After a batch of calls, call
    :meth:`finalize_batch` to produce a :class:`RewardTimingReport`. The
    wrapper also logs the human-readable summary via ``logger.warning`` for
    outliers and ``logger.info`` for the batch summary.

    Timeout enforcement uses ``signal.SIGALRM`` and is therefore only
    effective on POSIX platforms. On Windows the timeout is silently ignored
    (samples run to completion).
    """

    def __init__(self, reward_fn: Callable[[RewardRecord], float], config: RewardTimingConfig | None = None):
        self._fn = reward_fn
        self._config = config or RewardTimingConfig()
        self._config.validate()
        self._pending: list[RewardSampleTiming] = []

    @property
    def config(self) -> RewardTimingConfig:
        return self._config

    def __call__(self, record: RewardRecord) -> float:
        """Score one record, optionally timing the call.

        When timing is disabled, this is a transparent passthrough.
        """

        if not self._config.enabled:
            return self._fn(record)

        meta = record.metadata if hasattr(record, "metadata") else {}
        prompt_index = meta.get("prompt_index", -1)
        sample_index = meta.get("sample_index", -1)
        sample_id = f"p{prompt_index}_s{sample_index}"

        elapsed, timed_out, result = _timed_call(self._fn, record, self._config.timeout_s)
        timing = RewardSampleTiming(
            hook_name=self._config.hook_name,
            sample_id=sample_id,
            elapsed_s=elapsed,
            timed_out=timed_out,
        )
        self._pending.append(timing)

        if timed_out:
            logger.warning(
                "reward_timeout hook=%s sample=%s elapsed=%.4fs timeout=%.1fs",
                self._config.hook_name,
                sample_id,
                elapsed,
                self._config.timeout_s or 0.0,
            )
            return TIMEOUT_REWARD

        threshold = self._config.slow_threshold_s
        if threshold is not None and threshold > 0 and elapsed > threshold:
            logger.warning(
                "reward_slow hook=%s sample=%s elapsed=%.4fs threshold=%.1fs",
                self._config.hook_name,
                sample_id,
                elapsed,
                threshold,
            )

        return result

    def finalize_batch(self, step: int) -> RewardTimingReport | None:
        """Aggregate pending per-sample timings into a report and clear the buffer.

        Returns ``None`` when timing is disabled or no samples were recorded.
        """

        if not self._config.enabled or not self._pending:
            self._pending.clear()
            return None

        timings = self._pending
        self._pending = []

        elapsed_values = [t.elapsed_s for t in timings]
        total = sum(elapsed_values)
        count = len(elapsed_values)
        mean = total / count if count else 0.0
        max_elapsed = max(elapsed_values) if elapsed_values else 0.0
        p95 = _percentile(elapsed_values, 95) if elapsed_values else 0.0

        threshold = self._config.slow_threshold_s
        outliers = [
            t for t in timings
            if not t.timed_out and threshold is not None and threshold > 0 and t.elapsed_s > threshold
        ]
        timeouts = [t for t in timings if t.timed_out]

        report = RewardTimingReport(
            hook_name=self._config.hook_name,
            step=step,
            sample_timings=timings,
            total_elapsed_s=total,
            mean_elapsed_s=mean,
            max_elapsed_s=max_elapsed,
            p95_elapsed_s=p95,
            outliers=outliers,
            timeouts=timeouts,
        )

        logger.info(report.format_human())
        return report


def _timed_call(
    fn: Callable[[RewardRecord], float],
    record: RewardRecord,
    timeout_s: float | None,
) -> tuple[float, bool, float]:
    """Execute *fn(record)* with wall-clock timing and optional timeout.

    Returns ``(elapsed_seconds, timed_out, result)``. On timeout the function
    is interrupted (POSIX only) and ``timed_out`` is ``True``; *result* is
    ``TIMEOUT_REWARD`` in that case.
    """

    if timeout_s is not None and _has_sigalrm():
        return _timed_call_with_alarm(fn, record, timeout_s)

    start = time.perf_counter()
    result = fn(record)
    elapsed = time.perf_counter() - start
    return elapsed, False, float(result)


def _timed_call_with_alarm(
    fn: Callable[[RewardRecord], float],
    record: RewardRecord,
    timeout_s: float,
) -> tuple[float, bool, float]:
    """Timeout-enforced call using ``signal.SIGALRM`` (POSIX only)."""

    start = time.perf_counter()

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError("reward_fn timed out")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    old_itimer = signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        result = fn(record)
        return time.perf_counter() - start, False, float(result)
    except TimeoutError:
        return time.perf_counter() - start, True, TIMEOUT_REWARD
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        # Restore the previous itimer if one was active.
        if old_itimer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, old_itimer[0], old_itimer[1])


def _has_sigalrm() -> bool:
    """Return ``True`` if ``signal.SIGALRM`` is available (POSIX)."""

    return hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


def _percentile(values: list[float], percentile: float) -> float:
    """Return the *percentile*-th percentile from a list of floats.

    Uses nearest-rank interpolation. ``percentile`` is 0-100.
    """

    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (percentile / 100.0) * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    frac = rank - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac
