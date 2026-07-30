"""Per-sample reward profiling: timing, slow-sample flagging, per-batch timeout.

`RewardProfiler` wraps a user-supplied ``reward_fn`` and measures each
individual ``RewardRecord`` call, flags samples that exceed a configurable
absolute threshold, and enforces an optional per-batch wall-clock timeout.

When disabled (the default), ``score_batch`` degrades to a plain list
comprehension with zero ``perf_counter`` or dict-allocation overhead.

Thread-safety: when profiling is enabled and a ``batch_timeout_s`` is set,
each sample is executed inside a ``ThreadPoolExecutor`` thread so the
caller can enforce the timeout via ``future.result(timeout=...)``.  User
``reward_fn`` implementations must therefore be thread-safe.  Pure
computation reward functions (text parsing, scoring) are naturally safe;
functions with side effects (file I/O, global-state mutation) must be
made safe by the user.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass

from areno.api.rewards import RewardRecord

REWARD_HOOK_NAME = "reward_fn"


class RewardTimeoutError(RuntimeError):
    """Raised when per-batch wall-clock budget is exceeded.

    Carries the hook name, the sample identifier that was executing when
    the budget ran out, and the elapsed seconds so far.  The original
    error (if the reward_fn itself raised) is preserved in ``__cause__``.
    """

    def __init__(
        self,
        hook: str,
        prompt_index: int | None,
        sample_index: int | None,
        elapsed: float,
    ) -> None:
        self.hook = hook
        self.prompt_index = prompt_index
        self.sample_index = sample_index
        self.elapsed = elapsed
        super().__init__(
            f"reward hook '{hook}' timed out after {elapsed:.4f}s "
            f"at prompt_index={prompt_index} sample_index={sample_index}"
        )

    def __repr__(self) -> str:
        return (
            f"RewardTimeoutError(hook={self.hook!r}, "
            f"prompt_index={self.prompt_index}, sample_index={self.sample_index}, "
            f"elapsed={self.elapsed:.4f})"
        )


@dataclass(slots=True)
class RewardSampleTiming:
    """Timing record for a single reward_fn call."""

    prompt_index: int | None
    sample_index: int | None
    duration_s: float
    timed_out: bool = False


@dataclass(slots=True)
class RewardBatchProfile:
    """Aggregated timing profile for one batch of reward_fn calls."""

    total_s: float
    sample_timings: list[RewardSampleTiming]
    slow_samples: list[RewardSampleTiming]
    timeout_count: int

    def to_scalars(self) -> dict[str, float]:
        """Return a flat dict suitable for TensorBoard ``timing/reward_*`` scalars."""

        durations = [t.duration_s for t in self.sample_timings]
        return {
            "reward_slow_count": float(len(self.slow_samples)),
            "reward_max_s": float(max(durations)) if durations else 0.0,
            "reward_timeout_count": float(self.timeout_count),
            "reward_total_s": float(self.total_s),
        }

    def to_jsonl_records(self) -> list[dict]:
        """Return a list of dicts for jsonl artifact output.

        Each record contains only ``prompt_index``, ``sample_index``,
        ``duration_s``, and ``timed_out`` — never the prompt or completion
        text.
        """

        return [
            {
                "prompt_index": t.prompt_index,
                "sample_index": t.sample_index,
                "duration_s": t.duration_s,
                "timed_out": t.timed_out,
            }
            for t in self.sample_timings
        ]


class RewardProfiler:
    """Wraps a reward_fn with per-sample timing, slow-sample flagging, and per-batch timeout.

    Parameters
    ----------
    reward_fn:
        The user reward callable ``(RewardRecord) -> float``.
    enabled:
        When ``False`` (default), ``score_batch`` returns ``(rewards, None)``
        and avoids all timing overhead.
    slow_threshold_s:
        Absolute threshold in seconds.  Samples whose ``duration_s`` is
        ``>=`` this value are added to ``slow_samples``.  ``None`` disables
        slow-sample flagging.
    batch_timeout_s:
        Per-batch wall-clock budget in seconds.  If the cumulative time of
        all samples exceeds this value, a ``RewardTimeoutError`` is raised
        immediately with the current sample's identifier.  ``None`` disables
        the timeout.
    logger:
        Optional ``logging.Logger`` for slow-sample and timeout warnings.
    """

    def __init__(
        self,
        reward_fn: Callable[[RewardRecord], float],
        *,
        enabled: bool = False,
        slow_threshold_s: float | None = None,
        batch_timeout_s: float | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if slow_threshold_s is not None and slow_threshold_s <= 0:
            raise ValueError("reward_slow_threshold_s must be > 0 or None")
        if batch_timeout_s is not None and batch_timeout_s <= 0:
            raise ValueError("reward_batch_timeout_s must be > 0 or None")

        self._reward_fn = reward_fn
        self._enabled = enabled
        self._slow_threshold_s = slow_threshold_s
        self._batch_timeout_s = batch_timeout_s
        self._logger = logger

    @property
    def enabled(self) -> bool:
        return self._enabled

    def score_batch(self, records: list[RewardRecord]) -> tuple[list[float], RewardBatchProfile | None]:
        """Score every record through the wrapped reward_fn.

        Returns ``(rewards, profile)``.  When disabled, ``profile`` is
        ``None`` and no timing is performed.
        """

        if not self._enabled:
            rewards = [float(self._reward_fn(r)) for r in records]
            return rewards, None

        timings: list[RewardSampleTiming] = []
        rewards: list[float] = []
        batch_start = time.perf_counter()
        timeout_count = 0
        use_timeout = self._batch_timeout_s is not None

        executor = ThreadPoolExecutor(max_workers=1) if use_timeout else None
        try:
            for record in records:
                prompt_index = record.metadata.get("prompt_index")
                sample_index = record.metadata.get("sample_index")

                if use_timeout:
                    assert executor is not None
                    remaining = self._batch_timeout_s - (time.perf_counter() - batch_start)  # type: ignore[operator]
                    if remaining <= 0:
                        elapsed = time.perf_counter() - batch_start
                        raise RewardTimeoutError(
                            REWARD_HOOK_NAME, prompt_index, sample_index, elapsed
                        )

                    call_start = time.perf_counter()
                    future = executor.submit(self._reward_fn, record)
                    try:
                        result_val = future.result(timeout=remaining)
                    except FutureTimeout:
                        call_duration = time.perf_counter() - call_start
                        timeout_count += 1
                        timings.append(
                            RewardSampleTiming(
                                prompt_index=prompt_index,
                                sample_index=sample_index,
                                duration_s=call_duration,
                                timed_out=True,
                            )
                        )
                        elapsed = time.perf_counter() - batch_start
                        raise RewardTimeoutError(
                            REWARD_HOOK_NAME, prompt_index, sample_index, elapsed
                        ) from None
                    except Exception:
                        call_duration = time.perf_counter() - call_start
                        timings.append(
                            RewardSampleTiming(
                                prompt_index=prompt_index,
                                sample_index=sample_index,
                                duration_s=call_duration,
                            )
                        )
                        raise

                    call_duration = time.perf_counter() - call_start
                    rewards.append(float(result_val))
                    timings.append(
                        RewardSampleTiming(
                            prompt_index=prompt_index,
                            sample_index=sample_index,
                            duration_s=call_duration,
                        )
                    )
                else:
                    call_start = time.perf_counter()
                    result_val = float(self._reward_fn(record))
                    call_duration = time.perf_counter() - call_start
                    rewards.append(result_val)
                    timings.append(
                        RewardSampleTiming(
                            prompt_index=prompt_index,
                            sample_index=sample_index,
                            duration_s=call_duration,
                        )
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=False)

        total_s = time.perf_counter() - batch_start
        slow_samples = self._flag_slow(timings)
        profile = RewardBatchProfile(
            total_s=total_s,
            sample_timings=timings,
            slow_samples=slow_samples,
            timeout_count=timeout_count,
        )

        if self._logger is not None and slow_samples:
            slowest = max(slow_samples, key=lambda t: t.duration_s)
            self._logger.warning(
                "reward_timing hook=%s n=%d slow_count=%d slowest_sample_idx=%s slowest_time_s=%.4f",
                REWARD_HOOK_NAME,
                len(timings),
                len(slow_samples),
                slowest.sample_index,
                slowest.duration_s,
            )

        return rewards, profile

    def _flag_slow(self, timings: list[RewardSampleTiming]) -> list[RewardSampleTiming]:
        """Return the subset of timings that exceed ``slow_threshold_s``."""

        if self._slow_threshold_s is None:
            return []
        return [t for t in timings if t.duration_s >= self._slow_threshold_s]