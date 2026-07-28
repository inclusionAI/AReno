"""Bounded progress and timing for model-loading stages.

Issue #230: emit per-stage progress and elapsed time for the five model-loading
stages (reference resolution, config/tokenizer load, weight shard reading,
device placement, worker distribution). Output is human-readable and structured
so non-interactive consumers can parse it; only rank 0 logs to avoid spam in
multi-rank runs. The last completed stage is retained on failure so the caller
can report which stage and input caused the problem.

The tracker is intentionally dependency-free: it uses ``time.perf_counter`` and
the existing ``areno`` logger. It does not change any default behavior beyond
adding INFO-level log lines, so backward compatibility is preserved.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("areno.engine.load_progress")

# Fixed stage order lets callers and tests refer to stages by name without
# importing arbitrary enums; the tracker records the last completed one.
STAGE_REFERENCE_RESOLUTION = "reference_resolution"
STAGE_CONFIG_TOKENIZER = "config_tokenizer_load"
STAGE_WEIGHT_SHARD_READING = "weight_shard_reading"
STAGE_DEVICE_PLACEMENT = "device_placement"
STAGE_WORKER_DISTRIBUTION = "worker_distribution"


class ModelLoadTracker:
    """Record per-stage timing and emit bounded progress lines on rank 0.

    ``rank0`` gates logging so only one process prints in distributed runs;
    callers pass ``get_tp_context().rank == 0``. The tracker keeps
    ``last_completed_stage`` so a failing run can surface which stage and input
    caused the problem without re-running.
    """

    def __init__(self, *, rank0: bool = True):
        self._rank0 = bool(rank0)
        self.last_completed_stage: str | None = None
        self.start_time: float = time.perf_counter()

    @contextmanager
    def stage(self, name: str, *, detail: str | None = None) -> Iterator[None]:
        """Time one loading stage and log start/done lines on rank 0.

        On exit (success or exception) the elapsed time is logged and
        ``last_completed_stage`` is updated. Exceptions are re-raised unchanged
        after a structured ``status=failed`` line is emitted, so the original
        error is never hidden.
        """

        if self._rank0:
            prefix = f"model_load stage={name}"
            if detail:
                logger.info("%s status=start detail=%s", prefix, detail)
            else:
                logger.info("%s status=start", prefix)
        begin = time.perf_counter()
        try:
            yield
        except BaseException:
            elapsed = time.perf_counter() - begin
            if self._rank0:
                logger.info(
                    "model_load stage=%s status=failed elapsed=%.3fs",
                    name,
                    elapsed,
                )
            raise
        elapsed = time.perf_counter() - begin
        self.last_completed_stage = name
        if self._rank0:
            logger.info(
                "model_load stage=%s status=done elapsed=%.3fs",
                name,
                elapsed,
            )

    def summary(self) -> dict[str, object]:
        """Return a structured snapshot for non-interactive consumers.

        Includes the last completed stage and total elapsed time since the
        tracker was constructed, so the dashboard or CLI can surface a single
        record without parsing log lines.
        """

        return {
            "last_completed_stage": self.last_completed_stage,
            "total_elapsed_s": time.perf_counter() - self.start_time,
        }


@contextmanager
def tracked_stage(tracker: ModelLoadTracker | None, name: str, *, detail: str | None = None) -> Iterator[None]:
    """Time ``name`` on ``tracker`` when present, otherwise run untracked.

    Lets call sites keep a single line when tracking is optional (for example
    inside registry helpers that may be called outside a tracked load flow).
    """

    if tracker is None:
        yield
        return
    with tracker.stage(name, detail=detail):
        yield
