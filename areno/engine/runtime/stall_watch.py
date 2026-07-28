"""Stage stall watcher for RL post-training loops.

Tracks the wall-clock time since the last progress event for each of the five
logical stages (`loading`, `data`, `rollout`, `reward`, `training`) and emits a
rate-limited warning when a configurable idle interval is exceeded. Warnings
never stop the run; they only surface diagnostics through the existing `areno`
logger, the `train_stats` dict, and the dashboard state file.

Design notes:
- The watcher is a pure-Python object with an injectable clock and sink, so all
  behaviour is deterministic and testable on CPU without a GPU or real timers.
- ``interval_s == 0`` disables the watcher entirely (``tick``/``check`` become
  no-ops), which keeps the default path behaviour-identical to before.
- Internal watcher errors never bubble into the training loop: a failure inside
  ``tick``/``check`` is swallowed and reported via the sink, reporting the stage
  name and a short input summary without leaking full training samples.
- No external database, no background thread, no parallel subsystem: progress
  events are fed explicitly by the trainer loops at the same points that already
  call ``record_dashboard_state``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

# The five logical stages tracked by the watcher. These are the user-facing
# stage names; concrete trainer stage strings (``rollout_start``, ``train_end``
# ...) are mapped into these buckets by ``feed_stage_event``.
STALL_STAGES: tuple[str, ...] = ("loading", "data", "rollout", "reward", "training")

# Map of concrete dashboard stage strings -> logical stall stage. Only stages
# that represent meaningful forward progress are mapped; transitional stages
# such as ``max_steps_reached`` / ``epoch_end`` map to ``training`` because they
# still indicate the training loop advanced.
_STAGE_MAP: dict[str, str] = {
    # loading: only the explicit ``tick("loading")`` in ``Trainer.init()``
    # feeds this stage; no concrete dashboard stage maps to it because
    # ``epoch_start`` represents training-loop advance, not initial loading.
    # data
    "batch_loaded": "data",
    # rollout
    "rollout_start": "rollout",
    "rollout_end": "rollout",
    # reward
    "reward_done": "reward",
    "score_start": "reward",
    "score_end": "reward",
    # training
    "train_start": "training",
    "train_end": "training",
    "train_skip": "training",
    "logprob_score_start": "training",
    "logprob_score_end": "training",
    "old_logprob_score_start": "training",
    "old_logprob_score_end": "training",
    "value_score_start": "training",
    "value_score_end": "training",
    "advantage_start": "training",
    "advantage_end": "training",
    "save_checkpoint_start": "training",
    "save_checkpoint_end": "training",
    "max_steps_reached": "training",
    "epoch_end": "training",
}


class StallSink(Protocol):
    """Callable sink for stall warnings.

    A sink receives a ``StallWarning`` and is expected to surface it (e.g. via
    a logger, a metrics dict, or a dashboard state file). Sinks must not raise;
    watcher internals catch and discard any sink exception.
    """

    def __call__(self, warning: StallWarning) -> None: ...


@dataclass(slots=True)
class StallWatchConfig:
    """Configuration for the stage stall watcher.

    ``interval_s``:
        Idle seconds after which a stage is considered stalled. ``0.0``
        disables the watcher (safe default -> zero behaviour change).
    ``min_interval_s``:
        Minimum seconds between two warnings for the *same* stage. Prevents
        log spam when a long stall persists across many ``check`` calls.
    ``stages``:
        Subset of logical stages to track. Defaults to all five. Unknown
        stage names are rejected by ``validate``.
    ``now``:
        Clock callable used by the watcher; defaults to ``time.monotonic``.
        Tests inject a controllable fake clock.
    """

    interval_s: float = 0.0
    min_interval_s: float = 30.0
    stages: tuple[str, ...] = STALL_STAGES
    now: Callable[[], float] = field(default=time.monotonic, repr=False)

    def validate(self) -> None:
        """Raise ``ValueError`` with a descriptive message on invalid input.

        Called by the trainer before model/worker initialisation so
        misconfigured runs fail fast with an actionable error.
        """

        if self.interval_s < 0:
            raise ValueError(
                f"stall_watch.interval_s must be non-negative, got {self.interval_s}; use 0 to disable"
            )
        if self.min_interval_s < 0:
            raise ValueError(
                f"stall_watch.min_interval_s must be non-negative, got {self.min_interval_s}"
            )
        if 0 < self.interval_s < self.min_interval_s:
            raise ValueError(
                f"stall_watch.min_interval_s ({self.min_interval_s}) must not exceed "
                f"interval_s ({self.interval_s}) when the watcher is enabled"
            )
        unknown = [stage for stage in self.stages if stage not in STALL_STAGES]
        if unknown:
            raise ValueError(
                f"stall_watch.stages contains unknown stage(s) {unknown}; "
                f"valid stages are {list(STALL_STAGES)}"
            )

    @property
    def enabled(self) -> bool:
        """True when the watcher is active (``interval_s > 0``)."""

        return self.interval_s > 0


@dataclass(slots=True)
class StallWarning:
    """One emitted stall diagnostic.

    ``stage``: logical stage name.
    ``wait_s``: seconds elapsed since the last progress event for this stage.
    ``threshold_s``: configured idle threshold that was exceeded.
    """

    stage: str
    wait_s: float
    threshold_s: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict suitable for ``train_stats`` / dashboard."""

        return {
            "stall_stage": self.stage,
            "stall_wait_s": round(self.wait_s, 3),
            "stall_threshold_s": self.threshold_s,
        }


class StallWatcher:
    """Stateful tracker of per-stage idle time with rate-limited warnings.

    The watcher is intentionally single-threaded and synchronous: trainers call
    ``tick`` at known progress boundaries and ``check`` at convenient points
    (typically once per train step). No background thread is spawned.
    """

    def __init__(
        self,
        config: StallWatchConfig,
        *,
        sink: StallSink | None = None,
    ) -> None:
        self._cfg = config
        self._sink: StallSink = sink or _LoggerStallSink()
        # Per-stage last progress timestamp. Only stages in ``config.stages``
        # are tracked; others are ignored.
        self._last_tick: dict[str, float] = {}
        # Per-stage last warning timestamp (for rate limiting).
        self._last_warn: dict[str, float] = {}
        # NOTE: ``_last_tick`` starts empty on purpose. A stage only becomes
        # "tracked" after its first ``tick``; stages that never appear in the
        # run are never reported as stalled. This prevents a freshly-enabled
        # watcher from flooding the first ``check`` with false warnings for
        # stages that simply have not started yet (e.g. ``reward`` in SFT).

    @property
    def config(self) -> StallWatchConfig:
        return self._cfg

    def tick(self, stage: str) -> None:
        """Record a progress event for ``stage``, resetting its idle timer.

        Only the named stage's timer is reset; other stages keep their timers.
        Unknown stages (not in ``config.stages``) are silently ignored, so
        trainers can call ``tick`` unconditionally without per-stage guards.
        Errors inside ``tick`` never propagate to the caller.

        Special case: ticking any stage other than ``loading`` retires the
        ``loading`` timer. ``loading`` is a one-shot stage ticked once at the
        end of ``Trainer.init()``; once the run advances to rollout/training,
        a stale ``loading`` timer would otherwise fire forever. Removing it on
        the first non-loading tick keeps ``loading`` warnings meaningful
        (stuck between init and first rollout) without false positives later.
        """

        if not self._cfg.enabled or stage not in self._cfg.stages:
            return
        try:
            now = self._cfg.now()
            self._last_tick[stage] = now
            # Retire the one-shot ``loading`` stage once any other stage
            # advances, so it stops accumulating idle time after the run has
            # clearly progressed past initialisation.
            if stage != "loading" and "loading" in self._last_tick:
                self._last_tick.pop("loading", None)
                self._last_warn.pop("loading", None)
        except Exception:  # noqa: BLE001 - never let watcher crash the run
            self._safe_sink(
                StallWarning(stage=stage, wait_s=0.0, threshold_s=self._cfg.interval_s)
            )

    def feed_stage_event(self, stage_name: str) -> None:
        """Map a concrete dashboard stage string to a logical stage and tick.

        Unknown concrete stage names are ignored, so this is safe to call for
        every ``record_dashboard_state`` invocation without filtering.
        """

        logical = _STAGE_MAP.get(stage_name)
        if logical is not None:
            self.tick(logical)

    def check(self) -> list[StallWarning]:
        """Return and emit warnings for every stage currently over threshold.

        Rate limiting: a stage that was already warned within
        ``min_interval_s`` is not re-warned in the same call. Each emitted
        warning is passed to the sink and also returned so the caller can
        attach it to ``train_stats`` / dashboard state.
        """

        if not self._cfg.enabled:
            return []
        now = self._cfg.now()
        warnings: list[StallWarning] = []
        for stage in self._cfg.stages:
            try:
                last = self._last_tick.get(stage)
                if last is None:
                    continue
                wait = now - last
                if wait < self._cfg.interval_s:
                    continue
                last_warn = self._last_warn.get(stage)
                if last_warn is not None and (now - last_warn) < self._cfg.min_interval_s:
                    continue
                warning = StallWarning(stage=stage, wait_s=wait, threshold_s=self._cfg.interval_s)
                self._last_warn[stage] = now
                self._safe_sink(warning)
                warnings.append(warning)
            except Exception:  # noqa: BLE001 - isolate watcher failures
                continue
        return warnings

    def _safe_sink(self, warning: StallWarning) -> None:
        """Invoke the sink, swallowing any exception to protect the run."""

        try:
            self._sink(warning)
        except Exception:  # noqa: BLE001 - sink must not crash the trainer
            pass


class _LoggerStallSink:
    """Default sink: logs a single WARNING line per stall event.

    Uses the ``areno`` logger so the line respects ``ARENO_LOG_LEVEL`` and the
    handler installed by ``areno.engine.log.configure_default_logging``.
    """

    def __call__(self, warning: StallWarning) -> None:
        logger = logging.getLogger("areno.stall_watch")
        logger.warning(
            "stall stage=%s wait_s=%.1f threshold_s=%.1f",
            warning.stage,
            warning.wait_s,
            warning.threshold_s,
        )


def make_stall_watcher(config: StallWatchConfig | None) -> StallWatcher | None:
    """Build a watcher from an optional config, returning ``None`` when disabled.

    ``None`` config or ``interval_s == 0`` yields ``None`` so trainers can use a
    simple truthiness check instead of touching config internals.
    """

    if config is None or not config.enabled:
        return None
    return StallWatcher(config)


__all__ = [
    "STALL_STAGES",
    "StallSink",
    "StallWatchConfig",
    "StallWatcher",
    "StallWarning",
    "make_stall_watcher",
]

