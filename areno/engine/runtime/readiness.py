"""Serve-readiness state machine for AReno.

Exposes detailed readiness states during the serve startup sequence:
- model_loading: model is being loaded
- worker_ready: workers are initialized
- router_ready: router is ready
- minimal_probe: basic health check passed
- ready: fully ready to accept requests
- failed: failed state with error details
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class ReadinessState(enum.Enum):
    """Serve readiness states."""

    MODEL_LOADING = "model_loading"
    WORKER_READY = "worker_ready"
    ROUTER_READY = "router_ready"
    MINIMAL_PROBE = "minimal_probe"
    READY = "ready"
    FAILED = "failed"


# Valid state transitions
VALID_TRANSITIONS: dict[ReadinessState, set[ReadinessState]] = {
    ReadinessState.MODEL_LOADING: {
        ReadinessState.WORKER_READY,
        ReadinessState.FAILED,
    },
    ReadinessState.WORKER_READY: {
        ReadinessState.ROUTER_READY,
        ReadinessState.FAILED,
    },
    ReadinessState.ROUTER_READY: {
        ReadinessState.MINIMAL_PROBE,
        ReadinessState.FAILED,
    },
    ReadinessState.MINIMAL_PROBE: {
        ReadinessState.READY,
        ReadinessState.FAILED,
    },
    ReadinessState.READY: {ReadinessState.FAILED},
    ReadinessState.FAILED: set(),  # Terminal state
}

# State ordering for progress reporting
STATE_ORDER = [
    ReadinessState.MODEL_LOADING,
    ReadinessState.WORKER_READY,
    ReadinessState.ROUTER_READY,
    ReadinessState.MINIMAL_PROBE,
    ReadinessState.READY,
]


@dataclass
class StageInfo:
    """Information about a readiness stage."""

    state: str  # "pending", "in_progress", "completed", "failed"
    duration_ms: int | None = None
    error: str | None = None


@dataclass
class ReadinessStatus:
    """Full readiness status snapshot."""

    status: str  # "not_ready", "ready", "failed"
    current_stage: str
    stages: dict[str, StageInfo]
    last_completed_stage: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "current_stage": self.current_stage,
            "stages": {
                name: {
                    "state": info.state,
                    "duration_ms": info.duration_ms,
                    "error": info.error,
                }
                for name, info in self.stages.items()
            },
            "last_completed_stage": self.last_completed_stage,
            "error": self.error,
        }


class ReadinessStateMachine:
    """Thread-safe readiness state machine for serve lifecycle.

    Tracks the progression through serve startup stages and provides
    observability via callbacks, metrics, and status queries.
    """

    def __init__(
        self,
        enabled: bool = False,
        timeout_per_stage_seconds: float = 30.0,
        on_state_change: Callable[[ReadinessState, ReadinessState | None, float], None] | None = None,
    ):
        """Initialize the state machine.

        Args:
            enabled: Whether readiness tracking is enabled
            timeout_per_stage_seconds: Timeout for each stage
            on_state_change: Callback(old_state, new_state, duration_ms)
        """
        self._enabled = enabled
        self._timeout_per_stage_seconds = timeout_per_stage_seconds
        self._on_state_change = on_state_change

        self._current_state = ReadinessState.MODEL_LOADING if enabled else None
        self._previous_state: ReadinessState | None = None
        self._stage_start_time: float | None = time.time() if enabled else None
        self._stage_durations: dict[ReadinessState, float] = {}
        self._failed_stage: ReadinessState | None = None
        self._error_message: str | None = None
        self._lock = False  # Simple lock for async safety

    @property
    def enabled(self) -> bool:
        """Whether readiness tracking is enabled."""
        return self._enabled

    @property
    def current_state(self) -> ReadinessState | None:
        """Current readiness state."""
        return self._current_state

    def _acquire_lock(self) -> None:
        """Acquire simple spinlock (for single-threaded async use)."""
        import time

        while self._lock:
            time.sleep(0.001)
        self._lock = True

    def _release_lock(self) -> None:
        """Release lock."""
        self._lock = False

    def transition_to(self, new_state: ReadinessState, error: str | None = None) -> bool:
        """Transition to a new state.

        Args:
            new_state: The state to transition to
            error: Error message if transitioning to FAILED

        Returns:
            True if transition was successful, False otherwise
        """
        if not self._enabled:
            return False

        self._acquire_lock()
        try:
            old_state = self._current_state

            # Validate transition
            if old_state is not None and new_state not in VALID_TRANSITIONS.get(old_state, set()):
                # Allow transition to FAILED from any state
                if new_state != ReadinessState.FAILED:
                    return False

            # Record duration for completed stage
            if self._stage_start_time is not None and old_state is not None:
                duration = time.time() - self._stage_start_time
                self._stage_durations[old_state] = duration

                # Notify callback
                if self._on_state_change is not None:
                    try:
                        self._on_state_change(old_state, new_state, duration * 1000)
                    except Exception:
                        pass  # Don't let callback errors break state machine

            # Update state
            self._previous_state = old_state
            self._current_state = new_state
            self._stage_start_time = time.time()

            if new_state == ReadinessState.FAILED:
                self._failed_stage = old_state
                self._error_message = error or "Unknown error"

            return True
        finally:
            self._release_lock()

    def mark_stage_complete(self, stage: ReadinessState) -> None:
        """Mark the current stage as complete and transition to next.

        Args:
            stage: The stage that was completed (must match current)
        """
        if not self._enabled or self._current_state != stage:
            return

        # Determine next state
        current_idx = STATE_ORDER.index(stage) if stage in STATE_ORDER else -1
        if current_idx >= 0 and current_idx + 1 < len(STATE_ORDER):
            next_state = STATE_ORDER[current_idx + 1]
            self.transition_to(next_state)

    def mark_failed(self, stage: ReadinessState | None = None, error: str | None = None) -> None:
        """Mark a stage as failed.

        Args:
            stage: The stage that failed (defaults to current)
            error: Error message
        """
        if not self._enabled:
            return

        if stage is not None and self._current_state != stage:
            # Transition to the failing stage first, then fail
            self.transition_to(stage)

        self.transition_to(ReadinessState.FAILED, error)

    def check_timeout(self) -> bool:
        """Check if current stage has timed out.

        Returns:
            True if timed out and transitioned to FAILED
        """
        if not self._enabled or self._current_state is None:
            return False

        if self._current_state == ReadinessState.FAILED:
            return False

        if self._stage_start_time is None:
            return False

        elapsed = time.time() - self._stage_start_time
        if elapsed > self._timeout_per_stage_seconds:
            self.mark_failed(
                error=f"stage timeout after {int(elapsed * 1000)}ms (limit: {int(self._timeout_per_stage_seconds * 1000)}ms)"
            )
            return True

        return False

    def get_status(self) -> ReadinessStatus:
        """Get current readiness status.

        Returns:
            ReadinessStatus snapshot
        """
        if not self._enabled:
            return ReadinessStatus(
                status="not_enabled",
                current_stage="none",
                stages={},
                last_completed_stage=None,
                error=None,
            )

        stages: dict[str, StageInfo] = {}
        last_completed = None

        for state in STATE_ORDER:
            state_name = state.value

            if self._current_state == ReadinessState.FAILED and self._failed_stage == state:
                stages[state_name] = StageInfo(
                    state="failed",
                    duration_ms=self._get_duration_ms(state),
                    error=self._error_message,
                )
            elif state == self._current_state:
                stages[state_name] = StageInfo(
                    state="in_progress",
                    duration_ms=self._get_duration_ms(state) if self._stage_start_time else None,
                )
            elif state in self._stage_durations:
                stages[state_name] = StageInfo(
                    state="completed",
                    duration_ms=self._get_duration_ms(state),
                )
                last_completed = state_name
            else:
                stages[state_name] = StageInfo(state="pending")

        # Determine overall status
        if self._current_state == ReadinessState.FAILED:
            status = "failed"
        elif self._current_state == ReadinessState.READY:
            status = "ready"
        else:
            status = "not_ready"

        return ReadinessStatus(
            status=status,
            current_stage=self._current_state.value if self._current_state else "none",
            stages=stages,
            last_completed_stage=last_completed,
            error=self._error_message,
        )

    def _get_duration_ms(self, state: ReadinessState) -> int | None:
        """Get duration in milliseconds for a completed stage."""
        if state == self._current_state and self._stage_start_time is not None:
            return int((time.time() - self._stage_start_time) * 1000)
        if state in self._stage_durations:
            return int(self._stage_durations[state] * 1000)
        return None

    def get_state_for_metrics(self) -> int:
        """Get numeric state value for metrics.

        Returns:
            0=loading, 1=worker_ready, 2=router_ready, 3=probe, 4=ready, 5=failed
        """
        if not self._enabled or self._current_state is None:
            return 0

        mapping = {
            ReadinessState.MODEL_LOADING: 0,
            ReadinessState.WORKER_READY: 1,
            ReadinessState.ROUTER_READY: 2,
            ReadinessState.MINIMAL_PROBE: 3,
            ReadinessState.READY: 4,
            ReadinessState.FAILED: 5,
        }
        return mapping.get(self._current_state, 0)

    def get_stage_durations_for_metrics(self) -> dict[str, int]:
        """Get stage durations for metrics export.

        Returns:
            Dict mapping stage name to duration in ms (only completed stages)
        """
        if not self._enabled:
            return {}

        result = {}
        # Only include completed stages (in _stage_durations), not current stage
        for state, duration in self._stage_durations.items():
            result[state.value] = int(duration * 1000)

        return result
