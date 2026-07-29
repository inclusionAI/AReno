"""Two-stage graceful shutdown manager (issue #236).

Implements a signal-aware shutdown coordinator that:

1. **First signal** (SIGINT/SIGTERM): Sets a shutdown-requested flag,
   logs the reason, and starts a deadline timer. The main loop is
   expected to check ``should_stop`` at safe points and break.
   If the deadline expires before the loop exits, forces exit.
2. **Second signal**: Forces immediate exit via ``os._exit()``,
   preserving the initial termination reason in the exit message.

The module is pure Python with no external dependencies.  It never
replaces the original signal handlers until ``install()`` is called, and
``uninstall()`` restores the previous handlers (backward compatible).

Public API:

* :class:`GracefulShutdown` — coordinator object.
* :func:`format_shutdown_reason` — human-readable reason string.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("areno.engine.shutdown")

# Exit codes: 130 = terminated by SIGINT (128 + 2), 143 = SIGTERM (128 + 15).
_EXIT_CODE_SIGINT = 130
_EXIT_CODE_SIGTERM = 143

# Default deadline: 30 seconds after first signal before forced exit.
_DEFAULT_DEADLINE_S = 30.0


class ShutdownStage(str, Enum):
    """Which stage was interrupted when the shutdown was requested."""

    TRAINING = "training"
    ROLLOUT = "rollout"
    SERVING = "serving"
    IDLE = "idle"
    UNKNOWN = "unknown"


class ShutdownState(str, Enum):
    """Current state of the shutdown coordinator."""

    RUNNING = "running"
    SHUTDOWN_REQUESTED = "shutdown_requested"
    SHUTTING_DOWN = "shutting_down"
    COMPLETED = "completed"
    FORCED = "forced"


@dataclass
class ShutdownInfo:
    """Structured information about a shutdown event.

    Attributes:
        state: Current :class:`ShutdownState`.
        signal_number: The signal that triggered the shutdown (1st or 2nd).
        stage: The :class:`ShutdownStage` when the signal was received.
        reason: Human-readable reason string.
        timestamp: Monotonic time when the signal was received.
        first_signal: True if this was the first signal (graceful path).
    """

    state: ShutdownState
    signal_number: int
    stage: ShutdownStage
    reason: str
    timestamp: float
    first_signal: bool
    deadline: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "signal_number": self.signal_number,
            "stage": self.stage.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "first_signal": self.first_signal,
            "deadline": self.deadline,
        }


_SIGNAL_NAMES = {
    signal.SIGINT: "SIGINT",
    signal.SIGTERM: "SIGTERM",
}


def format_shutdown_reason(info: ShutdownInfo) -> str:
    """Return a human-readable shutdown reason string."""

    sig_name = _SIGNAL_NAMES.get(info.signal_number, f"signal {info.signal_number}")
    if info.first_signal:
        return (
            f"Graceful shutdown requested ({sig_name}) during {info.stage.value}. "
            f"Stopping new work, flushing outputs, and closing workers."
        )
    return f"Forced exit ({sig_name}) — second signal received during {info.stage.value}. Initial reason: {info.reason}"


class GracefulShutdown:
    """Two-stage graceful shutdown coordinator with deadline.

    On the first SIGINT or SIGTERM:
        - Sets ``shutdown_requested = True``.
        - Records the signal, stage, and reason.
        - Starts a deadline timer (default 30s). If the main loop does
          not exit before the deadline, forces exit.
        - Logs a message telling the user to wait or press Ctrl-C again.

    On the second signal:
        - Logs a forced-exit message preserving the initial reason.
        - Calls ``os._exit()`` immediately (cannot be caught or interrupted).

    The main loop should check ``should_stop`` at safe points::

        shutdown = GracefulShutdown()
        shutdown.install()
        try:
            for batch in training_loop:
                if shutdown.should_stop:
                    break
                train_step(batch)
        finally:
            shutdown.uninstall()
    """

    def __init__(
        self,
        *,
        deadline_s: float = _DEFAULT_DEADLINE_S,
        on_shutdown_requested: Callable[[ShutdownInfo], None] | None = None,
    ) -> None:
        if not math.isfinite(deadline_s) or deadline_s <= 0:
            raise ValueError("deadline_s must be a finite number greater than 0")
        self._state = ShutdownState.RUNNING
        self._first_info: ShutdownInfo | None = None
        self._current_stage: ShutdownStage = ShutdownStage.IDLE
        self._previous_handlers: dict[int, signal.Handlers | int] = {}
        self._installed = False
        self._deadline_s = deadline_s
        self._deadline_thread: threading.Thread | None = None
        self._deadline_stop = threading.Event()
        self._on_shutdown_requested = on_shutdown_requested

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shutdown_requested(self) -> bool:
        """True if a graceful shutdown has been requested."""

        return self._state in (ShutdownState.SHUTDOWN_REQUESTED, ShutdownState.SHUTTING_DOWN)

    @property
    def should_stop(self) -> bool:
        """True if the main loop should stop at the next safe point.

        This is the property that training loops should check.
        """

        return self.shutdown_requested

    @property
    def state(self) -> ShutdownState:
        return self._state

    @property
    def stage(self) -> ShutdownStage:
        if self._first_info is not None:
            return self._first_info.stage
        return self._current_stage

    @property
    def info(self) -> ShutdownInfo | None:
        return self._first_info

    @property
    def deadline_s(self) -> float:
        """The deadline in seconds after the first signal before forced exit."""

        return self._deadline_s

    def worker_payload(self) -> dict[str, Any] | None:
        """Return the initial shutdown event for propagation to worker ranks."""

        if self._first_info is None:
            return None
        return self._first_info.to_dict()

    @staticmethod
    def _log_event(event: str, info: ShutdownInfo) -> None:
        payload = {"event": event, **info.to_dict()}
        logger.info("shutdown_event=%s", json.dumps(payload, sort_keys=True))

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def set_stage(self, stage: ShutdownStage) -> None:
        """Update the current execution stage for better shutdown diagnostics."""

        self._current_stage = stage

    # ------------------------------------------------------------------
    # Signal handler installation
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Install signal handlers for SIGINT and SIGTERM."""

        if self._installed:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            prev = signal.signal(sig, self._handler)
            self._previous_handlers[sig] = prev
        self._installed = True
        logger.debug("Graceful shutdown handlers installed")

    def uninstall(self) -> None:
        """Restore the previous signal handlers."""

        if not self._installed:
            return
        self._cancel_deadline()
        for sig, prev in self._previous_handlers.items():
            try:
                signal.signal(sig, prev)
            except (OSError, ValueError):
                pass
        self._previous_handlers.clear()
        self._installed = False

    # ------------------------------------------------------------------
    # Deadline timer
    # ------------------------------------------------------------------

    def _start_deadline(self) -> None:
        """Start a background thread that forces exit after deadline_s."""

        self._deadline_stop.clear()
        self._deadline_thread = threading.Thread(target=self._deadline_worker, daemon=True)
        self._deadline_thread.start()

    def _cancel_deadline(self) -> None:
        """Cancel the deadline timer."""

        self._deadline_stop.set()
        if self._deadline_thread is not None:
            if self._deadline_thread is not threading.current_thread():
                self._deadline_thread.join(timeout=0.5)
            self._deadline_thread = None

    def _deadline_worker(self) -> None:
        """Background thread: wait for deadline, then force exit if still running."""

        if self._deadline_stop.wait(timeout=self._deadline_s):
            return  # Cancelled
        # Deadline expired — force exit.
        if self._state != ShutdownState.FORCED and self._first_info is not None:
            self._force_exit(self._first_info.signal_number, trigger="deadline")

    # ------------------------------------------------------------------
    # Signal handler
    # ------------------------------------------------------------------

    def _handler(self, signum: int, _frame: Any) -> None:
        """Internal signal handler implementing the two-stage logic."""

        now = time.monotonic()

        if self._first_info is None:
            # First signal: request graceful shutdown.
            reason = format_shutdown_reason(
                ShutdownInfo(
                    state=ShutdownState.SHUTDOWN_REQUESTED,
                    signal_number=signum,
                    stage=self._current_stage,
                    reason="",
                    timestamp=now,
                    first_signal=True,
                )
            )
            deadline = now + self._deadline_s
            self._first_info = ShutdownInfo(
                state=ShutdownState.SHUTDOWN_REQUESTED,
                signal_number=signum,
                stage=self._current_stage,
                reason=reason,
                timestamp=now,
                first_signal=True,
                deadline=deadline,
            )
            self._state = ShutdownState.SHUTDOWN_REQUESTED
            sig_name = _SIGNAL_NAMES.get(signum, f"signal {signum}")
            msg = (
                f"\n{sig_name} received during {self._current_stage.value}. "
                f"Stopping gracefully... (press Ctrl-C again to force exit, "
                f"or wait {self._deadline_s:.0f}s for auto-force)\n"
            )
            print(msg, file=sys.stderr, flush=True)
            logger.info(reason)
            self._log_event("shutdown_requested", self._first_info)
            if self._on_shutdown_requested is not None:
                try:
                    self._on_shutdown_requested(self._first_info)
                except Exception:
                    logger.exception("shutdown request callback failed")
            # Start deadline timer.
            self._start_deadline()
        else:
            self._force_exit(signum, trigger="second_signal", timestamp=now)

    def _force_exit(self, signum: int, *, trigger: str, timestamp: float | None = None) -> None:
        """Emit the forced event and terminate using the initial signal's exit code."""

        if self._first_info is None:
            return
        self._state = ShutdownState.FORCED
        self._cancel_deadline()
        forced_info = ShutdownInfo(
            state=ShutdownState.FORCED,
            signal_number=signum,
            stage=self._current_stage,
            reason=self._first_info.reason,
            timestamp=time.monotonic() if timestamp is None else timestamp,
            first_signal=False,
            deadline=self._first_info.deadline,
        )
        forced_reason = format_shutdown_reason(forced_info)
        print(f"\nForced exit ({trigger}): {forced_reason}\n", file=sys.stderr, flush=True)
        self._log_event("shutdown_forced", forced_info)
        self.uninstall()
        initial_signal = self._first_info.signal_number
        exit_code = _EXIT_CODE_SIGINT if initial_signal == signal.SIGINT else _EXIT_CODE_SIGTERM
        os._exit(exit_code)

    # ------------------------------------------------------------------
    # State transitions for the main loop
    # ------------------------------------------------------------------

    def begin_shutdown(self) -> None:
        """Transition from SHUTDOWN_REQUESTED to SHUTTING_DOWN."""

        if self._state == ShutdownState.SHUTDOWN_REQUESTED:
            self._state = ShutdownState.SHUTTING_DOWN
            logger.debug("Entering shutdown phase")

    def complete_shutdown(self) -> ShutdownInfo | None:
        """Mark shutdown as complete and uninstall handlers."""

        self._cancel_deadline()
        self.uninstall()
        if self._first_info is not None:
            self._state = ShutdownState.COMPLETED
            self._first_info.state = ShutdownState.COMPLETED
            self._log_event("shutdown_completed", self._first_info)
        return self._first_info

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> GracefulShutdown:
        self.install()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self._cancel_deadline()
        self.uninstall()

    # ------------------------------------------------------------------
    # Testing helpers
    # ------------------------------------------------------------------

    def _simulate_signal(self, signum: int) -> None:
        """Simulate receiving a signal for testing without actually sending one."""

        self._handler(signum, None)
