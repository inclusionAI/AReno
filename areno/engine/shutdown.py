"""Two-stage graceful shutdown manager (issue #236).

Implements a signal-aware shutdown coordinator that:

1. **First signal** (SIGINT/SIGTERM): Sets a shutdown-requested flag,
   logs the reason, and allows the main loop to reach a documented safe
   point, flush outputs, and close workers cleanly.
2. **Second signal**: Forces immediate exit via ``os._exit(130)``,
   preserving the initial termination reason in the exit message.

The module is pure Python with no external dependencies.  It never
replaces the original signal handlers until ``install()`` is called, and
``uninstall()`` restores the previous handlers (backward compatible).

Public API:

* :class:`GracefulShutdown` — coordinator object.
* :func:`format_shutdown_reason` — human-readable reason string.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("areno.engine.shutdown")

# Exit codes: 130 = terminated by SIGINT (128 + 2), 143 = SIGTERM (128 + 15).
_EXIT_CODE_SIGINT = 130
_EXIT_CODE_SIGTERM = 143


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
    FORCED = "forced"


@dataclass
class ShutdownInfo:
    """Structured information about a shutdown event.

    Attributes:
        state: Current :class:`ShutdownState`.
        signal_number: The signal that triggered the shutdown (1st or 2nd).
        stage: The :class:`ShutdownStage` when the signal was received.
        reason: Human-readable reason string.
        timestamp: Monotonic time when the signal was received (from
            ``time.monotonic()``).
        first_signal: True if this was the first signal (graceful path).
    """

    state: ShutdownState
    signal_number: int
    stage: ShutdownStage
    reason: str
    timestamp: float
    first_signal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "signal_number": self.signal_number,
            "stage": self.stage.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "first_signal": self.first_signal,
        }


_SIGNAL_NAMES = {
    signal.SIGINT: "SIGINT",
    signal.SIGTERM: "SIGTERM",
}


def format_shutdown_reason(info: ShutdownInfo) -> str:
    """Return a human-readable shutdown reason string.

    Args:
        info: A :class:`ShutdownInfo` describing the shutdown event.

    Returns:
        A formatted string suitable for logging or CLI output.
    """

    sig_name = _SIGNAL_NAMES.get(info.signal_number, f"signal {info.signal_number}")
    if info.first_signal:
        return (
            f"Graceful shutdown requested ({sig_name}) during {info.stage.value}. "
            f"Stopping new work, flushing outputs, and closing workers."
        )
    return f"Forced exit ({sig_name}) — second signal received during {info.stage.value}. Initial reason: {info.reason}"


class GracefulShutdown:
    """Two-stage graceful shutdown coordinator.

    On the first SIGINT or SIGTERM:
        - Sets ``shutdown_requested = True``.
        - Records the signal, stage, and reason.
        - Logs a message telling the user to wait or press Ctrl-C again to force.

    On the second signal:
        - Logs a forced-exit message preserving the initial reason.
        - Calls ``os._exit()`` immediately (cannot be caught or interrupted).

    Usage::

        shutdown = GracefulShutdown()
        shutdown.install()
        try:
            for batch in training_loop:
                if shutdown.shutdown_requested:
                    break
                train_step(batch)
        finally:
            shutdown.uninstall()
            flush_outputs()
    """

    def __init__(self) -> None:
        self._state = ShutdownState.RUNNING
        self._first_info: ShutdownInfo | None = None
        self._current_stage: ShutdownStage = ShutdownStage.IDLE
        self._previous_handlers: dict[int, signal.Handlers | int] = {}
        self._installed = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shutdown_requested(self) -> bool:
        """True if a graceful shutdown has been requested (first signal received)."""

        return self._state in (ShutdownState.SHUTDOWN_REQUESTED, ShutdownState.SHUTTING_DOWN)

    @property
    def state(self) -> ShutdownState:
        """Current shutdown state."""

        return self._state

    @property
    def stage(self) -> ShutdownStage:
        """The stage when the first signal was received, or current stage."""

        if self._first_info is not None:
            return self._first_info.stage
        return self._current_stage

    @property
    def info(self) -> ShutdownInfo | None:
        """Structured info about the first shutdown signal, or None."""

        return self._first_info

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def set_stage(self, stage: ShutdownStage) -> None:
        """Update the current execution stage for better shutdown diagnostics.

        Args:
            stage: The stage that is about to begin (training, rollout, serving).
        """

        self._current_stage = stage

    # ------------------------------------------------------------------
    # Signal handler installation
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Install signal handlers for SIGINT and SIGTERM.

        Saves the previous handlers so they can be restored by
        :meth:`uninstall`.  Calling ``install()`` twice without
        :meth:`uninstall` in between is a no-op.
        """

        if self._installed:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            prev = signal.signal(sig, self._handler)
            self._previous_handlers[sig] = prev
        self._installed = True
        logger.debug("Graceful shutdown handlers installed")

    def uninstall(self) -> None:
        """Restore the previous signal handlers.

        Safe to call even if :meth:`install` was never called.
        """

        if not self._installed:
            return
        for sig, prev in self._previous_handlers.items():
            try:
                signal.signal(sig, prev)
            except (OSError, ValueError):
                # In a non-main thread, signal.signal() raises ValueError.
                # This is expected and safe to ignore during teardown.
                pass
        self._previous_handlers.clear()
        self._installed = False

    # ------------------------------------------------------------------
    # Signal handler
    # ------------------------------------------------------------------

    def _handler(self, signum: int, _frame: Any) -> None:
        """Internal signal handler implementing the two-stage logic."""

        import time

        now = time.monotonic()

        if self._state in (ShutdownState.RUNNING, ShutdownState.SHUTDOWN_REQUESTED, ShutdownState.SHUTTING_DOWN):
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
                self._first_info = ShutdownInfo(
                    state=ShutdownState.SHUTDOWN_REQUESTED,
                    signal_number=signum,
                    stage=self._current_stage,
                    reason=reason,
                    timestamp=now,
                    first_signal=True,
                )
                self._state = ShutdownState.SHUTDOWN_REQUESTED
                sig_name = _SIGNAL_NAMES.get(signum, f"signal {signum}")
                msg = (
                    f"\n{sig_name} received during {self._current_stage.value}. "
                    f"Stopping gracefully... (press Ctrl-C again to force exit)\n"
                )
                # Use stderr to avoid interfering with any stdout output.
                print(msg, file=sys.stderr, flush=True)
                logger.info(reason)
            else:
                # Second signal: force exit.
                self._state = ShutdownState.FORCED
                forced_reason = format_shutdown_reason(
                    ShutdownInfo(
                        state=ShutdownState.FORCED,
                        signal_number=signum,
                        stage=self._current_stage,
                        reason=self._first_info.reason,
                        timestamp=now,
                        first_signal=False,
                    )
                )
                print(f"\nForced exit: {forced_reason}\n", file=sys.stderr, flush=True)
                # Restore default handler for the signal so a third Ctrl-C
                # (if it comes during os._exit) is not swallowed.
                self.uninstall()
                # Use os._exit so we cannot be interrupted by yet another signal.
                exit_code = _EXIT_CODE_SIGINT if signum == signal.SIGINT else _EXIT_CODE_SIGTERM
                os._exit(exit_code)

    # ------------------------------------------------------------------
    # State transitions for the main loop
    # ------------------------------------------------------------------

    def begin_shutdown(self) -> None:
        """Transition from SHUTDOWN_REQUESTED to SHUTTING_DOWN.

        Call this when the main loop has noticed ``shutdown_requested``
        and is about to flush outputs and close workers.
        """

        if self._state == ShutdownState.SHUTDOWN_REQUESTED:
            self._state = ShutdownState.SHUTTING_DOWN
            logger.debug("Entering shutdown phase")

    def complete_shutdown(self) -> ShutdownInfo | None:
        """Mark shutdown as complete and uninstall handlers.

        Returns:
            The :class:`ShutdownInfo` from the first signal, or ``None``
            if no shutdown was requested.
        """

        self.uninstall()
        if self._first_info is not None:
            self._state = ShutdownState.SHUTTING_DOWN
        return self._first_info

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> GracefulShutdown:
        self.install()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.uninstall()

    # ------------------------------------------------------------------
    # Testing helpers
    # ------------------------------------------------------------------

    def _simulate_signal(self, signum: int) -> None:
        """Simulate receiving a signal for testing without actually sending one.

        Args:
            signum: Signal number to simulate (e.g. ``signal.SIGINT``).
        """

        self._handler(signum, None)
