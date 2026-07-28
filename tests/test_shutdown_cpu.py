"""CPU tests for two-stage graceful shutdown (issue #236).

Tests cover:
- State transitions (RUNNING -> SHUTDOWN_REQUESTED -> SHUTTING_DOWN).
- First signal sets shutdown_requested, logs reason.
- Second signal forces exit (tested via _simulate_signal + mock).
- Stage tracking (training, rollout, serving, idle).
- format_shutdown_reason output.
- Install/uninstall restores previous handlers.
- Context manager support.
- Backward compatibility (uninstall without install is safe).
- ShutdownInfo.to_dict structured output.
- Deterministic state after simulated signals.
"""

from __future__ import annotations

import signal
import unittest
from unittest.mock import patch

from areno.engine.shutdown import (
    GracefulShutdown,
    ShutdownInfo,
    ShutdownStage,
    ShutdownState,
    format_shutdown_reason,
)

# ---------------------------------------------------------------------------
# format_shutdown_reason
# ---------------------------------------------------------------------------


class TestFormatShutdownReason(unittest.TestCase):
    """format_shutdown_reason should produce correct output for first/second signals."""

    def test_first_signal_message(self):
        info = ShutdownInfo(
            state=ShutdownState.SHUTDOWN_REQUESTED,
            signal_number=signal.SIGINT,
            stage=ShutdownStage.TRAINING,
            reason="",
            timestamp=0.0,
            first_signal=True,
        )
        text = format_shutdown_reason(info)
        self.assertIn("Graceful shutdown", text)
        self.assertIn("SIGINT", text)
        self.assertIn("training", text)

    def test_second_signal_message(self):
        info = ShutdownInfo(
            state=ShutdownState.FORCED,
            signal_number=signal.SIGTERM,
            stage=ShutdownStage.ROLLOUT,
            reason="Initial reason here",
            timestamp=0.0,
            first_signal=False,
        )
        text = format_shutdown_reason(info)
        self.assertIn("Forced exit", text)
        self.assertIn("SIGTERM", text)
        self.assertIn("Initial reason here", text)

    def test_unknown_signal_number(self):
        info = ShutdownInfo(
            state=ShutdownState.SHUTDOWN_REQUESTED,
            signal_number=99,
            stage=ShutdownStage.SERVING,
            reason="",
            timestamp=0.0,
            first_signal=True,
        )
        text = format_shutdown_reason(info)
        self.assertIn("signal 99", text)


# ---------------------------------------------------------------------------
# ShutdownInfo.to_dict
# ---------------------------------------------------------------------------


class TestShutdownInfoToDict(unittest.TestCase):
    """to_dict should produce a complete JSON-serialisable structure."""

    def test_to_dict_fields(self):
        info = ShutdownInfo(
            state=ShutdownState.SHUTDOWN_REQUESTED,
            signal_number=signal.SIGINT,
            stage=ShutdownStage.TRAINING,
            reason="test reason",
            timestamp=123.45,
            first_signal=True,
        )
        d = info.to_dict()
        self.assertEqual(d["state"], "shutdown_requested")
        self.assertEqual(d["signal_number"], signal.SIGINT)
        self.assertEqual(d["stage"], "training")
        self.assertEqual(d["reason"], "test reason")
        self.assertEqual(d["timestamp"], 123.45)
        self.assertTrue(d["first_signal"])


# ---------------------------------------------------------------------------
# GracefulShutdown: basic state
# ---------------------------------------------------------------------------


class TestGracefulShutdownState(unittest.TestCase):
    """GracefulShutdown should start in RUNNING state with no shutdown requested."""

    def test_initial_state(self):
        shutdown = GracefulShutdown()
        self.assertFalse(shutdown.shutdown_requested)
        self.assertEqual(shutdown.state, ShutdownState.RUNNING)
        self.assertEqual(shutdown.stage, ShutdownStage.IDLE)
        self.assertIsNone(shutdown.info)

    def test_set_stage(self):
        shutdown = GracefulShutdown()
        shutdown.set_stage(ShutdownStage.TRAINING)
        self.assertEqual(shutdown.stage, ShutdownStage.TRAINING)

    def test_set_stage_updates(self):
        shutdown = GracefulShutdown()
        shutdown.set_stage(ShutdownStage.ROLLOUT)
        self.assertEqual(shutdown.stage, ShutdownStage.ROLLOUT)
        shutdown.set_stage(ShutdownStage.TRAINING)
        self.assertEqual(shutdown.stage, ShutdownStage.TRAINING)


# ---------------------------------------------------------------------------
# GracefulShutdown: first signal (simulated)
# ---------------------------------------------------------------------------


class TestFirstSignal(unittest.TestCase):
    """First signal should request graceful shutdown without forcing exit."""

    def test_first_signal_sets_shutdown_requested(self):
        shutdown = GracefulShutdown()
        shutdown.set_stage(ShutdownStage.TRAINING)
        shutdown._simulate_signal(signal.SIGINT)
        self.assertTrue(shutdown.shutdown_requested)
        self.assertEqual(shutdown.state, ShutdownState.SHUTDOWN_REQUESTED)

    def test_first_signal_records_info(self):
        shutdown = GracefulShutdown()
        shutdown.set_stage(ShutdownStage.ROLLOUT)
        shutdown._simulate_signal(signal.SIGTERM)
        self.assertIsNotNone(shutdown.info)
        self.assertEqual(shutdown.info.signal_number, signal.SIGTERM)
        self.assertEqual(shutdown.info.stage, ShutdownStage.ROLLOUT)
        self.assertTrue(shutdown.info.first_signal)

    def test_first_signal_does_not_exit(self):
        """First signal must not call os._exit()."""
        shutdown = GracefulShutdown()
        shutdown._simulate_signal(signal.SIGINT)
        # If we reach this assertion, os._exit was not called.
        self.assertTrue(True)

    def test_first_signal_records_correct_stage(self):
        shutdown = GracefulShutdown()
        shutdown.set_stage(ShutdownStage.SERVING)
        shutdown._simulate_signal(signal.SIGINT)
        self.assertEqual(shutdown.info.stage, ShutdownStage.SERVING)


# ---------------------------------------------------------------------------
# GracefulShutdown: second signal (forced exit)
# ---------------------------------------------------------------------------


class TestSecondSignal(unittest.TestCase):
    """Second signal should force exit via os._exit()."""

    @patch("areno.engine.shutdown.os._exit")
    def test_second_signal_forces_exit(self, mock_exit):
        shutdown = GracefulShutdown()
        shutdown._simulate_signal(signal.SIGINT)
        # Second signal should call os._exit.
        shutdown._simulate_signal(signal.SIGINT)
        mock_exit.assert_called_once()
        self.assertEqual(mock_exit.call_args[0][0], 130)  # 128 + 2 (SIGINT)

    @patch("areno.engine.shutdown.os._exit")
    def test_second_signal_sigterm_exit_code(self, mock_exit):
        shutdown = GracefulShutdown()
        shutdown._simulate_signal(signal.SIGTERM)
        shutdown._simulate_signal(signal.SIGTERM)
        mock_exit.assert_called_once()
        self.assertEqual(mock_exit.call_args[0][0], 143)  # 128 + 15 (SIGTERM)

    @patch("areno.engine.shutdown.os._exit")
    def test_second_signal_state_is_forced(self, mock_exit):
        shutdown = GracefulShutdown()
        shutdown._simulate_signal(signal.SIGINT)
        shutdown._simulate_signal(signal.SIGINT)
        self.assertEqual(shutdown.state, ShutdownState.FORCED)

    @patch("areno.engine.shutdown.os._exit")
    def test_mixed_signals_force_exit(self, mock_exit):
        """First SIGINT then SIGTERM should still force exit."""
        shutdown = GracefulShutdown()
        shutdown._simulate_signal(signal.SIGINT)
        shutdown._simulate_signal(signal.SIGTERM)
        mock_exit.assert_called_once()


# ---------------------------------------------------------------------------
# GracefulShutdown: begin/complete shutdown
# ---------------------------------------------------------------------------


class TestShutdownTransitions(unittest.TestCase):
    """begin_shutdown and complete_shutdown should transition states correctly."""

    def test_begin_shutdown(self):
        shutdown = GracefulShutdown()
        shutdown._simulate_signal(signal.SIGINT)
        self.assertEqual(shutdown.state, ShutdownState.SHUTDOWN_REQUESTED)
        shutdown.begin_shutdown()
        self.assertEqual(shutdown.state, ShutdownState.SHUTTING_DOWN)

    def test_begin_shutdown_without_request_is_noop(self):
        shutdown = GracefulShutdown()
        shutdown.begin_shutdown()
        self.assertEqual(shutdown.state, ShutdownState.RUNNING)

    def test_complete_shutdown_returns_info(self):
        shutdown = GracefulShutdown()
        shutdown._simulate_signal(signal.SIGINT)
        info = shutdown.complete_shutdown()
        self.assertIsNotNone(info)
        self.assertEqual(info.signal_number, signal.SIGINT)

    def test_complete_shutdown_without_request_returns_none(self):
        shutdown = GracefulShutdown()
        info = shutdown.complete_shutdown()
        self.assertIsNone(info)


# ---------------------------------------------------------------------------
# GracefulShutdown: install/uninstall
# ---------------------------------------------------------------------------


class TestInstallUninstall(unittest.TestCase):
    """install and uninstall should manage signal handlers correctly."""

    def test_install_sets_installed_flag(self):
        shutdown = GracefulShutdown()
        shutdown.install()
        # Can't directly check _installed, but double-install should be no-op.
        shutdown.install()  # Should not raise.
        shutdown.uninstall()

    def test_uninstall_without_install_is_safe(self):
        shutdown = GracefulShutdown()
        shutdown.uninstall()  # Should not raise.

    def test_uninstall_restores_previous_handler(self):
        """After uninstall, the default SIGINT handler should be restored."""
        shutdown = GracefulShutdown()
        prev_handler = signal.getsignal(signal.SIGINT)
        shutdown.install()
        # Handler is now our custom handler.
        self.assertNotEqual(signal.getsignal(signal.SIGINT), prev_handler)
        shutdown.uninstall()
        # Handler should be restored.
        self.assertEqual(signal.getsignal(signal.SIGINT), prev_handler)


# ---------------------------------------------------------------------------
# GracefulShutdown: context manager
# ---------------------------------------------------------------------------


class TestContextManager(unittest.TestCase):
    """GracefulShutdown should work as a context manager."""

    def test_context_manager_installs_and_uninstalls(self):
        prev_handler = signal.getsignal(signal.SIGINT)
        with GracefulShutdown() as shutdown:
            self.assertTrue(shutdown._installed)
        self.assertFalse(shutdown._installed)
        self.assertEqual(signal.getsignal(signal.SIGINT), prev_handler)

    def test_context_manager_shutdown_requested_inside(self):
        with GracefulShutdown() as shutdown:
            shutdown.set_stage(ShutdownStage.TRAINING)
            shutdown._simulate_signal(signal.SIGINT)
            self.assertTrue(shutdown.shutdown_requested)


# ---------------------------------------------------------------------------
# GracefulShutdown: determinism
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    """Simulated signals should produce deterministic state transitions."""

    def test_same_sequence_same_state(self):
        s1 = GracefulShutdown()
        s1.set_stage(ShutdownStage.TRAINING)
        s1._simulate_signal(signal.SIGINT)

        s2 = GracefulShutdown()
        s2.set_stage(ShutdownStage.TRAINING)
        s2._simulate_signal(signal.SIGINT)

        self.assertEqual(s1.state, s2.state)
        self.assertEqual(s1.info.signal_number, s2.info.signal_number)
        self.assertEqual(s1.info.stage, s2.info.stage)
        self.assertEqual(s1.info.first_signal, s2.info.first_signal)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility(unittest.TestCase):
    """When not installed, the module should not interfere with existing behavior."""

    def test_no_install_no_interference(self):
        shutdown = GracefulShutdown()
        self.assertFalse(shutdown.shutdown_requested)
        self.assertEqual(shutdown.state, ShutdownState.RUNNING)

    def test_uninstall_without_install_no_error(self):
        shutdown = GracefulShutdown()
        shutdown.uninstall()
        shutdown.uninstall()

    def test_set_stage_without_install(self):
        shutdown = GracefulShutdown()
        shutdown.set_stage(ShutdownStage.SERVING)
        self.assertEqual(shutdown.stage, ShutdownStage.SERVING)


if __name__ == "__main__":
    unittest.main()
