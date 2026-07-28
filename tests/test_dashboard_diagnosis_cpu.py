"""CPU tests for failure diagnosis in the dashboard server."""

from __future__ import annotations

import unittest

from areno.dashboard.server import _diagnose_failure


class DiagnosisTest(unittest.TestCase):
    # ------------------------------------------------------------------
    # Traceback
    # ------------------------------------------------------------------

    def test_traceback_detected(self):
        logs = [
            "Starting training...",
            "Traceback (most recent call last):",
            '  File "train.py", line 42, in rollout',
            "    result = model.generate(...)",
            "RuntimeError: CUDA failure detected at rollout decode completion",
        ]
        diag = _diagnose_failure(logs, stage="rollout")
        self.assertTrue(diag["identified"])
        self.assertEqual(diag["type"], "traceback")
        self.assertEqual(diag["phase"], "rollout")
        self.assertIn("RuntimeError", diag["error"])
        self.assertTrue(any("Traceback" in line for line in diag["context"]))

    def test_teardown_traceback_filtered(self):
        """Tracebacks from atexit/__del__/cleanup are ignored."""
        logs = [
            "Training completed successfully.",
            "Error in atexit._run_exitfuncs:",
            "Traceback (most recent call last):",
            '  File "atexit.py", line 24, in _run_exitfuncs',
            "RuntimeError: cannot close after shutdown",
        ]
        diag = _diagnose_failure(logs)
        # Should fall through to generic (line has "RuntimeError:" and no teardown marker)
        # Actually the teardown filter should catch this and fall through to nothing
        # Since "atexit" is in the frame, the traceback is filtered, and the RuntimeError
        # line is inside the traceback (already consumed) — so generic won't pick it up either.
        # Expected: not identified
        self.assertFalse(diag["identified"])

    def test_first_traceback_wins(self):
        """Only the earliest actionable traceback is reported."""
        logs = [
            "Traceback (most recent call last):",
            '  File "data.py", line 10, in load',
            "KeyError: missing_column",
            "... more logs ...",
            "Traceback (most recent call last):",
            '  File "train.py", line 50, in train_step',
            "RuntimeError: gradient is NaN",
        ]
        diag = _diagnose_failure(logs)
        self.assertTrue(diag["identified"])
        self.assertEqual(diag["type"], "traceback")
        self.assertIn("KeyError", diag["error"])

    # ------------------------------------------------------------------
    # OOM
    # ------------------------------------------------------------------

    def test_oom_detected(self):
        logs = [
            "rollout decode progress: dp=0/4 active=8 ...",
            "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
            "Process exited with code 1",
        ]
        diag = _diagnose_failure(logs, stage="rollout")
        self.assertTrue(diag["identified"])
        self.assertEqual(diag["type"], "oom")
        self.assertIn("OutOfMemoryError", diag["error"])
        self.assertEqual(diag["phase"], "rollout")

    def test_oom_before_traceback(self):
        """OOM line comes before any traceback — OOM wins."""
        logs = [
            "CUDA out of memory",
            "Traceback (most recent call last):",
            '  File "train.py", line 42, in train_step',
            "RuntimeError: worker died",
        ]
        diag = _diagnose_failure(logs)
        self.assertTrue(diag["identified"])
        self.assertEqual(diag["type"], "oom")

    # ------------------------------------------------------------------
    # Model load
    # ------------------------------------------------------------------

    def test_model_load_failure_detected(self):
        logs = [
            "Loading checkpoint from Qwen/Qwen3-0.6B...",
            "RuntimeError: Can't load safetensors checkpoint: file not found",
        ]
        diag = _diagnose_failure(logs, stage="model_load")
        self.assertTrue(diag["identified"])
        self.assertEqual(diag["type"], "model_load")
        self.assertIn("file not found", diag["error"].lower())

    # ------------------------------------------------------------------
    # Data error
    # ------------------------------------------------------------------

    def test_data_error_detected(self):
        logs = [
            "Loading dataset gsm8k:main...",
            "KeyError: 'answer' column not found in dataset",
        ]
        diag = _diagnose_failure(logs)
        self.assertTrue(diag["identified"])
        self.assertEqual(diag["type"], "data")
        self.assertIn("KeyError", diag["error"])

    # ------------------------------------------------------------------
    # Distributed
    # ------------------------------------------------------------------

    def test_distributed_failure_detected(self):
        logs = [
            "rank 2: starting rollout...",
            "NCCL error: unhandled system error, rank=2",
        ]
        diag = _diagnose_failure(logs)
        self.assertTrue(diag["identified"])
        self.assertEqual(diag["type"], "distributed")
        self.assertIn("NCCL", diag["error"])

    # ------------------------------------------------------------------
    # Generic fallback
    # ------------------------------------------------------------------

    def test_generic_runtime_error(self):
        logs = [
            "Doing some work...",
            "RuntimeError: something went wrong during training",
            "Process finished.",
        ]
        diag = _diagnose_failure(logs)
        self.assertTrue(diag["identified"])
        self.assertEqual(diag["type"], "generic")
        self.assertIn("RuntimeError", diag["error"])

    # ------------------------------------------------------------------
    # Not identified
    # ------------------------------------------------------------------

    def test_no_error_returns_not_identified(self):
        logs = [
            "Starting training...",
            "Step 1: loss=0.5",
            "Step 2: loss=0.4",
            "Training finished.",
        ]
        diag = _diagnose_failure(logs)
        self.assertFalse(diag["identified"])

    def test_empty_logs_returns_not_identified(self):
        diag = _diagnose_failure([])
        self.assertFalse(diag["identified"])

    # ------------------------------------------------------------------
    # Context lines
    # ------------------------------------------------------------------

    def test_context_lines_around_error(self):
        logs = [
            "line 0",
            "line 1",
            "line 2",
            "line 3",
            "line 4",
            "RuntimeError: boom",
            "line 6",
            "line 7",
            "line 8",
            "line 9",
            "line 10",
        ]
        diag = _diagnose_failure(logs)
        self.assertTrue(diag["identified"])
        ctx = diag["context"]
        # 5 lines before + error line + 5 lines after = 11
        self.assertLessEqual(len(ctx), 11)
        self.assertIn("RuntimeError: boom", ctx)

    # ------------------------------------------------------------------
    # Phase propagation
    # ------------------------------------------------------------------

    def test_phase_propagated(self):
        logs = ["Traceback (most recent call last):", "  File ...", "RuntimeError: fail"]
        diag = _diagnose_failure(logs, stage="train")
        self.assertEqual(diag["phase"], "train")