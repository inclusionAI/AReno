from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from areno.api import metrics as metrics_mod
from areno.api.metrics import (
    MetricsRecorder,
    collect_train_batch_stats,
    init_rollout_stats,
    record_rollout_sequence_stats,
)
from areno.api.models import TrainSequence


class MetricsUtilityTest(unittest.TestCase):
    """Metric helper tests cover scalar extraction without TensorBoard writer IO."""

    def test_collect_train_batch_stats_filters_prompt_positions(self):
        """Only response positions should contribute logprob/advantage stats."""
        seq = TrainSequence(
            prompt_mask=[True, True, False, False],
            tokens=[1, 2, 3, 4],
            logprobs=[0.0, 0.0, -0.2, -0.4],
            advantages=[0.0, 0.0, 1.0, -1.0],
            reward=1.0,
        )

        stats = collect_train_batch_stats([seq])

        self.assertEqual(stats["rewards"], [1.0])
        self.assertEqual(stats["logprobs"], [-0.2, -0.4])
        self.assertEqual(stats["advantages"], [1.0, -1.0])
        self.assertEqual(stats["prompt_len"], [2])
        self.assertEqual(stats["response_len"], [2])

    def test_rollout_stats_accumulator_keeps_skip_counters(self):
        """The mutable stats accumulator carries prompt-skip counters forward."""
        stats = init_rollout_stats(skipped_long=2, total_skipped_long=5)

        record_rollout_sequence_stats(stats, prefix_len=3, response_logprobs=[-1.0], response_len=1)

        self.assertEqual(stats["skipped_long"], 2)
        self.assertEqual(stats["total_skipped_long"], 5)
        self.assertEqual(stats["seq_len"], [4])
        self.assertEqual(stats["logprobs"], [-1.0])

    def test_metrics_recorder_close_is_idempotent_context_cleanup(self):
        """MetricsRecorder should close the writer exactly once."""

        class FakeWriter:
            def __init__(self):
                self.close_count = 0

            def close(self):
                self.close_count += 1

        writer = FakeWriter()
        old_factory = metrics_mod.create_tensorboard_writer
        metrics_mod.create_tensorboard_writer = lambda _log_dir: writer
        try:
            with MetricsRecorder("/tmp/areno-test") as recorder:
                self.assertIs(recorder._writer, writer)
            recorder.close()
        finally:
            metrics_mod.create_tensorboard_writer = old_factory

        self.assertEqual(writer.close_count, 1)


class MetricsAtomicWriteTest(unittest.TestCase):
    """Verify that MetricsRecorder uses atomic writes for dashboard state."""

    def _make_recorder(self, log_dir: str) -> MetricsRecorder:
        class FakeWriter:
            def __init__(self):
                self.close_count = 0

            def close(self):
                self.close_count += 1

            def add_scalar(self, *args, **kwargs):
                pass

            def flush(self):
                pass

        writer = FakeWriter()
        old_factory = metrics_mod.create_tensorboard_writer
        metrics_mod.create_tensorboard_writer = lambda _log_dir: writer
        try:
            return MetricsRecorder(log_dir)
        finally:
            metrics_mod.create_tensorboard_writer = old_factory

    def test_record_dashboard_state_uses_atomic_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self._make_recorder(tmp)
            recorder.record_dashboard_state(stage="train", step=1)
            state_file = Path(tmp) / f"dashboard_state.{os.getpid()}.json"
            self.assertTrue(state_file.exists())
            data = json.loads(state_file.read_text())
            self.assertEqual(data["stage"], "train")
            self.assertEqual(data["step"], 1)
            # No .tmp file left behind.
            self.assertFalse((Path(str(state_file) + ".tmp")).exists())
            recorder.close()

    def test_record_dashboard_state_cleans_temp_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self._make_recorder(tmp)
            state_file = Path(tmp) / f"dashboard_state.{os.getpid()}.json"
            tmp_file = Path(str(state_file) + ".tmp")

            from areno.cli import atomic_io as atomic_io_mod
            original = atomic_io_mod.atomic_write_text

            def fail_atomic_write(path, content, **kwargs):
                if str(path) == str(state_file):
                    raise OSError("simulated failure")
                return original(path, content, **kwargs)

            with mock.patch.object(atomic_io_mod, "atomic_write_text", fail_atomic_write):
                try:
                    recorder.record_dashboard_state(stage="train", step=1)
                except OSError:
                    pass

            # No .tmp file left behind even on failure.
            self.assertFalse(tmp_file.exists())
            recorder.close()


if __name__ == "__main__":
    unittest.main()
