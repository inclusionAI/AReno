from __future__ import annotations

import unittest

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

    def test_close_calls_flush_before_close(self):
        """close() should call flush() before close() on the underlying writer."""

        class FakeWriter:
            def __init__(self):
                self.call_order: list[str] = []
                self.flush_count = 0
                self.close_count = 0

            def flush(self):
                self.call_order.append("flush")
                self.flush_count += 1

            def close(self):
                self.call_order.append("close")
                self.close_count += 1

        writer = FakeWriter()
        old_factory = metrics_mod.create_tensorboard_writer
        metrics_mod.create_tensorboard_writer = lambda _log_dir: writer
        try:
            recorder = MetricsRecorder("/tmp/areno-test")
            recorder.close()
        finally:
            metrics_mod.create_tensorboard_writer = old_factory

        self.assertEqual(writer.flush_count, 1)
        self.assertEqual(writer.close_count, 1)
        self.assertEqual(writer.call_order, ["flush", "close"])

    def test_close_survives_flush_failure(self):
        """close() should still call writer.close() even if flush() raises."""

        class FakeWriter:
            def __init__(self):
                self.close_count = 0
                self.flush_called = False

            def flush(self):
                self.flush_called = True
                raise OSError("disk full")

            def close(self):
                self.close_count += 1

        writer = FakeWriter()
        old_factory = metrics_mod.create_tensorboard_writer
        metrics_mod.create_tensorboard_writer = lambda _log_dir: writer
        try:
            recorder = MetricsRecorder("/tmp/areno-test")
            # Should not raise despite flush failure.
            recorder.close()
        finally:
            metrics_mod.create_tensorboard_writer = old_factory

        self.assertTrue(writer.flush_called)
        self.assertEqual(writer.close_count, 1)

    def test_close_idempotent_with_flush(self):
        """Multiple close() calls should only flush and close the writer once."""

        class FakeWriter:
            def __init__(self):
                self.flush_count = 0
                self.close_count = 0

            def flush(self):
                self.flush_count += 1

            def close(self):
                self.close_count += 1

        writer = FakeWriter()
        old_factory = metrics_mod.create_tensorboard_writer
        metrics_mod.create_tensorboard_writer = lambda _log_dir: writer
        try:
            recorder = MetricsRecorder("/tmp/areno-test")
            recorder.close()
            recorder.close()
            recorder.close()
        finally:
            metrics_mod.create_tensorboard_writer = old_factory

        self.assertEqual(writer.flush_count, 1)
        self.assertEqual(writer.close_count, 1)


if __name__ == "__main__":
    unittest.main()
