from __future__ import annotations

import unittest

from areno.api import metrics as metrics_mod
from areno.api.metrics import (
    MetricsRecorder,
    collect_train_batch_stats,
    init_rollout_stats,
    record_rollout_sequence_stats,
    record_training_stats,
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


class RewardComponentMetricsTest(unittest.TestCase):
    """Per-component reward keys flow through record_training_stats to TensorBoard."""

    def test_reward_component_keys_written_under_train_namespace(self):
        """`reward/<name>_mean` and `reward/<name>_invalid_count` in train_res land as
        `train/reward/<name>_*` scalars — the channel the trainer relies on without a
        dedicated writer branch."""

        class FakeWriter:
            def __init__(self):
                self.scalars: dict[str, tuple] = {}

            def add_scalar(self, tag, value, step):
                self.scalars[tag] = (value, step)

            def flush(self):
                pass

            def close(self):
                pass

        writer = FakeWriter()
        stats = init_rollout_stats()
        stats["rewards"] = [1.0, 0.0, 1.0]
        train_res = {
            "loss": 0.5,
            "reward/accuracy_reward_mean": 1.0,
            "reward/format_reward_mean": 0.0,
            "reward/format_reward_invalid_count": 2.0,
        }
        record_training_stats(writer, stats, step=3, train_res=train_res, train_batch=[], timings=None)

        # Component means and invalid counts mirror the train_res keys under `train/`.
        self.assertEqual(writer.scalars["train/reward/accuracy_reward_mean"], (1.0, 3))
        self.assertEqual(writer.scalars["train/reward/format_reward_mean"], (0.0, 3))
        self.assertEqual(writer.scalars["train/reward/format_reward_invalid_count"], (2.0, 3))
        # The ordinary backend metric and the batch-derived rollout mean still flow.
        self.assertEqual(writer.scalars["train/loss"], (0.5, 3))
        self.assertIn("rollout/rewards_mean", writer.scalars)


if __name__ == "__main__":
    unittest.main()
