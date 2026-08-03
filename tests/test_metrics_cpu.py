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


class TrainSequenceRewardComponentsTest(unittest.TestCase):
    """Tests for the reward_components field on TrainSequence."""

    def test_default_reward_components_is_empty_dict(self):
        """TrainSequence should default reward_components to an empty dict."""
        seq = TrainSequence()
        self.assertEqual(seq.reward_components, {})

    def test_reward_components_assigned_correctly(self):
        """reward_components should round-trip assigned values."""
        seq = TrainSequence(reward=0.85, reward_components={"correctness": 1.0, "format": 0.0})
        self.assertEqual(seq.reward_components["correctness"], 1.0)
        self.assertEqual(seq.reward_components["format"], 0.0)

    def test_reward_components_independent_per_instance(self):
        """Two TrainSequence instances should have independent reward_components."""
        seq1 = TrainSequence(reward_components={"a": 1.0})
        seq2 = TrainSequence(reward_components={"b": 2.0})
        self.assertNotEqual(seq1.reward_components, seq2.reward_components)
        self.assertNotIn("b", seq1.reward_components)


class CollectTrainBatchRewardComponentsTest(unittest.TestCase):
    """Tests for reward component collection in collect_train_batch_stats."""

    def test_stats_contain_reward_components_key(self):
        """init_rollout_stats should include a reward_components key."""
        stats = init_rollout_stats()
        self.assertIn("reward_components", stats)
        self.assertEqual(stats["reward_components"], {})

    def test_collect_gathers_reward_components(self):
        """collect_train_batch_stats should gather per-component values."""
        seq1 = TrainSequence(
            prompt_mask=[True, False],
            tokens=[1, 2],
            logprobs=[0.0, -0.1],
            advantages=[0.0, 1.0],
            reward=0.85,
            reward_components={"correctness": 1.0, "format": 0.0},
        )
        seq2 = TrainSequence(
            prompt_mask=[True, False],
            tokens=[1, 2],
            logprobs=[0.0, -0.2],
            advantages=[0.0, -1.0],
            reward=0.3,
            reward_components={"correctness": 0.0, "format": 1.0},
        )

        stats = collect_train_batch_stats([seq1, seq2])

        self.assertEqual(stats["reward_components"]["correctness"], [1.0, 0.0])
        self.assertEqual(stats["reward_components"]["format"], [0.0, 1.0])

    def test_collect_without_reward_components_produces_empty(self):
        """Sequences without reward_components should not create stats entries."""
        seq = TrainSequence(
            prompt_mask=[True, False],
            tokens=[1, 2],
            logprobs=[0.0, -0.1],
            advantages=[0.0, 1.0],
            reward=1.0,
        )

        stats = collect_train_batch_stats([seq])

        self.assertEqual(stats["reward_components"], {})

    def test_collect_mixed_composed_and_plain_sequences(self):
        """A mix of composed and plain sequences should only collect from composed ones."""
        seq_plain = TrainSequence(
            prompt_mask=[True, False],
            tokens=[1, 2],
            logprobs=[0.0, -0.1],
            advantages=[0.0, 1.0],
            reward=1.0,
        )
        seq_composed = TrainSequence(
            prompt_mask=[True, False],
            tokens=[1, 2],
            logprobs=[0.0, -0.2],
            advantages=[0.0, -1.0],
            reward=0.5,
            reward_components={"accuracy": 0.5},
        )

        stats = collect_train_batch_stats([seq_plain, seq_composed])

        self.assertEqual(stats["reward_components"], {"accuracy": [0.5]})


class RecordTrainingStatsRewardComponentsTest(unittest.TestCase):
    """Tests for reward component output in record_training_stats."""

    class _FakeWriter:
        def __init__(self):
            self.scalars = []

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

        def flush(self):
            pass

    def _make_stats_with_components(self):
        stats = init_rollout_stats()
        stats["rewards"] = [0.85, 0.3]
        stats["reward_components"] = {
            "correctness": [1.0, 0.0],
            "format": [0.0, 1.0],
        }
        return stats

    def test_outputs_reward_component_mean_and_std(self):
        """record_training_stats should write mean and std for each component."""
        writer = self._FakeWriter()
        stats = self._make_stats_with_components()
        train_res = {"loss": 0.5}

        from areno.api.metrics import record_training_stats

        record_training_stats(writer, stats, step=0, train_res=train_res, train_batch=[])

        tags = [tag for tag, _value, _step in writer.scalars]
        self.assertIn("rollout/reward_correctness_mean", tags)
        self.assertIn("rollout/reward_correctness_std", tags)
        self.assertIn("rollout/reward_format_mean", tags)
        self.assertIn("rollout/reward_format_std", tags)

    def test_outputs_correct_mean_values(self):
        """Component mean values should match hand calculation."""
        writer = self._FakeWriter()
        stats = self._make_stats_with_components()
        train_res = {}

        from areno.api.metrics import record_training_stats

        record_training_stats(writer, stats, step=0, train_res=train_res, train_batch=[])

        scalar_map = {tag: value for tag, value, _step in writer.scalars}
        # correctness: [1.0, 0.0] -> mean=0.5
        self.assertAlmostEqual(scalar_map["rollout/reward_correctness_mean"], 0.5, places=6)
        # format: [0.0, 1.0] -> mean=0.5
        self.assertAlmostEqual(scalar_map["rollout/reward_format_mean"], 0.5, places=6)

    def test_no_component_output_when_empty(self):
        """No component scalars should be written when reward_components is empty."""
        writer = self._FakeWriter()
        stats = init_rollout_stats()
        stats["rewards"] = [1.0]
        train_res = {}

        from areno.api.metrics import record_training_stats

        record_training_stats(writer, stats, step=0, train_res=train_res, train_batch=[])

        tags = [tag for tag, _value, _step in writer.scalars]
        component_tags = [t for t in tags if t.startswith("rollout/reward_") and t.endswith("_mean")]
        self.assertEqual(component_tags, [])


if __name__ == "__main__":
    unittest.main()
