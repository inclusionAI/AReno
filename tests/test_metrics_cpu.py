from __future__ import annotations

import unittest

from areno.api import metrics as metrics_mod
from areno.api.metrics import (
    MetricsRecorder,
    _effective_loss_token_count,
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


class EffectiveTrainableTokenTest(unittest.TestCase):
    """Effective-trainable-token accounting uses the loss mask, next-token shifted."""

    def test_effective_count_excludes_prompt_and_index_zero(self):
        """With no loss_mask, effective tokens are response positions past idx 0."""
        # prompt_mask=[T,T,F,F] -> response at idx 2,3; both count (idx0 skipped).
        count = _effective_loss_token_count([True, True, False, False], None)
        self.assertEqual(count, 2)

    def test_effective_count_honors_narrower_loss_mask(self):
        """loss_mask narrower than the response (e.g. masked tool-call span) drops those tokens."""
        # response at idx 2,3,4; loss_mask keeps only idx 2,3 -> 2 effective.
        count = _effective_loss_token_count([True, True, False, False, False], [False, False, True, True, False])
        self.assertEqual(count, 2)

    def test_effective_count_is_zero_for_all_prompt_sequence(self):
        """An all-prompt sequence contributes no loss targets."""
        count = _effective_loss_token_count([True, True, True], None)
        self.assertEqual(count, 0)

    def test_init_rollout_stats_carries_effective_token_keys(self):
        """The accumulator must expose the effective-token keys."""
        stats = init_rollout_stats()
        self.assertEqual(stats["effective_loss_tokens"], [])
        self.assertEqual(stats["total_input_tokens"], [])

    def test_collect_train_batch_stats_accounts_effective_and_total_tokens(self):
        """Default span (no loss_mask): effective = response tokens, total = sequence length."""
        seq = TrainSequence(
            prompt_mask=[True, True, False, False],
            tokens=[1, 2, 3, 4],
            logprobs=[0.0, 0.0, -0.2, -0.4],
            advantages=[0.0, 0.0, 1.0, -1.0],
            reward=1.0,
        )

        stats = collect_train_batch_stats([seq])

        self.assertEqual(stats["effective_loss_tokens"], [2])
        self.assertEqual(stats["total_input_tokens"], [4])

    def test_collect_train_batch_stats_loss_mask_narrows_effective_below_response_len(self):
        """A narrower loss_mask makes effective < response_len (the gap issue #227 fills)."""
        # response_len counts non-prompt positions (3); loss_mask drops idx 4 -> effective 2.
        seq = TrainSequence(
            prompt_mask=[True, True, False, False, False],
            loss_mask=[False, False, True, True, False],
            tokens=[1, 2, 3, 4, 5],
            logprobs=[0.0, 0.0, -0.2, -0.4, -0.6],
            advantages=[0.0, 0.0, 1.0, -1.0, 0.5],
            reward=1.0,
        )

        stats = collect_train_batch_stats([seq])

        self.assertEqual(stats["effective_loss_tokens"], [2])
        self.assertEqual(stats["response_len"], [3])
        self.assertTrue(stats["effective_loss_tokens"][0] < stats["response_len"][0])

    def test_collect_train_batch_stats_mean_effective_length_across_sequences(self):
        """Mean effective length is the per-sequence effective count averaged over the batch."""
        seq_a = TrainSequence(prompt_mask=[True, True, False, False], tokens=[1, 2, 3, 4], reward=1.0)
        # response at idx 3,4 (skip idx0) -> 2 effective.
        seq_b = TrainSequence(prompt_mask=[True, False, False, False, False], tokens=[1, 3, 4, 5, 6], reward=1.0)

        stats = collect_train_batch_stats([seq_a, seq_b])

        self.assertEqual(stats["effective_loss_tokens"], [2, 4])
        self.assertAlmostEqual(sum(stats["effective_loss_tokens"]) / len(stats["effective_loss_tokens"]), 3.0)

    def test_collect_train_batch_stats_empty_batch_keeps_empty_accumulators(self):
        """An empty batch must not raise and leaves the effective-token lists empty."""
        stats = collect_train_batch_stats([])

        self.assertEqual(stats["effective_loss_tokens"], [])
        self.assertEqual(stats["total_input_tokens"], [])

    def test_collect_train_batch_stats_clamps_length_on_trailing_token_mismatch(self):
        """tokens longer than prompt_mask must not index past the mask."""
        # prompt_mask covers 3 tokens, tokens has 5 -> total clamps to 3, effective over 3.
        seq = TrainSequence(
            prompt_mask=[True, False, False],
            tokens=[1, 2, 3, 4, 5],
            logprobs=[0.0, -0.2, -0.4],
            advantages=[0.0, 1.0, -1.0],
            reward=1.0,
        )

        stats = collect_train_batch_stats([seq])

        self.assertEqual(stats["total_input_tokens"], [3])
        # response at idx 1,2 (skip idx0) -> 2 effective.
        self.assertEqual(stats["effective_loss_tokens"], [2])

    def test_record_training_stats_emits_effective_token_scalars(self):
        """record_training_stats writes the four effective-token scalars for a real batch."""
        recorded: dict[str, float] = {}

        class FakeWriter:
            def add_scalar(self, tag, value, _step):
                recorded[tag] = float(value)

            def flush(self):
                pass

        seq = TrainSequence(
            prompt_mask=[True, True, False, False],
            loss_mask=[False, False, True, True],
            tokens=[1, 2, 3, 4],
            logprobs=[0.0, 0.0, -0.2, -0.4],
            advantages=[0.0, 0.0, 1.0, -1.0],
            reward=1.0,
        )
        stats = collect_train_batch_stats([seq])

        metrics_mod.record_training_stats(FakeWriter(), stats, step=1, train_res={}, train_batch=[seq])

        self.assertEqual(recorded["rollout/effective_loss_tokens"], 2.0)
        self.assertEqual(recorded["rollout/total_input_tokens"], 4.0)
        self.assertEqual(recorded["rollout/masked_tokens"], 2.0)
        self.assertEqual(recorded["rollout/effective_length_mean"], 2.0)

    def test_record_training_stats_empty_batch_skips_effective_token_scalars(self):
        """An empty batch records num_sequences=0 and skips the effective-token scalars (no 0/0)."""
        recorded: dict[str, float] = {}

        class FakeWriter:
            def add_scalar(self, tag, value, _step):
                recorded[tag] = float(value)

            def flush(self):
                pass

        stats = collect_train_batch_stats([])

        metrics_mod.record_training_stats(FakeWriter(), stats, step=1, train_res={}, train_batch=[])

        self.assertEqual(recorded["rollout/num_sequences"], 0.0)
        self.assertNotIn("rollout/effective_loss_tokens", recorded)
        self.assertNotIn("rollout/effective_length_mean", recorded)


if __name__ == "__main__":
    unittest.main()
