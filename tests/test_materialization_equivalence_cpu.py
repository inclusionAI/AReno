"""CPU equivalence tests for the policy-only materialization refactor.

The rollout->train materialization path in ``PolicyOnlyTrainer`` was changed
to (a) decode completions with one batched tokenizer call, (b) reuse the
per-sample prefix/response lists across the reward record and the
``TrainSequence``, (c) compute group-relative advantages with one vectorized
numpy pass, and (d) amortize rollout-logprob statistics instead of keeping one
float per response token. These tests pin the behavior-preservation contract:
the vectorized advantage math agrees with the per-group reference, the
amortized statistics reproduce the exact mean, and batched decoding matches
per-completion decoding.
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from areno.api.advantages import compute_batch_group_advantages
from areno.api.rewards import compute_group_advantages
from areno.api.trainers.policy_only import _batch_decode, _LogprobStats, _rollout_logprob_mean


class _FakeTokenizerBase:
    """Tiny tokenizer stub used to exercise the decode call sites."""

    eos_token_id = 0

    def decode(self, tokens) -> str:
        return "".join(f"t{int(token)}" for token in tokens)


class _FakeTokenizer(_FakeTokenizerBase):
    def batch_decode(self, token_lists) -> list[str]:
        return [self.decode(tokens) for tokens in token_lists]


class _FakeTokenizerNoBatch(_FakeTokenizerBase):
    """Tokenizer without a batch_decode entry point (fallback path)."""


class BatchGroupAdvantagesTest(unittest.TestCase):
    """Vectorized group advantages must match the per-group reference."""

    def test_matches_per_group_reference(self):
        rng = random.Random(0)
        for group_sizes in ([4], [2, 2], [3, 5, 2], [1, 1, 1], [6, 4, 3, 2]):
            rewards = [rng.uniform(-1.0, 1.0) for _ in range(sum(group_sizes))]
            expected: list[float] = []
            offset = 0
            for size in group_sizes:
                expected.extend(compute_group_advantages(rewards[offset : offset + size]))
                offset += size
            actual = compute_batch_group_advantages(rewards, group_sizes)
            self.assertEqual(len(actual), len(rewards))
            np.testing.assert_allclose(
                np.asarray(actual, dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_constant_reward_groups(self):
        """Groups with identical rewards must produce zero advantages."""

        actual = compute_batch_group_advantages([3.0, 3.0, 3.0, 5.0, 5.0], [3, 2])
        np.testing.assert_allclose(actual, np.zeros(5), atol=1e-7)

    def test_empty_groups(self):
        self.assertEqual(compute_batch_group_advantages([], []), [])

    def test_raises_on_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            compute_batch_group_advantages([1.0, 2.0], [3])
        with self.assertRaises(ValueError):
            compute_batch_group_advantages([1.0, 2.0], [0, 2])


class LogprobStatsTest(unittest.TestCase):
    """Amortized statistics must reproduce the exact list mean."""

    def test_mean_matches_exact(self):
        stats = _LogprobStats()
        values = [0.1, -0.2, 0.3, 0.0, -0.5, 0.7]
        stats.add(values)
        self.assertAlmostEqual(float(stats.mean), float(np.mean(values)), places=12)
        self.assertTrue(stats)

    def test_incremental_adds(self):
        stats = _LogprobStats()
        stats.add([1.0, 2.0])
        stats.add([3.0])
        self.assertAlmostEqual(float(stats.mean), 2.0, places=12)
        self.assertEqual(stats._count, 3)

    def test_empty_stats(self):
        stats = _LogprobStats()
        self.assertFalse(stats)
        self.assertIsNone(stats.mean)

    def test_rollout_logprob_mean_handles_list_and_stats(self):
        values = [0.1, -0.2, 0.3]
        self.assertAlmostEqual(float(_rollout_logprob_mean(list(values))), float(np.mean(values)), places=12)
        stats = _LogprobStats()
        stats.add(values)
        self.assertAlmostEqual(float(_rollout_logprob_mean(stats)), float(np.mean(values)), places=12)
        self.assertIsNone(_rollout_logprob_mean(None))
        self.assertIsNone(_rollout_logprob_mean([]))


class BatchDecodeTest(unittest.TestCase):
    """Batched decoding must equal per-completion decoding."""

    def test_batch_decode_matches_per_item(self):
        tokenizer = _FakeTokenizer()
        token_lists = [[1, 2, 3], [4], [5, 6]]
        batched = _batch_decode(tokenizer, token_lists)
        expected = [tokenizer.decode(tokens) for tokens in token_lists]
        self.assertEqual(batched, expected)

    def test_fallback_without_batch_decode(self):
        tokenizer = _FakeTokenizerNoBatch()
        token_lists = [[1, 2], [3, 4, 5]]
        self.assertEqual(_batch_decode(tokenizer, token_lists), ["t1t2", "t3t4t5"])


if __name__ == "__main__":
    unittest.main()
