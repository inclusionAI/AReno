from __future__ import annotations

import unittest

import torch

from areno.engine.data import SamplingParams
from areno.engine.data import sampling


class SamplingTest(unittest.TestCase):
    """Sampling tests cover CPU-only filtering and stop-token helpers."""

    def test_greedy_respects_min_new_tokens_and_suppression(self):
        """Greedy sampling should mask EOS and suppressed ids before argmax."""
        logits = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
        params = SamplingParams(temperature=0.0, min_new_tokens=2, suppress_token_ids=(1,))

        token = sampling._sample(logits, params, torch.device("cpu"), eos_token_id=0, sample_step=1)

        self.assertEqual(token.tolist(), [2])

    def test_stochastic_sampling_is_seeded(self):
        """A fixed sampling seed should produce reproducible token draws."""
        logits = torch.tensor([[0.1, 0.2, 3.0, 0.0]])
        params = SamplingParams(temperature=0.8, top_k=2, seed=123)
        gen1 = sampling._make_sample_generator(params, torch.device("cpu"))
        gen2 = sampling._make_sample_generator(params, torch.device("cpu"))

        sample1 = sampling._sample(logits, params, torch.device("cpu"), generator=gen1)
        sample2 = sampling._sample(logits, params, torch.device("cpu"), generator=gen2)

        self.assertEqual(sample1.tolist(), sample2.tolist())

    def test_top_k_limits_candidates(self):
        """Top-k filtering should prevent lower-ranked tokens from sampling."""
        logits = torch.tensor([[9.0, 8.0, 1.0, 0.0]])
        params = SamplingParams(temperature=1.0, top_k=2, seed=0)
        gen = sampling._make_sample_generator(params, torch.device("cpu"))

        sample = sampling._sample(logits, params, torch.device("cpu"), generator=gen)

        self.assertIn(sample.item(), {0, 1})

    def test_degenerate_probs_fall_back_to_argmax(self):
        """All-zero/invalid probability rows should become one-hot fallbacks."""
        probs = torch.zeros(2, 4)
        fallback = torch.tensor([2, 1])

        sanitized = sampling._sanitize_probs(probs, fallback)

        self.assertTrue(torch.equal(sanitized.argmax(dim=-1), fallback))
        self.assertTrue(torch.allclose(sanitized.sum(dim=-1), torch.ones(2)))

    def test_stop_tokens_are_deduped_and_truncation_keeps_stop(self):
        """Stop-token handling should dedupe ids and keep the matched stop token."""
        params = SamplingParams(stop_token_ids=(2, 3))
        stop_ids = sampling._stop_token_ids(params, eos_token_id=(1, 2))

        rows, reasons = sampling._truncate_generated([[4, 2, 5], [6, 7]], stop_ids)

        self.assertEqual(stop_ids, (1, 2, 3))
        self.assertEqual(rows, [[4, 2], [6, 7]])
        self.assertEqual(reasons, ["stop", "length"])

    def test_tokens_match_any_handles_empty_and_multiple_ids(self):
        """Token matching should handle empty stop sets and multiple ids."""
        tokens = torch.tensor([[1, 2, 3], [4, 5, 1]])

        empty = sampling._tokens_match_any(tokens, ())
        matched = sampling._tokens_match_any(tokens, (1, 5))

        self.assertFalse(empty.any())
        self.assertEqual(matched.tolist(), [[True, False, False], [False, True, True]])


if __name__ == "__main__":
    unittest.main()
