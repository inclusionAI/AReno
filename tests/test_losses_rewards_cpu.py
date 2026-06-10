from __future__ import annotations

import unittest

import torch

from areno.api.advantages import _compute_gae_python
from areno.api.rewards import compute_group_advantages
from areno.api.loss_fns.grpo import grpo_loss_fn
from areno.api.loss_fns.ppo import _kl_penalty, ppo_loss_fn


class AdvantageAndRewardTest(unittest.TestCase):
    """Reward and advantage tests validate normalization and GAE fallback math."""

    def test_compute_group_advantages_normalizes_rewards(self):
        """Group rewards should standardize to zero mean and unit variance."""
        advantages = compute_group_advantages([1.0, 2.0, 3.0])

        tensor = torch.tensor(advantages)
        self.assertAlmostEqual(float(tensor.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(tensor.std(unbiased=False)), 1.0, places=6)

    def test_compute_group_advantages_handles_constant_rewards(self):
        """Constant reward groups should produce zero advantages."""
        advantages = compute_group_advantages([3.0, 3.0, 3.0])

        self.assertEqual(advantages, [0.0, 0.0, 0.0])

    def test_compute_gae_python_matches_manual_terminal_reward(self):
        """The Python GAE fallback should match hand-computed terminal returns."""
        advantages, returns = _compute_gae_python([0.0, 1.0], [0.2, 0.3], gamma=1.0, lam=1.0)

        self.assertEqual(len(advantages), 2)
        self.assertAlmostEqual(advantages[1], 0.7, places=6)
        self.assertAlmostEqual(advantages[0], 0.8, places=6)
        self.assertAlmostEqual(returns[0], 1.0, places=6)
        self.assertAlmostEqual(returns[1], 1.0, places=6)

    def test_compute_gae_rejects_mismatched_lengths(self):
        """GAE requires one critic value for every reward timestep."""
        with self.assertRaisesRegex(ValueError, "one value per reward"):
            _compute_gae_python([1.0, 2.0], [0.0], gamma=1.0, lam=1.0)


class LossFunctionTest(unittest.TestCase):
    """GRPO/PPO loss tests cover padded and packed response layouts."""

    def test_grpo_padded_loss_backpropagates(self):
        """Padded GRPO should produce finite loss and gradients."""
        data_pack = {
            "prompt_mask": torch.tensor([[True, True, False, False]]),
            "advantages": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "logprobs": torch.tensor([[0.0, 0.0, -0.3, -0.4]]),
        }
        logprobs = torch.tensor([[0.0, -0.2, -0.3]], requires_grad=True)

        loss, stats = grpo_loss_fn(data_pack, logprobs)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logprobs.grad)
        self.assertIn("policy_loss", stats)

    def test_grpo_packed_loss_groups_tokens_by_sequence(self):
        """Packed GRPO should aggregate response tokens by sequence id."""
        data_pack = {
            "packed_response_mask": torch.tensor([True, True, True]),
            "packed_seq_ids": torch.tensor([0, 0, 1]),
            "packed_num_sequences": 2,
            "packed_advantages": torch.tensor([1.0, 1.0, -1.0]),
            "packed_logprobs": torch.tensor([-0.1, -0.2, -0.3]),
        }
        logprobs = torch.tensor([-0.1, -0.2, -0.3], requires_grad=True)

        loss, stats = grpo_loss_fn(data_pack, logprobs)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logprobs.grad)
        self.assertEqual(float(stats["response_len"]), 1.5)

    def test_ppo_padded_loss_reports_kl_and_clip_stats(self):
        """Padded PPO should emit KL and clipping diagnostics."""
        data_pack = {
            "prompt_mask": torch.tensor([[True, True, False, False]]),
            "advantages": torch.tensor([[0.0, 0.0, 1.0, -1.0]]),
            "logprobs": torch.tensor([[0.0, 0.0, -0.3, -0.4]]),
            "ref_logprobs": torch.tensor([[0.0, 0.0, -0.25, -0.35]]),
        }
        logprobs = torch.tensor([[0.0, -0.2, -0.45]], requires_grad=True)

        loss, stats = ppo_loss_fn(data_pack, logprobs, use_kl_loss=True, kl_loss_coef=0.01)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logprobs.grad)
        self.assertIn("pg_clipfrac", stats)
        self.assertIn("ppo_kl", stats)

    def test_ppo_packed_loss_reports_ratio_stats(self):
        """Packed PPO should still report policy ratio metrics."""
        data_pack = {
            "packed_response_mask": torch.tensor([True, True]),
            "packed_logprobs": torch.tensor([-0.2, -0.4]),
            "packed_ref_logprobs": torch.tensor([-0.25, -0.45]),
            "packed_advantages": torch.tensor([1.0, -1.0]),
        }
        logprobs = torch.tensor([-0.1, -0.5], requires_grad=True)

        loss, stats = ppo_loss_fn(data_pack, logprobs, use_kl_loss=True, kl_loss_coef=0.01)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logprobs.grad)
        self.assertIn("ratio_mean", stats)

    def test_kl_penalty_variants_are_finite(self):
        """Supported KL penalty variants should be finite on simple inputs."""
        logprob = torch.tensor([-0.2, -0.5])
        ref = torch.tensor([-0.3, -0.4])

        for name in ("kl", "abs", "mse", "low_var_kl"):
            value = _kl_penalty(logprob, ref, name)
            self.assertTrue(torch.isfinite(value).all())
        with self.assertRaises(NotImplementedError):
            _kl_penalty(logprob, ref, "unknown")


if __name__ == "__main__":
    unittest.main()
