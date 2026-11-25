from __future__ import annotations

import math
import unittest

import torch

from areno.api.backend.common import TrainMetric
from areno.experimental.async_policy.losses import async_grpo_loss_fn, async_gspo_loss_fn


def _single_sequence_pack(old_logprobs, advantages):
    return {
        "packed_response_mask": torch.tensor([True] * len(old_logprobs)),
        "packed_seq_ids": torch.tensor([0] * len(old_logprobs)),
        "packed_num_sequences": 1,
        "packed_advantages": torch.tensor(advantages, dtype=torch.float32),
        "packed_logprobs": torch.tensor(old_logprobs, dtype=torch.float32),
    }


class AsyncGrpoLossTest(unittest.TestCase):
    """Async GRPO loss should importance-weight against rollout logprobs."""

    def test_async_grpo_loss_backpropagates(self):
        """Packed async GRPO should produce finite loss and gradients."""
        data_pack = {
            "packed_response_mask": torch.tensor([False, True, True]),
            "packed_seq_ids": torch.tensor([0, 0, 0]),
            "packed_num_sequences": 1,
            "packed_advantages": torch.tensor([0.0, 1.0, 1.0]),
            "packed_logprobs": torch.tensor([0.0, -0.3, -0.4]),
        }
        logprobs = torch.tensor([0.0, -0.2, -0.3], requires_grad=True)

        loss, stats = async_grpo_loss_fn(data_pack, logprobs)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logprobs.grad)
        self.assertTrue(torch.isfinite(logprobs.grad).all())
        self.assertAlmostEqual(float(loss.detach()), -math.exp(0.1), places=4)
        self.assertAlmostEqual(float(stats[TrainMetric.RATIO_MEAN]), math.exp(0.1), places=4)

    def test_async_grpo_loss_ratio_uses_rollout_logprobs(self):
        """Stale rollout logprobs must drive the importance ratio."""
        logprobs = torch.tensor([-0.2, -0.3], requires_grad=False)

        on_policy_pack = _single_sequence_pack([-0.2, -0.3], [1.0, 1.0])
        stale_pack = _single_sequence_pack([-0.3, -0.4], [1.0, 1.0])

        on_policy_loss, _ = async_grpo_loss_fn(on_policy_pack, logprobs)
        stale_loss, _ = async_grpo_loss_fn(stale_pack, logprobs)

        self.assertAlmostEqual(float(on_policy_loss), -1.0, places=6)
        self.assertAlmostEqual(float(stale_loss), -math.exp(0.1), places=4)

    def test_async_grpo_loss_clips_upward_stale_ratio(self):
        """Ratios above 1 + clip_eps on positive advantages use the clip."""
        data_pack = _single_sequence_pack([-0.7], [1.0])
        logprobs = torch.tensor([-0.2], requires_grad=False)

        loss, stats = async_grpo_loss_fn(data_pack, logprobs, clip_eps=0.2)

        self.assertAlmostEqual(float(loss), -1.2, places=6)
        self.assertAlmostEqual(float(stats["pg_clipfrac"]), 1.0, places=6)

    def test_async_grpo_loss_clips_downward_stale_ratio(self):
        """Ratios below 1 - clip_eps on negative advantages use the clip."""
        data_pack = _single_sequence_pack([-0.7], [-1.0])
        logprobs = torch.tensor([-1.2], requires_grad=False)

        loss, stats = async_grpo_loss_fn(data_pack, logprobs, clip_eps=0.2)

        self.assertAlmostEqual(float(loss), 0.8, places=6)
        self.assertAlmostEqual(float(stats["pg_clipfrac"]), 1.0, places=6)

    def test_async_grpo_loss_mask_blocks_masked_response_gradients(self):
        """Masked response tokens must not receive gradients."""
        data_pack = {
            "packed_response_mask": torch.tensor([False, True, False]),
            "packed_seq_ids": torch.tensor([0, 0, 0]),
            "packed_num_sequences": 1,
            "packed_advantages": torch.tensor([0.0, 1.0, 100.0]),
            "packed_logprobs": torch.tensor([0.0, -0.3, -0.4]),
        }
        logprobs = torch.tensor([0.0, -0.2, -0.3], requires_grad=True)

        loss, _ = async_grpo_loss_fn(data_pack, logprobs)
        loss.backward()

        self.assertEqual(float(logprobs.grad[0]), 0.0)
        self.assertEqual(float(logprobs.grad[2]), 0.0)
        self.assertNotEqual(float(logprobs.grad[1]), 0.0)


class AsyncGspoLossTest(unittest.TestCase):
    """Async GSPO loss should importance-weight at sequence granularity."""

    def test_async_gspo_loss_on_policy_matches_grouped_advantage(self):
        """With fresh rollout logprobs the sequence ratio is exactly one."""
        data_pack = {
            "packed_response_mask": torch.tensor([True, True, True]),
            "packed_seq_ids": torch.tensor([0, 0, 1]),
            "packed_num_sequences": 2,
            "packed_advantages": torch.tensor([1.0, 1.0, -0.5]),
            "packed_logprobs": torch.tensor([-0.1, -0.2, -0.3]),
        }
        logprobs = torch.tensor([-0.1, -0.2, -0.3], requires_grad=False)

        loss, stats = async_gspo_loss_fn(data_pack, logprobs)

        self.assertAlmostEqual(float(loss), -0.25, places=6)
        self.assertAlmostEqual(float(stats[TrainMetric.RATIO_MEAN]), 1.0, places=6)
        self.assertAlmostEqual(float(stats["pg_clipfrac"]), 0.0, places=6)

    def test_async_gspo_loss_uses_sequence_level_stale_ratio(self):
        """Stale rollouts shift the length-normalized sequence ratio."""
        data_pack = {
            "packed_response_mask": torch.tensor([True, True, True]),
            "packed_seq_ids": torch.tensor([0, 0, 1]),
            "packed_num_sequences": 2,
            "packed_advantages": torch.tensor([1.0, 1.0, -0.5]),
            "packed_logprobs": torch.tensor([-0.6, -0.7, -0.8]),
        }
        logprobs = torch.tensor([-0.1, -0.2, -0.3], requires_grad=False)

        loss, stats = async_gspo_loss_fn(data_pack, logprobs, clip_eps=3e-4)

        ratio = math.exp(0.5)
        seq0 = min(ratio * 1.0, (1.0 + 3e-4) * 1.0)
        seq1 = min(ratio * -0.5, (1.0 - 3e-4) * -0.5)
        expected = -(seq0 + seq1) / 2.0
        self.assertAlmostEqual(float(loss), expected, places=6)
        # seq0 clips at 1 + clip_eps; seq1 keeps the unclipped stale ratio.
        self.assertAlmostEqual(float(stats["pg_clipfrac"]), 0.5, places=6)

    def test_async_gspo_loss_backpropagates(self):
        """Packed async GSPO should produce finite loss and gradients."""
        data_pack = {
            "packed_response_mask": torch.tensor([True, True, True]),
            "packed_seq_ids": torch.tensor([0, 0, 1]),
            "packed_num_sequences": 2,
            "packed_advantages": torch.tensor([1.0, 1.0, -1.0]),
            "packed_logprobs": torch.tensor([-0.1, -0.2, -0.3]),
        }
        logprobs = torch.tensor([-0.2, -0.3, -0.4], requires_grad=True)

        loss, _ = async_gspo_loss_fn(data_pack, logprobs)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logprobs.grad)
        self.assertTrue(torch.isfinite(logprobs.grad).all())


if __name__ == "__main__":
    unittest.main()
