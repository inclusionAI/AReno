from __future__ import annotations

import unittest

import torch

from areno.api.loss_fns.dpo import dpo_loss_fn
from areno.api.loss_fns.sft import sft_loss_fn


class SftLossTest(unittest.TestCase):
    """SFT loss tests cover padded and packed response masking."""

    def test_sft_padded_loss_ignores_prompt_tokens(self):
        """Padded SFT should train only positions after the prompt."""
        data_pack = {"prompt_mask": torch.tensor([[True, True, False, False]])}
        logprobs = torch.tensor([[-5.0, -0.25, -0.75]], requires_grad=True)

        loss, stats = sft_loss_fn(data_pack, logprobs)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.5, places=6)
        self.assertEqual(float(stats["sft_target_tokens"]), 2.0)
        self.assertIsNotNone(logprobs.grad)
        self.assertEqual(float(logprobs.grad[0, 0]), 0.0)

    def test_sft_padded_loss_honors_loss_mask(self):
        """Padded SFT should not train response tokens masked by loss_mask."""
        data_pack = {
            "prompt_mask": torch.tensor([[True, False, False, False]]),
            "loss_mask": torch.tensor([[False, True, False, True]]),
        }
        logprobs = torch.tensor([[-1.0, -100.0, -3.0]], requires_grad=True)

        loss, stats = sft_loss_fn(data_pack, logprobs)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 2.0, places=6)
        self.assertEqual(float(stats["sft_target_tokens"]), 2.0)
        self.assertIsNotNone(logprobs.grad)
        self.assertAlmostEqual(float(logprobs.grad[0, 0]), -0.5, places=6)
        self.assertEqual(float(logprobs.grad[0, 1]), 0.0)
        self.assertAlmostEqual(float(logprobs.grad[0, 2]), -0.5, places=6)

    def test_sft_packed_loss_uses_response_mask(self):
        """Packed SFT should use the provided flattened response mask exactly."""
        data_pack = {"packed_response_mask": torch.tensor([False, True, True])}
        logprobs = torch.tensor([-9.0, -0.2, -0.4], requires_grad=True)

        loss, stats = sft_loss_fn(data_pack, logprobs)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.3, places=6)
        self.assertEqual(float(stats["sft_target_tokens"]), 2.0)
        self.assertEqual(float(logprobs.grad[0]), 0.0)

    def test_sft_padded_and_packed_loss_masks_agree(self):
        """Padded loss_mask and packed_response_mask should select the same tokens."""
        padded_pack = {
            "prompt_mask": torch.tensor([[True, False, False, False]]),
            "loss_mask": torch.tensor([[False, True, False, True]]),
        }
        packed_pack = {"packed_response_mask": torch.tensor([True, False, True])}
        padded_logprobs = torch.tensor([[-1.0, -100.0, -3.0]])
        packed_logprobs = torch.tensor([-1.0, -100.0, -3.0])

        padded_loss, padded_stats = sft_loss_fn(padded_pack, padded_logprobs)
        packed_loss, packed_stats = sft_loss_fn(packed_pack, packed_logprobs)

        self.assertAlmostEqual(float(padded_loss), float(packed_loss), places=6)
        self.assertEqual(float(padded_stats["sft_target_tokens"]), float(packed_stats["sft_target_tokens"]))
        self.assertAlmostEqual(
            float(padded_stats["sft_logprob_mean"]), float(packed_stats["sft_logprob_mean"]), places=6
        )

    def test_sft_accumulation_uses_global_token_mean_when_annotated(self):
        """SFT accumulation should weight microbatches by target-token count."""
        short_logprobs = torch.tensor([-1.0], requires_grad=True)
        long_logprobs = torch.tensor([-3.0, -3.0, -3.0], requires_grad=True)

        short_loss, _ = sft_loss_fn({"packed_response_mask": torch.tensor([True])}, short_logprobs)
        long_loss, _ = sft_loss_fn({"packed_response_mask": torch.tensor([True, True, True])}, long_logprobs)
        microbatch_mean = (short_loss + long_loss) / 2

        annotated_short_loss, _ = sft_loss_fn(
            {
                "packed_response_mask": torch.tensor([True]),
                "_sft_total_target_tokens": 4,
                "_sft_grad_scale": 2,
            },
            short_logprobs,
        )
        annotated_long_loss, _ = sft_loss_fn(
            {
                "packed_response_mask": torch.tensor([True, True, True]),
                "_sft_total_target_tokens": 4,
                "_sft_grad_scale": 2,
            },
            long_logprobs,
        )
        global_token_mean = (annotated_short_loss + annotated_long_loss) / 2

        self.assertAlmostEqual(float(microbatch_mean.detach()), 2.0, places=6)
        self.assertAlmostEqual(float(global_token_mean.detach()), 2.5, places=6)

    def test_sft_backend_annotations_group_target_tokens(self):
        """Backend SFT annotations describe one optimizer-step accumulation group."""
        from areno.api.backend.areno.backend import _annotate_sft_token_mean_packs

        packs = [{}, {}]
        _annotate_sft_token_mean_packs(packs, [1, 3], gradient_accumulation_steps=None)

        self.assertEqual(packs[0]["_sft_total_target_tokens"], 4)
        self.assertEqual(packs[1]["_sft_total_target_tokens"], 4)
        self.assertEqual(packs[0]["_sft_grad_scale"], 2)
        self.assertEqual(packs[1]["_sft_grad_scale"], 2)


class DpoLossTest(unittest.TestCase):
    """DPO tests validate pair ordering and packed sequence aggregation."""

    def test_dpo_padded_loss_prefers_chosen_response(self):
        """Chosen/rejected rows are paired by order in padded DPO batches."""
        data_pack = {
            "prompt_mask": torch.tensor(
                [
                    [True, False, False],
                    [True, False, False],
                ]
            ),
            "ref_logprobs": torch.tensor(
                [
                    [0.0, -0.2, -0.2],
                    [0.0, -0.2, -0.2],
                ]
            ),
        }
        logprobs = torch.tensor([[-0.1, -0.1], [-1.0, -1.0]], requires_grad=True)

        loss, stats = dpo_loss_fn(data_pack, logprobs, beta=1.0)
        loss.backward()

        self.assertLess(float(loss.detach()), 0.7)
        self.assertEqual(float(stats["dpo_accuracy"]), 1.0)
        self.assertGreater(float(stats["dpo_margin"]), 0.0)
        self.assertIsNotNone(logprobs.grad)

    def test_dpo_packed_loss_rejects_odd_sequence_count(self):
        """Packed DPO cannot split an odd number of sequences into pairs."""
        data_pack = {
            "packed_response_mask": torch.tensor([True]),
            "packed_seq_ids": torch.tensor([0]),
            "packed_num_sequences": 1,
            "packed_ref_logprobs": torch.tensor([-0.1]),
        }
        logprobs = torch.tensor([-0.1], requires_grad=True)

        with self.assertRaisesRegex(ValueError, "even number"):
            dpo_loss_fn(data_pack, logprobs)

    def test_dpo_packed_loss_accumulates_tokens_per_sequence(self):
        """Packed DPO should sum token logprobs back into per-sequence scores."""
        data_pack = {
            "packed_response_mask": torch.tensor([True, True, True, True]),
            "packed_seq_ids": torch.tensor([0, 0, 1, 1]),
            "packed_num_sequences": 2,
            "packed_ref_logprobs": torch.tensor([-0.2, -0.2, -0.2, -0.2]),
        }
        logprobs = torch.tensor([-0.1, -0.1, -0.8, -0.8], requires_grad=True)

        loss, stats = dpo_loss_fn(data_pack, logprobs, beta=1.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(stats["dpo_response_len"]), 2.0)
        self.assertEqual(float(stats["dpo_accuracy"]), 1.0)


if __name__ == "__main__":
    unittest.main()
