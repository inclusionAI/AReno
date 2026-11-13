from __future__ import annotations

import unittest

import torch

from areno.api.algorithms import opd_loss_fn as registered_opd_loss
from areno.api.backend.cuda.losses import opd_loss_fn
from areno.api.backend.cuda.training import annotate_opd_token_mean_packs


def _pack(*, mask, old, teacher):
    return {
        "packed_response_mask": torch.tensor(mask, dtype=torch.float32),
        "packed_logprobs": torch.tensor(old, dtype=torch.float32),
        "packed_ref_logprobs": torch.tensor(teacher, dtype=torch.float32),
        "packed_seq_ids": torch.zeros(len(mask), dtype=torch.long),
        "packed_num_sequences": 1,
    }


class OpdLossTest(unittest.TestCase):
    """OPD loss is pure tensor math; these tests never touch a CUDA device."""

    def test_registry_wrapper_is_a_distinct_callable(self):
        self.assertTrue(callable(registered_opd_loss))
        self.assertIsNot(registered_opd_loss, opd_loss_fn)

    def test_pg_zero_signal_gives_zero_loss_and_zero_gradient(self):
        pack = _pack(mask=[0, 1, 1, 0], old=[0, -1, -2, 0], teacher=[0, -1, -2, 0])
        logprobs = torch.tensor([0.0, -1.0, -2.0, 0.0], requires_grad=True)

        loss, stats = opd_loss_fn(pack, logprobs)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertAlmostEqual(stats["opd_advantage_mean"].item(), 0.0, places=6)
        loss.backward()
        self.assertTrue(torch.allclose(logprobs.grad, torch.zeros(4)))

    def test_pg_loss_value_and_gradient_direction(self):
        # teacher is more confident than the rollout policy on both response
        # tokens, so the student logprobs should be pushed upwards.
        pack = _pack(mask=[0, 1, 1, 0], old=[0, -1, -2, 0], teacher=[0, -0.5, -1.5, 0])
        logprobs = torch.tensor([0.0, -1.0, -2.0, 0.0], requires_grad=True)

        loss, stats = opd_loss_fn(pack, logprobs)

        # advantage = [0, 0.5, 0.5, 0]; masked_mean divides by valid_count=2,
        # so loss = -(0.5*-1 + 0.5*-2)/2 = 0.75
        self.assertAlmostEqual(loss.item(), 0.75, places=6)
        loss.backward()
        self.assertAlmostEqual(logprobs.grad[1].item(), -0.25, places=6)
        self.assertAlmostEqual(logprobs.grad[2].item(), -0.25, places=6)
        self.assertEqual(logprobs.grad[0].item(), 0.0)
        self.assertEqual(logprobs.grad[3].item(), 0.0)

    def test_pg_prompt_and_padding_do_not_affect_loss(self):
        base = _pack(mask=[0, 1, 1, 0], old=[0, -1, -2, 0], teacher=[0, -0.5, -1.5, 0])
        noisy = _pack(mask=[0, 1, 1, 0], old=[999, -1, -2, -999], teacher=[-999, -0.5, -1.5, 999])
        student = torch.tensor([0.0, -1.0, -2.0, 0.0], requires_grad=True)

        loss_a, _ = opd_loss_fn(base, student)
        loss_b, _ = opd_loss_fn(noisy, student)

        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=6)

    def test_clipfrac_is_mask_weighted_not_plain_mean(self):
        # Only one of the two response tokens is clipped. A plain .mean() over
        # all four positions would report 0.25 instead of 0.5.
        pack = _pack(mask=[0, 1, 1, 0], old=[0, -1, -2, 0], teacher=[0, 100, -2, 0])
        logprobs = torch.tensor([0.0, -1.0, -2.0, 0.0], requires_grad=True)

        _, stats = opd_loss_fn(pack, logprobs, opd_clip=5.0)

        self.assertAlmostEqual(stats["opd_clipfrac"].item(), 0.5, places=6)

    def test_pg_clip_bounds_advantage(self):
        pack = _pack(mask=[0, 1, 0, 0], old=[0, -1, 0, 0], teacher=[0, 100, 0, 0])
        logprobs = torch.tensor([0.0, -1.0, 0.0, 0.0], requires_grad=True)

        _, stats = opd_loss_fn(pack, logprobs, opd_clip=5.0)

        self.assertAlmostEqual(stats["opd_advantage_mean"].item(), 5.0, places=6)

    def test_k3_zero_when_teacher_matches_student(self):
        pack = _pack(mask=[0, 1, 1, 0], old=[0, -1, -2, 0], teacher=[0, -1, -2, 0])
        logprobs = torch.tensor([0.0, -1.0, -2.0, 0.0], requires_grad=True)

        loss, _ = opd_loss_fn(pack, logprobs, opd_loss_type="k3")

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_k3_never_clamps_loss_output(self):
        # delta = 3 gives exp(3)-3-1 ~= 16.1; clamping the loss output to 10
        # would zero the gradient here.
        pack = _pack(mask=[0, 1, 0, 0], old=[0, -1, 0, 0], teacher=[0, 2.0, 0, 0])
        logprobs = torch.tensor([0.0, -1.0, 0.0, 0.0], requires_grad=True)

        loss, _ = opd_loss_fn(pack, logprobs, opd_loss_type="k3")

        self.assertGreater(loss.item(), 10.0)
        loss.backward()
        self.assertNotAlmostEqual(logprobs.grad[1].item(), 0.0, places=3)
        self.assertLess(logprobs.grad[1].item(), 0.0)

    def test_k3_bounds_exponent_input_without_flattening_loss(self):
        # delta = 10 exceeds the exponent bound of 5, so exp() is capped at
        # exp(5), but the objective itself stays far above any loss clamp.
        pack = _pack(mask=[0, 1, 0, 0], old=[0, -1, 0, 0], teacher=[0, 9.0, 0, 0])
        logprobs = torch.tensor([0.0, -1.0, 0.0, 0.0], requires_grad=True)

        loss, _ = opd_loss_fn(pack, logprobs, opd_loss_type="k3")

        self.assertGreater(loss.item(), 100.0)

    def test_k3_zero_gradient_beyond_exponent_bound(self):
        # delta = 10 sits past the bound, so the token contributes no gradient
        # even though the loss value itself is not clamped.
        pack = _pack(mask=[0, 1, 0, 0], old=[0, -1, 0, 0], teacher=[0, 9.0, 0, 0])
        logprobs = torch.tensor([0.0, -1.0, 0.0, 0.0], requires_grad=True)

        loss, _ = opd_loss_fn(pack, logprobs, opd_loss_type="k3")
        loss.backward()

        self.assertGreater(loss.item(), 100.0)
        self.assertEqual(logprobs.grad[1].item(), 0.0)

    def test_k3_gradient_pushes_student_toward_teacher(self):
        pack = _pack(mask=[0, 1, 0, 0], old=[0, -1, 0, 0], teacher=[0, -0.5, 0, 0])
        logprobs = torch.tensor([0.0, -1.0, 0.0, 0.0], requires_grad=True)

        loss, _ = opd_loss_fn(pack, logprobs, opd_loss_type="k3")

        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertLess(logprobs.grad[1].item(), 0.0)

    def test_missing_teacher_signal_raises(self):
        pack = _pack(mask=[0, 1, 1, 0], old=[0, -1, -2, 0], teacher=[0, -1, -2, 0])
        del pack["packed_ref_logprobs"]
        logprobs = torch.tensor([0.0, -1.0, -2.0, 0.0], requires_grad=True)

        with self.assertRaisesRegex(ValueError, "teacher logprobs"):
            opd_loss_fn(pack, logprobs)

    def test_non_finite_teacher_signal_raises(self):
        pack = _pack(mask=[0, 1, 1, 0], old=[0, -1, -2, 0], teacher=[0, float("nan"), -2, 0])
        logprobs = torch.tensor([0.0, -1.0, -2.0, 0.0], requires_grad=True)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            opd_loss_fn(pack, logprobs)

    def test_unknown_loss_type_raises(self):
        pack = _pack(mask=[0, 1, 1, 0], old=[0, -1, -2, 0], teacher=[0, -1, -2, 0])
        logprobs = torch.tensor([0.0, -1.0, -2.0, 0.0], requires_grad=True)

        with self.assertRaisesRegex(ValueError, "unsupported opd_loss_type"):
            opd_loss_fn(pack, logprobs, opd_loss_type="mse")

    def test_coef_scales_loss_linearly(self):
        pack = _pack(mask=[0, 1, 1, 0], old=[0, -1, -2, 0], teacher=[0, -0.5, -1.5, 0])
        logprobs = torch.tensor([0.0, -1.0, -2.0, 0.0], requires_grad=True)

        loss_a, _ = opd_loss_fn(pack, logprobs, opd_coef=1.0)
        loss_b, _ = opd_loss_fn(pack, logprobs, opd_coef=2.0)

        self.assertAlmostEqual(loss_b.item(), 2 * loss_a.item(), places=6)

    def test_token_weighting_matches_merged_batch(self):
        # The same 21-token data, once split into 3 + 18 token packs and once
        # kept as one pack. After the engine divides each pack loss by its grad
        # scale, both paths must produce identical gradients for pg and k3.
        old = [0.1 * (index % 5) for index in range(21)]
        teacher = [old[index] + (0.4 if index % 2 == 0 else -0.3) for index in range(21)]
        student = [-0.7 - 0.01 * index for index in range(21)]

        for loss_type in ("pg", "k3"):
            with self.subTest(loss_type=loss_type):
                merged = _pack(mask=[1] * 21, old=old, teacher=teacher)
                short = _pack(mask=[1] * 3, old=old[:3], teacher=teacher[:3])
                long = _pack(mask=[1] * 18, old=old[3:], teacher=teacher[3:])
                annotate_opd_token_mean_packs([short, long], [3, 18], gradient_accumulation_steps=None)

                merged_logprobs = torch.tensor(student, requires_grad=True)
                short_logprobs = torch.tensor(student[:3], requires_grad=True)
                long_logprobs = torch.tensor(student[3:], requires_grad=True)

                merged_loss, _ = opd_loss_fn(merged, merged_logprobs, opd_loss_type=loss_type)
                short_loss, _ = opd_loss_fn(short, short_logprobs, opd_loss_type=loss_type)
                long_loss, _ = opd_loss_fn(long, long_logprobs, opd_loss_type=loss_type)

                merged_loss.backward()
                ((short_loss + long_loss) / 2).backward()

                self.assertTrue(torch.allclose(merged_logprobs.grad[:3], short_logprobs.grad, atol=1e-6))
                self.assertTrue(torch.allclose(merged_logprobs.grad[3:], long_logprobs.grad, atol=1e-6))

    def test_opd_metrics_are_weighted_by_response_tokens(self):
        from areno.api.backend.common import LOGP_METRIC_WEIGHT, reduce_microbatch_metrics

        reduced = reduce_microbatch_metrics(
            [
                {"opd_loss": 1.0, "opd_clipfrac": 0.1, LOGP_METRIC_WEIGHT: 3.0},
                {"opd_loss": 2.0, "opd_clipfrac": 0.2, LOGP_METRIC_WEIGHT: 18.0},
            ]
        )

        self.assertAlmostEqual(reduced["opd_loss"], (3 * 1.0 + 18 * 2.0) / 21, places=6)
        self.assertAlmostEqual(reduced["opd_clipfrac"], (3 * 0.1 + 18 * 0.2) / 21, places=6)


if __name__ == "__main__":
    unittest.main()
