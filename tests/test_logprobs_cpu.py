from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from areno.engine.runtime import logprobs as logprob_ops
from tests.helpers import PatchedContext, single_tp_context


class LogprobTest(unittest.TestCase):
    """Logprob tests patch TP context to exercise single-rank CPU math."""

    def test_vocab_parallel_selected_logprobs_matches_full_softmax_single_tp(self):
        """Single-TP selected logprobs should equal regular full-vocab softmax."""
        logits = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.25, 0.0]], requires_grad=True)
        labels = torch.tensor([1, 2])
        expected = torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1)

        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            actual = logprob_ops.vocab_parallel_selected_logprobs(logits, labels)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_vocab_parallel_selected_logprobs_backward_matches_cross_entropy(self):
        """Custom selected-logprob backward should match torch autograd."""
        logits = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.25, 0.0]], requires_grad=True)
        labels = torch.tensor([1, 2])
        ref_logits = logits.detach().clone().requires_grad_(True)
        expected = torch.log_softmax(ref_logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1)
        expected.sum().backward()

        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            actual = logprob_ops.vocab_parallel_selected_logprobs(logits, labels)
        actual.sum().backward()

        self.assertTrue(torch.allclose(logits.grad, ref_logits.grad, atol=1e-6))

    def test_next_token_logprobs_aligns_with_next_tokens(self):
        """Padded next-token scores should align logits[t] with tokens[t + 1]."""
        logits = torch.tensor([[[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [2.0, 0.0, 1.0]]])
        tokens = torch.tensor([[0, 1, 2]])
        expected = torch.stack(
            [
                torch.log_softmax(logits[0, 0], dim=-1)[1],
                torch.log_softmax(logits[0, 1], dim=-1)[2],
            ]
        ).view(1, 2)

        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            actual = logprob_ops.next_token_logprobs(logits, tokens)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_packed_next_token_logprobs_skips_sequence_tails(self):
        """Packed scoring should not emit labels for each sequence tail token."""
        logits = torch.tensor(
            [
                [
                    [1.0, 2.0, 0.0],
                    [0.0, 3.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [0.5, 0.0, 1.5],
                    [1.0, 1.0, 2.0],
                ]
            ]
        )
        tokens = torch.tensor([0, 1, 2, 0, 2])
        cu_seqlens = torch.tensor([0, 3, 5])
        expected = torch.tensor(
            [
                torch.log_softmax(logits[0, 0], dim=-1)[1],
                torch.log_softmax(logits[0, 1], dim=-1)[2],
                torch.log_softmax(logits[0, 3], dim=-1)[2],
            ]
        )

        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            actual = logprob_ops.packed_next_token_logprobs(logits, tokens, cu_seqlens)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_empty_vocab_logprob_path_returns_empty_tensor(self):
        """Empty microbatches should return an empty tensor instead of failing."""
        logits = torch.empty(0, 3)
        labels = torch.empty(0, dtype=torch.long)

        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            actual = logprob_ops.vocab_parallel_selected_logprobs(logits, labels)

        self.assertEqual(actual.numel(), 0)

    def test_forward_selected_logprobs_chunks_vocab_exp(self):
        """Forward-only selected logprobs should avoid full-vocab exp tensors."""
        logits = torch.tensor(
            [
                [1.0, 2.0, -1.0, 0.25, 0.75],
                [0.5, -0.25, 0.0, 3.0, -2.0],
            ]
        )
        labels = torch.tensor([1, 3])
        expected = torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1)
        exp_widths = []
        real_exp = torch.exp

        def tracked_exp(value):
            exp_widths.append(value.shape[-1])
            return real_exp(value)

        with patch.object(logprob_ops.torch, "exp", side_effect=tracked_exp):
            actual = logprob_ops._selected_logprobs_components_forward(
                logits,
                labels,
                vocab_start=0,
                group=None,
                world_size=1,
                vocab_chunk_size=2,
            )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(exp_widths)
        self.assertLessEqual(max(exp_widths), 2)

    def test_training_selected_logprobs_chunks_vocab_exp(self):
        """Training should recompute softmax without full-vocab FP32 probabilities."""
        logits = torch.randn(2, 9000, requires_grad=True)
        labels = torch.tensor([1, 8999])
        expected = torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1)
        exp_widths = []
        real_exp = torch.exp

        def tracked_exp(value):
            exp_widths.append(value.shape[-1])
            return real_exp(value)

        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            with patch.object(logprob_ops.torch, "exp", side_effect=tracked_exp):
                actual = logprob_ops.vocab_parallel_selected_logprobs(logits, labels)
                actual.sum().backward()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
        self.assertTrue(exp_widths)
        self.assertLessEqual(max(exp_widths), 8192)

    def test_checkpointed_lm_head_logprobs_match_full_projection_gradients(self):
        """Checkpointed packed scoring should preserve LM-head and hidden gradients."""
        hidden = torch.randn(1, 6, 4, requires_grad=True)
        head = torch.nn.Linear(4, 7, bias=False)
        tokens = torch.tensor([[0, 1, 2, 3, 4, 5]])
        cu_seqlens = torch.tensor([0, 3, 6])
        ref_hidden = hidden.detach().clone().requires_grad_(True)
        ref_head = torch.nn.Linear(4, 7, bias=False)
        ref_head.load_state_dict(head.state_dict())
        positions = torch.tensor([0, 1, 3, 4])
        labels = torch.tensor([1, 2, 4, 5])
        softcap = 2.0
        ref_logits = ref_head(ref_hidden.reshape(-1, 4)[positions])
        ref_logits = softcap * torch.tanh(ref_logits / softcap)
        expected = torch.log_softmax(ref_logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1)
        expected.sum().backward()

        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            actual = logprob_ops.packed_next_token_logprobs_from_hidden(
                hidden,
                tokens,
                cu_seqlens,
                head,
                logit_softcap=softcap,
                chunk_size=2,
            )
        actual.sum().backward()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(torch.allclose(hidden.grad, ref_hidden.grad, atol=1e-6))
        self.assertTrue(torch.allclose(head.weight.grad, ref_head.weight.grad, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
