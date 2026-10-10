"""Rowwise selected log-probs must be independent of token chunk geometry."""

import pytest
import torch

from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.engine.runtime.logprobs import vocab_parallel_selected_logprobs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA row-reduction kernel")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("vocab", [17, 128, 8192, 8193, 157184])
def test_selected_logprobs_chunk_geometry_cuda(dtype, vocab):
    previous = get_tp_context()
    set_tp_context(TPContext(0, 1, torch.device("cuda", 0), None))
    try:
        torch.manual_seed(53)
        # A column stride also checks the reduction's explicit input layout.
        logits = torch.randn(37, vocab * 2, device="cuda", dtype=dtype)[:, ::2].requires_grad_()
        labels = torch.randint(vocab, (37,), device="cuda")
        whole = vocab_parallel_selected_logprobs(logits, labels)
        chunks = torch.cat(
            [vocab_parallel_selected_logprobs(logits[i : i + 7], labels[i : i + 7]) for i in range(0, 37, 7)]
        )
        torch.testing.assert_close(chunks, whole, atol=0, rtol=0)
        reference = logits.double().log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        torch.testing.assert_close(whole.double(), reference, atol=3e-6, rtol=1e-6)
        upstream = torch.linspace(-1, 1, 37, device="cuda")
        expected = torch.autograd.grad(whole, logits, upstream)[0]
        actual = torch.autograd.grad(chunks, logits, upstream)[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if dtype == torch.float32:
            reference_gradient = torch.autograd.grad(reference, logits, upstream.double())[0]
            torch.testing.assert_close(expected, reference_gradient, atol=2e-6, rtol=3e-6)
        with torch.no_grad():
            forward_only = vocab_parallel_selected_logprobs(logits, labels)
        torch.testing.assert_close(forward_only, whole, atol=0, rtol=0)
    finally:
        set_tp_context(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA selected log-probs")
def test_selected_logprobs_preserves_batched_inputs_cuda():
    previous = get_tp_context()
    set_tp_context(TPContext(0, 1, torch.device("cuda", 0), None))
    try:
        logits = torch.randn(2, 19, 17, device="cuda", requires_grad=True)
        labels = torch.randint(17, (2, 19), device="cuda")
        output = vocab_parallel_selected_logprobs(logits, labels)
        reference = logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
        torch.testing.assert_close(output, reference, atol=2e-6, rtol=1e-6)
        actual = torch.autograd.grad(output.sum(), logits)[0]
        expected = torch.autograd.grad(reference.sum(), logits)[0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    finally:
        set_tp_context(previous)
