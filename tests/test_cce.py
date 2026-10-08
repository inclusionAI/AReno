"""CUDA numerical contracts for unfiltered cut selected log-probabilities."""

import pytest
import torch
import torch.nn.functional as F

from areno.accel.cce import cut_logprobs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CCE requires CUDA")


def reference(x, w, labels, cap):
    logits = F.linear(x, w)
    if cap:
        logits = cap * torch.tanh(logits / cap)
    return logits.float().log_softmax(-1).gather(1, labels[:, None]).squeeze(1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("cap", [0.0, 3.0])
@pytest.mark.parametrize("train_weight", [False, True])
def test_cce_outputs_and_all_gradients(dtype, cap, train_weight):
    torch.manual_seed(32)
    # Odd sizes exercise token, vocabulary, and reduction-tail masks.
    x = (torch.randn(35, 65, device="cuda", dtype=dtype) * 0.3).requires_grad_()
    w = (torch.randn(259, 65, device="cuda", dtype=dtype) * 0.2).requires_grad_(train_weight)
    labels = torch.randint(259, (35,), device="cuda")
    labels[:4] = torch.tensor([0, 127, 128, 258], device="cuda")
    upstream = torch.randn(35, device="cuda")
    # Include negative and masked upstream values, as required by policy losses.
    upstream[::3] = 0
    expected = reference(x, w, labels, cap)
    params = (x, w) if train_weight else (x,)
    ref_grads = torch.autograd.grad(expected, params, upstream)
    actual = cut_logprobs(x, w, labels, softcap=cap)
    grads = torch.autograd.grad(actual, params, upstream)
    atol = 2e-5 if dtype == torch.float32 else (8e-3 if dtype == torch.bfloat16 else 1e-3)
    rtol = 2e-4 if dtype == torch.float32 else (2e-2 if dtype == torch.bfloat16 else 4e-3)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for grad, ref in zip(grads, ref_grads, strict=True):
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, ref, atol=atol, rtol=rtol)


def test_cce_noncontiguous_empty_and_graph():
    torch.manual_seed(42)
    x = torch.randn(64, 8, device="cuda").T.requires_grad_()
    w = torch.randn(64, 257, device="cuda").T.requires_grad_()
    labels = torch.arange(8, device="cuda")
    torch.testing.assert_close(cut_logprobs(x, w, labels), reference(x, w, labels, 0), atol=2e-5, rtol=2e-5)
    empty = cut_logprobs(x[:0], w, labels[:0])
    dx, dw = torch.autograd.grad(empty.sum(), (x, w))
    assert not dx.count_nonzero() and not dw.count_nonzero()
    # Warm up on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            cut_logprobs(x, w, labels).sum().backward()
            x.grad = w.grad = None
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = cut_logprobs(x, w, labels)
        out.sum().backward()
    graph.replay()
    torch.testing.assert_close(out, reference(x, w, labels, 0), atol=2e-5, rtol=2e-5)


def test_cce_large_vocab_ling_fp32_head():
    torch.manual_seed(12)
    x = torch.randn(37, 1536, device="cuda", requires_grad=True)
    w = torch.randn(157184, 1536, device="cuda") * 0.02
    labels = torch.randint(w.shape[0], (37,), device="cuda")
    expected = reference(x, w, labels, 0)
    actual = cut_logprobs(x, w, labels)
    upstream = torch.randn_like(expected)
    (dx_ref,) = torch.autograd.grad(expected, x, upstream)
    (dx,) = torch.autograd.grad(actual, x, upstream)
    print({"logprob_max_abs": (actual - expected).abs().max().item(), "dx_max_abs": (dx - dx_ref).abs().max().item()})
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(dx, dx_ref, atol=2e-5, rtol=2e-4)
