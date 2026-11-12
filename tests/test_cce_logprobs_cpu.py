"""CPU tests for fused vocabulary-projection logprobs (issue #588).

Every check compares the streaming fused path against (a) a literal
full-logits reference written out with ``F.linear`` + ``log_softmax`` and
(b) the existing checkpointed ``packed_next_token_logprobs_from_hidden``
path, for forward values and backward gradients under arbitrary upstream
token gradients. One test pins the memory property: no ``[chunk, vocab]``
tensor may be saved for backward.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from areno.engine.runtime.cce_logprobs import packed_next_token_logprobs_from_hidden_fused
from areno.engine.runtime.logprobs import packed_next_token_logprobs_from_hidden

VOCAB = 130  # not divisible by the tile size -> exercises a partial last tile
HIDDEN_DIM = 16
TILE = 32
CU_SEQLENS = (0, 5, 9, 12)  # rows of 5/4/3 tokens -> 9 action sites


class StubLMHead(torch.nn.Module):
    """Bias-free linear head matching ``VocabParallelLMHead``'s projection."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


def _naive_packed_logprobs(hidden, tokens, cu_seqlens, weight, softcap=0.0):
    """Literal full-logits reference: materialize [sites, vocab], log-softmax, select."""

    flat = tokens.reshape(-1)
    cu = cu_seqlens.to(device=tokens.device, dtype=torch.long)
    action_count = flat.numel() - (cu.numel() - 1)
    if action_count == 0:
        return torch.empty(0, device=flat.device, dtype=torch.float32)
    keep = torch.ones(flat.numel(), device=flat.device, dtype=torch.bool)
    keep[cu[1:] - 1] = False
    positions = torch.arange(flat.numel(), device=flat.device)[keep]
    labels = flat[positions + 1]
    h = hidden.reshape(-1, hidden.shape[-1])[positions]
    logits = F.linear(h, weight)
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _make_inputs(dtype, *, seed=0, requires_grad=True):
    torch.manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (3, 4))
    cu_seqlens = torch.tensor(CU_SEQLENS)
    hidden = (torch.randn(3, 4, HIDDEN_DIM) * 0.5).to(dtype).requires_grad_(requires_grad)
    weight = torch.nn.Parameter(torch.randn(VOCAB, HIDDEN_DIM).to(dtype) * 0.05, requires_grad=requires_grad)
    return tokens, cu_seqlens, hidden, weight


def _backward(loss):
    grad = torch.linspace(-1.0, 1.0, loss.numel())
    (loss * grad).sum().backward()
    return grad


DTYPES = [torch.float32, torch.bfloat16]
TOLS = {
    torch.float32: {"atol": 1e-4, "rtol": 1e-4},
    torch.bfloat16: {"atol": 3e-2, "rtol": 3e-2},
}
GRAD_TOLS = {
    torch.float32: {"atol": 1e-4, "rtol": 1e-3},
    torch.bfloat16: {"atol": 5e-2, "rtol": 5e-2},
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("softcap", [0.0, 30.0])
@pytest.mark.parametrize("chunk_size", [1, 4, 256])
def test_forward_matches_naive_full_logits(dtype, softcap, chunk_size):
    tokens, cu_seqlens, hidden, weight = _make_inputs(dtype)
    fused = packed_next_token_logprobs_from_hidden_fused(
        hidden, tokens, cu_seqlens, StubLMHead(weight),
        logit_softcap=softcap or None, chunk_size=chunk_size, vocab_tile_size=TILE,
    )
    naive = _naive_packed_logprobs(hidden, tokens, cu_seqlens, weight, softcap=softcap)
    assert fused.shape == naive.shape == (9,)
    assert torch.allclose(fused.float(), naive.float(), **TOLS[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_backward_matches_naive_full_logits(dtype, softcap):
    tokens, cu_seqlens, hidden, weight = _make_inputs(dtype)
    fused = packed_next_token_logprobs_from_hidden_fused(
        hidden, tokens, cu_seqlens, StubLMHead(weight),
        logit_softcap=softcap or None, vocab_tile_size=TILE,
    )
    _backward(fused)
    fused_hidden_grad = hidden.grad.float().clone()
    fused_weight_grad = weight.grad.float().clone()

    tokens, cu_seqlens, hidden2, weight2 = _make_inputs(dtype)
    naive = _naive_packed_logprobs(hidden2, tokens, cu_seqlens, weight2, softcap=softcap)
    _backward(naive)
    assert torch.allclose(fused_hidden_grad, hidden2.grad.float(), **GRAD_TOLS[dtype])
    assert torch.allclose(fused_weight_grad, weight2.grad.float(), **GRAD_TOLS[dtype])


@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_matches_existing_checkpointed_path(softcap):
    tokens, cu_seqlens, hidden, weight = _make_inputs(torch.float32)
    fused = packed_next_token_logprobs_from_hidden_fused(
        hidden, tokens, cu_seqlens, StubLMHead(weight),
        logit_softcap=softcap or None, vocab_tile_size=TILE,
    )
    _backward(fused)
    fused_hidden_grad = hidden.grad.clone()
    fused_weight_grad = weight.grad.clone()

    tokens, cu_seqlens, hidden2, weight2 = _make_inputs(torch.float32)
    baseline = packed_next_token_logprobs_from_hidden(
        hidden2, tokens, cu_seqlens, StubLMHead(weight2),
        logit_softcap=softcap or None,
    )
    _backward(baseline)
    assert torch.allclose(fused, baseline, atol=1e-4, rtol=1e-4)
    assert torch.allclose(fused_hidden_grad, hidden2.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(fused_weight_grad, weight2.grad, atol=1e-4, rtol=1e-3)


def test_no_full_vocab_tensor_saved_for_backward():
    tokens, cu_seqlens, hidden, weight = _make_inputs(torch.float32)
    saved_shapes = []

    def pack(t):
        saved_shapes.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = packed_next_token_logprobs_from_hidden_fused(
            hidden, tokens, cu_seqlens, StubLMHead(weight), vocab_tile_size=TILE,
        )
    action_count, vocab = out.numel(), weight.shape[0]
    materialized = [s for s in saved_shapes if len(s) >= 2 and s[0] == action_count and s[1] == vocab]
    assert not materialized, f"full-vocab logits saved for backward: {materialized}"
    _backward(out)  # gradients still flow through the streamed tiles


def test_label_at_vocab_edges():
    torch.manual_seed(1)
    tokens = torch.tensor([[5, 0, VOCAB - 1, 3, 8, VOCAB - 1, 0, 2]])
    cu_seqlens = torch.tensor([0, 8])
    hidden = torch.randn(1, 8, HIDDEN_DIM, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(VOCAB, HIDDEN_DIM) * 0.05)
    fused = packed_next_token_logprobs_from_hidden_fused(
        hidden, tokens, cu_seqlens, StubLMHead(weight), vocab_tile_size=TILE,
    )
    naive = _naive_packed_logprobs(hidden, tokens, cu_seqlens, weight)
    assert torch.allclose(fused, naive, atol=1e-4, rtol=1e-4)
    _backward(fused)
    fused_hidden_grad = hidden.grad.clone()
    fused_weight_grad = weight.grad.clone()
    hidden2 = hidden.detach().clone().requires_grad_(True)
    weight2 = torch.nn.Parameter(weight.detach().clone())
    _backward(_naive_packed_logprobs(hidden2, tokens, cu_seqlens, weight2))
    assert torch.allclose(fused_hidden_grad, hidden2.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(fused_weight_grad, weight2.grad, atol=1e-4, rtol=1e-3)


def test_single_action_site_and_empty_pack():
    torch.manual_seed(2)
    weight = torch.nn.Parameter(torch.randn(VOCAB, HIDDEN_DIM) * 0.05)
    # One row of two tokens -> exactly one action site.
    hidden = torch.randn(1, 2, HIDDEN_DIM, requires_grad=True)
    tokens = torch.tensor([[9, 4]])
    fused = packed_next_token_logprobs_from_hidden_fused(
        hidden, tokens, torch.tensor([0, 2]), StubLMHead(weight), vocab_tile_size=TILE,
    )
    naive = _naive_packed_logprobs(hidden, tokens, torch.tensor([0, 2]), weight)
    assert fused.shape == naive.shape == (1,)
    assert torch.allclose(fused, naive, atol=1e-4, rtol=1e-4)
    # Single-token rows produce no action sites; the empty result matches the
    # checkpointed path's contract (plain empty FP32 tensor, no graph).
    hidden1 = torch.randn(1, 1, HIDDEN_DIM, requires_grad=True)
    out = packed_next_token_logprobs_from_hidden_fused(
        hidden1, torch.tensor([[3]]), torch.tensor([0, 1]), StubLMHead(weight),
    )
    assert out.shape == (0,) and out.dtype == torch.float32


def test_tied_weight_gradient_accumulates():
    torch.manual_seed(3)
    tokens = torch.randint(0, VOCAB, (3, 4))
    cu_seqlens = torch.tensor(CU_SEQLENS)
    scale = torch.randn(3, 4, HIDDEN_DIM)
    weight = torch.nn.Parameter(torch.randn(VOCAB, HIDDEN_DIM) * 0.05)
    hidden = torch.randn(3, 4, HIDDEN_DIM, requires_grad=True)
    head = packed_next_token_logprobs_from_hidden_fused(
        hidden, tokens, cu_seqlens, StubLMHead(weight), vocab_tile_size=TILE,
    )
    emb = F.embedding(tokens.reshape(-1), weight).reshape(3, 4, HIDDEN_DIM)
    (head.sum() + (emb * scale).sum()).backward()
    fused_grad = weight.grad.clone()

    weight2 = torch.nn.Parameter(weight.detach().clone())
    hidden2 = hidden.detach().clone().requires_grad_(True)
    head2 = _naive_packed_logprobs(hidden2, tokens, cu_seqlens, weight2)
    emb2 = F.embedding(tokens.reshape(-1), weight2).reshape(3, 4, HIDDEN_DIM)
    (head2.sum() + (emb2 * scale).sum()).backward()
    assert torch.allclose(fused_grad, weight2.grad, atol=1e-4, rtol=1e-3)


def test_frozen_head_returns_no_weight_grad():
    tokens, cu_seqlens, hidden, _ = _make_inputs(torch.float32)
    weight = torch.nn.Parameter(torch.randn(VOCAB, HIDDEN_DIM) * 0.05, requires_grad=False)
    out = packed_next_token_logprobs_from_hidden_fused(
        hidden, tokens, cu_seqlens, StubLMHead(weight), vocab_tile_size=TILE,
    )
    _backward(out)
    assert weight.grad is None
    hidden2 = hidden.detach().clone().requires_grad_(True)
    _backward(_naive_packed_logprobs(hidden2, tokens, cu_seqlens, weight))
    assert torch.allclose(hidden.grad, hidden2.grad, atol=1e-4, rtol=1e-3)
