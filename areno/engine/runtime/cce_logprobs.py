"""Fused vocabulary-projection selected logprobs (issue #588).

The deferred LM-head path in :mod:`areno.engine.runtime.logprobs` still
materializes ``[chunk, vocab]`` logits for every position chunk before the
selected-logprob reduction. This module fuses the vocabulary projection into
the loss so full-vocabulary logits are never materialized: logits are streamed
in vocab tiles of ``vocab_tile_size`` rows and the log-softmax normalizer is
accumulated with a running max / exp-sum. The peak head activation becomes
``[chunk, vocab_tile]`` instead of ``[chunk, vocab]``.

Forward values and backward gradients match the materialized reference up to
floating-point summation order. The autograd function recomputes each vocab
tile in backward, so nothing larger than one tile is kept alive: the saved
tensors are the hidden states, the head weight, and one logsumexp row per
token. Weight gradients accumulate in FP32 across tiles and token chunks and
are rounded to the parameter dtype once, and a frozen head (weight without
``requires_grad``) simply receives no weight gradient.

Scope of the first delivery: single-GPU, bias-free linear head (all current
heads), packed next-token scoring with the same mask/reduction semantics as
``packed_next_token_logprobs_from_hidden``. Tensor-parallel vocab shards fall
back to the existing path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from areno.engine.runtime.logprobs import _packed_action_sites


class _FusedVocabProjectionLogprobs(torch.autograd.Function):
    """Selected logprobs with the vocab projection streamed per tile.

    Forward computes ``logit[label] - logsumexp(logits)`` per row without ever
    forming the full row of logits; backward re-derives each tile's softmax
    contribution from the saved logsumexp and streams the hidden / weight
    gradients through the same tiles.
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, softcap, vocab_tile_size):
        n = hidden.shape[0]
        vocab = weight.shape[0]
        tile = int(vocab_tile_size)
        h = hidden.to(weight.dtype)
        running_max = torch.full((n,), -float("inf"), dtype=torch.float32, device=hidden.device)
        running_sum = torch.zeros(n, dtype=torch.float32, device=hidden.device)
        target = torch.zeros(n, dtype=torch.float32, device=hidden.device)
        for v0 in range(0, vocab, tile):
            v1 = min(v0 + tile, vocab)
            logits = F.linear(h, weight[v0:v1])
            if softcap:
                logits = softcap * torch.tanh(logits / softcap)
            logits32 = logits.float()
            tile_max = logits32.max(dim=-1).values
            new_max = torch.maximum(running_max, tile_max)
            running_sum = running_sum * torch.exp(running_max - new_max)
            running_sum = running_sum + torch.exp(logits32 - new_max.unsqueeze(-1)).sum(dim=-1)
            running_max = new_max
            in_tile = (labels >= v0) & (labels < v1)
            cols = (labels - v0).clamp(min=0, max=(v1 - v0) - 1).unsqueeze(-1)
            picked = logits32.gather(-1, cols).squeeze(-1)
            target = torch.where(in_tile, picked, target)
        lse = running_max + running_sum.log()
        ctx.save_for_backward(hidden, weight, labels, lse)
        ctx.softcap = float(softcap or 0.0)
        ctx.tile = tile
        return target - lse

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, labels, lse = ctx.saved_tensors
        cap = ctx.softcap
        tile = ctx.tile
        n, hidden_dim = hidden.shape
        vocab = weight.shape[0]
        h = hidden.to(weight.dtype)
        g = grad_output.float()
        need_hidden, need_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_hidden = torch.zeros(n, hidden_dim, dtype=torch.float32, device=hidden.device) if need_hidden else None
        grad_weight = torch.zeros(vocab, hidden_dim, dtype=torch.float32, device=weight.device) if need_weight else None
        for v0 in range(0, vocab, tile):
            v1 = min(v0 + tile, vocab)
            w_tile = weight[v0:v1]
            w_tile32 = w_tile.float()
            logits = F.linear(h, w_tile)
            if cap:
                logits = cap * torch.tanh(logits / cap)
            # dL/dlogit = g * (onehot(label) - softmax(logits)); the softmax
            # part is rebuilt tile-wise from the saved full-row logsumexp.
            dlogits = torch.exp(logits.float() - lse.unsqueeze(-1)).mul_(-g.unsqueeze(-1))
            in_tile = (labels >= v0) & (labels < v1)
            cols = (labels - v0).clamp(min=0, max=(v1 - v0) - 1).unsqueeze(-1)
            dlogits.scatter_add_(-1, cols, (in_tile.to(dlogits.dtype) * g).unsqueeze(-1))
            if cap:
                # y = cap * tanh(z / cap) => dy/dz = 1 - (y / cap)^2.
                dlogits = dlogits * (1.0 - (logits.float() / cap) ** 2)
            if need_hidden:
                grad_hidden += dlogits @ w_tile32
            if need_weight:
                grad_weight[v0:v1] = dlogits.transpose(0, 1) @ h.float()
        if need_hidden:
            grad_hidden = grad_hidden.to(hidden.dtype)
        if need_weight:
            grad_weight = grad_weight.to(weight.dtype)
        return grad_hidden, grad_weight, None, None, None


def packed_next_token_logprobs_from_hidden_fused(
    hidden_states: torch.Tensor,
    tokens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lm_head,
    *,
    logit_softcap: float | None = None,
    chunk_size: int = 256,
    vocab_tile_size: int = 32768,
) -> torch.Tensor:
    """Score packed tokens by streaming the vocabulary projection per tile.

    Same contract as
    :func:`areno.engine.runtime.logprobs.packed_next_token_logprobs_from_hidden`
    (per-site FP32 selected logprobs over packed varlen rows) but the head
    output is never materialized wider than ``vocab_tile_size``. ``chunk_size``
    bounds the token axis exactly like the checkpointed path.
    """

    flat_tokens, _, positions, action_count = _packed_action_sites(tokens, cu_seqlens)
    selected = torch.empty(action_count, device=hidden_states.device, dtype=torch.float32)
    if action_count == 0:
        return selected
    labels = flat_tokens[positions + 1]
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    cap = float(logit_softcap or 0.0)
    weight = lm_head.weight
    if torch.is_grad_enabled() and weight.dtype in (torch.bfloat16, torch.float16):
        projection_dtype = lm_head(flat_hidden[:0].to(weight.dtype).unsqueeze(0)).dtype
        if projection_dtype == torch.float32:
            # Some heads project in FP32 despite BF16 weights. Cast once for
            # the whole loss graph so partial weight gradients sum in FP32 and
            # round to the parameter dtype a single time.
            weight = weight.float()
    for start in range(0, action_count, chunk_size):
        end = min(start + chunk_size, action_count)
        chunk_hidden = flat_hidden.index_select(0, positions[start:end])
        selected[start:end] = _FusedVocabProjectionLogprobs.apply(
            chunk_hidden,
            weight,
            labels[start:end],
            cap,
            vocab_tile_size,
        )
    return selected
