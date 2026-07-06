"""Unified causal attention shim for consistency diagnostics.

This backend intentionally routes train full-forward, rollout prefill, and
rollout decode through the same CUDA forward kernel. It is slower than
flash-attn, but keeps causal/window masking and softmax accumulation identical
across paths so rollout old-logp and train logp can be compared without mixing
attention implementations.
"""

from __future__ import annotations

import torch

from areno.accel._extension import extension as _extension


def _window_left(window_left: int | None) -> int:
    return -1 if window_left is None else int(window_left)


def _scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    return float(softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5)


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_start: int,
    window_left: int,
    softmax_scale: float,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) * softmax_scale
    q_pos = torch.arange(query_start, query_start + q.shape[-2], device=q.device).view(1, 1, q.shape[-2], 1)
    k_pos = torch.arange(k.shape[-2], device=k.device).view(1, 1, 1, k.shape[-2])
    mask = k_pos <= q_pos
    if window_left >= 0:
        mask = mask & (k_pos >= q_pos - window_left)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def _reference_paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_left: int,
    softmax_scale: float,
) -> torch.Tensor:
    outs = []
    q_heads = q.shape[1]
    for idx in range(q.shape[0]):
        length = int(cache_seqlens[idx].item()) + 1
        blocks = block_table[idx]
        positions = torch.arange(length, device=q.device)
        block_cols = torch.div(positions, k_cache.shape[1], rounding_mode="floor")
        block_offsets = positions % k_cache.shape[1]
        block_ids = blocks[block_cols]
        k = k_cache[block_ids, block_offsets]
        v = v_cache[block_ids, block_offsets]
        k = k.repeat_interleave(q_heads // k.shape[1], dim=1).transpose(0, 1).unsqueeze(0)
        v = v.repeat_interleave(q_heads // v.shape[1], dim=1).transpose(0, 1).unsqueeze(0)
        outs.append(
            _reference_attention(
                q[idx : idx + 1].unsqueeze(2),
                k,
                v,
                query_start=length - 1,
                window_left=window_left,
                softmax_scale=softmax_scale,
            ).squeeze(2)
        )
    return torch.cat(outs, dim=0)


class _ArenoCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_start: int,
        window_left: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        out = _extension().areno_causal_attention_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            int(query_start),
            int(window_left),
            float(softmax_scale),
        )
        ctx.save_for_backward(q, k, v, out)
        ctx.query_start = int(query_start)
        ctx.window_left = int(window_left)
        ctx.softmax_scale = float(softmax_scale)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        q, k, v, out = ctx.saved_tensors
        dq, dk, dv = _extension().areno_causal_attention_backward(
            grad_output.contiguous(),
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out.contiguous(),
            ctx.query_start,
            ctx.window_left,
            ctx.softmax_scale,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


@torch._dynamo.disable
def areno_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_start: int = 0,
    window_left: int | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Apply causal attention to ``(batch, heads, seqlen, head_dim)`` tensors."""

    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("areno_causal_attention requires CUDA q, k, and v tensors")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("areno_causal_attention expects q/k/v tensors shaped (batch, heads, seqlen, head_dim)")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("areno_causal_attention batch size mismatch")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("areno_causal_attention head count mismatch")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("areno_causal_attention head dim mismatch")
    return _ArenoCausalAttention.apply(q, k, v, int(query_start), _window_left(window_left), _scale(q, softmax_scale))


class _ArenoVarlenCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        window_left: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        out = _extension().areno_varlen_causal_attention_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens.contiguous(),
            int(window_left),
            float(softmax_scale),
        )
        ctx.save_for_backward(q, k, v, out, cu_seqlens)
        ctx.window_left = int(window_left)
        ctx.softmax_scale = float(softmax_scale)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        q, k, v, out, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = _extension().areno_varlen_causal_attention_backward(
            grad_output.contiguous(),
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out.contiguous(),
            cu_seqlens.contiguous(),
            ctx.window_left,
            ctx.softmax_scale,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


@torch._dynamo.disable
def areno_varlen_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    window_left: int | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Apply packed causal attention to flat ``(tokens, heads, head_dim)`` tensors."""

    if not (q.is_cuda and k.is_cuda and v.is_cuda and cu_seqlens.is_cuda):
        raise RuntimeError("areno_varlen_causal_attention requires CUDA q, k, v, and cu_seqlens tensors")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("areno_varlen_causal_attention expects q/k/v tensors shaped (tokens, heads, head_dim)")
    if cu_seqlens.dim() != 1 or cu_seqlens.dtype != torch.int32:
        raise ValueError("areno_varlen_causal_attention expects an int32 1D cu_seqlens tensor")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("areno_varlen_causal_attention token count mismatch")
    if k.shape[1] != v.shape[1] or q.shape[1] % k.shape[1] != 0:
        raise ValueError("areno_varlen_causal_attention expects q heads to be divisible by kv heads")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise ValueError("areno_varlen_causal_attention head dim mismatch")
    return _ArenoVarlenCausalAttention.apply(q, k, v, cu_seqlens, _window_left(window_left), _scale(q, softmax_scale))


class _ArenoPagedCausalAttentionDecode(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k_update: torch.Tensor,
        v_update: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        window_left: int,
        num_splits: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        out = _extension().areno_paged_causal_attention_decode_forward(
            q.contiguous(),
            k_update.contiguous(),
            v_update.contiguous(),
            k_cache.contiguous(),
            v_cache.contiguous(),
            block_table.contiguous(),
            cache_seqlens.contiguous(),
            int(window_left),
            int(num_splits),
            float(softmax_scale),
        )
        ctx.save_for_backward(q, k_cache, v_cache, block_table, cache_seqlens)
        ctx.window_left = int(window_left)
        ctx.num_splits = int(num_splits)
        ctx.softmax_scale = float(softmax_scale)
        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, torch.Tensor, None, None, None, None, None]:
        q, k_cache, v_cache, block_table, cache_seqlens = ctx.saved_tensors
        with torch.enable_grad():
            q_ref = q.detach().float().requires_grad_(True)
            k_ref = k_cache.detach().float().requires_grad_(True)
            v_ref = v_cache.detach().float().requires_grad_(True)
            out = _reference_paged_decode_attention(
                q_ref,
                k_ref,
                v_ref,
                block_table,
                cache_seqlens,
                ctx.window_left,
                ctx.softmax_scale,
            )
            dq, dk, dv = torch.autograd.grad(out, (q_ref, k_ref, v_ref), grad_output.float())
        return dq.to(q.dtype), None, None, dk.to(k_cache.dtype), dv.to(v_cache.dtype), None, None, None, None, None


@torch._dynamo.disable
def areno_paged_causal_attention_decode(
    q: torch.Tensor,
    k_update: torch.Tensor,
    v_update: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    window_left: int | None = None,
    num_splits: int = 8,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Apply single-token paged-cache causal attention to ``(batch, heads, dim)`` Q."""

    if not (
        q.is_cuda
        and k_update.is_cuda
        and v_update.is_cuda
        and k_cache.is_cuda
        and v_cache.is_cuda
        and block_table.is_cuda
        and cache_seqlens.is_cuda
    ):
        raise RuntimeError("areno_paged_causal_attention_decode requires CUDA tensors")
    if q.dim() != 3:
        raise ValueError("areno_paged_causal_attention_decode expects q shaped (batch, heads, head_dim)")
    if k_update.dim() != 3 or v_update.dim() != 3:
        raise ValueError("areno_paged_causal_attention_decode expects k/v updates shaped (batch, kv_heads, head_dim)")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("areno_paged_causal_attention_decode expects 4D paged cache tensors")
    if block_table.dim() != 2 or cache_seqlens.dim() != 1:
        raise ValueError("areno_paged_causal_attention_decode expects block_table 2D and cache_seqlens 1D")
    if block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise ValueError("areno_paged_causal_attention_decode expects int32 block_table and cache_seqlens")
    return _ArenoPagedCausalAttentionDecode.apply(
        q,
        k_update,
        v_update,
        k_cache,
        v_cache,
        block_table,
        cache_seqlens,
        _window_left(window_left),
        int(num_splits),
        _scale(q, softmax_scale),
    )
