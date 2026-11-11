"""AReno-owned fused attention sinks with backward and paged inference.

The softmax includes a persistent zero-valued per-head sentinel. CPU math is
an explicit reference; CUDA forward/backward never build a dense score matrix.
"""

from __future__ import annotations

import torch


def sink_sequence_starts(length, device, sequence_ids=None):
    positions = torch.arange(length, device=device, dtype=torch.long)
    if sequence_ids is None:
        return torch.zeros_like(positions)
    starts = torch.where(
        torch.cat((torch.ones(1, device=device, dtype=torch.bool), sequence_ids[1:] != sequence_ids[:-1])), positions, 0
    )
    return starts.cummax(0).values


class _SinkAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, starts, window):
        from areno.accel.kernels.sink_attention import dense_forward

        q, k, v = (x.contiguous() for x in (q, k, v))
        out, lse = dense_forward(q, k, v, sinks, starts, window)
        ctx.save_for_backward(q, k, v, sinks, starts, out, lse)
        ctx.window = window
        return out

    @staticmethod
    def backward(ctx, grad):
        from areno.accel.kernels.sink_attention import dense_backward

        return (*dense_backward(*ctx.saved_tensors, grad, ctx.window), None, None)


def sink_attention(q, k, v, sinks, sequence_starts=None, sequence_ids=None, window=None):
    if q.is_cuda:
        if sequence_starts is None:
            sequence_starts = sink_sequence_starts(q.shape[1], q.device, sequence_ids)
        return _SinkAttention.apply(q, k, v, sinks, sequence_starts, window)
    batch, length, heads, dim = q.shape
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    repeat = heads // k.shape[1]
    k = k.repeat_interleave(repeat, 1)
    v = v.repeat_interleave(repeat, 1)
    rows = torch.arange(length, device=q.device)[:, None]
    cols = torch.arange(length, device=q.device)[None, :]
    valid = cols <= rows
    if window is not None:
        valid = valid & (cols > rows - window)
    if sequence_ids is not None:
        valid = valid & (sequence_ids[:, None] == sequence_ids[None, :])
    scores = (q.float() @ k.float().transpose(-1, -2)) * dim**-0.5
    scores = scores.masked_fill(~valid, -float("inf"))
    logits = torch.cat((scores, sinks.view(1, heads, 1, 1).expand(batch, -1, length, -1)), -1)
    out = logits.softmax(-1)[..., :length] @ v.float()
    return out.to(q.dtype).transpose(1, 2).contiguous()


def paged_sink_attention(q, k_cache, v_cache, sinks, block_table, sequence_ids, positions, window=None):
    from areno.accel.kernels.sink_attention import paged_sink_forward

    return paged_sink_forward(q.contiguous(), k_cache, v_cache, sinks, block_table, sequence_ids, positions, window)
