"""Exact (unfiltered) cut selected-log-probability autograd for CUDA heads."""

from __future__ import annotations

import torch
import torch.distributed as dist


class _CutLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, softcap, vocab_start, group, world_size):
        from areno.accel.kernels.cce import forward_kernel

        n, d = hidden.shape
        v = weight.shape[0]
        lse = torch.empty(n, device=hidden.device, dtype=torch.float32)
        target = torch.zeros_like(lse)
        # Bound the partial reduction workspace independently of sequence length.
        for begin in range(0, n, 1024):
            end = min(begin + 1024, n)
            size = end - begin
            partial = torch.empty(((v + 127) // 128, size), device=hidden.device, dtype=torch.float32)
            forward_kernel[((size + 31) // 32, (v + 127) // 128)](
                hidden[begin:end],
                weight,
                labels[begin:end],
                partial,
                target[begin:end],
                size,
                v,
                d,
                vocab_start,
                32,
                128,
                64,
                softcap,
            )
            lse[begin:end] = torch.logsumexp(partial, dim=0)
        if world_size > 1:
            maximum = lse.clone()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
            sums = (lse - maximum).exp()
            dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
            lse = maximum + sums.log()
            dist.all_reduce(target, op=dist.ReduceOp.SUM, group=group)
        ctx.save_for_backward(hidden, weight, labels, lse)
        ctx.softcap, ctx.vocab_start = softcap, vocab_start
        return target - lse

    @staticmethod
    def backward(ctx, grad_output):
        from areno.accel.kernels.cce import backward_kernel

        hidden, weight, labels, lse = ctx.saved_tensors
        n, d = hidden.shape
        v = weight.shape[0]
        need_x, need_w = ctx.needs_input_grad[:2]
        # FP32 reductions avoid repeated low-precision rounding across tiles.
        dx = torch.zeros_like(hidden, dtype=torch.float32) if need_x else None
        dw = torch.zeros_like(weight, dtype=torch.float32) if need_w else None
        if n and (need_x or need_w):
            backward_kernel[((n + 31) // 32, (v + 127) // 128)](
                hidden,
                weight,
                labels,
                lse,
                grad_output.contiguous(),
                dx,
                dw,
                n,
                v,
                d,
                ctx.vocab_start,
                32,
                128,
                64,
                ctx.softcap,
                need_x,
                need_w,
            )
        return (
            dx.to(hidden.dtype) if need_x else None,
            dw.to(weight.dtype) if need_w else None,
            None,
            None,
            None,
            None,
            None,
        )


@torch._dynamo.disable
def cut_logprobs(hidden, weight, labels, *, softcap=0.0, vocab_start=0, group=None, world_size=1):
    """Return one log-prob per row; TP input-gradient reduction belongs to the caller.

    Projection and softcap preserve the head's dtype rounding. All vocabulary
    entries contribute; no probability/gradient threshold is applied. CUDA
    atomics can change the final FP32 reduction order between executions.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or labels.shape != hidden.shape[:1]:
        raise ValueError("CCE expects hidden [tokens, hidden], weight [vocab, hidden], labels [tokens]")
    if hidden.shape[1] != weight.shape[1] or not weight.shape[0] or not weight.shape[1]:
        raise ValueError("CCE requires matching nonempty hidden dimensions and a nonempty vocabulary")
    if hidden.device.type != "cuda" or weight.device != hidden.device or labels.device != hidden.device:
        raise ValueError("CCE requires hidden, weight and labels on the same CUDA device")
    if hidden.dtype != weight.dtype or hidden.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("CCE requires matching float32, bfloat16 or float16 hidden and weight")
    if labels.dtype != torch.long:
        raise ValueError("CCE labels must have dtype int64")
    return _CutLogprobs.apply(
        hidden.contiguous(),
        weight.contiguous(),
        labels.contiguous(),
        float(softcap),
        int(vocab_start),
        group,
        int(world_size),
    )
