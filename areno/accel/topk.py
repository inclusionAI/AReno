"""Fused ARENO softmax top-k router."""

from __future__ import annotations

import torch

from areno.accel._extension import extension as _extension


class _TopKSoftmax(torch.autograd.Function):
    """Autograd glue for the ARENO softmax top-k router kernel."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, top_k: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        idx, weight = _extension().areno_topk_softmax_forward(logits.contiguous(), int(top_k), bool(renormalize))
        ctx.save_for_backward(logits, idx)
        ctx.renormalize = bool(renormalize)
        return idx, weight

    @staticmethod
    def backward(ctx, grad_idx: torch.Tensor, grad_weight: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        del grad_idx
        logits, idx = ctx.saved_tensors
        grad_logits = _extension().areno_topk_softmax_backward(grad_weight.contiguous(), logits.contiguous(), idx.contiguous(), ctx.renormalize)
        return grad_logits, None, None


@torch._dynamo.disable
def areno_topk_softmax(logits: torch.Tensor, top_k: int, renormalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(topk_idx, topk_weight)`` using ARENO CUDA kernels."""

    if not logits.is_cuda:
        raise RuntimeError("areno_topk_softmax requires CUDA logits")
    if logits.dim() != 2:
        raise ValueError(f"areno_topk_softmax logits must have shape (tokens, experts), got {tuple(logits.shape)}")
    return _TopKSoftmax.apply(logits, int(top_k), bool(renormalize))
