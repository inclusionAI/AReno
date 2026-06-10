"""Fused grouped top-k router kernel for MoE routing decisions.

Implements DeepSeek-style "group sigmoid + bias top-k" routing in a single
CUDA kernel: experts are divided into ``num_groups`` groups, ``topk_group``
groups are selected per token, and finally ``top_k`` experts are picked from
those groups using sigmoid scores plus a learned ``expert_bias`` adjustment.
The kernel returns both the expert indices and their normalized routing
weights ready for the permute / grouped-GEMM pipeline.
"""
import torch

from areno.accel._extension import extension as _extension


@torch._dynamo.disable
def areno_grouped_topk_router(
    logits: torch.Tensor,
    expert_bias: torch.Tensor,
    top_k: int,
    num_groups: int,
    topk_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route tokens with grouped top-k using sigmoid scores and expert-bias selection.

    ``logits`` shape: ``(tokens, experts)``; ``expert_bias`` is a float32
    per-expert score offset. Returns ``(topk_idx, topk_weight)`` with shapes
    ``(tokens, top_k)`` for downstream permute / unpermute kernels.
    """
    if not logits.is_cuda or not expert_bias.is_cuda:
        raise RuntimeError("areno_grouped_topk_router requires CUDA logits and expert_bias")
    if logits.dim() != 2:
        raise ValueError(f"logits must have shape (tokens, experts), got {tuple(logits.shape)}")
    if expert_bias.dtype != torch.float32:
        raise TypeError("areno_grouped_topk_router expert_bias must be float32")
    topk_idx, topk_weight = _extension().areno_grouped_topk_router(
        logits.contiguous(),
        expert_bias.contiguous(),
        int(top_k),
        int(num_groups),
        int(topk_group),
    )
    return topk_idx, topk_weight
