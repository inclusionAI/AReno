"""Stable Python surface over the areno.accel fused kernels.

Re-exports fused activation wrappers and the Triton-based kernels
(fused MoE experts, grouped RMSNorm with sigmoid gate, segmented linear
attention). Adds two small utilities used throughout the layer code:

- ``log_once`` / ``warn_once``: emit a logger message exactly once per
  process for a given key, so kernel-selection diagnostics do not flood
  training logs.
- ``can_use_cuda_kernel``: CUDA-device gate used by kernel-selection code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from areno.accel._extension import extension
from areno.accel.activations import areno_gelu_tanh_and_mul, areno_silu_and_mul
from areno.accel.attention import (
    areno_causal_attention,
    areno_paged_causal_attention_decode,
    areno_varlen_causal_attention,
)
from areno.accel.utils import can_use_cuda_kernel, is_cuda_graph_capturing, log_once, on_kernel_device, warn_once

__all__ = [
    "Any",
    "FusedMoeConfig",
    "SegLaMeta",
    "areno_fused_experts",
    "can_use_cuda_kernel",
    "fused_moe_is_available",
    "is_cuda_graph_capturing",
    "log_once",
    "rms_norm_gate_fwd",
    "seg_la_fwd",
    "areno_gelu_tanh_and_mul",
    "areno_causal_attention",
    "areno_paged_causal_attention_decode",
    "areno_varlen_causal_attention",
    "areno_silu_and_mul",
    "warn_once",
]


@dataclass(slots=True)
class FusedMoeConfig:
    """Static configuration for one fused MoE layer.

    Attributes:
        num_experts: Total number of experts in this layer.
        hidden_size: Model hidden dimension (size of the per-token MoE input
            and the final per-token output).
        intermediate_size: Width of the MLP expansion inside each expert,
            i.e. the column count of ``w1`` (before the SiLU+mul halving).
        top_k: Number of experts each token is routed to.
        routed_scaling_factor: Multiplier applied during the top-k sum-reduce
            (DeepSeek-style routed-expert rescaling).
        block_size_m: M tile size of the grouped matmul. Tokens are padded to
            multiples of this so each tile sees one expert exclusively.
        block_size_n: N tile size (output features per program).
        block_size_k: K tile size (reduction inner dim per iteration).
        group_size_m: L2-friendly M-supergroup factor; controls the
            ``pid_m / pid_n`` swizzle inside the matmul kernel.
    """

    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    routed_scaling_factor: float = 1.0
    block_size_m: int = 16
    block_size_n: int = 64
    block_size_k: int = 64
    group_size_m: int = 8


@dataclass
class SegLaMeta:
    """Per-batch metadata required to dispatch segmented linear attention.

    The kernel processes many variable-length requests in one launch; this
    struct packs the descriptors. ``q_offsets`` gives the start index of each
    request inside the flattened (sum_l, heads, head_dim) Q/K/V tensors,
    ``q_lengths`` gives each request's length. ``s_offsets`` is the slot id
    in the persistent state pool (or ``-1`` to skip an entry), and
    ``s_scales`` is 0 for the very first prefill chunk of a request (state
    zero-initialised inside the kernel) or 1 for continuation chunks (state
    loaded from ``S``).
    """

    batch_size: int  # batch size, num of requests
    max_q_length: int  # max(seq_lens)
    q_offsets: torch.Tensor  # [bs+1], query_start_locations,
    s_offsets: torch.Tensor  # [bs], slot_ids
    q_lengths: torch.Tensor  # [bs], query length
    s_scales: torch.Tensor  # [bs], prefill = 0, decode = 1
    s_offsets_stride: int = 0
    q_offsets_stride: int = 0
    s_scales_stride: int = 0
    decay_scales_stride: int = 0
    mask: torch.Tensor | None = None  # Currently not supported


def areno_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    config: FusedMoeConfig,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    if not on_kernel_device(hidden_states, w1, w2, topk_weights, topk_ids):
        raise RuntimeError("fused_experts requires CUDA or NPU tensors on the same device")
    if hidden_states.device.type == "npu":
        return extension(hidden_states.device).areno_fused_experts(
            hidden_states, w1, w2, topk_weights, topk_ids, config, activation=activation
        )
    from areno.accel.kernels.fused_moe import fused_experts

    return fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, config, activation=activation)


def fused_moe_is_available():
    # The CUDA implementation exposes this same unconditional capability.
    # Importing its Triton module here would also initialize it on NPU hosts.
    return True


def rms_norm_gate_fwd(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if not on_kernel_device(x, gate, weight):
        raise RuntimeError("rms_norm_gate_fwd requires CUDA or NPU tensors on the same device")
    if x.device.type == "npu":
        return extension(x.device).rms_norm_gate_fwd(x, gate, weight, eps)
    from areno.accel.kernels.group_rmsnorm import rms_norm_gate_fwd as implementation

    return implementation(x, gate, weight, eps)


def seg_la_fwd(q, k, v, s, decay_scales, meta, caches=None, softmax_scale=None):
    if not on_kernel_device(
        q, k, v, s, decay_scales, meta.q_offsets, meta.s_offsets, meta.q_lengths, meta.s_scales, meta.mask, caches
    ):
        raise RuntimeError("seg_la_fwd requires CUDA or NPU tensors on the same device")
    if q.device.type == "npu":
        return extension(q.device).seg_la_fwd(q, k, v, s, decay_scales, meta, caches, softmax_scale)
    from areno.accel.kernels.seg_la import seg_la_fwd as implementation

    return implementation(q, k, v, s, decay_scales, meta, caches, softmax_scale)
