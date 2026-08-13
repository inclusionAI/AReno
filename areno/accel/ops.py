"""Stable Python surface over the areno.accel fused kernels.

Re-exports fused activation wrappers and the Triton-based kernels
(fused MoE experts, grouped RMSNorm with sigmoid gate, segmented linear
attention). Also exposes small utilities used throughout the layer code:

- ``log_once`` / ``warn_once``: emit a logger message exactly once per
  process for a given key, so kernel-selection diagnostics do not flood
  training logs.
- ``can_use_cuda_kernel``: CUDA-device gate used by kernel-selection code.
"""

from __future__ import annotations

from typing import Any

import torch

from areno.accel.activations import areno_gelu_tanh_and_mul, areno_silu_and_mul
from areno.accel.attention import (
    areno_causal_attention,
    areno_paged_causal_attention_decode,
    areno_varlen_causal_attention,
)
from areno.accel.kernels.fused_moe import FusedMoeConfig
from areno.accel.kernels.fused_moe import fused_experts as areno_fused_experts
from areno.accel.kernels.fused_moe import is_available as fused_moe_is_available
from areno.accel.kernels.group_rmsnorm import rms_norm_gate_fwd
from areno.accel.kernels.seg_la import SegLaMeta, seg_la_fwd
from areno.engine.log import log_once, warn_once


@torch._dynamo.disable
def is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
    """True if the tensor lives on CUDA and we are inside a graph capture."""

    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()


@torch._dynamo.disable
def can_use_cuda_kernel(tensor: torch.Tensor, name: str, *, allow_sm121: bool = False) -> bool:
    """Decide whether to take the fused kernel path for ``tensor``.

    Returns False only on non-CUDA tensors. ``name`` and ``allow_sm121`` are
    kept for compatibility with existing call sites.
    """

    if not tensor.is_cuda:
        return False
    return True


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
