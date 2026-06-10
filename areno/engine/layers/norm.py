"""Normalization layers (vanilla and gated RMSNorm).

`RMSNorm` is the standard root-mean-square norm with a learnable scale,
delegating to the areno.accel fused kernel. `GroupRMSNormSigmoidGate`
implements a fused per-group RMSNorm followed by a sigmoid gate, used by
linear-attention style blocks; the weight is sharded so each TP rank holds
only ``groups_per_rank`` groups.
"""

from __future__ import annotations

import torch
from torch import nn

from areno.accel import areno_rmsnorm
from areno.engine.layers.linear import mark_tensor_parallel_parameter
from areno.accel.ops import can_use_cuda_kernel, log_once, rms_norm_gate_fwd


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel scale.

    The scale weight is stored in fp32 regardless of the activation dtype
    to preserve numerical precision; the fused kernel performs the cast
    internally. By default the weight is replicated across the TP group but
    flagged as sequence-parallel, so gradient hooks know to all-reduce
    contributions from each sequence shard.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        tensor_model_parallel: bool = False,
        sequence_parallel: bool = True,
        tp_grad_allreduce: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        mark_tensor_parallel_parameter(
            self.weight,
            tensor_model_parallel,
            sequence_parallel=sequence_parallel,
            tp_grad_allreduce=tp_grad_allreduce,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _areno_rmsnorm_no_compile(x, self.weight, self.eps)


@torch._dynamo.disable
def _areno_rmsnorm_no_compile(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Dynamo-opaque wrapper around the fused RMSNorm kernel."""

    return areno_rmsnorm(x, weight, eps)


class GroupRMSNormSigmoidGate(nn.Module):
    """Fused group-wise RMSNorm followed by a sigmoid gate.

    Splits the hidden dim into ``group_norm_size`` equal-width groups and
    normalizes each group independently, then multiplies by ``sigmoid(gate)``
    using a fused Triton kernel. The group axis is the TP shard axis: each
    rank owns ``groups_per_rank = group_norm_size // tp_size`` groups, so
    the weight is shaped ``(groups_per_rank, group_width)``.
    """

    def __init__(self, hidden_size: int, group_norm_size: int, tp_size: int, eps: float):
        super().__init__()
        if group_norm_size <= 1:
            raise ValueError("group_norm_size must be > 1 for grouped gate norm")
        if tp_size > group_norm_size or group_norm_size % tp_size != 0:
            raise ValueError(f"group_norm_size={group_norm_size} must be divisible by tp_size={tp_size}")
        if hidden_size % group_norm_size != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by group_norm_size={group_norm_size}")
        self.group_norm_size = group_norm_size
        # Local groups per rank after sharding the group axis across TP.
        self.groups_per_rank = group_norm_size // tp_size
        # Channels inside each normalization group.
        self.group_width = hidden_size // group_norm_size
        self.weight = nn.Parameter(torch.ones(self.groups_per_rank, self.group_width))
        # Weight is TP-sharded along the group axis, no SP gradient sum needed.
        mark_tensor_parallel_parameter(self.weight, True, sequence_parallel=False)
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        # Reshape last dim into (groups_per_rank, group_width) for the kernel.
        x = x.view(*shape[:-1], self.groups_per_rank, self.group_width)
        gate = gate.view(*shape[:-1], self.groups_per_rank, self.group_width)
        if rms_norm_gate_fwd is None or not can_use_cuda_kernel(x, "fused group RMSNorm sigmoid gate kernel"):
            raise RuntimeError("ARENO group RMSNorm sigmoid gate requires the fused CUDA kernel")
        log_once("group_rmsnorm_sigmoid_gate", "using fused group RMSNorm sigmoid gate kernel")
        # Flatten the leading dims into a single batch so the kernel only
        # sees a 3D (B, groups, width) tensor.
        out, _ = rms_norm_gate_fwd(
            x.reshape(-1, self.groups_per_rank, self.group_width),
            gate.reshape(-1, self.groups_per_rank, self.group_width),
            self.weight.to(dtype=x.dtype),
            self.eps,
        )
        return out.view(shape)
