"""Pure Torch operations that define OLMo 2 decoder semantics."""

from __future__ import annotations

import torch
from torch import nn


def projected_rms_norm(
    hidden_states: torch.Tensor,
    squared_sum: torch.Tensor,
    weight: torch.Tensor,
    global_size: int,
    eps: float,
) -> torch.Tensor:
    """Normalize a TP shard using the squared sum over the global projection."""

    input_dtype = hidden_states.dtype
    scale = torch.rsqrt(squared_sum / global_size + eps)
    return (hidden_states.float() * scale * weight).to(dtype=input_dtype)


def post_norm_residual(residual: torch.Tensor, output: torch.Tensor, norm: nn.Module) -> torch.Tensor:
    """Apply OLMo 2's output normalization before the residual addition."""

    return residual + norm(output)
