"""SwiGLU gated MLP block.

Implements the standard `down(SiLU(gate) * up)` pattern using a fused
column-parallel `gate_up` projection and a row-parallel `down` projection.
The SiLU-and-multiply is dispatched to the areno.accel fused kernel.
"""

from __future__ import annotations

import torch
from torch import nn

from areno.engine.config import ModelConfig
from areno.engine.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from areno.accel.ops import areno_silu_and_mul, log_once


class GatedMLP(nn.Module):
    """SwiGLU MLP with TP-fused gate/up and TP row-parallel down.

    ``gate_up_proj`` is a merged column-parallel linear so gate and up share
    one matmul; the result is fed through the fused SiLU-multiply kernel and
    the activation is shrunk back to hidden_size by the row-parallel down.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Two stacked column-parallel projections (gate, up) sharing input,
        # each sized intermediate_size and sharded across the TP group.
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            (config.intermediate_size, config.intermediate_size),
            bias=False,
        )
        # Row-parallel projection that all-reduces (or reduce-scatters under
        # sequence parallelism) the partial sums.
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gate||up matmul; result layout is [..., gate_local | up_local].
        gate_up = self.gate_up_proj(x)
        log_once("areno_silu_and_mul", "using ARENO fused silu_and_mul kernel")
        # SwiGLU: hidden = SiLU(gate) * up, computed in a single kernel.
        hidden = _areno_silu_and_mul_no_compile(gate_up)
        return self.down_proj(hidden)


@torch._dynamo.disable
def _areno_silu_and_mul_no_compile(x: torch.Tensor) -> torch.Tensor:
    """Dynamo-opaque wrapper so the fused kernel survives torch.compile."""

    return areno_silu_and_mul(x)
