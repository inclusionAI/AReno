"""Bailing/Kimi Delta Attention accel entry points."""

from __future__ import annotations

import torch

from areno.accel.kernels.kda_fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from areno.accel.kernels.kda_fla.kda import chunk_kda


@torch._dynamo.disable
def areno_kda_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=state_indices,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
        cu_seqlens=cu_seqlens,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )


@torch._dynamo.disable
def areno_kda_recurrent_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    return fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=raw_gate,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=beta,
        initial_state_source=state,
        initial_state_indices=state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        is_kda=True,
        lower_bound=lower_bound,
    )


__all__ = ["areno_kda_chunk", "areno_kda_recurrent_update"]
