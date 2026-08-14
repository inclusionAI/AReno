"""AReno KDA wrappers for flash-linear-attention.

This module keeps the Bailing V3 call contract and only overrides the KDA
pieces that need AReno-specific behavior.  Shared kernels stay imported from
FLA so we do not vendor the full FLA operator stack.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def activate_kda_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Apply SGLang/FLA KDA gate activation without the chunk cumulative sum."""

    head_count, key_dim = raw_gate.shape[-2:]
    if a_log.numel() != head_count or dt_bias.numel() != head_count * key_dim:
        raise ValueError("KDA gate parameters do not match gate shape")
    rate = a_log.float().exp().view(1, 1, head_count, 1)
    gate_input = raw_gate.float() + dt_bias.float().view(1, 1, head_count, key_dim)
    if lower_bound is not None:
        decay = lower_bound * torch.sigmoid(rate * gate_input)
    else:
        decay = -rate * F.softplus(gate_input)
    return decay.to(dtype=raw_gate.dtype)


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    if initial_state_indices is not None and initial_state is not None:
        initial_state = initial_state.index_select(0, initial_state_indices)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if A_log is None or dt_bias is None:
        raise ValueError("KDA prefill requires A_log and dt_bias")
    decay = activate_kda_gate(g, A_log, dt_bias, lower_bound)

    if use_qk_l2norm_in_kernel:
        q_float = q.float()
        k_float = k.float()
        q = (q_float * torch.rsqrt((q_float * q_float).sum(dim=-1, keepdim=True) + 1e-6)).to(q.dtype)
        k = (k_float * torch.rsqrt((k_float * k_float).sum(dim=-1, keepdim=True) + 1e-6)).to(k.dtype)

    fla_initial_state = initial_state.transpose(-1, -2).contiguous() if initial_state is not None else None
    out, final_state = fla_chunk_kda(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=decay.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=fla_initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    if final_state is not None:
        final_state = final_state.transpose(-1, -2).contiguous()
    return out, final_state


__all__ = ["activate_kda_gate", "chunk_kda"]
