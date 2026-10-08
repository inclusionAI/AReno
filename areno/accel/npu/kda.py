"""KDA state/layout adapters; Ascend FLA owns the recurrent kernels."""

import torch

# This wrapper contains device-independent Torch preparation and calls FLA.
# Keep its gate rounding, normalization and training backward contract shared.
from areno.accel.kernels.kda_fla.kda import chunk_kda


def fused_sigmoid_gating_delta_rule_update(
    *,
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    scale,
    use_qk_l2norm_in_kernel,
    cu_seqlens,
    is_kda,
    lower_bound,
):
    from fla.ops.kda import fused_recurrent_kda

    if not is_kda or softplus_beta != 1.0 or softplus_threshold != 20.0:
        raise ValueError("the AReno KDA adapter requires KDA with the standard softplus gate")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, a, b, A_log, dt_bias, initial_state_source)):
        raise RuntimeError("KDA recurrent update is inference-only; use areno_kda_chunk for training")
    indices = initial_state_indices.to(dtype=torch.long)
    initial = initial_state_source.index_select(0, indices)
    out, final = fused_recurrent_kda(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=a.reshape(*v.shape[:3], q.shape[-1]).contiguous(),
        beta=b.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial,
        output_final_state=True,
        state_v_first=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    initial_state_source.index_copy_(0, indices, final.to(initial_state_source.dtype))
    return out


__all__ = ["chunk_kda", "fused_sigmoid_gating_delta_rule_update"]
