"""TorchDynamo-safe wrappers shared by hybrid GatedDeltaNet adapters."""

from __future__ import annotations

import torch

from areno.accel import (
    areno_depthwise_causal_conv1d_silu,
    areno_depthwise_causal_conv1d_silu_decode,
    areno_packed_depthwise_causal_conv1d_silu,
    areno_rmsnorm_silu_gate,
    areno_sigmoid,
    areno_softplus,
)

try:
    from fla.modules.convolution import causal_conv1d as _fla_causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule

    _HAVE_FLA_GDN = True
except ImportError:
    _fla_causal_conv1d = None
    _fla_chunk_gated_delta_rule = None
    _fla_fused_recurrent_gated_delta_rule = None
    _HAVE_FLA_GDN = False


def _require_fla_gdn() -> None:
    if not _HAVE_FLA_GDN:
        raise ImportError("GatedDeltaNet adapters require flash-linear-attention (fla)")


@torch._dynamo.disable
def _areno_depthwise_causal_conv1d_silu_no_compile(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return areno_depthwise_causal_conv1d_silu(x, weight)


@torch._dynamo.disable
def _areno_packed_depthwise_causal_conv1d_silu_no_compile(x: torch.Tensor, weight: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    return areno_packed_depthwise_causal_conv1d_silu(x, weight, cu_seqlens)


@torch._dynamo.disable
def _areno_depthwise_causal_conv1d_silu_decode_no_compile(
    current: torch.Tensor,
    history: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return areno_depthwise_causal_conv1d_silu_decode(current, history, weight)


@torch._dynamo.disable
def _fla_causal_conv1d_no_compile(*args, **kwargs) -> torch.Tensor:
    _require_fla_gdn()
    result = _fla_causal_conv1d(*args, **kwargs)
    return result[0] if isinstance(result, tuple) else result


def _l2norm_no_autotune(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    return (x_float * torch.rsqrt((x_float * x_float).sum(dim=-1, keepdim=True) + eps)).to(dtype=x.dtype)


@torch._dynamo.disable
def _fla_chunk_gated_delta_rule_no_compile(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    _require_fla_gdn()
    args, kwargs = _prepare_fla_gdn_qk(*args, **kwargs)
    return _fla_chunk_gated_delta_rule(*args, **kwargs)


@torch._dynamo.disable
def _fla_fused_recurrent_gated_delta_rule_no_compile(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    _require_fla_gdn()
    args, kwargs = _prepare_fla_gdn_qk(*args, **kwargs)
    return _fla_fused_recurrent_gated_delta_rule(*args, **kwargs)


def _prepare_fla_gdn_qk(*args, **kwargs):
    if args:
        q, k, v, *rest = args
        args = (q, k, v, *rest)
    else:
        q = kwargs["q"]
        k = kwargs["k"]
        v = kwargs["v"]
    if kwargs.pop("use_qk_l2norm_in_kernel", False):
        q = _l2norm_no_autotune(q)
        k = _l2norm_no_autotune(k)
    if q.shape[2] != v.shape[2]:
        if v.shape[2] % q.shape[2] != 0:
            raise ValueError(f"cannot repeat q/k heads from {q.shape[2]} to {v.shape[2]}")
        repeat = v.shape[2] // q.shape[2]
        q = q.repeat_interleave(repeat, dim=2).contiguous()
        k = k.repeat_interleave(repeat, dim=2).contiguous()
    kwargs["use_qk_l2norm_in_kernel"] = False
    if args:
        args = (q, k, v, *args[3:])
    else:
        kwargs["q"] = q
        kwargs["k"] = k
    return args, kwargs


@torch._dynamo.disable
def _areno_sigmoid_no_compile(x: torch.Tensor) -> torch.Tensor:
    return areno_sigmoid(x)


@torch._dynamo.disable
def _areno_softplus_no_compile(x: torch.Tensor) -> torch.Tensor:
    return areno_softplus(x)


@torch._dynamo.disable
def _areno_rmsnorm_silu_gate_no_compile(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return areno_rmsnorm_silu_gate(x, gate, weight, eps)


__all__ = [
    "_areno_depthwise_causal_conv1d_silu_decode_no_compile",
    "_areno_depthwise_causal_conv1d_silu_no_compile",
    "_areno_packed_depthwise_causal_conv1d_silu_no_compile",
    "_areno_rmsnorm_silu_gate_no_compile",
    "_areno_sigmoid_no_compile",
    "_areno_softplus_no_compile",
    "_fla_causal_conv1d_no_compile",
    "_fla_chunk_gated_delta_rule_no_compile",
    "_fla_fused_recurrent_gated_delta_rule_no_compile",
    "_l2norm_no_autotune",
    "_prepare_fla_gdn_qk",
    "_require_fla_gdn",
]
