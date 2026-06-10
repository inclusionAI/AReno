"""Fused depthwise causal Conv1d + SiLU kernels for short-range token mixing.

Used by the linear-attention / Mamba-style layers in areno to mix recent
tokens within each channel. The kernel performs the convolution with explicit
causal padding (no future leak) and applies SiLU in the same pass; weights are
always coerced to ``float32`` so the CUDA path matches the reference layout.
The ``*_decode`` entry point handles the single-token autoregressive case
using a separately maintained history cache.
"""
import torch

from areno.accel._extension import extension as _extension


def _check_weight_shape(weight: torch.Tensor) -> None:
    """Validate depthwise weight layout (channels, 1, kernel_size)."""
    if weight.dim() != 3 or weight.shape[1] != 1:
        raise ValueError(f"weight must have shape (channels, 1, kernel), got {tuple(weight.shape)}")


def _kernel_weight(weight: torch.Tensor) -> torch.Tensor:
    """Cast convolution weights to float32 as required by the CUDA kernel."""
    _check_weight_shape(weight)
    return weight if weight.dtype == torch.float32 else weight.float()


class _DepthwiseCausalConv1dSilu(torch.autograd.Function):
    """Autograd glue for fused depthwise causal conv1d + SiLU.

    Forward returns the activated output; ``preact`` is saved so the backward
    pass can recompute the SiLU derivative cheaply.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out, preact = _extension().areno_depthwise_causal_conv1d_silu_forward(x.contiguous(), weight.contiguous())
        ctx.save_for_backward(x, weight, preact)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, weight, preact = ctx.saved_tensors
        grad_input, grad_weight = _extension().areno_depthwise_causal_conv1d_silu_backward(
            grad_output.contiguous(),
            x.contiguous(),
            weight.contiguous(),
            preact,
        )
        return grad_input, grad_weight


class _PackedDepthwiseCausalConv1dSilu(torch.autograd.Function):
    """Autograd glue for packed varlen depthwise causal conv1d + SiLU."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32).contiguous()
        out, preact = _extension().areno_packed_depthwise_causal_conv1d_silu_forward(
            x.contiguous(),
            weight.contiguous(),
            cu_seqlens,
        )
        ctx.save_for_backward(x, weight, cu_seqlens, preact)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight, cu_seqlens, preact = ctx.saved_tensors
        grad_input, grad_weight = _extension().areno_packed_depthwise_causal_conv1d_silu_backward(
            grad_output.contiguous(),
            x.contiguous(),
            weight.contiguous(),
            cu_seqlens,
            preact,
        )
        return grad_input, grad_weight, None


@torch._dynamo.disable
def areno_depthwise_causal_conv1d_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply depthwise causal conv1d followed by SiLU for (batch, seqlen, channels) tensors.

    The kernel handles arbitrary ``seqlen`` and uses left-padding to keep the
    convolution causal. Falls through to the inference path when autograd is
    disabled to avoid stashing the pre-activation tensor.
    """
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("areno_depthwise_causal_conv1d_silu requires CUDA input and weight")
    if x.dim() != 3:
        raise ValueError(f"input must have shape (batch, seqlen, channels), got {tuple(x.shape)}")
    weight = _kernel_weight(weight)
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"channel mismatch: input={x.shape[-1]} weight={weight.shape[0]}")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return _DepthwiseCausalConv1dSilu.apply(x, weight)
    out, _ = _extension().areno_depthwise_causal_conv1d_silu_forward(x.contiguous(), weight.contiguous())
    return out


@torch._dynamo.disable
def areno_packed_depthwise_causal_conv1d_silu(x: torch.Tensor, weight: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Apply depthwise causal conv1d followed by SiLU to packed (1, tokens, channels) tensors."""
    if not x.is_cuda or not weight.is_cuda or not cu_seqlens.is_cuda:
        raise RuntimeError("areno_packed_depthwise_causal_conv1d_silu requires CUDA tensors")
    if x.dim() != 3 or x.shape[0] != 1:
        raise ValueError(f"input must have shape (1, tokens, channels), got {tuple(x.shape)}")
    weight = _kernel_weight(weight)
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"channel mismatch: input={x.shape[-1]} weight={weight.shape[0]}")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return _PackedDepthwiseCausalConv1dSilu.apply(x, weight, cu_seqlens)
    out, _ = _extension().areno_packed_depthwise_causal_conv1d_silu_forward(
        x.contiguous(),
        weight.contiguous(),
        cu_seqlens.to(device=x.device, dtype=torch.int32).contiguous(),
    )
    return out


@torch._dynamo.disable
def areno_depthwise_causal_conv1d_silu_decode(current: torch.Tensor, history: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply one-token depthwise causal conv1d followed by SiLU for decode.

    ``current`` is the new token activations with shape ``(rows, channels)``
    and ``history`` provides the previous ``kernel_size - 1`` tokens shaped
    ``(rows, channels, kernel - 1)``. Returns the activated single-step output;
    callers are responsible for shifting ``history``.
    """
    if not current.is_cuda or not history.is_cuda or not weight.is_cuda:
        raise RuntimeError("areno_depthwise_causal_conv1d_silu_decode requires CUDA tensors")
    if current.dim() != 2:
        raise ValueError(f"current must have shape (rows, channels), got {tuple(current.shape)}")
    if history.dim() != 3:
        raise ValueError(f"history must have shape (rows, channels, kernel - 1), got {tuple(history.shape)}")
    weight = _kernel_weight(weight)
    out, _ = _extension().areno_depthwise_causal_conv1d_silu_decode(
        current.contiguous(),
        history.contiguous(),
        weight.contiguous(),
    )
    return out
