"""Fused RMSNorm variants used throughout areno transformer blocks.

Three variants are exposed: vanilla ``areno_rmsnorm``, an optional-scale form
where the gain vector may be omitted, and a fused
``areno_rmsnorm_silu_gate`` that combines normalization, a SiLU-gated branch,
and channel scaling in a single CUDA pass. All variants persist a per-row
``inv_rms`` tensor across forward/backward so the gradient kernel avoids the
extra reduction.
"""
import torch

from areno.accel._extension import extension as _extension


def _kernel_weight(weight: torch.Tensor) -> torch.Tensor:
    """Cast the gain vector to float32 to match the CUDA kernel expectation."""
    return weight if weight.dtype == torch.float32 else weight.float()


class _RMSNorm(torch.autograd.Function):
    """Autograd glue for fused RMSNorm with mandatory channel scale."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        out, inv_rms = _extension().areno_rmsnorm_forward(x.contiguous(), weight.contiguous(), float(eps))
        ctx.save_for_backward(x, weight, inv_rms)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight, inv_rms = ctx.saved_tensors
        grad_input, grad_weight = _extension().areno_rmsnorm_backward(
            grad_output.contiguous(),
            x.contiguous(),
            weight.contiguous(),
            inv_rms,
        )
        return grad_input, grad_weight, None


class _OptionalScaleRMSNorm(torch.autograd.Function):
    """Autograd glue for RMSNorm where the channel scale may be absent."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
        use_scale = weight is not None
        # Kernel takes a concrete tensor even when scale is unused; pass empty.
        kernel_weight = weight if use_scale else torch.empty(0, device=x.device, dtype=torch.float32)
        out, inv_rms = _extension().areno_optional_scale_rmsnorm_forward(
            x.contiguous(),
            kernel_weight.contiguous(),
            float(eps),
            use_scale,
        )
        ctx.save_for_backward(x, kernel_weight, inv_rms)
        ctx.use_scale = use_scale
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        x, weight, inv_rms = ctx.saved_tensors
        grad_input, grad_weight = _extension().areno_optional_scale_rmsnorm_backward(
            grad_output.contiguous(),
            x.contiguous(),
            weight.contiguous(),
            inv_rms,
            ctx.use_scale,
        )
        return grad_input, grad_weight if ctx.use_scale else None, None


class _RMSNormSiluGate(torch.autograd.Function):
    """Autograd glue for fused RMSNorm(x) * SiLU(gate) * weight."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        out, inv_rms = _extension().areno_rmsnorm_silu_gate_forward(
            x.contiguous(),
            gate.contiguous(),
            weight.contiguous(),
            float(eps),
        )
        ctx.save_for_backward(x, gate, weight, inv_rms)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        x, gate, weight, inv_rms = ctx.saved_tensors
        grad_input, grad_gate, grad_weight = _extension().areno_rmsnorm_silu_gate_backward(
            grad_output.contiguous(),
            x.contiguous(),
            gate.contiguous(),
            weight.contiguous(),
            inv_rms,
        )
        return grad_input, grad_gate, grad_weight, None


@torch._dynamo.disable
def areno_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply ARENO RMSNorm with fused CUDA forward and backward on CUDA tensors.

    Normalizes over the last dimension. Output dtype matches ``x``; ``weight``
    is internally cast to float32 to match the kernel signature.
    """
    if not x.is_cuda:
        raise RuntimeError("areno_rmsnorm requires CUDA input")
    weight = _kernel_weight(weight)
    return _RMSNorm.apply(x, weight, float(eps))


@torch._dynamo.disable
def areno_optional_scale_rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Apply RMSNorm with an optional channel scale using an ARENO CUDA kernel.

    Passing ``weight=None`` skips the per-channel multiply entirely.
    """
    if not x.is_cuda:
        raise RuntimeError("areno_optional_scale_rmsnorm requires CUDA input")
    if weight is not None:
        weight = _kernel_weight(weight)
    return _OptionalScaleRMSNorm.apply(x, weight, float(eps))


@torch._dynamo.disable
def areno_rmsnorm_silu_gate(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMSNorm, SiLU gate, and channel scale in one ARENO CUDA kernel.

    Computes ``rmsnorm(x) * silu(gate) * weight``; ``x`` and ``gate`` must
    share shape, ``weight`` is the per-channel gain over the last dimension.
    """
    if not x.is_cuda or not gate.is_cuda:
        raise RuntimeError("areno_rmsnorm_silu_gate requires CUDA input and gate")
    if x.shape != gate.shape:
        raise ValueError(f"input/gate shape mismatch: {tuple(x.shape)} vs {tuple(gate.shape)}")
    weight = _kernel_weight(weight)
    return _RMSNormSiluGate.apply(x, gate, weight, float(eps))
