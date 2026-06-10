"""Fused activation kernels (SiLU, GELU-tanh, sigmoid, softplus, gated variants).

Each wrapper dispatches to the ARENO CUDA kernel via the compiled extension and
preserves autograd by routing through ``torch.autograd.Function`` whenever the
input requires gradients. The ``*_and_mul`` variants implement the common
"gated MLP" pattern where the last input dimension is split in two and the
first half is activated then element-wise multiplied with the second half,
producing an output with half the last-dimension size.
"""
import torch

from areno.accel._extension import extension as _extension


def _activation_out(x: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """Allocate or validate the half-width output tensor for ``*_and_mul`` ops."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"activation input last dimension must be even, got {x.shape[-1]}")
    hidden = x.shape[-1] // 2
    expected_shape = (*x.shape[:-1], hidden)
    if out is None:
        return torch.empty(expected_shape, device=x.device, dtype=x.dtype)
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"activation output shape must be {expected_shape}, got {tuple(out.shape)}")
    return out


def _can_use_cuda_extension(x: torch.Tensor) -> bool:
    """Guard that the input lives on CUDA; the kernels have no CPU path."""
    if not x.is_cuda:
        raise RuntimeError("ARENO activation kernels require CUDA tensors")
    return True


class _SiluMul(torch.autograd.Function):
    """Autograd glue for fused SiLU(x[..., :H]) * x[..., H:]."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        out = _activation_out(x, None)
        _extension().areno_silu_and_mul(out, x.contiguous())
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        grad_input = torch.empty_like(x)
        _extension().areno_d_silu_and_mul(grad_input, grad_output.contiguous(), x.contiguous())
        return (grad_input,)


class _Silu(torch.autograd.Function):
    """Autograd glue for the element-wise SiLU kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        out = _extension().areno_silu(x.contiguous())
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        return (_extension().areno_d_silu(grad_output.contiguous(), x.contiguous()),)


class _Sigmoid(torch.autograd.Function):
    """Autograd glue for the element-wise sigmoid kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        out = _extension().areno_sigmoid(x.contiguous())
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (out,) = ctx.saved_tensors
        return (_extension().areno_d_sigmoid(grad_output.contiguous(), out.contiguous()),)


class _Softplus(torch.autograd.Function):
    """Autograd glue for the element-wise softplus kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        out = _extension().areno_softplus(x.contiguous())
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        return (_extension().areno_d_softplus(grad_output.contiguous(), x.contiguous()),)


class _GeluTanhMul(torch.autograd.Function):
    """Autograd glue for fused tanh-approx GELU(x[..., :H]) * x[..., H:]."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        out = _activation_out(x, None)
        _extension().areno_gelu_tanh_and_mul(out, x.contiguous())
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        grad_input = torch.empty_like(x)
        _extension().areno_d_gelu_tanh_and_mul(grad_input, grad_output.contiguous(), x.contiguous())
        return (grad_input,)


@torch._dynamo.disable
def areno_silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Apply SiLU to the first half of the last dimension and multiply by the second half.

    Input shape (..., 2H) -> output shape (..., H). When an ``out`` tensor is
    supplied the autograd path is skipped and the kernel writes in-place.
    """
    result = _activation_out(x, out)
    _can_use_cuda_extension(x)
    # Only enter the autograd Function when we actually need gradients; the
    # plain kernel path is used for inference / when an out buffer is given.
    if out is None and torch.is_grad_enabled() and x.requires_grad:
        return _SiluMul.apply(x)
    _extension().areno_silu_and_mul(result, x.contiguous())
    return result


@torch._dynamo.disable
def areno_silu(x: torch.Tensor) -> torch.Tensor:
    """Apply SiLU with an ARENO CUDA kernel."""
    _can_use_cuda_extension(x)
    if torch.is_grad_enabled() and x.requires_grad:
        return _Silu.apply(x)
    return _extension().areno_silu(x.contiguous())


@torch._dynamo.disable
def areno_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid with an ARENO CUDA kernel."""
    _can_use_cuda_extension(x)
    if torch.is_grad_enabled() and x.requires_grad:
        return _Sigmoid.apply(x)
    return _extension().areno_sigmoid(x.contiguous())


@torch._dynamo.disable
def areno_softplus(x: torch.Tensor) -> torch.Tensor:
    """Apply softplus(beta=1, threshold=20) with an ARENO CUDA kernel."""
    _can_use_cuda_extension(x)
    if torch.is_grad_enabled() and x.requires_grad:
        return _Softplus.apply(x)
    return _extension().areno_softplus(x.contiguous())


@torch._dynamo.disable
def areno_gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Apply tanh-approximate GELU to the first half and multiply by the second half.

    Input shape (..., 2H) -> output shape (..., H). Matches the GeGLU
    formulation used by Gemma-family MLPs.
    """
    result = _activation_out(x, out)
    _can_use_cuda_extension(x)
    if out is None and torch.is_grad_enabled() and x.requires_grad:
        return _GeluTanhMul.apply(x)
    _extension().areno_gelu_tanh_and_mul(result, x.contiguous())
    return result
