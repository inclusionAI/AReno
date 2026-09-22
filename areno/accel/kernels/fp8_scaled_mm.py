"""FP8 decode-linear backend built on torch._scaled_mm (A8W8).

Computes y = x @ (e4m3(w) * w_scale)^T on Hopper/Ada and later, where cuBLASLt
reads the 1-byte FP8 weight directly. _scaled_mm rejects a bf16 activation, so
the activation is quantized to E4M3 per call. The weight payload comes from
areno.engine.quantization.mark_fp8_weight and is refreshed in place per decode
session, so tensors captured by decode CUDA graphs keep their storage.
"""

from __future__ import annotations

import torch

from areno.accel.utils import FP8_E4M3_MAX

try:  # triton ships with torch on Linux; keep the module importable without it
    import triton
    import triton.language as tl

    @triton.jit
    def _quantize_act_kernel(x_ptr, scale_ptr, out_ptr, numel, E4M3_MAX: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < numel
        values = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # The scale comes from the largest magnitude, not the largest signed
        # value, so a negative tail cannot saturate; padding lanes load as 0.0.
        amax = tl.max(tl.abs(values), axis=0)
        scale = tl.where(amax > 0.0, amax / E4M3_MAX, 1.0)
        tl.store(scale_ptr, scale)
        tl.store(
            out_ptr + offs,
            tl.clamp(values / scale, -E4M3_MAX, E4M3_MAX).to(tl.float8e4nv),
            mask=mask,
        )

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# One program covers the whole tensor, so this caps the block size. Measured on H20
# under CUDA-graph replay the fused kernel beats the torch-op sequence 5-7x at decode
# sizes (up to 12288 elements), is break-even at 65536 and 2-17x slower beyond, so this
# is a real boundary rather than a conservative guess.
_FUSED_QUANT_MAX_ELEMENTS = 1 << 15


def scaled_mm_available(device: torch.device | int) -> bool:
    """Whether torch._scaled_mm (FP8 A8W8) can run on `device`."""

    if not torch.cuda.is_available() or not hasattr(torch, "_scaled_mm"):
        return False
    if isinstance(device, int):
        index = device
    elif device.type != "cuda":
        return False
    else:
        index = torch.cuda.current_device() if device.index is None else device.index
    return torch.cuda.get_device_capability(index) >= (8, 9)


def _quantize_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 quantize an activation into (fp8, float32 scalar scale)."""

    numel = x.numel()
    if _HAS_TRITON and x.is_cuda and numel <= _FUSED_QUANT_MAX_ELEMENTS:
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty((), dtype=torch.float32, device=x.device)
        _quantize_act_kernel[(1,)](x, scale, out, numel, E4M3_MAX=FP8_E4M3_MAX, BLOCK=triton.next_power_of_2(numel))
        return out, scale

    # Compute the scale in float32: a bf16 amax would round the division and
    # disagree with the kernel, which promotes to float32 before reducing.
    amax = x.abs().amax().to(torch.float32)
    scale = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
    return (x.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn), scale


def quantized_fp8_scaled_mm(x: torch.Tensor, fp8_weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """FP8 decode linear over an E4M3 payload, returning bf16.

    x is an (M, K) bf16 activation, fp8_weight an (N, K) E4M3 weight and
    weight_scale its per-tensor float32 scale.
    """

    if not scaled_mm_available(x.device):
        raise RuntimeError(
            f"torch._scaled_mm (FP8 A8W8) is unavailable on {x.device}; it requires compute capability >= 8.9"
        )
    assert fp8_weight.dtype == torch.float8_e4m3fn, f"fp8 payload must be float8_e4m3fn, got {fp8_weight.dtype}"
    activation_fp8, activation_scale = _quantize_act(x.contiguous())
    # cuBLASLt wants mat2 column-major (K, N): transpose the view, do not copy.
    return torch._scaled_mm(activation_fp8, fp8_weight.t(), activation_scale, weight_scale, out_dtype=torch.bfloat16)
