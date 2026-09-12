"""FP8 decode-linear backends for ``_areno_linear_forward``.

The current backend runs ``y = x @ (e4m3(w) * scale)^T`` through
``torch._scaled_mm`` (cuBLASLt) on Hopper/Ada (cc >= 8.9), which reads the
1-byte FP8 weight directly -- the decode memory-bound win: ~1.6-2x per linear
at Qwen3-8B shapes under CUDA graphs on H20. ``_scaled_mm`` is A8W8 (it
rejects a bf16 activation), so the activation is quantized to E4M3 per call;
a fused Triton kernel keeps that cheap for small (decode) activations and
larger (prefill) tensors use the torch-op sequence.

The weight payload is produced by ``areno.engine.quantization.mark_fp8_weight``
and refreshed in place per decode session, so tensors captured by decode CUDA
graphs keep their storage. Decode-only: FP8 has no backward, so this must not
enter a training graph. Adding another backend (e.g. a different instruction
target or format) means a sibling module selected by device capability in
``_areno_linear_forward``.
"""

from __future__ import annotations

import torch

try:  # triton ships with torch on Linux; keep the module importable without it
    import triton
    import triton.language as tl

    @triton.jit
    def _quantize_act_kernel(x_ptr, scale_ptr, out_ptr, numel, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        m = offs < numel
        v = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        amax = tl.max(tl.where(m, v, float("-inf")), axis=0)
        scale = amax / 448.0
        scale = tl.where(scale > 0, scale, 1.0)
        tl.store(scale_ptr, scale)
        tl.store(out_ptr + offs, tl.clamp(v / scale, -448.0, 448.0).to(tl.float8e4nv), mask=m)

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - torch-on-Linux always bundles triton
    _HAS_TRITON = False


_AVAILABLE: bool | None = None


def scaled_mm_available() -> bool:
    """``torch._scaled_mm`` (FP8 A8W8) needs Hopper/Ada (compute capability >= 8.9)."""
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = bool(
            hasattr(torch, "_scaled_mm") and torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 9)
        )
    return _AVAILABLE


# Largest numel the single-launch activation quantizer can cover. Decode
# activations are tiny (M x K), so one block computes the amax and quantizes in
# a single kernel; bigger (prefill) tensors use the torch-op sequence below.
_KQUANT_MAX_NUMEL = 1 << 15


def _quantize_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 quantize of the activation -> (fp8, scale)."""
    numel = x.numel()
    if _HAS_TRITON and x.is_cuda and numel <= _KQUANT_MAX_NUMEL:
        block = 1024
        while block < numel:
            block *= 2
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty((), dtype=torch.float32, device=x.device)
        _quantize_act_kernel[(1,)](x, scale, out, numel, BLOCK=block)
        return out, scale
    amax = x.abs().amax()
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax)).to(torch.float32)
    return (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


def quantized_fp8_scaled_mm(x: torch.Tensor, w_u8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """Hopper FP8 decode linear: ``y = x @ (e4m3(w) * w_scale)^T`` (A8W8).

    ``x`` is bf16 (M, K) and is quantized to E4M3 here; ``w_u8`` is the FP8
    weight (N, K) with the TP-consistent per-tensor ``w_scale``.
    """
    if not scaled_mm_available():
        raise RuntimeError("quantized_fp8_scaled_mm requires torch._scaled_mm (Hopper/Ada, cc >= 8.9)")
    if w_u8.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"fp8 weight payload must be float8_e4m3fn, got {w_u8.dtype}")
    xq, x_scale = _quantize_act(x.contiguous())
    # cuBLASLt wants mat2 column-major (K, N): transpose the view, do NOT
    # make it contiguous.
    return torch._scaled_mm(xq, w_u8.t(), x_scale, w_scale, out_dtype=torch.bfloat16)
