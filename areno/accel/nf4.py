"""Frozen NF4 weights with nested dynamic 8-bit quantization of block scales.

NF4 blocks contain 64 contiguous weights. Their mean-centered scales use
256-scale blocks, matching QLoRA's double-quantization layout. Only packed
weights and nested metadata persist; backward reconstructs one projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from areno.engine.optim.dynamic_quant import SIGNED_DYNAMIC_MAP

# QLoRA Appendix E: asymmetric normal quantiles, including an exact zero.
NF4_VALUES = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


class NF4Weight(nn.Module):
    """Serializable packed weight; deliberately owns no trainable parameter."""

    @torch.no_grad()
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        if weight.ndim not in (2, 3) or not weight.numel():
            raise ValueError("NF4 requires a nonempty linear or grouped-linear weight")
        if weight.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError("NF4 requires a floating point weight")
        self.shape = tuple(weight.shape)
        self.compute_dtype = weight.dtype
        flat = weight.detach().contiguous().view(-1)
        count = flat.numel()
        scales = torch.empty((count + 63) // 64, device=weight.device, dtype=torch.float32)
        packed = torch.empty((count + 1) // 2, device=weight.device, dtype=torch.uint8)
        code = torch.tensor(NF4_VALUES, device=weight.device, dtype=torch.float32)
        thresholds = (code[1:] + code[:-1]) / 2
        # Bound loading-time scratch; never expand the entire checkpoint to FP32.
        for start in range(0, count, 1024 * 1024):
            end = min(start + 1024 * 1024, count)
            values = F.pad(flat[start:end].float(), (0, (start - end) % 64)).view(-1, 64)
            scale = values.abs().amax(-1)
            scales[start // 64 : (end + 63) // 64] = scale
            norm = values / scale.clamp_min(torch.finfo(torch.float32).tiny)[:, None]
            indices = torch.bucketize(norm, thresholds).flatten().to(torch.uint8)
            pairs = (indices[::2] << 4) | indices[1::2]
            packed[start // 2 : (end + 1) // 2] = pairs[: (end - start + 1) // 2]
        offset = scales.mean()
        centered = F.pad(scales - offset, (0, -scales.numel() % 256)).view(-1, 256)
        nested_absmax = centered.abs().amax(-1)
        dynamic = torch.tensor(SIGNED_DYNAMIC_MAP, device=weight.device, dtype=torch.float32)
        normalized = centered / nested_absmax.clamp_min(torch.finfo(torch.float32).tiny)[:, None]
        nested = torch.bucketize(normalized, (dynamic[1:] + dynamic[:-1]) / 2).flatten()
        self.register_buffer("packed", packed)
        self.register_buffer("scale_codes", nested[: scales.numel()].to(torch.uint8))
        self.register_buffer("scale_absmax", nested_absmax)
        self.register_buffer("scale_offset", offset)
        self.register_buffer("codebook", code)
        self.register_buffer("scale_codebook", dynamic)

    @property
    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers())

    def dequantize(self) -> torch.Tensor:
        if self.packed.is_cuda:
            from areno.accel.kernels.nf4 import dequantize

            return dequantize(self)
        # CPU reference also enables format/round-trip tests without a GPU.
        count = math.prod(self.shape)
        indices = torch.stack((self.packed >> 4, self.packed & 15), dim=-1).flatten()[:count].long()
        blocks = torch.arange(self.scale_codes.numel(), device=self.packed.device) // 256
        scales = self.scale_codebook[self.scale_codes.long()] * self.scale_absmax[blocks] + self.scale_offset
        values = self.codebook[indices] * scales.repeat_interleave(64)[:count]
        return values.to(self.compute_dtype).view(self.shape)


class _NF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quantized, bias):
        ctx.quantized = quantized
        ctx.has_bias = bias is not None
        return F.linear(x, quantized.dequantize(), bias)

    @staticmethod
    def backward(ctx, grad):
        dx = grad @ ctx.quantized.dequantize() if ctx.needs_input_grad[0] else None
        db = grad.reshape(-1, grad.shape[-1]).sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, None, db


@torch._dynamo.disable
def nf4_linear(x, quantized, bias=None):
    return _NF4Linear.apply(x, quantized, bias)


class _NF4GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quantized, counts):
        from areno.accel._extension import extension

        ctx.quantized, ctx.input_shape = quantized, x.shape
        ctx.save_for_backward(counts)
        return extension(x.device).areno_grouped_linear_forward_counts(
            x.contiguous(), quantized.dequantize(), counts.contiguous()
        )

    @staticmethod
    def backward(ctx, grad):
        from areno.accel._extension import extension

        (counts,) = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            # Input values are unused when the frozen weight gradient is disabled.
            shape_only_input = grad.new_empty(ctx.input_shape)
            dx, _ = extension(grad.device).areno_grouped_linear_backward_counts(
                grad.contiguous(), shape_only_input, ctx.quantized.dequantize(), counts.contiguous(), True, False
            )
        return dx, None, None


@torch._dynamo.disable
def nf4_grouped_linear(x, quantized, counts):
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts, device=x.device, dtype=torch.int64)
    return _NF4GroupedLinear.apply(x, quantized, counts)
