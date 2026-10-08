"""Decode packed NF4 and nested scales in a single bandwidth-bound pass."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant(Packed, Codes, Absmax, Offset, Codebook, ScaleCodebook, Out, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    byte = tl.load(Packed + i // 2, i < N, other=0).to(tl.int32)
    code = tl.where(i % 2 == 0, byte >> 4, byte & 15)
    scale_code = tl.load(Codes + i // 64, i < N, other=0).to(tl.int32)
    scale = tl.load(ScaleCodebook + scale_code) * tl.load(Absmax + i // (64 * 256), i < N, other=0)
    scale = scale + tl.load(Offset)
    value = tl.load(Codebook + code) * scale
    tl.store(Out + i, value, i < N)


def dequantize(weight):
    out = torch.empty(weight.shape, device=weight.packed.device, dtype=weight.compute_dtype)
    n = math.prod(weight.shape)
    _dequant[(triton.cdiv(n, 1024),)](
        weight.packed,
        weight.scale_codes,
        weight.scale_absmax,
        weight.scale_offset,
        weight.codebook,
        weight.scale_codebook,
        out,
        n,
        1024,
        enable_fp_fusion=False,
    )
    return out
