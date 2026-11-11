"""Linear-time MXFP4 checkpoint materialization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode(B, S, O, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(B + i // 2, i < N, 0)
    code = tl.where(i % 2 == 0, b & 15, b >> 4)
    x = code & 7
    magnitude = tl.where(
        x < 2, x * 0.5, tl.where(x < 4, (x - 2) * 0.5 + 1, tl.where(x < 6, x - 4 + 2, (x - 6) * 2 + 4))
    )
    sign = tl.where(code >= 8, -1.0, 1.0)
    exponent = tl.load(S + i // 32, i < N, 127).to(tl.int32) - 127
    value = sign * magnitude * tl.exp2(exponent.to(tl.float32))
    tl.store(O + i, value, i < N)


def decode(blocks, scales, dtype):
    out = torch.empty(blocks.numel() * 2, device=blocks.device, dtype=dtype)
    _decode[(triton.cdiv(out.numel(), 1024),)](blocks, scales, out, out.numel(), 1024)
    return out
