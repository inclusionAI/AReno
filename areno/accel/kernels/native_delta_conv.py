"""Packed causal convolution for native delta attention with parallel filter gradients."""

import torch
import triton
import triton.language as tl

from areno.accel._extension import extension


@triton.jit
def _silu_adjoint(g, z):
    sig = tl.sigmoid(z)
    return g * (sig * (1.0 + z * (1.0 - sig)))


@triton.jit
def _backward(
    X,
    W,
    CU,
    Z,
    DY,
    DX,
    PART,
    T,
    C: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    SX: tl.constexpr,
    SEARCH: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    block = tl.program_id(0)
    t = block * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    pos = t[:, None] * C + c[None, :]
    mask = (t[:, None] < T) & (c[None, :] < C)
    if S == 1:
        start, end = 0, T
    else:
        lo = tl.full((BT,), 0, tl.int32)
        for bit in tl.static_range(SEARCH - 1, -1, -1):
            mid = tl.minimum(lo + (1 << bit), S - 1)
            lo = tl.where(tl.load(CU + mid) <= t, mid, lo)
        start, end = tl.load(CU + lo), tl.load(CU + lo + 1)
    dz = _silu_adjoint(tl.load(DY + pos, mask, 0).to(tl.float32), tl.load(Z + pos, mask, 0))
    dx = tl.full((BT, BC), 0, tl.float32)
    for delay in tl.static_range(K):
        tap = K - 1 - delay
        valid = (t - delay >= start)[:, None] & mask
        x = tl.load(X + (t[:, None] - delay) * SX + c[None, :], valid, 0).to(tl.float32)
        dw = tl.sum(dz * x, 0)
        tl.store(PART + (block * C + c) * K + tap, dw, c < C)
        if delay == 0:
            future = dz
        else:
            future_pos = pos + delay * C
            valid = (t + delay < end)[:, None] & mask
            future = _silu_adjoint(tl.load(DY + future_pos, valid, 0).to(tl.float32), tl.load(Z + future_pos, valid, 0))
        dx += future * tl.load(W + c * K + tap, c < C, 0)[None, :]
    tl.store(DX + pos, dx, mask)


class _PackedConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, cu):
        result, preact = extension(x.device).areno_packed_depthwise_causal_conv1d_silu_forward(
            x.contiguous(), weight, cu
        )
        ctx.save_for_backward(x, weight, cu, preact)
        return result

    @staticmethod
    def backward(ctx, dout):
        x, weight, cu, preact = ctx.saved_tensors
        _, tokens, channels = x.shape
        taps = weight.shape[-1]
        blocks = triton.cdiv(tokens, 64)
        dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        partial = weight.new_empty((blocks, channels, taps))
        _backward[(blocks, triton.cdiv(channels, 64))](
            x,
            weight,
            cu,
            preact,
            dout.contiguous(),
            dx,
            partial,
            tokens,
            channels,
            taps,
            cu.numel() - 1,
            x.stride(1),
            (cu.numel() - 2).bit_length(),
            64,
            64,
            num_warps=4,
        )
        return dx, partial.sum(0).view_as(weight), None


def native_packed_conv(x, weight, cu):
    if x.stride(-1) != 1:
        x = x.contiguous()
    return _PackedConv.apply(x, weight.float().contiguous(), cu.to(device=x.device, dtype=torch.int32).contiguous())
