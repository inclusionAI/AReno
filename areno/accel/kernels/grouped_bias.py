"""Fused routed bias and bounded interleaved SwiGLU, with fp32 bias reductions."""

import torch
import triton
import triton.language as tl


@triton.jit
def _bias(
    X,
    B,
    ENDS,
    O,
    G,
    DB,
    N: tl.constexpr,
    C: tl.constexpr,
    E: tl.constexpr,
    ER: tl.constexpr,
    BLOCK: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    es = tl.arange(0, ER)
    ends = tl.load(ENDS + es, es < E, 2147483647)
    expert = tl.sum((row >= ends).to(tl.int32), 0)
    valid = (expert < E) & (cols < C)
    if BACKWARD:
        grad = tl.load(G + row * C + cols, valid, 0).to(tl.float32)
        tl.store(O + row * C + cols, grad, cols < C)
        tl.atomic_add(DB + expert * C + cols, grad, valid)
    else:
        value = tl.load(X + row * C + cols, valid, 0).to(tl.float32)
        bias = tl.load(B + expert * C + cols, valid, 0).to(tl.float32)
        tl.store(O + row * C + cols, value + bias, cols < C)


@triton.jit
def _activation(
    X, O, G, DX, N: tl.constexpr, ALPHA: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr, BACKWARD: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rawg = tl.load(X + 2 * i, i < N, 0).to(tl.float32)
    rawu = tl.load(X + 2 * i + 1, i < N, 0).to(tl.float32)
    gate = tl.minimum(rawg, LIMIT)
    up = tl.minimum(tl.maximum(rawu, -LIMIT), LIMIT)
    sigmoid = 1 / (1 + tl.exp(-gate * ALPHA))
    glu = gate * sigmoid
    if BACKWARD:
        grad = tl.load(G + i, i < N, 0).to(tl.float32)
        dg = grad * (up + 1) * (sigmoid + ALPHA * gate * sigmoid * (1 - sigmoid)) * (rawg <= LIMIT)
        du = grad * glu * ((rawu >= -LIMIT) & (rawu <= LIMIT))
        tl.store(DX + 2 * i, dg, i < N)
        tl.store(DX + 2 * i + 1, du, i < N)
    else:
        tl.store(O + i, (up + 1) * glu, i < N)


def bias_forward(x, bias, counts):
    ends = counts.cumsum(0)
    out = torch.empty_like(x)
    _bias[(x.shape[0], triton.cdiv(x.shape[1], 256))](
        x,
        bias,
        ends,
        out,
        x,
        bias,
        x.shape[0],
        x.shape[1],
        bias.shape[0],
        triton.next_power_of_2(bias.shape[0]),
        256,
        False,
    )
    return out, ends


def bias_backward(grad, ends, shape):
    grad = grad.contiguous()
    dx = torch.empty_like(grad)
    db = torch.zeros(shape, device=grad.device, dtype=torch.float32)
    _bias[(grad.shape[0], triton.cdiv(grad.shape[1], 256))](
        grad,
        db,
        ends,
        dx,
        grad,
        db,
        grad.shape[0],
        grad.shape[1],
        shape[0],
        triton.next_power_of_2(shape[0]),
        256,
        True,
    )
    return dx, db.to(grad.dtype)


def activation_forward(x, alpha, limit):
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
    _activation[(triton.cdiv(out.numel(), 256),)](x, out, out, x, out.numel(), alpha, limit, 256, False)
    return out


def activation_backward(x, grad, alpha, limit):
    grad = grad.contiguous()
    dx = torch.empty_like(x)
    _activation[(triton.cdiv(grad.numel(), 256),)](x, grad, grad, dx, grad.numel(), alpha, limit, 256, True)
    return dx
