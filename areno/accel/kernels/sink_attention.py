"""Paged causal/window attention with a persistent per-query-head sink."""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged(
    Q,
    K,
    V,
    S,
    BT,
    SEQ,
    POS,
    O,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    seq = tl.load(SEQ + token)
    pos = tl.load(POS + token)
    d = tl.arange(0, D)
    query = tl.load(Q + token * H * D + head * D + d).to(tl.float32)
    maximum = tl.load(S + head).to(tl.float32)
    denominator = tl.full((), 1.0, tl.float32)
    acc = tl.full((D,), 0.0, tl.float32)
    start = 0
    if WINDOW > 0:
        start = tl.maximum(0, pos - WINDOW + 1)
    for base in range(start, pos + 1, BLOCK):
        indices = base + tl.arange(0, BLOCK)
        valid = indices <= pos
        pages = tl.load(BT + seq * WIDTH + indices // PAGE, valid, 0)
        offsets = ((pages[:, None] * PAGE + indices[:, None] % PAGE) * HK + head // (H // HK)) * D + d[None, :]
        keys = tl.load(K + offsets, valid[:, None], 0).to(tl.float32)
        values = tl.load(V + offsets, valid[:, None], 0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], 1) * (D**-0.5)
        scores = tl.where(valid, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 0))
        correction = tl.exp(maximum - new_max)
        probabilities = tl.exp(scores - new_max)
        acc = acc * correction + tl.sum(probabilities[:, None] * values, 0)
        denominator = denominator * correction + tl.sum(probabilities, 0)
        maximum = new_max
    tl.store(O + token * H * D + head * D + d, acc / denominator)


def paged_sink_forward(q, k, v, sinks, table, sequences, positions, window):
    tokens, heads, dim = q.shape
    out = torch.empty_like(q)
    _paged[(tokens, heads)](
        q,
        k,
        v,
        sinks,
        table,
        sequences,
        positions,
        out,
        heads,
        k.shape[2],
        dim,
        k.shape[1],
        table.shape[1],
        window or 0,
        32,
        num_warps=4,
    )
    return out


@triton.jit
def _dense_forward(
    Q,
    K,
    V,
    S,
    START,
    O,
    L,
    N: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    batch = row // N
    position = row % N
    d = tl.arange(0, D)
    query = tl.load(Q + row * H * D + head * D + d).to(tl.float32)
    maximum = tl.load(S + head).to(tl.float32)
    denom = tl.full((), 1.0, tl.float32)
    acc = tl.full((D,), 0.0, tl.float32)
    begin = tl.load(START + position)
    if WINDOW > 0:
        begin = tl.maximum(begin, position - WINDOW + 1)
    for base in range(begin, position + 1, BLOCK):
        positions = base + tl.arange(0, BLOCK)
        valid = positions <= position
        offsets = ((batch * N + positions[:, None]) * HK + head // (H // HK)) * D + d[None, :]
        keys = tl.load(K + offsets, valid[:, None], 0).to(tl.float32)
        values = tl.load(V + offsets, valid[:, None], 0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], 1) * (D**-0.5)
        scores = tl.where(valid, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 0))
        scale = tl.exp(maximum - new_max)
        probability = tl.exp(scores - new_max)
        acc = acc * scale + tl.sum(probability[:, None] * values, 0)
        denom = denom * scale + tl.sum(probability, 0)
        maximum = new_max
    tl.store(O + row * H * D + head * D + d, acc / denom)
    tl.store(L + row * H + head, maximum + tl.log(denom))


@triton.jit
def _dense_backward(
    Q,
    K,
    V,
    S,
    START,
    O,
    L,
    G,
    DQ,
    DK,
    DV,
    DS,
    N: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    batch = row // N
    position = row % N
    d = tl.arange(0, D)
    query = tl.load(Q + row * H * D + head * D + d).to(tl.float32)
    output = tl.load(O + row * H * D + head * D + d).to(tl.float32)
    grad = tl.load(G + row * H * D + head * D + d).to(tl.float32)
    lse = tl.load(L + row * H + head)
    delta = tl.sum(output * grad, 0)
    ds = -tl.exp(tl.load(S + head).to(tl.float32) - lse) * delta
    tl.atomic_add(DS + head, ds)
    begin = tl.load(START + position)
    if WINDOW > 0:
        begin = tl.maximum(begin, position - WINDOW + 1)
    dq = tl.full((D,), 0.0, tl.float32)
    for base in range(begin, position + 1, BLOCK):
        positions = base + tl.arange(0, BLOCK)
        valid = positions <= position
        offsets = ((batch * N + positions[:, None]) * HK + head // (H // HK)) * D + d[None, :]
        keys = tl.load(K + offsets, valid[:, None], 0).to(tl.float32)
        values = tl.load(V + offsets, valid[:, None], 0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], 1) * (D**-0.5)
        probability = tl.where(valid, tl.exp(scores - lse), 0.0)
        dp = tl.sum(values * grad[None, :], 1)
        dscores = probability * (dp - delta) * (D**-0.5)
        dq += tl.sum(dscores[:, None] * keys, 0)
        tl.atomic_add(DK + offsets, dscores[:, None] * query[None, :], valid[:, None])
        tl.atomic_add(DV + offsets, probability[:, None] * grad[None, :], valid[:, None])
    tl.store(DQ + row * H * D + head * D + d, dq)


def dense_forward(q, k, v, sinks, starts, window):
    b, n, h, d = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((b, n, h), device=q.device, dtype=torch.float32)
    _dense_forward[(b * n, h)](q, k, v, sinks, starts, out, lse, n, h, k.shape[2], d, window or 0, 32, num_warps=4)
    return out, lse


def dense_backward(q, k, v, sinks, starts, out, lse, grad, window):
    b, n, h, d = q.shape
    dq = torch.empty_like(q)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    ds = torch.zeros_like(sinks, dtype=torch.float32)
    _dense_backward[(b * n, h)](
        q,
        k,
        v,
        sinks,
        starts,
        out,
        lse,
        grad.contiguous(),
        dq,
        dk,
        dv,
        ds,
        n,
        h,
        k.shape[2],
        d,
        window or 0,
        32,
        num_warps=4,
    )
    return dq, dk.to(k.dtype), dv.to(v.dtype), ds.to(sinks.dtype)
