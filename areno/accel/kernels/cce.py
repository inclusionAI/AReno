"""Tiled vocabulary projection and selected log-softmax, without logits in DRAM.

No gradient filtering: every vocabulary tile participates in both passes.
FP32 accumulation and projection-dtype rounding match a materialized head.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _scores(
    X,
    W,
    rows,
    cols,
    N: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    CAP: tl.constexpr,
):
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BV), 0, tl.float32)
    for k in range(triton.cdiv(D, BK)):
        dims = k * BK + ks
        x = tl.load(X + rows[:, None] * D + dims[None, :], (rows[:, None] < N) & (dims[None, :] < D), other=0)
        w = tl.load(W + cols[None, :] * D + dims[:, None], (cols[None, :] < V) & (dims[:, None] < D), other=0)
        acc += tl.dot(x, w, input_precision="tf32x3")
    raw = acc.to(X.dtype.element_ty).to(tl.float32)
    tanh = tl.full((BM, BV), 0, tl.float32)
    if CAP > 0:
        scaled = (raw / CAP).to(X.dtype.element_ty).to(tl.float32)
        tanh = libdevice.tanh(scaled).to(X.dtype.element_ty).to(tl.float32)
        raw = (CAP * tanh).to(X.dtype.element_ty).to(tl.float32)
    return raw, tanh


@triton.jit
def forward_kernel(
    X,
    W,
    Labels,
    Partial,
    Target,
    N: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    START: tl.constexpr,
    BM: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    CAP: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BV + tl.arange(0, BV)
    logits, _ = _scores(X, W, rows, cols, N, V, D, BM, BV, BK, CAP)
    logits = tl.where(cols[None, :] < V, logits, -float("inf"))
    maximum = tl.max(logits, axis=1)
    lse = maximum + tl.log(tl.sum(tl.exp(logits - maximum[:, None]), axis=1))
    tl.store(Partial + tl.program_id(1) * N + rows, lse, rows < N)
    labels = tl.load(Labels + rows, rows < N, other=-1) - START
    target = tl.sum(tl.where(cols[None, :] == labels[:, None], logits, 0.0), axis=1)
    owns = (labels >= tl.program_id(1) * BV) & (labels < tl.minimum((tl.program_id(1) + 1) * BV, V))
    tl.store(Target + rows, target, (rows < N) & owns)


@triton.jit
def backward_kernel(
    X,
    W,
    Labels,
    LSE,
    Grad,
    DX,
    DW,
    N: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    START: tl.constexpr,
    BM: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    CAP: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_W: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BV + tl.arange(0, BV)
    logits, tanh = _scores(X, W, rows, cols, N, V, D, BM, BV, BK, CAP)
    lse = tl.load(LSE + rows, rows < N, other=0)
    grad = tl.load(Grad + rows, rows < N, other=0)
    labels = tl.load(Labels + rows, rows < N, other=-1) - START
    dl = (-tl.exp(logits - lse[:, None]) * grad[:, None]).to(X.dtype.element_ty)
    dl = (dl.to(tl.float32) + tl.where(cols[None, :] == labels[:, None], grad[:, None], 0.0)).to(X.dtype.element_ty)
    if CAP > 0:
        dl = (dl.to(tl.float32) * CAP).to(X.dtype.element_ty)
        dl = (dl.to(tl.float32) * (1.0 - tanh * tanh)).to(X.dtype.element_ty)
        dl = (dl.to(tl.float32) / CAP).to(X.dtype.element_ty)
    dl = tl.where((rows[:, None] < N) & (cols[None, :] < V), dl, 0.0)
    ks = tl.arange(0, BK)
    for k in range(triton.cdiv(D, BK)):
        dims = k * BK + ks
        if NEED_X:
            w = tl.load(W + cols[:, None] * D + dims[None, :], (cols[:, None] < V) & (dims[None, :] < D), other=0)
            dx = tl.dot(dl, w, input_precision="tf32x3")
            tl.atomic_add(
                DX + rows[:, None] * D + dims[None, :], dx, (rows[:, None] < N) & (dims[None, :] < D), sem="relaxed"
            )
        if NEED_W:
            x = tl.load(X + rows[:, None] * D + dims[None, :], (rows[:, None] < N) & (dims[None, :] < D), other=0)
            dw = tl.dot(tl.trans(dl), x, input_precision="tf32x3")
            tl.atomic_add(
                DW + cols[:, None] * D + dims[None, :], dw, (cols[:, None] < V) & (dims[None, :] < D), sem="relaxed"
            )
