"""Linear projection with token-count-independent K reduction for the native backend."""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear(
    X,
    W,
    B,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + ks
        x = tl.load(X + rows[:, None] * K + k[None, :], (rows[:, None] < M) & (k[None, :] < K), 0)
        w = tl.load(W + cols[:, None] * K + k[None, :], (cols[:, None] < N) & (k[None, :] < K), 0)
        acc = tl.dot(x, tl.trans(w), acc, input_precision="ieee")
    if BIAS:
        acc += tl.load(B + cols, cols < N, 0)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _router(X, W, B, Y, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BIAS: tl.constexpr):
    row = tl.program_id(1)
    cols = tl.program_id(0) * 2 + tl.arange(0, 2)
    k = tl.arange(0, BK)
    x = tl.load(X + row * K + k, k < K, 0)
    w = tl.load(W + cols[:, None] * K + k[None, :], (cols[:, None] < N) & (k[None, :] < K), 0)
    out = tl.sum(w * x[None, :], 1)
    if BIAS:
        out += tl.load(B + cols, cols < N, 0)
    tl.store(Y + row * N + cols, out, cols < N)


class InvariantLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias):
        flat = x.reshape(-1, w.shape[1]).contiguous()
        rows = flat.shape[0]
        out = x.new_empty((*x.shape[:-1], w.shape[0]))
        # Large, aligned BF16 matrices on GB10 use the tensor-core GEMM path.
        # Fix M for each launch so packed sequence lengths cannot choose a
        # different algorithm. Tails use the token-invariant Triton kernel.
        if (
            rows >= 2048
            and x.dtype == torch.bfloat16
            and bias is None
            and w.shape[0] >= 1024
            and w.shape[0] % 128 == 0
            and w.shape[1] % 128 == 0
            and torch.cuda.get_device_capability(x.device) == (12, 1)
        ):
            covered = rows // 2048 * 2048
            dest = out.view(rows, w.shape[0])
            for start in range(0, covered, 2048):
                torch.mm(flat[start : start + 2048], w.T, out=dest[start : start + 2048])
            flat = flat[covered:]
            rows -= covered
        else:
            covered = 0
        tail = out.view(-1, w.shape[0])[covered:]
        if rows and x.dtype == torch.float32 and w.shape[0] <= 256 and w.shape[1] <= 4096:
            _router[(triton.cdiv(w.shape[0], 2), rows)](
                flat,
                w.contiguous(),
                w if bias is None else bias,
                tail,
                *w.shape,
                triton.next_power_of_2(w.shape[1]),
                bias is not None,
            )
        elif rows:
            bm, bn, bk, warps = 16, 64, 128, 4
            if rows < 16 and x.dtype != torch.float32 and w.shape[0] >= 8192:
                bk, warps = 256, 8
            if rows >= 128 and x.dtype != torch.float32:
                bm, bn, bk, warps = (128, 256, 64, 8) if w.shape[0] >= 256 else (64, 64, 32, 4)
            _linear[(triton.cdiv(rows, bm), triton.cdiv(w.shape[0], bn))](
                flat,
                w.contiguous(),
                w if bias is None else bias,
                tail,
                rows,
                *w.shape,
                bias is not None,
                bm,
                bn,
                bk,
                num_warps=warps,
            )
        ctx.save_for_backward(x, w)
        return out

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        flat = dy.reshape(-1, w.shape[0])
        return (
            dy @ w if ctx.needs_input_grad[0] else None,
            flat.T @ x.reshape(-1, w.shape[1]) if ctx.needs_input_grad[1] else None,
            flat.sum(0) if ctx.needs_input_grad[2] else None,
        )
