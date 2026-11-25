"""CUDA row reductions with geometry independent of the token schedule."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _row_logsumexp_kernel(
    LOGITS,
    MAXIMUM,
    OUTPUT,
    WIDTH: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOCAL_SUM: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    maximum = tl.load(MAXIMUM + row)
    total = tl.full((), 0, tl.float32)
    for start in range(0, WIDTH, BLOCK):
        values = tl.load(
            LOGITS + row * ROW_STRIDE + (start + columns) * COL_STRIDE,
            mask=start + columns < WIDTH,
            other=-float("inf"),
        ).to(tl.float32)
        total += tl.sum(tl.exp(values - maximum), 0)
    tl.store(OUTPUT + row, total if LOCAL_SUM else maximum + tl.log(total))


def row_logsumexp(
    logits: torch.Tensor, maximum: torch.Tensor, vocab_chunk_size: int, *, local_sum: bool = False
) -> torch.Tensor:
    output = torch.empty_like(maximum, dtype=torch.float32)
    block = min(triton.next_power_of_2(logits.shape[-1]), 1 << (vocab_chunk_size.bit_length() - 1))
    _row_logsumexp_kernel[(logits.shape[0],)](
        logits,
        maximum,
        output,
        logits.shape[-1],
        logits.stride(0),
        logits.stride(1),
        block,
        local_sum,
        num_warps=4 if block <= 1024 else 8,
    )
    return output


@triton.jit
def _packed_logprob_grad_kernel(
    LOGITS,
    SITES,
    LSE,
    LABELS,
    UPSTREAM,
    GRAD,
    WIDTH: tl.constexpr,
    RS: tl.constexpr,
    CS: tl.constexpr,
    GS: tl.constexpr,
    VOCAB_START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    site = tl.load(SITES + row)
    valid = site >= 0
    value = tl.load(LOGITS + row * RS + cols * CS, valid & (cols < WIDTH), other=0).to(tl.float32)
    lse = tl.load(LSE + site, valid, other=0)
    dy = tl.load(UPSTREAM + site * GS, valid, other=0)
    label = tl.load(LABELS + site, valid, other=-1)
    dtype = LOGITS.dtype.element_ty
    # Match the eager VJP's cast before adding the target contribution.
    base = (libdevice.exp(value - lse) * -dy).to(dtype)
    dx = tl.where(cols + VOCAB_START == label, base + dy.to(dtype), base)
    tl.store(GRAD + row * WIDTH + cols, tl.where(valid, dx, 0), cols < WIDTH)


def packed_logprob_gradient(logits, sites, lse, labels, upstream, vocab_start):
    rows, width = logits.shape
    grad = torch.empty((rows, width), dtype=logits.dtype, device=logits.device)
    _packed_logprob_grad_kernel[(rows, triton.cdiv(width, 1024))](
        logits,
        sites,
        lse,
        labels.contiguous(),
        upstream,
        grad,
        width,
        *logits.stride(),
        upstream.stride(0),
        vocab_start,
        1024,
        enable_fp_fusion=False,
    )
    return grad
