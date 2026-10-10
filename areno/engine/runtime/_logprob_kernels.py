"""CUDA row reductions with geometry independent of the token schedule."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_logsumexp_kernel(
    LOGITS,
    MAXIMUM,
    OUTPUT,
    WIDTH: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
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
    tl.store(OUTPUT + row, maximum + tl.log(total))


def row_logsumexp(logits: torch.Tensor, maximum: torch.Tensor, vocab_chunk_size: int) -> torch.Tensor:
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
        num_warps=4 if block <= 1024 else 8,
    )
    return output
