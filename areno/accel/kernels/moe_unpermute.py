"""Deterministic expert-major to token-major reduction."""

import torch
import triton
import triton.language as tl


@triton.jit
def _unpermute_kernel(
    X,
    ORDER,
    OFFSETS,
    OUTPUT,
    HIDDEN: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(OFFSETS + token)
    end = tl.load(OFFSETS + token + 1)
    total = tl.full((BLOCK,), 0, tl.float32)
    for cursor in range(begin, end):
        row = tl.load(ORDER + cursor)
        values = tl.load(X + row * ROW_STRIDE + channels * COL_STRIDE, mask=channels < HIDDEN, other=0).to(tl.float32)
        total += values
    tl.store(OUTPUT + token * HIDDEN + channels, total, mask=channels < HIDDEN)


def deterministic_unpermute(x: torch.Tensor, token_index: torch.Tensor, tokens: int, hidden: int) -> torch.Tensor:
    # Expert-major order fixes the sum's expert order, even when atomic
    # permutation assigned token rows differently inside each expert group.
    # Integer counts are deterministic and keep all shapes known in graphs.
    order = torch.argsort(token_index, stable=True)
    counts = torch.zeros(tokens, dtype=torch.long, device=x.device)
    counts.scatter_add_(0, token_index, torch.ones_like(token_index))
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    output = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    if tokens and hidden:
        _unpermute_kernel[(tokens, triton.cdiv(hidden, 256))](
            x,
            order,
            offsets,
            output,
            hidden,
            x.stride(0),
            x.stride(1),
            256,
        )
    return output
