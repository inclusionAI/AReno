"""Fixed GEMM tiles keep expert projections independent of token chunk sizes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _list_offsets(Tokens, Tiles, Sizes: tl.constexpr):
    # List counts are static graph metadata. Populate them on device without
    # a blocking host-to-device copy during CUDA graph capture.
    tokens, tiles = 0, 0
    tl.store(Tokens, tokens)
    tl.store(Tiles, tiles)
    for expert in tl.static_range(len(Sizes)):
        tokens += Sizes[expert]
        tiles += tl.cdiv(Sizes[expert], 16)
        tl.store(Tokens + expert + 1, tokens)
        tl.store(Tiles + expert + 1, tiles)


@triton.jit
def _grouped_mm(X, W, Y, Tokens, Tiles, E: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BE: tl.constexpr):
    tile = tl.program_id(0)
    if tile >= tl.load(Tiles + E):
        return
    experts = tl.arange(0, BE)
    starts = tl.load(Tiles + experts, experts < E, other=2147483647)
    expert = tl.sum((starts <= tile).to(tl.int32)).to(tl.int64) - 1
    begin = tl.load(Tokens + expert)
    end = tl.load(Tokens + expert + 1)
    rows = begin + (tile - tl.load(Tiles + expert)) * 16 + tl.arange(0, 16)
    columns = tl.program_id(1) * 64 + tl.arange(0, 64)
    inner = tl.arange(0, 32)
    accumulator = tl.zeros((16, 64), tl.float32)
    for step in range(tl.cdiv(K, 32)):
        ks = step * 32 + inner
        x = tl.load(X + rows[:, None] * K + ks[None, :], (rows[:, None] < end) & (ks[None, :] < K), 0)
        w = tl.load(
            W + expert * N * K + columns[None, :] * K + ks[:, None],
            (columns[None, :] < N) & (ks[:, None] < K),
            0,
        )
        accumulator += tl.dot(x, w)
    tl.store(Y + rows[:, None] * N + columns[None, :], accumulator, (rows[:, None] < end) & (columns[None, :] < N))


def grouped_linear(x, weight, counts):
    # Preserve the native counts entry's host-validation contract. Both its
    # list and tensor variants already materialize counts before CUDA GEMMs.
    if isinstance(counts, torch.Tensor):
        if counts.device != x.device or counts.ndim != 1 or counts.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("tokens_per_expert must be a same-device int32 or int64 vector")
        sizes = counts.cpu().tolist()
    else:
        sizes = list(counts)
    if x.ndim != 2 or weight.ndim != 3 or x.shape[1] != weight.shape[2]:
        raise RuntimeError("areno_grouped_linear input and weight shape mismatch")
    if len(sizes) != weight.shape[0] or any(size < 0 for size in sizes) or sum(sizes) != x.shape[0]:
        raise RuntimeError("tokens_per_expert must match expert count and sum to input rows")
    if weight.device != x.device or weight.dtype != x.dtype:
        raise RuntimeError("areno_grouped_linear input and weight must have the same device and dtype")
    x, weight = x.contiguous(), weight.contiguous()
    output = x.new_empty((x.shape[0], weight.shape[1]))
    if not output.numel():
        return output
    with torch.cuda.device(x.device):
        if isinstance(counts, torch.Tensor):
            counts = counts.to(dtype=torch.int64)
            tokens = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            tiles = torch.cat((counts.new_zeros(1), ((counts + 15) // 16).cumsum(0)))
        else:
            tokens = torch.empty(len(sizes) + 1, device=x.device, dtype=torch.int64)
            tiles = torch.empty_like(tokens)
            _list_offsets[(1,)](tokens, tiles, tuple(sizes), num_warps=1)
        _grouped_mm[(triton.cdiv(x.shape[0], 16) + len(sizes), triton.cdiv(weight.shape[1], 64))](
            x,
            weight,
            output,
            tokens,
            tiles,
            len(sizes),
            weight.shape[1],
            weight.shape[2],
            triton.next_power_of_2(len(sizes)),
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    return output
