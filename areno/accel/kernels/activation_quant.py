"""Group-wise 2/4/8-bit codes for saved activations.

The CUDA counterpart of the PyTorch path in
``areno.engine.runtime.activation_compression``; both produce identical codes.
Each group of ``GROUP`` contiguous elements stores its FP32 minimum, its FP32
step ``(max - min) / (2**BITS - 1)`` and the codes
``floor((x - min) / step + u)``. ``u`` is a uniform draw from a keyed
counter-based hash of the element index (stochastic rounding), or 0.5 for
rounding to nearest. Elements whose channel (index modulo ``row``) is flagged in
``KEEP`` are stored elsewhere by the caller: they are excluded from the group
range and get code 0. Code ``j`` of a group occupies bits
``BITS * (j % (8 // BITS))`` of byte ``j // (8 // BITS)``.
"""

import torch
import triton
import triton.language as tl

_BLOCK_GROUPS = 64


@triton.jit
def _mul32(h, c: tl.constexpr):
    # (h * c) mod 2**32 for h, c < 2**32 without int64 overflow.
    return ((h & 0xFFFF) * c + ((((h >> 16) * c) & 0xFFFF) << 16)) & 0xFFFFFFFF


@triton.jit
def _lowbias32(h):
    # C. Wellons' lowbias32 integer hash of h < 2**32.
    h = h ^ (h >> 16)
    h = _mul32(h, 0x7FEB352D)
    h = h ^ (h >> 15)
    h = _mul32(h, 0x846CA68B)
    return h ^ (h >> 16)


@triton.jit
def _uniform(index, seed):
    # Hash the counter before keying it: XOR-ing the seed into the raw index
    # would only permute indices, reusing the same draws across seeds.
    h = _lowbias32(_lowbias32(index & 0xFFFFFFFF) ^ seed)
    return (h >> 8).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _quantize_kernel(
    X,
    KEEP,
    CODES,
    MINIMUM,
    SCALE,
    numel,
    groups_total,
    row,
    seed,
    GROUP: tl.constexpr,
    BITS: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    HAS_KEEP: tl.constexpr,
    ROW_ALIGNED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    groups = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    columns = tl.arange(0, GROUP).to(tl.int64)
    index = groups[:, None] * GROUP + columns[None, :]
    valid = index < numel
    used = valid
    if HAS_KEEP:
        if ROW_ALIGNED:
            # Groups never straddle rows: one modulo per group, not per element.
            channel = ((groups * GROUP) % row)[:, None] + columns[None, :]
        else:
            channel = index % row
        used = valid & (tl.load(KEEP + channel, mask=valid, other=0) == 0)
    x = tl.load(X + index, mask=valid, other=0.0).to(tl.float32)
    minimum = tl.min(tl.where(used, x, float("inf")), axis=1)
    maximum = tl.max(tl.where(used, x, float("-inf")), axis=1)
    # Groups without quantized elements store a zero range.
    minimum = tl.where(minimum == float("inf"), 0.0, minimum)
    maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    levels = (1 << BITS) - 1
    scale = tl.math.div_rn(maximum - minimum, tl.full((BLOCK,), levels, tl.float32))
    scale = tl.where(scale > 0, scale, 1.0)
    q = tl.math.div_rn(x - minimum[:, None], scale[:, None])
    if STOCHASTIC:
        q = q + _uniform(index, seed)
    else:
        q = q + 0.5
    codes = tl.minimum(tl.maximum(tl.floor(q), 0.0), levels * 1.0).to(tl.int32)
    codes = tl.where(used, codes, 0)
    stored = groups < groups_total
    tl.store(MINIMUM + groups, minimum, mask=stored)
    tl.store(SCALE + groups, scale, mask=stored)
    if BITS == 8:
        packed = codes
    elif BITS == 4:
        low, high = tl.split(tl.reshape(codes, (BLOCK, GROUP // 2, 2)))
        packed = low | (high << 4)
    else:
        even, odd = tl.split(tl.reshape(codes, (BLOCK, GROUP // 4, 2, 2)))
        code0, code2 = tl.split(even)
        code1, code3 = tl.split(odd)
        packed = code0 | (code1 << 2) | (code2 << 4) | (code3 << 6)
    width: tl.constexpr = GROUP * BITS // 8
    offsets = groups[:, None] * width + tl.arange(0, width).to(tl.int64)[None, :]
    tl.store(CODES + offsets, packed.to(tl.uint8), mask=stored[:, None])


@triton.jit
def _dequantize_kernel(
    CODES,
    MINIMUM,
    SCALE,
    OUT,
    numel,
    groups_total,
    GROUP: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    groups = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    stored = groups < groups_total
    width: tl.constexpr = GROUP * BITS // 8
    offsets = groups[:, None] * width + tl.arange(0, width).to(tl.int64)[None, :]
    packed = tl.load(CODES + offsets, mask=stored[:, None], other=0).to(tl.int32)
    if BITS == 8:
        codes = packed
    elif BITS == 4:
        codes = tl.reshape(tl.join(packed & 15, packed >> 4), (BLOCK, GROUP))
    else:
        even = tl.join(packed & 3, (packed >> 4) & 3)
        odd = tl.join((packed >> 2) & 3, (packed >> 6) & 3)
        codes = tl.reshape(tl.join(even, odd), (BLOCK, GROUP))
    minimum = tl.load(MINIMUM + groups, mask=stored, other=0.0)
    scale = tl.load(SCALE + groups, mask=stored, other=0.0)
    values = codes.to(tl.float32) * scale[:, None] + minimum[:, None]
    index = groups[:, None] * GROUP + tl.arange(0, GROUP).to(tl.int64)[None, :]
    tl.store(OUT + index, values.to(OUT.dtype.element_ty), mask=index < numel)


def quantize_groups(
    x: torch.Tensor,
    keep: torch.Tensor | None,
    bits: int,
    group_size: int,
    *,
    stochastic: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed codes, per-group minima and steps of contiguous ``x``.

    ``keep`` is an optional uint8 flag per channel (last dimension) marking
    channels the caller stores separately.
    """

    numel = x.numel()
    groups = triton.cdiv(numel, group_size)
    codes = torch.empty(groups * group_size * bits // 8, dtype=torch.uint8, device=x.device)
    minimum = torch.empty(groups, dtype=torch.float32, device=x.device)
    scale = torch.empty(groups, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(groups, _BLOCK_GROUPS),)
    _quantize_kernel[grid](
        x,
        codes if keep is None else keep,
        codes,
        minimum,
        scale,
        numel,
        groups,
        x.shape[-1],
        seed,
        GROUP=group_size,
        BITS=bits,
        STOCHASTIC=stochastic,
        HAS_KEEP=keep is not None,
        ROW_ALIGNED=x.shape[-1] % group_size == 0,
        BLOCK=_BLOCK_GROUPS,
    )
    return codes, minimum, scale


def dequantize_groups(
    codes: torch.Tensor,
    minimum: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Write the reconstruction of ``quantize_groups`` codes into contiguous ``out``."""

    groups = minimum.numel()
    grid = (triton.cdiv(groups, _BLOCK_GROUPS),)
    _dequantize_kernel[grid](
        codes,
        minimum,
        scale,
        out,
        out.numel(),
        groups,
        GROUP=group_size,
        BITS=bits,
        BLOCK=_BLOCK_GROUPS,
        # Round code * step and + minimum separately, like the PyTorch path.
        enable_fp_fusion=False,
    )
    return out
