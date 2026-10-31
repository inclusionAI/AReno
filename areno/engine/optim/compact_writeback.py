"""Bounded fused writeback of an already-computed GaLore FP32 update.

Moments/projection stay in their original implementation. Explicit fma matches
PyTorch CUDA add_(update, alpha=...), following a separately rounded decay mul.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _writeback(MODEL, LOW, CARRY, UPDATE, INVALID, count, factor, alpha, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < count
    model = tl.load(MODEL + index, mask, other=0)
    high = model.to(tl.uint16, bitcast=True).to(tl.uint32)
    carry_byte = tl.load(CARRY + index // 8, mask, other=0).to(tl.uint32)
    carry = (carry_byte >> (index % 8)) & 1
    low = tl.load(LOW + index, mask, other=0).to(tl.uint32)
    words = (((high - carry) & 65535) << 16) | low
    master = words.to(tl.float32, bitcast=True)
    delta = tl.load(UPDATE + index, mask, other=0)
    if SCALE is not None:
        delta = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[delta, tl.cast(SCALE, tl.float32)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    decayed = master * factor
    result = tl.fma(delta, alpha, decayed)
    result_words = result.to(tl.uint32, bitcast=True)
    rounded = result.to(tl.bfloat16, fp_downcast_rounding="rtne")
    rounded_high = rounded.to(tl.uint16, bitcast=True).to(tl.uint32)
    truncated_high = result_words >> 16
    difference = (rounded_high - truncated_high) & 65535
    invalid = mask & (((result_words & 0x7F800000) == 0x7F800000) | (difference > 1))
    if tl.sum(invalid.to(tl.int32), 0) > 0:
        tl.atomic_or(INVALID, 1)
    tl.store(MODEL + index, rounded, mask)
    tl.store(LOW + index, (result_words & 65535).to(tl.uint16), mask)
    bit = tl.where(mask, difference, 0).to(tl.uint32) << (index % 8)
    packed = tl.sum(tl.reshape(bit, (BLOCK // 8, 8)), 1).to(tl.uint8)
    byte_index = tl.program_id(0) * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
    tl.store(CARRY + byte_index, packed, byte_index < (count + 7) // 8)


def compact_writeback(model, storage, update, offset, factor, alpha, invalid, *, scale=None):
    if offset % 8 or (update.numel() % 8 and offset + update.numel() != storage.numel):
        raise RuntimeError("compact writes require byte-aligned tile boundaries")
    flat_update = update.contiguous().view(-1)
    _writeback[(triton.cdiv(flat_update.numel(), 1024),)](
        model.view(-1)[offset:],
        storage.low_bits[offset:],
        storage.round_up_bits[offset // 8 :],
        flat_update,
        invalid,
        flat_update.numel(),
        factor,
        alpha,
        SCALE=scale,
        BLOCK=1024,
        enable_fp_fusion=False,
    )
