"""Compact, lossless storage for a logical FP32 master weight.

For BF16 model parameters, the rounded BF16 value already stores the upper
half of the corresponding FP32 value.  Persisting another FP32 tensor wastes
those two bytes.  ``BF16MasterStorage`` stores only the original FP32 low
16 bits plus one packed bit recording whether round-to-nearest incremented the
BF16 high half.  Together with the live BF16 model weight this reconstructs the
logical FP32 master bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class BF16MasterStorage:
    """Low bits and packed BF16 rounding carries for one flat master shard."""

    low_bits: torch.Tensor
    round_up_bits: torch.Tensor

    @property
    def numel(self) -> int:
        return int(self.low_bits.numel())

    @property
    def nbytes(self) -> int:
        return self.low_bits.numel() * self.low_bits.element_size() + self.round_up_bits.numel()

    def to(self, device: torch.device | str) -> BF16MasterStorage:
        return BF16MasterStorage(
            low_bits=self.low_bits.to(device=device),
            round_up_bits=self.round_up_bits.to(device=device),
        )

    @classmethod
    def zeros(cls, numel: int, *, device: torch.device | str) -> BF16MasterStorage:
        """Represent a master that is exactly equal to its BF16 model tensor."""

        return cls(
            low_bits=torch.zeros(numel, device=device, dtype=torch.uint16),
            round_up_bits=torch.zeros((numel + 7) // 8, device=device, dtype=torch.uint8),
        )


def encode_fp32_master(master: torch.Tensor) -> tuple[torch.Tensor, BF16MasterStorage]:
    """Split a contiguous FP32 tensor into rounded BF16 high and exact low bits."""

    if master.dtype != torch.float32:
        raise TypeError(f"FP32 master encoding requires float32 input, got {master.dtype}")
    flat = master.detach().contiguous().view(-1)
    model = flat.to(dtype=torch.bfloat16)
    words = flat.view(torch.int32)
    truncated_high = torch.bitwise_and(torch.bitwise_right_shift(words, 16), 0xFFFF)
    rounded_high = torch.bitwise_and(model.view(torch.int16).to(dtype=torch.int32), 0xFFFF)
    round_up = rounded_high.ne(truncated_high)
    low_bits = torch.bitwise_and(words, 0xFFFF).to(dtype=torch.uint16)
    storage = BF16MasterStorage(low_bits=low_bits, round_up_bits=_pack_bits(round_up))
    return model.view(master.shape), storage


def decode_fp32_master(model: torch.Tensor, storage: BF16MasterStorage) -> torch.Tensor:
    """Reconstruct the logical FP32 master bit-for-bit from BF16 + metadata."""

    if model.dtype != torch.bfloat16:
        raise TypeError(f"FP32 master decoding requires bfloat16 model input, got {model.dtype}")
    flat = model.detach().contiguous().view(-1)
    if flat.numel() != storage.numel:
        raise ValueError(f"model has {flat.numel()} values but master storage has {storage.numel()}")
    rounded_high = torch.bitwise_and(flat.view(torch.int16).to(dtype=torch.int32), 0xFFFF)
    round_up = _unpack_bits(storage.round_up_bits, flat.numel()).to(dtype=torch.int32)
    original_high = torch.bitwise_and(rounded_high - round_up, 0xFFFF)
    words = torch.bitwise_or(
        torch.bitwise_left_shift(original_high, 16),
        storage.low_bits.to(dtype=torch.int32),
    )
    return words.view(torch.float32).view(model.shape)


def decode_fp32_master_slice(
    model: torch.Tensor,
    storage: BF16MasterStorage,
    offset: int,
) -> torch.Tensor:
    """Decode a contiguous model slice whose metadata starts at ``offset``."""

    if model.dtype != torch.bfloat16:
        raise TypeError(f"FP32 master decoding requires bfloat16 model input, got {model.dtype}")
    flat = model.detach().contiguous().view(-1)
    end = offset + flat.numel()
    if offset < 0 or end > storage.numel:
        raise ValueError(f"master slice [{offset}, {end}) is outside storage with {storage.numel} values")
    rounded_high = torch.bitwise_and(flat.view(torch.int16).to(dtype=torch.int32), 0xFFFF)
    round_up = _unpack_bits_range(storage.round_up_bits, offset, flat.numel()).to(dtype=torch.int32)
    original_high = torch.bitwise_and(rounded_high - round_up, 0xFFFF)
    words = torch.bitwise_or(
        torch.bitwise_left_shift(original_high, 16),
        storage.low_bits.narrow(0, offset, flat.numel()).to(dtype=torch.int32),
    )
    return words.view(torch.float32).view(model.shape)


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack a flat boolean tensor into little-endian uint8 bits."""

    flat = bits.detach().contiguous().view(-1).to(dtype=torch.int64)
    if flat.numel() == 0:
        return torch.empty(0, device=flat.device, dtype=torch.uint8)
    padded_numel = ((flat.numel() + 7) // 8) * 8
    if padded_numel != flat.numel():
        padded = torch.zeros(padded_numel, device=flat.device, dtype=torch.int64)
        padded[: flat.numel()].copy_(flat)
        flat = padded
    shifts = torch.arange(8, device=flat.device, dtype=torch.int64)
    return (flat.view(-1, 8) * torch.bitwise_left_shift(torch.ones_like(shifts), shifts)).sum(dim=1).to(torch.uint8)


def _unpack_bits(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack ``numel`` little-endian bits into a flat boolean tensor."""

    if packed.dtype != torch.uint8:
        raise TypeError(f"packed rounding bits must be uint8, got {packed.dtype}")
    if packed.numel() < (numel + 7) // 8:
        raise ValueError("packed rounding bits are shorter than the requested output")
    if numel == 0:
        return torch.empty(0, device=packed.device, dtype=torch.bool)
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64)
    unpacked = torch.bitwise_and(
        torch.bitwise_right_shift(packed.to(dtype=torch.int64).unsqueeze(1), shifts),
        1,
    )
    return unpacked.reshape(-1)[:numel].to(dtype=torch.bool)


def _unpack_bits_range(packed: torch.Tensor, offset: int, numel: int) -> torch.Tensor:
    """Unpack an arbitrary contiguous bit range without expanding the prefix."""

    if packed.dtype != torch.uint8:
        raise TypeError(f"packed rounding bits must be uint8, got {packed.dtype}")
    if offset < 0 or numel < 0 or offset + numel > packed.numel() * 8:
        raise ValueError("requested rounding bit range is outside the packed tensor")
    if numel == 0:
        return torch.empty(0, device=packed.device, dtype=torch.bool)
    indexes = torch.arange(offset, offset + numel, device=packed.device, dtype=torch.int64)
    selected = packed.index_select(0, torch.div(indexes, 8, rounding_mode="floor")).to(dtype=torch.int64)
    return torch.bitwise_and(torch.bitwise_right_shift(selected, indexes.remainder(8)), 1).to(dtype=torch.bool)


__all__ = ["BF16MasterStorage", "decode_fp32_master", "decode_fp32_master_slice", "encode_fp32_master"]
