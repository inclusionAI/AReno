"""Independent PyTorch reference for blockwise low-bit activation storage.

This module deliberately has no training-runtime integration. It defines a small,
deterministic format used to check activation-compression correctness on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

SUPPORTED_BIT_WIDTHS = (2, 4, 8)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@dataclass(frozen=True)
class CompressedActivation:
    """Owned activation payload plus all metadata needed for reconstruction.

    ``packed`` stores offset-binary codes in consecutive low-order bit groups.
    ``scales`` uses FP32 for FP16/BF16/FP32 inputs and FP64 for FP64 inputs.
    A fallback payload owns an uncompressed ``raw`` copy and empty packed data.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    bit_width: int
    block_size: int
    chunk_size: int
    source_nbytes: int
    packed: torch.Tensor
    scales: torch.Tensor
    raw: torch.Tensor | None
    max_abs_error: float
    fallback_reason: str | None = None

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def used_fallback(self) -> bool:
        return self.raw is not None

    @property
    def scale_count(self) -> int:
        return self.scales.numel()

    @property
    def packed_nbytes(self) -> int:
        return self.packed.numel() * self.packed.element_size()

    @property
    def scale_nbytes(self) -> int:
        return self.scales.numel() * self.scales.element_size()

    @property
    def raw_nbytes(self) -> int:
        return 0 if self.raw is None else self.raw.numel() * self.raw.element_size()

    @property
    def storage_nbytes(self) -> int:
        """Persistent tensor payload bytes, excluding Python object overhead."""
        return self.packed_nbytes + self.scale_nbytes + self.raw_nbytes

    @property
    def reconstruction_workspace_nbytes(self) -> int:
        """Explicit code and padded reconstruction buffers, excluding output.

        Per-chunk arithmetic temporaries and allocator overhead are intentionally
        excluded; this is deterministic logical accounting, not peak RSS/VRAM.
        """
        if self.used_fallback:
            return 0
        padded_numel = self.scale_count * self.block_size
        return self.numel + padded_numel * self.scales.element_size()

    def restore(self) -> torch.Tensor:
        """Restore an owned tensor with the original shape, dtype, and device."""
        if self.raw is not None:
            return self.raw.clone(memory_format=torch.preserve_format)

        compute_dtype = self.scales.dtype
        padded_numel = self.scale_count * self.block_size
        values = torch.empty(padded_numel, dtype=compute_dtype, device=self.device)
        codes = _unpack_codes(self.packed, self.numel, self.bit_width)
        qmax = (1 << (self.bit_width - 1)) - 1
        chunk_values = _whole_blocks_per_chunk(self.chunk_size, self.block_size)

        for start in range(0, self.numel, chunk_values):
            end = min(start + chunk_values, self.numel)
            block_end = (end + self.block_size - 1) // self.block_size
            padded_end = block_end * self.block_size
            block_scales = self.scales[start // self.block_size : block_end]
            expanded_scales = block_scales.repeat_interleave(self.block_size)[: end - start]
            quantized = codes[start:end].to(compute_dtype) - qmax
            values[start:end] = quantized * expanded_scales
            if padded_end > end:
                values[end:padded_end].zero_()

        # Clone so the returned tensor does not retain the padded workspace.
        return values[: self.numel].view(self.shape).to(self.dtype).clone()


def compress_activation(
    tensor: torch.Tensor,
    *,
    bit_width: int = 4,
    block_size: int = 64,
    chunk_size: int = 4096,
    max_abs_error: float | None = None,
) -> CompressedActivation:
    """Compress a floating activation, or retain it exactly on fallback.

    Blocks are formed over the contiguous flattened input. ``chunk_size`` only
    controls traversal and is rounded up to whole blocks; it cannot change block
    membership. If ``max_abs_error`` is exceeded, the quantized payload is
    discarded and a bit-exact owned copy is retained instead.
    """
    _validate_arguments(tensor, bit_width, block_size, chunk_size, max_abs_error)
    shape = tuple(tensor.shape)
    compute_dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    common = {
        "shape": shape,
        "dtype": tensor.dtype,
        "device": tensor.device,
        "bit_width": bit_width,
        "block_size": block_size,
        "chunk_size": chunk_size,
        "source_nbytes": tensor.numel() * tensor.element_size(),
    }

    if not bool(torch.isfinite(tensor).all().item()):
        return _fallback_payload(
            tensor,
            compute_dtype=compute_dtype,
            observed_error=math.nan,
            reason="nonfinite_input",
            **common,
        )

    flat = tensor.detach().contiguous().view(-1)
    numel = flat.numel()
    scale_count = (numel + block_size - 1) // block_size
    qmax = (1 << (bit_width - 1)) - 1
    scales = torch.empty(scale_count, dtype=compute_dtype, device=tensor.device)
    codes = torch.empty(numel, dtype=torch.uint8, device=tensor.device)
    chunk_values = _whole_blocks_per_chunk(chunk_size, block_size)
    tiny = torch.finfo(compute_dtype).tiny

    for start in range(0, numel, chunk_values):
        end = min(start + chunk_values, numel)
        block_end = (end + block_size - 1) // block_size
        padded_end = block_end * block_size
        block_values = torch.zeros(padded_end - start, dtype=compute_dtype, device=tensor.device)
        block_values[: end - start] = flat[start:end].to(compute_dtype)
        blocks = block_values.view(-1, block_size)
        absmax = blocks.abs().amax(dim=1)
        block_scales = torch.where(
            absmax > 0,
            (absmax / qmax).clamp_min(tiny),
            torch.zeros_like(absmax),
        )
        scales[start // block_size : block_end] = block_scales
        normalized = blocks / block_scales.clamp_min(tiny).unsqueeze(1)
        quantized = torch.round(normalized).clamp_(-qmax, qmax)
        block_codes = (quantized + qmax).to(torch.uint8).view(-1)
        codes[start:end] = block_codes[: end - start]

    packed = _pack_codes(codes, bit_width)
    candidate = CompressedActivation(
        **common,
        packed=packed,
        scales=scales,
        raw=None,
        max_abs_error=0.0,
    )
    restored = candidate.restore()
    if numel:
        error = (restored.view(-1).to(compute_dtype) - flat.to(compute_dtype)).abs().max().item()
    else:
        error = 0.0

    if not math.isfinite(error):
        return _fallback_payload(
            tensor,
            compute_dtype=compute_dtype,
            observed_error=error,
            reason="nonfinite_reconstruction",
            **common,
        )
    if max_abs_error is not None and error > max_abs_error:
        return _fallback_payload(
            tensor,
            compute_dtype=compute_dtype,
            observed_error=error,
            reason="error_budget_exceeded",
            **common,
        )

    return CompressedActivation(
        **common,
        packed=packed,
        scales=scales,
        raw=None,
        max_abs_error=error,
    )


def restore_activation(payload: CompressedActivation) -> torch.Tensor:
    """Functional counterpart to :meth:`CompressedActivation.restore`."""
    return payload.restore()


def _fallback_payload(
    tensor: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
    observed_error: float,
    reason: str,
    **metadata,
) -> CompressedActivation:
    return CompressedActivation(
        **metadata,
        packed=torch.empty(0, dtype=torch.uint8, device=tensor.device),
        scales=torch.empty(0, dtype=compute_dtype, device=tensor.device),
        raw=tensor.detach().clone(memory_format=torch.preserve_format),
        max_abs_error=observed_error,
        fallback_reason=reason,
    )


def _validate_arguments(
    tensor: torch.Tensor,
    bit_width: int,
    block_size: int,
    chunk_size: int,
    max_abs_error: float | None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported activation dtype: {tensor.dtype}")
    for name, value in (("bit_width", bit_width), ("block_size", block_size), ("chunk_size", chunk_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if bit_width not in SUPPORTED_BIT_WIDTHS:
        raise ValueError(f"bit_width must be one of {SUPPORTED_BIT_WIDTHS}")
    if block_size <= 0 or chunk_size <= 0:
        raise ValueError("block_size and chunk_size must be positive")
    if max_abs_error is not None:
        if math.isnan(max_abs_error) or max_abs_error < 0:
            raise ValueError("max_abs_error must be nonnegative or None")


def _whole_blocks_per_chunk(chunk_size: int, block_size: int) -> int:
    return max(block_size, (chunk_size // block_size) * block_size)


def _pack_codes(codes: torch.Tensor, bit_width: int) -> torch.Tensor:
    values_per_byte = 8 // bit_width
    padding = (-codes.numel()) % values_per_byte
    if padding:
        codes = torch.cat((codes, codes.new_zeros(padding)))
    weights = torch.arange(values_per_byte, dtype=torch.int16, device=codes.device) * bit_width
    weights = torch.bitwise_left_shift(torch.ones_like(weights), weights)
    pairs = codes.view(-1, values_per_byte).to(torch.int16)
    return (pairs * weights.unsqueeze(0)).sum(dim=1).to(torch.uint8)


def _unpack_codes(packed: torch.Tensor, numel: int, bit_width: int) -> torch.Tensor:
    values_per_byte = 8 // bit_width
    shifts = torch.arange(values_per_byte, dtype=torch.uint8, device=packed.device) * bit_width
    mask = (1 << bit_width) - 1
    codes = (packed.unsqueeze(1) >> shifts) & mask
    return codes.reshape(-1)[:numel]
