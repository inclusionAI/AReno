"""Independent, pure PyTorch KIVI reference for CPU correctness tests.

This module verifies quantization semantics only. Codes intentionally remain
unpacked in uint8 tensors, and the cache is not connected to AReno inference.
Inputs and materialized caches use ``(batch, heads, tokens, head_dim)`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_SUPPORTED_ATTENTION_TYPE = "softmax"
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_bits(bits: int) -> None:
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 8:
        raise ValueError("bits must be an integer in [2, 8]")


def _validate_group_size(group_size: int) -> None:
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")


def _validate_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be 4D with shape (batch, heads, tokens, head_dim)")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype float16, bfloat16, or float32")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0 or tensor.shape[3] <= 0:
        raise ValueError(f"{name} batch, heads, and head_dim must be nonempty")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class KiviQuantizedBlock:
    """Unpacked quantized values plus their affine quantization metadata."""

    codes: torch.Tensor
    scales: torch.Tensor
    minima: torch.Tensor
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    group_size: int
    layout: str

    @torch.no_grad()
    def dequantize(self) -> torch.Tensor:
        """Reconstruct this block in its original shape and dtype."""

        values = self.codes.float() * self.scales + self.minima
        if self.layout == "key":
            work_shape = (
                self.original_shape[0],
                self.original_shape[1],
                self.original_shape[3],
                self.original_shape[2],
            )
            values = values.reshape(work_shape).transpose(-1, -2)
        else:
            values = values.reshape(self.original_shape)
        return values.to(self.original_dtype)


@torch.no_grad()
def _quantize(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    layout: str,
) -> KiviQuantizedBlock:
    _validate_bits(bits)
    _validate_group_size(group_size)
    _validate_tensor(tensor, layout)
    if tensor.shape[2] == 0:
        raise ValueError(f"{layout} must contain at least one token")

    work = tensor.transpose(-1, -2) if layout == "key" else tensor
    if work.shape[-1] % group_size != 0:
        axis = "tokens" if layout == "key" else "head_dim"
        raise ValueError(f"{layout} {axis} must be divisible by group_size")

    grouped = work.float().reshape(*work.shape[:-1], work.shape[-1] // group_size, group_size)
    minima = grouped.amin(dim=-1, keepdim=True)
    maxima = grouped.amax(dim=-1, keepdim=True)
    levels = (1 << bits) - 1
    scales = (maxima - minima) / levels
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.round((grouped - minima) / safe_scales).clamp(0, levels).to(torch.uint8)
    return KiviQuantizedBlock(
        codes=codes,
        scales=scales,
        minima=minima,
        original_shape=tuple(tensor.shape),
        original_dtype=tensor.dtype,
        group_size=group_size,
        layout=layout,
    )


def quantize_key_per_channel(
    key: torch.Tensor, *, bits: int = 2, group_size: int = 32
) -> KiviQuantizedBlock:
    """Quantize each key channel over groups of tokens."""

    return _quantize(key, bits=bits, group_size=group_size, layout="key")


def quantize_value_per_token(
    value: torch.Tensor, *, bits: int = 2, group_size: int = 32
) -> KiviQuantizedBlock:
    """Quantize each value token over groups of head-dimension channels."""

    return _quantize(value, bits=bits, group_size=group_size, layout="value")


class KiviReferenceCache:
    """Single-sequence append-only KIVI cache with an exact residual window.

    Only complete groups older than the residual window are quantized. At most
    ``group_size - 1`` older tokens remain in a separate pending buffer until
    they form a complete group. The most recent ``residual_length`` tokens are
    always retained without quantization.
    """

    def __init__(
        self,
        *,
        bits: int = 2,
        group_size: int = 32,
        residual_length: int = 128,
        attention_type: str = _SUPPORTED_ATTENTION_TYPE,
    ) -> None:
        _validate_bits(bits)
        _validate_group_size(group_size)
        if not isinstance(residual_length, int) or isinstance(residual_length, bool) or residual_length < 0:
            raise ValueError("residual_length must be a nonnegative integer")
        if attention_type != _SUPPORTED_ATTENTION_TYPE:
            raise ValueError(
                f"unsupported attention type {attention_type!r}; "
                f"KIVI reference supports only {_SUPPORTED_ATTENTION_TYPE!r}"
            )
        self.bits = bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.attention_type = attention_type
        self.reset()

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def quantized_tokens(self) -> int:
        return sum(block.original_shape[2] for block in self._key_blocks)

    @property
    def pending_tokens(self) -> int:
        return 0 if self._pending_key is None else self._pending_key.shape[2]

    @property
    def residual_tokens(self) -> int:
        return 0 if self._residual_key is None else self._residual_key.shape[2]

    @property
    def key_blocks(self) -> tuple[KiviQuantizedBlock, ...]:
        return tuple(self._key_blocks)

    @property
    def value_blocks(self) -> tuple[KiviQuantizedBlock, ...]:
        return tuple(self._value_blocks)

    @property
    def residual(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._residual_key, self._residual_value

    def reset(self) -> None:
        """Drop all tensor state while retaining the configuration."""

        self._key_blocks: list[KiviQuantizedBlock] = []
        self._value_blocks: list[KiviQuantizedBlock] = []
        self._pending_key: torch.Tensor | None = None
        self._pending_value: torch.Tensor | None = None
        self._residual_key: torch.Tensor | None = None
        self._residual_value: torch.Tensor | None = None
        self._num_tokens = 0
        self._update_shape: tuple[int, int, int] | None = None
        self._dtype: torch.dtype | None = None
        self._device: torch.device | None = None

    def _validate_update(self, key: torch.Tensor, value: torch.Tensor) -> None:
        _validate_tensor(key, "key")
        _validate_tensor(value, "value")
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        if key.shape[2] == 0:
            raise ValueError("an update must contain at least one token")
        if key.dtype != value.dtype:
            raise TypeError("key and value must have the same dtype")
        if key.device != value.device:
            raise ValueError("key and value must be on the same device")

        update_shape = (key.shape[0], key.shape[1], key.shape[3])
        if self._update_shape is not None and update_shape != self._update_shape:
            raise ValueError("update batch, heads, and head_dim must match the existing cache")
        if self._dtype is not None and key.dtype != self._dtype:
            raise TypeError("update dtype must match the existing cache")
        if self._device is not None and key.device != self._device:
            raise ValueError("update device must match the existing cache")

    @torch.no_grad()
    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append a prefill or decode update in chronological token order."""

        self._validate_update(key, value)
        if self._update_shape is None:
            self._update_shape = (key.shape[0], key.shape[1], key.shape[3])
            self._dtype = key.dtype
            self._device = key.device

        update_length = key.shape[2]
        key_parts = [part for part in (self._pending_key, self._residual_key, key) if part is not None]
        value_parts = [part for part in (self._pending_value, self._residual_value, value) if part is not None]
        keys = torch.cat(key_parts, dim=2)
        values = torch.cat(value_parts, dim=2)

        stable_tokens = max(0, keys.shape[2] - self.residual_length)
        flush_tokens = (stable_tokens // self.group_size) * self.group_size
        for start in range(0, flush_tokens, self.group_size):
            stop = start + self.group_size
            self._key_blocks.append(
                quantize_key_per_channel(
                    keys[:, :, start:stop], bits=self.bits, group_size=self.group_size
                )
            )
            self._value_blocks.append(
                quantize_value_per_token(
                    values[:, :, start:stop], bits=self.bits, group_size=self.group_size
                )
            )

        remaining_keys = keys[:, :, flush_tokens:]
        remaining_values = values[:, :, flush_tokens:]
        pending_tokens = stable_tokens - flush_tokens
        self._pending_key = remaining_keys[:, :, :pending_tokens].detach().clone() if pending_tokens else None
        self._pending_value = remaining_values[:, :, :pending_tokens].detach().clone() if pending_tokens else None
        self._residual_key = remaining_keys[:, :, pending_tokens:].detach().clone()
        self._residual_value = remaining_values[:, :, pending_tokens:].detach().clone()
        if self._residual_key.shape[2] == 0:
            self._residual_key = None
            self._residual_value = None
        self._num_tokens += update_length

    @torch.no_grad()
    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full reconstructed cache in original shape and dtype."""

        if self._update_shape is None or self._dtype is None:
            raise RuntimeError("cannot materialize an empty cache with unknown tensor metadata")
        key_parts = [block.dequantize() for block in self._key_blocks]
        value_parts = [block.dequantize() for block in self._value_blocks]
        key_parts.extend(part for part in (self._pending_key, self._residual_key) if part is not None)
        value_parts.extend(part for part in (self._pending_value, self._residual_value) if part is not None)
        return torch.cat(key_parts, dim=2), torch.cat(value_parts, dim=2)
