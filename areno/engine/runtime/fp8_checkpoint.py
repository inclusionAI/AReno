"""FP8 storage for tensors saved at activation-checkpoint boundaries.

The model forward still consumes BF16 tensors.  Only the copy retained by
autograd for the later checkpoint recomputation is quantized, so this module
does not change the original forward graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


@dataclass(slots=True)
class FP8CheckpointTensor:
    """Quantized tensor payload returned from a saved-tensor pack hook."""

    values: torch.Tensor
    scales: torch.Tensor
    shape: torch.Size
    dtype: torch.dtype
    group_size: int


@dataclass(slots=True)
class _MemoryStats:
    boundaries: int = 0
    fallback_boundaries: int = 0
    warmup_boundaries: int = 0
    original_bytes: int = 0
    stored_bytes: int = 0


_STATS = _MemoryStats()
_RANGE_AMAX: dict[int, torch.Tensor] = {}
_ROUNDING_GENERATORS: dict[str, torch.Generator] = {}
_LOCK = Lock()


def reset_fp8_checkpoint_stats() -> None:
    """Reset process-local counters collected for the next train microbatch."""

    with _LOCK:
        # Reset in place. Compiled saved-tensor pack paths can retain the
        # original stats object captured during their first trace; rebinding
        # ``_STATS`` would make those paths update a stale object while the
        # metrics reader observes a new, permanently-zero object.
        _STATS.boundaries = 0
        _STATS.fallback_boundaries = 0
        _STATS.warmup_boundaries = 0
        _STATS.original_bytes = 0
        _STATS.stored_bytes = 0


def fp8_checkpoint_metrics() -> dict[str, float]:
    """Return checkpoint-boundary compression counters as training metrics."""

    with _LOCK:
        stats = _MemoryStats(
            boundaries=_STATS.boundaries,
            fallback_boundaries=_STATS.fallback_boundaries,
            warmup_boundaries=_STATS.warmup_boundaries,
            original_bytes=_STATS.original_bytes,
            stored_bytes=_STATS.stored_bytes,
        )
    reduction = 0.0
    if stats.original_bytes:
        reduction = 1.0 - stats.stored_bytes / stats.original_bytes
    return {
        "fp8_ckpt_boundaries": float(stats.boundaries),
        "fp8_ckpt_fallback_boundaries": float(stats.fallback_boundaries),
        "fp8_ckpt_warmup_boundaries": float(stats.warmup_boundaries),
        "fp8_ckpt_original_bytes": float(stats.original_bytes),
        "fp8_ckpt_stored_bytes": float(stats.stored_bytes),
        "fp8_ckpt_storage_reduction": float(reduction),
    }


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _record_compressed(original: torch.Tensor, packed: FP8CheckpointTensor) -> None:
    with _LOCK:
        _STATS.boundaries += 1
        _STATS.original_bytes += _tensor_bytes(original)
        _STATS.stored_bytes += _tensor_bytes(packed.values) + _tensor_bytes(packed.scales)


def _record_fallback(tensor: torch.Tensor, *, warmup: bool = False) -> None:
    with _LOCK:
        _STATS.fallback_boundaries += 1
        _STATS.warmup_boundaries += int(warmup)
        size = _tensor_bytes(tensor)
        _STATS.original_bytes += size
        _STATS.stored_bytes += size


def _stochastic_round(values: torch.Tensor) -> torch.Tensor:
    """Add unbiased E4M3-bin noise without advancing the caller's RNG."""

    key = str(values.device)
    with _LOCK:
        generator = _ROUNDING_GENERATORS.get(key)
        if generator is None:
            generator = torch.Generator(device=values.device)
            generator.manual_seed(0xA8E0 + (values.device.index or 0))
            _ROUNDING_GENERATORS[key] = generator
        random = torch.rand(values.shape, dtype=values.dtype, device=values.device, generator=generator)
    magnitude = values.abs()
    normal_step = torch.pow(2.0, torch.floor(torch.log2(magnitude.clamp_min(2.0**-9))) - 3.0)
    step = normal_step.clamp_min(2.0**-9)
    noise = (random - 0.5) * step
    return torch.where(magnitude == 0, values, values + noise)


def pack_fp8_checkpoint_tensor(
    tensor: torch.Tensor,
    *,
    group_size: int = 128,
    stochastic: bool = False,
) -> FP8CheckpointTensor:
    """Quantize a BF16 checkpoint tensor with E4M3 group-wise scales.

    ``group_size=0`` selects one scale per token.  A non-divisible hidden
    dimension falls back to per-token scaling rather than retaining padding.
    """

    if tensor.dtype != torch.bfloat16:
        raise TypeError("FP8 checkpoint compression requires a BF16 tensor")
    if tensor.ndim < 2 or tensor.numel() == 0:
        raise ValueError("FP8 checkpoint compression requires a non-empty tensor with ndim >= 2")
    if group_size not in {0, 128, 256}:
        raise ValueError("FP8 checkpoint group_size must be one of: 0, 128, 256")

    hidden_size = tensor.shape[-1]
    actual_group_size = hidden_size if group_size == 0 or hidden_size % group_size else group_size
    rows = tensor.reshape(-1, hidden_size).float()
    groups = rows.reshape(rows.shape[0], -1, actual_group_size)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(amax == 0, torch.ones_like(amax), amax / _FP8_MAX)
    normalized = (groups / scales).clamp(-_FP8_MAX, _FP8_MAX)
    if stochastic:
        normalized = _stochastic_round(normalized).clamp(-_FP8_MAX, _FP8_MAX)
    values = normalized.to(_FP8_DTYPE)
    return FP8CheckpointTensor(
        values=values,
        scales=scales,
        shape=tensor.shape,
        dtype=tensor.dtype,
        group_size=actual_group_size,
    )


def unpack_fp8_checkpoint_tensor(packed: FP8CheckpointTensor) -> torch.Tensor:
    """Restore a checkpoint payload to its original shape and dtype."""

    restored = packed.values.float() * packed.scales
    return restored.reshape(packed.shape).to(packed.dtype)


def _layer_owner(layer_fn: Any) -> Any:
    if isinstance(layer_fn, torch.nn.Module):
        return layer_fn
    return getattr(layer_fn, "__self__", layer_fn)


def checkpoint_layer_index(layer_fn: Any) -> int | None:
    """Resolve a stable decoder/routed-expert index from a checkpoint callable."""

    owner = _layer_owner(layer_fn)
    for candidate in (owner, getattr(owner, "self_attn", None), getattr(owner, "attention", None)):
        if candidate is None:
            continue
        for name in ("layer_idx", "routing_layer_slot"):
            value = getattr(candidate, name, None)
            if isinstance(value, int):
                return value
    return None


def checkpoint_saved_tensor_hooks(
    layer_fn: Any,
    *,
    group_size: int,
    stochastic: bool,
    warmup: bool,
    fallback_layers: tuple[int, ...],
):
    """Build hooks that compress only the input saved by a checkpoint call."""

    layer_index = checkpoint_layer_index(layer_fn)
    force_fallback = layer_index is not None and layer_index in fallback_layers

    def pack(tensor: torch.Tensor):
        if tensor.dtype != torch.bfloat16 or tensor.ndim < 2 or tensor.numel() == 0:
            return tensor
        if warmup:
            if layer_index is not None:
                value = tensor.detach().abs().amax()
                with _LOCK:
                    previous = _RANGE_AMAX.get(layer_index)
                    _RANGE_AMAX[layer_index] = value if previous is None else torch.maximum(previous, value)
            _record_fallback(tensor, warmup=True)
            return tensor
        if force_fallback:
            _record_fallback(tensor)
            return tensor
        try:
            packed = pack_fp8_checkpoint_tensor(tensor, group_size=group_size, stochastic=stochastic)
        except (RuntimeError, TypeError, ValueError):
            _record_fallback(tensor)
            return tensor
        _record_compressed(tensor, packed)
        return packed

    def unpack(value):
        if isinstance(value, FP8CheckpointTensor):
            return unpack_fp8_checkpoint_tensor(value)
        return value

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)
