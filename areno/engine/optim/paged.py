"""Adam variants whose persistent moments use real CUDA managed allocations.

The driver owns page residency and eviction. No separate CPU copy is kept.
Managed storage is outside PyTorch's caching allocator and must be reported
separately in memory comparisons, especially on unified-memory machines.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import torch

from areno.engine.optim.adamw_4bit import AdamW4bit
from areno.engine.optim.adamw_8bit import AdamW8bit
from areno.engine.optim.adamw_fp32_master import AdamWFP32Master


def managed_copy(tensor):
    if tensor is None or not tensor.numel() or getattr(tensor, "_areno_managed", False):
        return tensor
    if tensor.device.type != "cuda":
        raise ValueError("paged optimizer state must reside on a CUDA managed allocation")
    from areno.accel._extension import extension

    result = extension(tensor.device).areno_managed_empty_like(tensor.contiguous())
    result.copy_(tensor)
    result._areno_managed = True
    return result


class _PagedState:
    """Preserve the existing DP/FP32-master update and serialization contracts."""

    @torch.no_grad()
    def _ensure_bucket_state(self, bucket, *args):
        super()._ensure_bucket_state(bucket, *args)
        state = args[0] if args else bucket
        names = [field.name for field in fields(state)] if is_dataclass(state) else vars(state)
        for name in names:
            if not name.startswith("exp_avg"):
                continue
            value = getattr(state, name)
            if isinstance(value, torch.Tensor):
                setattr(state, name, managed_copy(value))
        if bucket.master_storage is not None:
            bucket.master_storage.low_bits = managed_copy(bucket.master_storage.low_bits)
            bucket.master_storage.round_up_bits = managed_copy(bucket.master_storage.round_up_bits)

    def managed_memory_bytes(self):
        """Actual persistent managed bytes, omitted by torch.cuda memory counters."""
        tensors = []
        for state in getattr(self, "_states", self.buckets):
            names = [field.name for field in fields(state)] if is_dataclass(state) else vars(state)
            tensors.extend(getattr(state, name) for name in names if name.startswith("exp_avg"))
        for bucket in self.buckets:
            if bucket.master_storage is not None:
                tensors.extend((bucket.master_storage.low_bits, bucket.master_storage.round_up_bits))
        tensors.extend(getattr(self, "_factored_second_moments", {}).values())
        return sum(t.numel() * t.element_size() for t in tensors if getattr(t, "_areno_managed", False))


class PagedAdamWFP32Master(_PagedState, AdamWFP32Master):
    pass


class PagedAdamW8bit(_PagedState, AdamW8bit):
    pass


class PagedAdamW4bit(_PagedState, AdamW4bit):
    def _ensure_factored_second_moment(self, parameter):
        value = managed_copy(super()._ensure_factored_second_moment(parameter))
        self._factored_second_moments[id(parameter)] = value
        return value
