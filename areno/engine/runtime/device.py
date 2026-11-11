"""Device-specific hooks used by the shared Torch engine."""

from types import ModuleType
from typing import Protocol

import torch


class Completion(Protocol):
    """A stream or event that owns an outstanding device transfer."""

    def synchronize(self) -> None: ...


def accelerator_module(device: torch.device) -> ModuleType | None:
    """Resolve the worker's registered accelerator without probing other hardware."""

    return getattr(torch, device.type) if device.type in {"cuda", "npu"} else None


def configure_unified_memory_budget(device: torch.device) -> float | None:
    """Reject oversized Spark allocator requests before they reach the driver.

    MST only checks pressure between chunks; it cannot bound a single forward
    or temporary allocation. The inherited CLI budget also caps PyTorch's
    allocator, leaving physical memory available to the OS and harness.
    """
    import logging
    import os

    if device.type != "cuda":
        return None
    properties = torch.cuda.get_device_properties(device)
    if "GB10" not in properties.name:
        return None
    fraction = float(os.environ.get("ARENO_MST_MEM_FRAC", "0.9"))
    if not 0 < fraction <= 1:
        raise ValueError("CUDA memory fraction must be in (0, 1]")
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    logging.getLogger(__name__).info(
        "Spark CUDA allocator budget %.3f (%.2f GiB); swap is excluded",
        fraction,
        fraction * properties.total_memory / (1024**3),
    )
    return fraction
