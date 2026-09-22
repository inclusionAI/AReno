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
