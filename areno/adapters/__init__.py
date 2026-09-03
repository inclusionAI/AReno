"""Native adapter runtime exposed by AReno.

Keep the configuration type importable on MLX-only installations while
loading the Torch-backed adapter implementation only when it is requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from areno.adapters.config import LoraConfig

if TYPE_CHECKING:
    from areno.adapters.lora import AdapterRegistry, LoraSlot, initialize_lora


def __getattr__(name: str) -> Any:
    if name not in {"AdapterRegistry", "LoraSlot", "initialize_lora"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    implementation = import_module("areno.adapters.lora")
    value = getattr(implementation, name)
    globals()[name] = value
    return value


__all__ = ["AdapterRegistry", "LoraConfig", "LoraSlot", "initialize_lora"]
