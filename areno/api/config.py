"""Typed areno backend configuration and selection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from areno.api.models import BackendType


@dataclass(slots=True)
class ArenoConfig:
    """Typed backend config for the local/process based areno backend.

    `tp_size`/`dp_size` describe the parallelism layout used by `ArenoEngine`
    (when `dp_size` is None the backend infers it from world size / tp size).
    The `optimizer` and `runtime` dicts are passed verbatim to the engine's
    `OptimizerConfig`/`RuntimeConfig` so any new tuning knob can be added
    without changing this file.
    """

    model_path: str | None = None
    tp_size: int = 1
    dp_size: int | None = None
    devices: list[int] | None = None
    dummy_load: bool = False
    optimizer: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    max_running_prompts: int = 64
    decode_progress_interval_s: float = 10.0


BackendConfig = ArenoConfig


def resolve_backend_type(backend_type: BackendType | None, custom_config: Any) -> BackendType:
    """Choose the backend from an explicit value or the default Areno."""

    del custom_config
    if backend_type is not None:
        return backend_type
    return BackendType.Areno


def coerce_backend_config(backend_type: BackendType, custom_config: Any) -> BackendConfig | None:
    """Validate that the config dataclass matches the selected backend.

    Returning ``None`` for a missing config lets the backend fall back to its
    own defaults; passing a mismatched dataclass raises early so misconfigured
    runs fail at construction time rather than during training.
    """

    if custom_config is None:
        return None
    if backend_type == BackendType.Areno and isinstance(custom_config, ArenoConfig):
        return custom_config
    raise TypeError(f"{backend_type.value} requires its typed backend config dataclass, got {type(custom_config)!r}")
