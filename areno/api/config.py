"""Typed areno backend configuration and selection helpers."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any

from areno.api.models import BackendType


@dataclass(slots=True)
class CudaConfig:
    """Typed backend config for the local/process based CUDA backend.

    `tp_size`/`dp_size` describe the parallelism layout used by `ArenoEngine`
    (when `dp_size` is None the backend infers it from world size / tp size).
    `sequence_parallel` is a tri-state checkpoint override: None preserves the
    model adapter's value while True/False explicitly replace it.
    The `optimizer` and `runtime` dicts are passed verbatim to the engine's
    `OptimizerConfig`/`RuntimeConfig` so any new tuning knob can be added
    without changing this file.
    """

    model_path: str | None = None
    tp_size: int = 1
    sequence_parallel: bool | None = None
    dp_size: int | None = None
    devices: list[int] | None = None
    rollout_tp_size: int | None = None
    rollout_devices: list[int] | None = None
    policy_sync_bucket_mb: int = 64
    dummy_load: bool = False
    optimizer: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    max_running_prompts: int = 64
    decode_progress_interval_s: float = 10.0

    def uses_separate_rollout_engine(self) -> bool:
        """Return whether rollout runs on its own CUDA device partition."""

        return self.rollout_devices is not None

    def resolved_rollout_tp_size(self) -> int:
        """Return the explicit rollout TP size or the training TP size."""

        return self.tp_size if self.rollout_tp_size is None else self.rollout_tp_size


@dataclass(slots=True)
class MlxConfig:
    """Configuration for the in-process MLX/MLX-LM backend."""

    model_path: str | None = None
    adapter_path: str | None = None
    optimizer: dict[str, Any] = field(default_factory=dict)
    max_running_prompts: int = 32
    completion_batch_size: int = 32
    prefill_batch_size: int = 8
    prefill_step_size: int = 2048
    max_kv_size: int | None = None
    decode_progress_interval_s: float = 10.0
    keep_rollout_state: bool = True
    logits_chunk_size: int = 4096
    compile_train_step: bool = True
    gradient_checkpointing: bool = True


BackendConfig = CudaConfig | MlxConfig


def default_backend_type() -> BackendType:
    """Select the native backend for the current host platform."""

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Linux":
        return BackendType.CUDA
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return BackendType.MLX
    raise RuntimeError(
        f"AReno has no native backend for {system}/{machine}; CUDA requires Linux and MLX requires Apple Silicon"
    )


def resolve_backend_type(backend_type: BackendType | None, custom_config: Any) -> BackendType:
    """Choose an explicit backend or the native backend for this platform."""

    del custom_config
    if backend_type is not None:
        return backend_type
    return default_backend_type()


def coerce_backend_config(backend_type: BackendType, custom_config: Any) -> BackendConfig | None:
    """Validate that the config dataclass matches the selected backend.

    Returning ``None`` for a missing config lets the backend fall back to its
    own defaults; passing a mismatched dataclass raises early so misconfigured
    runs fail at construction time rather than during training.
    """

    if custom_config is None:
        return None
    if backend_type == BackendType.CUDA and isinstance(custom_config, CudaConfig):
        return custom_config
    if backend_type == BackendType.MLX and isinstance(custom_config, MlxConfig):
        return custom_config
    raise TypeError(f"{backend_type.value} requires its typed backend config dataclass, got {type(custom_config)!r}")
