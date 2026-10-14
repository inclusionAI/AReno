"""Configuration for large-scale VLA post-training with a world-model backend."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SlurmResources:
    """Single-node resources used by the staged world-model workflow."""

    partition: str
    train_qos: str | None = None
    eval_qos: str | None = None
    train_time_limit: str = "3-00:00:00"
    eval_time_limit: str = "06:00:00"
    constraint: str | None = None
    account: str | None = None
    gpus: int = 8
    cpus: int = 64
    memory: str = "500G"
    node: str | None = None
    disable_gpu_binding: bool = False

    def validate(self) -> None:
        if self.gpus < 1:
            raise ValueError("slurm.gpus must be >= 1")
        if self.cpus < 1:
            raise ValueError("slurm.cpus must be >= 1")
        for name in (
            "partition",
            "train_time_limit",
            "eval_time_limit",
            "memory",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"slurm.{name} must be non-empty")
        for name in ("train_qos", "eval_qos", "constraint", "account", "node"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"slurm.{name} must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class WorldModelVLAConfig:
    """Frozen-Wan/OpenVLA post-training configuration.

    AReno owns the experiment lifecycle while the first backend delegates the
    three-role distributed execution to a compatible RLinf source checkout.
    """

    rlinf_root: Path
    rlinf_python: Path
    world_model_path: Path
    policy_path: Path
    output_dir: Path
    cache_root: Path
    slurm: SlurmResources
    stage_ends: tuple[int, ...] = (2, 20, 40, 60, 80, 100)
    actor_micro_batch_size: int = 1
    eval_trajectories: int = 496
    target_success_rate: float = 0.775
    ray_object_store_bytes: int = 16 * 1024**3
    verify_sha256: bool = False
    gpu_monitor: bool = True
    gpu_monitor_interval: int = 2
    require_rlinf_compatibility_patchset: bool = False
    setup_script: Path | None = None
    train_overrides: tuple[str, ...] = ()
    eval_overrides: tuple[str, ...] = ()
    extra_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "rlinf_root",
            "rlinf_python",
            "world_model_path",
            "policy_path",
            "output_dir",
            "cache_root",
        ):
            path = Path(getattr(self, name)).expanduser()
            # Keep the virtual-environment path so sibling commands such as
            # bin/ray and optional sources next to the environment remain discoverable.
            normalized = Path(os.path.abspath(path)) if name == "rlinf_python" else path.resolve()
            object.__setattr__(self, name, normalized)
        if self.setup_script is not None:
            object.__setattr__(self, "setup_script", Path(self.setup_script).expanduser().resolve())
        object.__setattr__(self, "stage_ends", tuple(int(step) for step in self.stage_ends))
        object.__setattr__(self, "train_overrides", tuple(str(value) for value in self.train_overrides))
        object.__setattr__(self, "eval_overrides", tuple(str(value) for value in self.eval_overrides))
        object.__setattr__(self, "extra_env", {str(key): str(value) for key, value in self.extra_env.items()})
        self.validate()

    def validate(self) -> None:
        self.slurm.validate()
        previous = 0
        if not self.stage_ends:
            raise ValueError("stage_ends must not be empty")
        for step in self.stage_ends:
            if step <= previous:
                raise ValueError("stage_ends must be a strictly increasing sequence of positive integers")
            previous = step
        if self.actor_micro_batch_size < 1:
            raise ValueError("actor_micro_batch_size must be >= 1")
        if self.eval_trajectories < 1:
            raise ValueError("eval_trajectories must be >= 1")
        if self.eval_trajectories % self.slurm.gpus:
            raise ValueError("eval_trajectories must be divisible by slurm.gpus")
        if not 0.0 <= self.target_success_rate <= 1.0:
            raise ValueError("target_success_rate must be between 0 and 1")
        if self.ray_object_store_bytes < 1:
            raise ValueError("ray_object_store_bytes must be positive")
        if self.gpu_monitor_interval < 1:
            raise ValueError("gpu_monitor_interval must be >= 1")
        for name in ("verify_sha256", "gpu_monitor", "require_rlinf_compatibility_patchset"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if any(not value.strip() for value in (*self.train_overrides, *self.eval_overrides)):
            raise ValueError("Hydra overrides must be non-empty strings")
        if any(not key.strip() for key in self.extra_env):
            raise ValueError("extra_env keys must be non-empty strings")

    @classmethod
    def from_json(cls, path: str | Path) -> WorldModelVLAConfig:
        source = Path(path).expanduser().resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("world-model VLA config must contain a JSON object")
        data = dict(data)
        slurm_data = data.pop("slurm", None)
        if not isinstance(slurm_data, dict):
            raise ValueError("slurm must contain a JSON object with at least a partition")
        for name in (
            "rlinf_root",
            "rlinf_python",
            "world_model_path",
            "policy_path",
            "output_dir",
            "cache_root",
            "setup_script",
        ):
            value = data.get(name)
            if value is not None and not Path(value).expanduser().is_absolute():
                data[name] = source.parent / Path(value).expanduser()
        try:
            data["slurm"] = SlurmResources(**slurm_data)
        except TypeError as error:
            raise ValueError(f"invalid slurm configuration: {error}") from error
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in (
            "rlinf_root",
            "rlinf_python",
            "world_model_path",
            "policy_path",
            "output_dir",
            "cache_root",
            "setup_script",
        ):
            value = data[name]
            data[name] = None if value is None else str(value)
        data["stage_ends"] = list(self.stage_ends)
        data["train_overrides"] = list(self.train_overrides)
        data["eval_overrides"] = list(self.eval_overrides)
        return data

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
        try:
            temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = ["SlurmResources", "WorldModelVLAConfig"]
