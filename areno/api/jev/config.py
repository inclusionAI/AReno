"""Narrow configs; leave the public runtime/config defaults untouched."""

import math
from dataclasses import dataclass, fields
from pathlib import Path

from areno.api.config import CudaConfig, MlxConfig
from areno.api.trainer_config import TrainerConfig


@dataclass
class JevMlxConfig(MlxConfig):
    decision_head: bool = True
    head_lr: float = 2e-4
    decision_checkpoint: str | None = None
    max_seq_len: int = 1024

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lora is None and not self.inference_only:
            raise ValueError("JEV requires LoRA so the backbone and head train together")
        if not math.isfinite(self.head_lr) or self.head_lr <= 0:
            raise ValueError("head_lr must be positive")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >=2")


@dataclass
class JevCudaConfig(CudaConfig):
    decision_head: bool = True
    head_lr: float = 2e-4
    decision_checkpoint: str | None = None
    max_seq_len: int = 1024
    inference_only: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.head_lr) or self.head_lr <= 0 or self.max_seq_len < 2:
            raise ValueError("JEV needs positive head_lr and max_seq_len>=2")
        if self.uses_separate_rollout_engine():
            raise ValueError("offline JEV uses one training engine")
        self.runtime = {
            **self.runtime,
            "decision_head": {
                "head_lr": self.head_lr,
                "checkpoint": self.decision_checkpoint,
                "max_seq_len": self.max_seq_len,
            },
        }


@dataclass(slots=True)
class JevTrainerConfig(TrainerConfig):
    mini_bs: int = 1
    max_seq_len: int = 1024
    seed: int = 17
    head_lr: float = 2e-4
    brier_weight: float = 0.5

    def __post_init__(self) -> None:
        TrainerConfig.__post_init__(self)
        if self.backend not in {"mlx", "cuda"}:
            raise ValueError("JEV requires backend=mlx or cuda")
        if self.backend == "mlx" and (self.world_size != 1 or self.tp_size != 1):
            raise ValueError("MLX JEV requires world_size=1, tp_size=1")
        if self.world_size % self.tp_size:
            raise ValueError("world_size must be divisible by tp_size")
        if self.max_context_len is not None:
            self.max_seq_len = self.max_context_len
        if (
            self.save_path is None
            or self.save_interval < 1
            or self.max_seq_len < 2
            or self.batch_size < 1
            or self.mini_bs < 1
        ):
            raise ValueError("JEV needs save_path and positive save_interval, max_seq_len>=2 and batch_size")
        if not math.isfinite(self.brier_weight) or self.brier_weight < 0:
            raise ValueError("brier_weight must be nonnegative")

    def mlx_config(self) -> JevMlxConfig:
        base = TrainerConfig.mlx_config(self)
        values = {field.name: getattr(base, field.name) for field in fields(base)}
        checkpoint = self.lora.adapter_path if self.lora is not None else None
        if checkpoint is not None and not (Path(checkpoint) / "score_head.safetensors").is_file():
            checkpoint = None
        return JevMlxConfig(
            **values, head_lr=self.head_lr, max_seq_len=self.max_seq_len, decision_checkpoint=checkpoint
        )

    def cuda_config(self) -> JevCudaConfig:
        base = TrainerConfig.cuda_config(self)
        values = {field.name: getattr(base, field.name) for field in fields(base)}
        checkpoint = self.resume_checkpoint()
        return JevCudaConfig(
            **values, head_lr=self.head_lr, max_seq_len=self.max_seq_len, decision_checkpoint=checkpoint
        )

    def resume_checkpoint(self) -> str | None:
        checkpoint = self.lora.adapter_path if self.lora is not None else self.ckpt
        if checkpoint is not None and (Path(checkpoint) / "score_head.safetensors").is_file():
            return str(checkpoint)
        return None
