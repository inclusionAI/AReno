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
        if self.lora is None:
            raise ValueError("JEV requires LoRA so the backbone and head train together")
        if not math.isfinite(self.head_lr) or self.head_lr <= 0:
            raise ValueError("head_lr must be positive")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >=2")


@dataclass(slots=True)
class JevCudaConfig(CudaConfig):
    """CUDA config for JEV: the engine owns the scoring head.

    The head is attached by `areno.engine.score_head` when `runtime.score_head`
    is set, trained as its own optimizer LR group (`optimizer.score_head_lr`,
    with an optional head-only warmup), and written next to the HF backbone as
    `score_head.safetensors`. With native LoRA the backbone stays frozen and the
    head trains alongside the adapters, so a checkpoint is the PEFT adapter
    directory with `score_head.safetensors` written into it. Resuming a JEV run
    loads the head from the LoRA adapter directory when it carries one, and
    otherwise from `model_path` (a full-weight JEV checkpoint).
    """

    head_lr: float = 2e-4
    score_head_warmup_steps: int = 0
    max_seq_len: int = 1024

    def __post_init__(self) -> None:
        # `CudaConfig` is a slots dataclass; the slots rewrite replaces the
        # class object, so a zero-arg `super()` cannot resolve. Call the base
        # implementation unbound instead.
        CudaConfig.__post_init__(self)
        if not math.isfinite(self.head_lr) or self.head_lr <= 0:
            raise ValueError("head_lr must be positive")
        if self.score_head_warmup_steps < 0:
            raise ValueError("score_head_warmup_steps must be non-negative")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >=2")
        # The engine reads these through RuntimeConfig/OptimizerConfig.
        self.runtime = {**self.runtime, "score_head": True}
        self.optimizer = {
            **self.optimizer,
            "score_head_lr": self.head_lr,
            "score_head_warmup_steps": self.score_head_warmup_steps,
        }


@dataclass(slots=True)
class JevTrainerConfig(TrainerConfig):
    mini_bs: int = 1
    max_seq_len: int = 1024
    seed: int = 17
    head_lr: float = 2e-4
    brier_weight: float = 0.5
    score_head_warmup_steps: int = 0

    def __post_init__(self) -> None:
        TrainerConfig.__post_init__(self)
        if self.backend not in {"mlx", "cuda"}:
            raise ValueError("JEV requires backend=mlx or backend=cuda")
        if self.backend == "mlx" and (self.world_size != 1 or self.tp_size != 1):
            raise ValueError("JEV on MLX requires world_size=1 and tp_size=1")
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
        if self.score_head_warmup_steps < 0:
            raise ValueError("score_head_warmup_steps must be non-negative")

    def _checkpoint(self) -> str | None:
        checkpoint = self.lora.adapter_path if self.lora is not None else None
        if checkpoint is not None and not (Path(checkpoint) / "score_head.safetensors").is_file():
            checkpoint = None
        return checkpoint

    def mlx_config(self) -> JevMlxConfig:
        base = TrainerConfig.mlx_config(self)
        values = {field.name: getattr(base, field.name) for field in fields(base)}
        return JevMlxConfig(
            **values, head_lr=self.head_lr, max_seq_len=self.max_seq_len, decision_checkpoint=self._checkpoint()
        )

    def cuda_config(self) -> JevCudaConfig:
        base = TrainerConfig.cuda_config(self)
        values = {field.name: getattr(base, field.name) for field in fields(base)}
        values["optimizer"] = dict(values["optimizer"])
        values["runtime"] = dict(values["runtime"])
        return JevCudaConfig(
            **values,
            head_lr=self.head_lr,
            score_head_warmup_steps=self.score_head_warmup_steps,
            max_seq_len=self.max_seq_len,
        )