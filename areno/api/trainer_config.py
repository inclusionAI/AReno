"""Dataclass trainer configurations consumed by `build_trainer`.

`TrainerConfig` captures fields common to every training algorithm.
`RolloutTrainerConfig` adds sampling/rollout fields. `PolicyTrainerConfig`
adds reward-function wiring and GSPO/GRPO clipping for policy-gradient RL.
`DPOTrainerConfig` adds the frozen reference policy and DPO temperature.
`PPOTrainerConfig` extends the policy config with the extra knobs PPO requires:
role checkpoints, KL/PPO clipping, value loss weighting, GAE constants, and a
critic warmup window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from areno.adapters.config import LoraConfig
from areno.api.defaults import DEFAULT_METRICS_LOG_DIR


@dataclass(slots=True)
class TrainerConfig:
    """Common runtime settings shared by all trainers.

    The defaults match the defaults used by the project's `train.py` CLI so a
    bare ``TrainerConfig(...)`` can drive simple supervised trainers.
    """

    algo: str
    ckpt: str
    dataset_path: str
    backend: str | None = None
    base_model_name_or_path: str | None = field(default=None, kw_only=True)
    model_hub: str = "modelscope"
    dataset_loader_fn: str | None = None
    save_path: str | None = None
    save_interval: int = 100
    epochs: int = 10
    max_steps: int | None = None
    tp_size: int = 4
    sequence_parallel: bool | None = None
    world_size: int = 8
    train_devices: list[int] | None = None
    batch_size: int = 32
    mini_bs: int = 16
    score_micro_bs: int = 8
    gradient_accumulation_steps: int | None = None
    max_prompt_tokens: int = 1024
    max_new_tokens: int = 3071
    max_context_len: int | None = None
    optimizer_lr: float = 1.0e-6
    optimizer_min_lr: float = 1.0e-7
    lr_decay_steps: int = 1000
    lr_decay_style: str = "cosine"
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    weight_decay: float = 1.0e-2
    grad_clip_norm: float = 1.0
    adam_8bit: bool = False
    unfreeze_multimodal_tower: bool = False
    unfreeze_multimodal_projector: bool = False
    multimodal_tower_lr: float | None = None
    multimodal_tower_min_lr: float | None = None
    multimodal_tower_lr_decay_steps: int | None = None
    multimodal_tower_lr_decay_style: str | None = None
    multimodal_projector_lr: float | None = None
    multimodal_projector_min_lr: float | None = None
    multimodal_projector_lr_decay_steps: int | None = None
    multimodal_projector_lr_decay_style: str | None = None
    activation_checkpointing: bool = True
    keep_rollout_state: bool = True
    optimizer_state_offload: str | bool = "none"
    optimizer_state_offload_dir: str | None = None
    optimizer_state_offload_batch_size: int = 1
    eager_decode: bool = False
    attn_backend: str = "flash"
    metrics_log_dir: str | None = DEFAULT_METRICS_LOG_DIR
    agent_fn: str | None = None
    agent_timeout_s: float = 300.0
    train_tool_results: bool = False
    chat_template_enable_thinking: bool | None = None
    lora: LoraConfig | None = None
    reference_mode: Literal["independent", "reuse_actor_base"] = "independent"

    def __post_init__(self) -> None:
        if self.backend is None:
            from areno.api.config import default_backend_type

            self.backend = default_backend_type().value.lower()
        else:
            self.backend = self.backend.lower()
        if self.backend not in {"cuda", "mlx"}:
            raise ValueError("backend must be one of: cuda, mlx")
        if self.attn_backend not in {"flash", "native"}:
            raise ValueError("attn_backend must be one of: flash, native")
        if self.model_hub not in {"hf", "modelscope"}:
            raise ValueError("model_hub must be one of: hf, modelscope")
        if isinstance(self.optimizer_state_offload, bool):
            self.optimizer_state_offload = "cpu" if self.optimizer_state_offload else "none"
        if self.optimizer_state_offload not in {"none", "cpu", "disk"}:
            raise ValueError("optimizer_state_offload must be one of: none, cpu, disk")
        if self.optimizer_state_offload == "disk" and not self.optimizer_state_offload_dir:
            raise ValueError("optimizer_state_offload_dir is required for disk offload")
        if self.optimizer_state_offload_batch_size < 1:
            raise ValueError("optimizer_state_offload_batch_size must be positive")
        if self.optimizer_state_offload != "none" and self.backend != "cuda":
            raise ValueError("optimizer_state_offload is only supported by the CUDA backend")
        self._validate_multimodal_optimizer_group(
            "tower",
            self.unfreeze_multimodal_tower,
            self.multimodal_tower_lr,
            self.multimodal_tower_min_lr,
            self.multimodal_tower_lr_decay_steps,
            self.multimodal_tower_lr_decay_style,
        )
        self._validate_multimodal_optimizer_group(
            "projector",
            self.unfreeze_multimodal_projector,
            self.multimodal_projector_lr,
            self.multimodal_projector_min_lr,
            self.multimodal_projector_lr_decay_steps,
            self.multimodal_projector_lr_decay_style,
        )
        if self.lora is not None and self.backend != "cuda":
            raise ValueError("native LoRA is only supported by the CUDA backend")

    @staticmethod
    def _validate_multimodal_optimizer_group(
        group: str,
        enabled: bool,
        lr: float | None,
        min_lr: float | None,
        decay_steps: int | None,
        decay_style: str | None,
    ) -> None:
        if lr is not None and lr <= 0:
            raise ValueError(f"multimodal_{group}_lr must be positive")
        if min_lr is not None and min_lr < 0:
            raise ValueError(f"multimodal_{group}_min_lr must be non-negative")
        if decay_steps is not None and decay_steps <= 0:
            raise ValueError(f"multimodal_{group}_lr_decay_steps must be positive")
        if decay_style is not None and decay_style not in {"constant", "linear", "cosine"}:
            raise ValueError(f"multimodal_{group}_lr_decay_style must be one of: constant, linear, cosine")
        if not enabled and any(value is not None for value in (lr, min_lr, decay_steps, decay_style)):
            raise ValueError(f"multimodal {group} LR options require unfreeze_multimodal_{group}=True")

    def optimizer_config(self) -> dict:
        """Build the optimizer dict consumed by the backend config."""

        return {
            "lr": self.optimizer_lr,
            "min_lr": self.optimizer_min_lr,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_decay_style": self.lr_decay_style,
            "betas": (self.optimizer_beta1, self.optimizer_beta2),
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
            "adam_8bit": self.adam_8bit,
            "unfreeze_multimodal_tower": self.unfreeze_multimodal_tower,
            "unfreeze_multimodal_projector": self.unfreeze_multimodal_projector,
            "multimodal_tower_lr": self.multimodal_tower_lr,
            "multimodal_tower_min_lr": self.multimodal_tower_min_lr,
            "multimodal_tower_lr_decay_steps": self.multimodal_tower_lr_decay_steps,
            "multimodal_tower_lr_decay_style": self.multimodal_tower_lr_decay_style,
            "multimodal_projector_lr": self.multimodal_projector_lr,
            "multimodal_projector_min_lr": self.multimodal_projector_min_lr,
            "multimodal_projector_lr_decay_steps": self.multimodal_projector_lr_decay_steps,
            "multimodal_projector_lr_decay_style": self.multimodal_projector_lr_decay_style,
        }

    def backend_type(self):
        """Return the selected execution backend without importing it eagerly."""

        from areno.api.models import BackendType

        return BackendType.MLX if self.backend.lower() == "mlx" else BackendType.CUDA

    def backend_config(self):
        """Build the typed configuration for the selected backend."""

        if self.backend.lower() == "mlx":
            return self.mlx_config()
        return self.cuda_config()

    def mlx_config(self):
        """Build the MLX backend config using common optimizer settings."""

        from areno.api.config import MlxConfig

        return MlxConfig(
            optimizer=self.optimizer_config(),
            keep_rollout_state=self.keep_rollout_state,
            compile_train_step=True,
            gradient_checkpointing=self.activation_checkpointing,
        )

    def cuda_config(self):
        """Build the backend config exposed by this trainer config.

        Imported lazily so consumers that never touch areno (e.g. the verl
        wrapper) avoid pulling in its dependency tree.
        """

        from areno.api.config import CudaConfig

        return CudaConfig(
            base_model_name_or_path=self.base_model_name_or_path,
            tp_size=self.tp_size,
            sequence_parallel=self.sequence_parallel,
            devices=self.train_devices,
            optimizer=self.optimizer_config(),
            runtime={
                "activation_checkpointing": self.activation_checkpointing,
                "keep_rollout_state": self.keep_rollout_state,
                "optimizer_state_offload": self.optimizer_state_offload,
                "optimizer_state_offload_dir": self.optimizer_state_offload_dir,
                "optimizer_state_offload_batch_size": self.optimizer_state_offload_batch_size,
                "eager_decode": self.eager_decode,
                "attn_backend": self.attn_backend,
            },
            lora=self.lora,
            reference_mode=self.reference_mode,
        )


@dataclass(slots=True)
class RolloutTrainerConfig(TrainerConfig):
    """Sampling/rollout settings used by online RL trainers."""

    n_samples: int = 8
    greedy: bool = False
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    max_running_prompts: int | None = None
    rollout_tp_size: int | None = None
    rollout_devices: list[int] | None = None
    policy_sync_bucket_mb: int = 64

    def resolved_max_running_prompts(self) -> int:
        """Return explicit or full-batch rollout concurrency."""

        if self.max_running_prompts is not None:
            return self.max_running_prompts
        return max(self.batch_size * self.n_samples, 1)

    def cuda_config(self):
        """Build backend config including rollout cache capacity."""

        from areno.api.config import CudaConfig

        return CudaConfig(
            base_model_name_or_path=self.base_model_name_or_path,
            tp_size=self.tp_size,
            sequence_parallel=self.sequence_parallel,
            devices=self.train_devices,
            rollout_tp_size=self.rollout_tp_size,
            rollout_devices=self.rollout_devices,
            policy_sync_bucket_mb=self.policy_sync_bucket_mb,
            max_running_prompts=self.resolved_max_running_prompts(),
            optimizer=self.optimizer_config(),
            runtime={
                "activation_checkpointing": self.activation_checkpointing,
                "keep_rollout_state": self.keep_rollout_state,
                "optimizer_state_offload": self.optimizer_state_offload,
                "optimizer_state_offload_dir": self.optimizer_state_offload_dir,
                "optimizer_state_offload_batch_size": self.optimizer_state_offload_batch_size,
                "eager_decode": self.eager_decode,
                "attn_backend": self.attn_backend,
            },
            lora=self.lora,
            reference_mode=self.reference_mode,
        )

    def mlx_config(self):
        """Build MLX config with rollout concurrency from this trainer."""

        from areno.api.config import MlxConfig

        max_running = self.resolved_max_running_prompts()
        return MlxConfig(
            optimizer=self.optimizer_config(),
            max_running_prompts=max_running,
            completion_batch_size=max_running,
            prefill_batch_size=min(max_running, 8),
            keep_rollout_state=self.keep_rollout_state,
            compile_train_step=True,
            gradient_checkpointing=self.activation_checkpointing,
        )


@dataclass(slots=True)
class PolicyTrainerConfig(RolloutTrainerConfig):
    """Reward-driven policy trainer configuration for GSPO/GRPO."""

    reward_fn_path: str | None = None
    gspo_clip_eps: float = 3.0e-4
    grpo_clip_eps: float = 0.2


@dataclass(slots=True)
class DPOTrainerConfig(TrainerConfig):
    """DPO role configuration.

    DPO uses the trainable policy plus one frozen reference policy. Preference
    rows are materialized as consecutive chosen/rejected `TrainSequence` pairs,
    and `dpo_beta` controls the logistic margin temperature.
    """

    ref_ckpt: str | None = None
    dpo_beta: float = 0.1


@dataclass(slots=True)
class PPOTrainerConfig(PolicyTrainerConfig):
    """PPO role configuration.

    Actor is the trainable policy. Ref, reward, and critic are independent
    roles owned by the trainer. Their load/offload lifecycle must stay behind
    backend/trainer boundaries; algorithm code should not call memory movement
    APIs directly.
    """

    ref_ckpt: str | None = None
    reward_ckpt: str | None = None
    critic_ckpt: str | None = None
    role_device: str | None = None
    critic_lr: float = 1e-5
    kl_coef: float = 0.02
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    clip_eps: float = 0.2
    clip_ratio_c: float = 3.0
    value_clip_eps: float = 0.5
    value_loss_coef: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95
    # The first `critic_warmup_steps` steps train only the critic so the value
    # baseline is calibrated before the actor starts using its advantages.
    critic_warmup_steps: int = 20
