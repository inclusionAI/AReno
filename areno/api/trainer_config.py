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

from dataclasses import dataclass

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
    dataset_loader_fn: str | None = None
    save_path: str | None = None
    save_interval: int = 100
    epochs: int = 10
    max_steps: int | None = None
    tp_size: int = 4
    world_size: int = 8
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
    activation_checkpointing: bool = True
    keep_rollout_state: bool = True
    eager_decode: bool = False
    attn_backend: str = "flash"
    metrics_log_dir: str | None = DEFAULT_METRICS_LOG_DIR
    agent_fn: str | None = None
    agent_timeout_s: float = 300.0
    train_tool_results: bool = False
    chat_template_enable_thinking: bool | None = None

    def __post_init__(self) -> None:
        if self.attn_backend not in {"flash", "native"}:
            raise ValueError("attn_backend must be one of: flash, native")

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
        }

    def areno_config(self):
        """Build the backend config exposed by this trainer config.

        Imported lazily so consumers that never touch areno (e.g. the verl
        wrapper) avoid pulling in its dependency tree.
        """

        from areno.api.config import ArenoConfig

        return ArenoConfig(
            tp_size=self.tp_size,
            optimizer=self.optimizer_config(),
            runtime={
                "activation_checkpointing": self.activation_checkpointing,
                "keep_rollout_state": self.keep_rollout_state,
                "eager_decode": self.eager_decode,
                "attn_backend": self.attn_backend,
            },
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

    def resolved_max_running_prompts(self) -> int:
        """Return explicit or full-batch rollout concurrency."""

        if self.max_running_prompts is not None:
            return self.max_running_prompts
        return max(self.batch_size * self.n_samples, 1)

    def areno_config(self):
        """Build backend config including rollout cache capacity."""

        from areno.api.config import ArenoConfig

        return ArenoConfig(
            tp_size=self.tp_size,
            max_running_prompts=self.resolved_max_running_prompts(),
            optimizer=self.optimizer_config(),
            runtime={
                "activation_checkpointing": self.activation_checkpointing,
                "keep_rollout_state": self.keep_rollout_state,
                "eager_decode": self.eager_decode,
                "attn_backend": self.attn_backend,
            },
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
