from __future__ import annotations

import pytest
from click import UsageError

from areno.api.trainer_config import DPOTrainerConfig, PolicyTrainerConfig, PPOTrainerConfig, TrainerConfig
from areno.cli.train import _trainer_config_from_options


def test_train_config_requires_ckpt():
    with pytest.raises(UsageError, match="--ckpt is required"):
        _trainer_config_from_options(**_options(ckpt=None, algo="sft"))


def test_train_config_requires_dataset_path():
    with pytest.raises(UsageError, match="--dataset-path is required"):
        _trainer_config_from_options(**_options(dataset_path=None, algo="sft"))


def test_train_config_requires_reward_source_for_rollout_algorithms():
    with pytest.raises(UsageError, match="--reward-fn-path or --reward-ckpt is required"):
        _trainer_config_from_options(**_options(algo="gspo", reward_fn_path=None, reward_ckpt=None))


def test_train_config_unknown_algorithm_message_lists_registered_algorithms():
    with pytest.raises(UsageError, match=r"unknown algorithm 'bogus'; registered: .*dpo.*gspo.*ppo.*sft"):
        _trainer_config_from_options(**_options(algo="bogus"))


def test_train_config_requires_world_size_divisible_by_tp_size():
    with pytest.raises(UsageError, match="--world-size must be divisible by --tp-size"):
        _trainer_config_from_options(**_options(algo="sft", world_size=3, tp_size=2))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("save_interval", 0, "--save-interval must be positive"),
        ("epochs", 0, "--epochs must be positive"),
        ("tp_size", 0, "--tp-size must be positive"),
        ("world_size", 0, "--world-size must be positive"),
        ("batch_size", 0, "--batch-size must be positive"),
        ("n_samples", 0, "--n-samples must be positive"),
        ("mini_bs", 0, "--mini-bs must be positive"),
        ("gradient_accumulation_steps", 0, "--gradient-accumulation-steps must be positive"),
        ("max_prompt_tokens", 0, "--max-prompt-tokens must be positive"),
        ("max_new_tokens", 0, "--max-new-tokens must be positive"),
        ("max_running_prompts", 0, "--max-running-prompts must be positive"),
        ("agent_timeout_s", 0.0, "--agent-timeout-s must be positive"),
        ("lr", 0.0, "--lr must be positive"),
        ("min_lr", -0.1, "--min-lr must be non-negative"),
        ("lr_decay_steps", 0, "--lr-decay-steps must be positive"),
        ("adam_beta1", 0.0, "--adam-beta1 must be positive"),
        ("adam_beta2", 0.0, "--adam-beta2 must be positive"),
        ("weight_decay", -0.1, "--weight-decay must be non-negative"),
        ("grad_clip_norm", 0.0, "--grad-clip-norm must be positive"),
        ("temperature", 0.0, "--temperature must be positive"),
        ("top_p", 0.0, "--top-p must be positive"),
        ("gspo_clip_eps", 0.0, "--gspo-clip-eps must be positive"),
    ],
)
def test_train_config_validates_common_positive_fields(field, value, message):
    with pytest.raises(UsageError, match=message):
        _trainer_config_from_options(**_options(algo="gspo", **{field: value}))


def test_train_config_validates_grpo_clip_eps_only_for_grpo():
    with pytest.raises(UsageError, match="--grpo-clip-eps must be positive"):
        _trainer_config_from_options(**_options(algo="grpo", grpo_clip_eps=0.0))


def test_train_config_does_not_validate_unused_rollout_clip_eps():
    gspo = _trainer_config_from_options(**_options(algo="gspo", grpo_clip_eps=0.0))
    ppo = _trainer_config_from_options(**_options(algo="ppo", gspo_clip_eps=0.0, grpo_clip_eps=0.0))

    assert isinstance(gspo, PolicyTrainerConfig)
    assert isinstance(ppo, PPOTrainerConfig)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dpo_beta", 0.0, "--dpo-beta must be positive"),
    ],
)
def test_train_config_validates_dpo_positive_fields(field, value, message):
    with pytest.raises(UsageError, match=message):
        _trainer_config_from_options(**_options(algo="dpo", **{field: value}))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("critic_lr", 0.0, "--critic-lr must be positive"),
        ("kl_loss_coef", 0.0, "--kl-loss-coef must be positive"),
        ("clip_eps", 0.0, "--clip-eps must be positive"),
        ("clip_ratio_c", 0.0, "--clip-ratio-c must be positive"),
        ("value_clip_eps", 0.0, "--value-clip-eps must be positive"),
        ("value_loss_coef", 0.0, "--value-loss-coef must be positive"),
        ("gamma", 0.0, "--gamma must be positive"),
        ("lam", 0.0, "--lam must be positive"),
    ],
)
def test_train_config_validates_ppo_positive_fields(field, value, message):
    with pytest.raises(UsageError, match=message):
        _trainer_config_from_options(**_options(algo="ppo", **{field: value}))


def test_train_config_builds_sft_shape_without_rollout_or_role_fields():
    cfg = _trainer_config_from_options(**_options(algo="sft", reward_fn_path=None, reward_ckpt=None, min_lr=0.0))

    assert type(cfg) is TrainerConfig
    assert cfg.algo == "sft"
    assert cfg.ckpt == "actor"
    assert cfg.dataset_path == "dataset"
    assert cfg.optimizer_min_lr == 0.0
    assert cfg.batch_size == 2
    assert cfg.mini_bs == 1
    assert not hasattr(cfg, "n_samples")
    assert not hasattr(cfg, "reward_fn_path")
    assert not hasattr(cfg, "ref_ckpt")


def test_train_config_builds_dpo_shape_and_ref_ckpt():
    cfg = _trainer_config_from_options(
        **_options(algo="dpo", reward_fn_path=None, reward_ckpt=None, ref_ckpt="reference", dpo_beta=0.25)
    )

    assert isinstance(cfg, DPOTrainerConfig)
    assert cfg.algo == "dpo"
    assert cfg.ref_ckpt == "reference"
    assert cfg.dpo_beta == 0.25
    assert not hasattr(cfg, "n_samples")
    assert not hasattr(cfg, "reward_fn_path")


@pytest.mark.parametrize(
    ("algo", "clip_attr", "clip_value"), [("gspo", "gspo_clip_eps", 0.123), ("grpo", "grpo_clip_eps", 0.456)]
)
def test_train_config_builds_policy_shape_for_gspo_and_grpo(algo, clip_attr, clip_value):
    cfg = _trainer_config_from_options(
        **_options(
            algo=algo,
            reward_fn_path="reward.py",
            n_samples=3,
            max_running_prompts=12,
            temperature=0.7,
            top_k=10,
            top_p=0.9,
            **{clip_attr: clip_value},
        )
    )

    assert isinstance(cfg, PolicyTrainerConfig)
    assert type(cfg) is PolicyTrainerConfig
    assert cfg.algo == algo
    assert cfg.reward_fn_path == "reward.py"
    assert cfg.n_samples == 3
    assert cfg.resolved_max_running_prompts() == 12
    assert cfg.temperature == 0.7
    assert cfg.top_k == 10
    assert cfg.top_p == 0.9
    assert getattr(cfg, clip_attr) == clip_value
    assert not hasattr(cfg, "ref_ckpt")


def test_train_config_builds_ppo_shape_and_role_checkpoints():
    cfg = _trainer_config_from_options(
        **_options(
            algo="ppo",
            reward_fn_path="reward.py",
            ref_ckpt="reference",
            reward_ckpt="reward-model",
            critic_ckpt="critic",
            critic_lr=2e-5,
            use_kl_loss=False,
            kl_loss_coef=0.02,
            kl_loss_type="mse",
            clip_eps=0.3,
            clip_ratio_c=4.0,
            value_clip_eps=0.6,
            value_loss_coef=0.7,
            gamma=0.99,
            lam=0.9,
            critic_warmup_steps=5,
        )
    )

    assert isinstance(cfg, PPOTrainerConfig)
    assert cfg.algo == "ppo"
    assert cfg.ref_ckpt == "reference"
    assert cfg.reward_ckpt == "reward-model"
    assert cfg.critic_ckpt == "critic"
    assert cfg.critic_lr == 2e-5
    assert cfg.use_kl_loss is False
    assert cfg.kl_loss_coef == 0.02
    assert cfg.kl_loss_type == "mse"
    assert cfg.clip_eps == 0.3
    assert cfg.clip_ratio_c == 4.0
    assert cfg.value_clip_eps == 0.6
    assert cfg.value_loss_coef == 0.7
    assert cfg.gamma == 0.99
    assert cfg.lam == 0.9
    assert cfg.critic_warmup_steps == 5


def test_train_config_reward_ckpt_satisfies_rollout_preflight_without_local_reward_fn():
    cfg = _trainer_config_from_options(**_options(algo="gspo", reward_fn_path=None, reward_ckpt="reward-model"))

    assert isinstance(cfg, PolicyTrainerConfig)
    assert cfg.reward_fn_path is None
    assert not hasattr(cfg, "reward_ckpt")


def test_train_config_ppo_preserves_reward_ckpt_as_role_checkpoint():
    cfg = _trainer_config_from_options(**_options(algo="ppo", reward_fn_path=None, reward_ckpt="reward-model"))

    assert isinstance(cfg, PPOTrainerConfig)
    assert cfg.reward_fn_path is None
    assert cfg.reward_ckpt == "reward-model"


def _options(**overrides):
    defaults = dict(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        dataset_loader_fn=None,
        reward_fn_path="reward.py",
        save_path="save",
        save_interval=10,
        epochs=2,
        tp_size=1,
        world_size=1,
        batch_size=2,
        n_samples=2,
        mini_bs=1,
        gradient_accumulation_steps=None,
        max_prompt_tokens=128,
        max_new_tokens=16,
        greedy=False,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        max_running_prompts=None,
        lr=1e-6,
        min_lr=1e-7,
        lr_decay_steps=100,
        lr_decay_style="cosine",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_8bit=False,
        weight_decay=1e-2,
        grad_clip_norm=1.0,
        activation_checkpointing=True,
        drop_rollout_state=False,
        eager_decode=False,
        metrics_log_dir=None,
        agent_fn=None,
        agent_timeout_s=300.0,
        train_tool_results=False,
        gspo_clip_eps=3.0e-4,
        grpo_clip_eps=0.2,
        ref_ckpt=None,
        dpo_beta=0.1,
        reward_ckpt=None,
        critic_ckpt=None,
        critic_lr=1e-5,
        use_kl_loss=True,
        kl_loss_coef=0.001,
        kl_loss_type="low_var_kl",
        clip_eps=0.2,
        clip_ratio_c=3.0,
        value_clip_eps=0.5,
        value_loss_coef=0.5,
        gamma=1.0,
        lam=0.95,
        critic_warmup_steps=20,
    )
    defaults.update(overrides)
    return defaults
