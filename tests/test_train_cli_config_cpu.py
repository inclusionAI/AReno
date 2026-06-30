from __future__ import annotations

from dataclasses import replace

import pytest
from click import UsageError, unstyle
from click.testing import CliRunner

from areno.api.trainer_config import DPOTrainerConfig, PolicyTrainerConfig, PPOTrainerConfig, TrainerConfig
from areno.cli import train as train_cli
from areno.cli.train import (
    TRAIN_OPTION_GROUPS,
    _callable_name,
    _format_summary_section,
    _format_training_config_summary,
    _trainer_config_from_options,
)


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
        ("max_context_len", 0, "--max-context-len must be positive"),
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


def test_train_config_validates_tune_params_for_rollout_algorithms():
    with pytest.raises(UsageError, match="--tune-params currently supports rollout-based algorithms"):
        _trainer_config_from_options(**_options(algo="sft", reward_fn_path=None, reward_ckpt=None, tune_params=True))


def test_train_config_validates_mem_frac():
    with pytest.raises(UsageError, match=r"--mem-frac must be in \(0, 1\]"):
        _trainer_config_from_options(**_options(algo="gspo", mem_frac=1.2))


def test_train_config_validates_tune_max_samples():
    with pytest.raises(UsageError, match="--tune-max-samples must be positive"):
        _trainer_config_from_options(**_options(algo="gspo", tune_max_samples=0))


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
    cfg = _trainer_config_from_options(
        **_options(algo="sft", reward_fn_path=None, reward_ckpt=None, min_lr=0.0, attn_backend="native")
    )

    assert type(cfg) is TrainerConfig
    assert cfg.algo == "sft"
    assert cfg.ckpt == "actor"
    assert cfg.dataset_path == "dataset"
    assert cfg.optimizer_min_lr == 0.0
    assert cfg.attn_backend == "native"
    assert cfg.areno_config().runtime["attn_backend"] == "native"
    assert cfg.batch_size == 2
    assert cfg.mini_bs == 1
    assert not hasattr(cfg, "n_samples")
    assert not hasattr(cfg, "reward_fn_path")
    assert not hasattr(cfg, "ref_ckpt")


def test_train_config_disable_thinking_sets_chat_template_option():
    default_cfg = _trainer_config_from_options(**_options(algo="sft", reward_fn_path=None, reward_ckpt=None))
    disabled_cfg = _trainer_config_from_options(
        **_options(algo="sft", reward_fn_path=None, reward_ckpt=None, disable_thinking=True)
    )

    assert default_cfg.chat_template_enable_thinking is None
    assert disabled_cfg.chat_template_enable_thinking is False


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
            n_samples=3,
            max_running_prompts=12,
            max_context_len=512,
            temperature=0.7,
            top_k=10,
            top_p=0.9,
            **{clip_attr: clip_value},
        )
    )

    assert isinstance(cfg, PolicyTrainerConfig)
    assert type(cfg) is PolicyTrainerConfig
    assert cfg.algo == algo
    assert cfg.reward_fn_path is None
    assert cfg.n_samples == 3
    assert cfg.resolved_max_running_prompts() == 12
    assert cfg.max_context_len == 512
    assert cfg.temperature == 0.7
    assert cfg.top_k == 10
    assert cfg.top_p == 0.9
    assert getattr(cfg, clip_attr) == clip_value
    assert not hasattr(cfg, "ref_ckpt")


def test_train_config_builds_ppo_shape_and_role_checkpoints():
    cfg = _trainer_config_from_options(
        **_options(
            algo="ppo",
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


def test_training_config_summary_shows_resolved_values_and_warning():
    cfg = _trainer_config_from_options(
        **_options(
            algo="gspo",
            reward_fn_path=None,
            reward_ckpt="reward-model",
            save_path=None,
            world_size=8,
            tp_size=2,
            batch_size=4,
            n_samples=3,
            max_running_prompts=None,
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            lr=2e-6,
            min_lr=0.0,
            metrics_log_dir="/tmp/metrics",
        )
    )

    summary = _format_training_config_summary(cfg, reward_ckpt="reward-model")

    assert "AReno training config" in summary
    assert "Algorithm\n---------" in summary
    assert "name              gspo" in summary
    assert "default_loss      gspo_loss_fn" in summary
    assert "requires_rollout  yes" in summary
    assert "ckpt            actor" in summary
    assert "dataset_path    dataset" in summary
    assert "reward_fn       none" in summary
    assert "reward_ckpt     reward-model" in summary
    assert "dp_size       4" in summary
    assert "attn_backend  flash" in summary
    assert "max_running_prompts  12" in summary
    assert "sampling             greedy=no, temperature=0.7, top_k=20, top_p=0.9" in summary
    assert "optimizer                    lr=2e-06, min_lr=0.0, decay=cosine/100" in summary
    assert "metrics_log_dir  /tmp/metrics" in summary
    assert "WARNING: no checkpoint output path configured (--save-path)" in summary


def test_training_config_summary_warns_about_native_attention_fallback(monkeypatch):
    cfg = _trainer_config_from_options(**_options(algo="sft", reward_fn_path=None, reward_ckpt=None, world_size=1))
    monkeypatch.setattr(
        train_cli,
        "flash_attention_unsupported_gpu_reason",
        lambda devices: "Tesla T4 cc 7.5",
    )

    summary = _format_training_config_summary(cfg)

    assert summary.startswith("AReno training config\nWARNING: flash-attn does not support")
    assert "AReno will use attn_backend='native'" in summary
    assert "attn_backend  native (auto fallback from flash: Tesla T4 cc 7.5)" in summary


def test_training_config_summary_can_colorize_output():
    cfg = _trainer_config_from_options(**_options(algo="sft", reward_fn_path=None, reward_ckpt=None))

    summary = _format_training_config_summary(cfg, color=True)

    assert "\x1b[" in summary
    assert "AReno training config" in summary


def test_training_config_summary_wraps_for_narrow_terminals(monkeypatch):
    monkeypatch.setattr(
        train_cli.shutil, "get_terminal_size", lambda fallback: train_cli.shutil.os.terminal_size((48, 24))
    )
    cfg = _trainer_config_from_options(
        **_options(
            algo="sft",
            reward_fn_path=None,
            reward_ckpt=None,
            metrics_log_dir="/tmp/areno/a/very/long/path/that/should/wrap",
        )
    )

    summary = _format_training_config_summary(cfg)

    lines = summary.splitlines()
    row_idx = next(idx for idx, line in enumerate(lines) if line.startswith("  metrics_log_dir"))
    assert lines[row_idx].startswith("  metrics_log_dir  ")
    assert lines[row_idx + 1].startswith(" " * 19)


def test_training_config_summary_marks_non_rollout_fields_not_applicable():
    cfg = _trainer_config_from_options(**_options(algo="sft", reward_fn_path=None, reward_ckpt=None))

    summary = _format_training_config_summary(cfg)

    assert "name              sft" in summary
    assert "requires_rollout  no" in summary
    assert "reward_fn       n/a" in summary
    assert "n_samples            n/a" in summary
    assert "max_running_prompts  n/a" in summary
    assert "sampling             n/a" in summary


def test_training_config_summary_handles_invalid_tp_size_defensively():
    cfg = TrainerConfig(algo="sft", ckpt="actor", dataset_path="dataset", tp_size=0, world_size=8)

    summary = _format_training_config_summary(cfg)

    assert "tp_size       0" in summary
    assert "dp_size       n/a" in summary


def test_training_config_summary_section_handles_empty_rows():
    assert _format_summary_section("Empty", [], color=False) == ["", "Empty", "-----"]


def test_training_config_summary_callable_name_handles_callable_objects():
    class CallableLoss:
        def __call__(self):
            return None

    assert _callable_name(CallableLoss()) == "CallableLoss"


def test_train_command_prints_summary_before_run(monkeypatch):
    events = []

    def fake_run(config):
        events.append(("run", config.algo))

    monkeypatch.setattr(train_cli, "run", fake_run)

    result = CliRunner().invoke(
        train_cli.train_command,
        [
            "--algo",
            "sft",
            "--ckpt",
            "actor",
            "--dataset-path",
            "dataset",
            "--world-size",
            "2",
            "--tp-size",
            "1",
            "--save-path",
            "out",
        ],
    )

    assert result.exit_code == 0, result.output
    output = unstyle(result.output)
    assert output.startswith("AReno training config\n")
    assert "dp_size       2" in output
    assert "save_path        out" in output
    assert "WARNING: no checkpoint output path configured" not in output
    assert events == [("run", "sft")]


def test_train_command_tunes_params_before_summary_and_run(monkeypatch):
    from areno.cli.auto_tune import AutoTuneCandidate, AutoTuneMeasurement, AutoTuneResult

    events = []

    def fake_auto_tune(config, *, mem_frac, auto_max_samples):
        events.append(("tune", config.batch_size, config.n_samples, config.mini_bs, mem_frac, auto_max_samples))
        tuned = replace(
            config,
            tp_size=2,
            batch_size=1,
            n_samples=4,
            mini_bs=2,
            max_running_prompts=4,
            adam_8bit=True,
            keep_rollout_state=False,
        )
        return AutoTuneResult(
            config=tuned,
            measurement=AutoTuneMeasurement(
                candidate=AutoTuneCandidate(
                    tp_size=2,
                    batch_size=1,
                    n_samples=4,
                    mini_bs=2,
                    max_running_prompts=4,
                    adam_8bit=True,
                    keep_rollout_state=False,
                ),
                peak_mem_frac=0.81,
                ok=True,
            ),
            measurements=(),
        )

    def fake_run(config):
        events.append(
            ("run", config.batch_size, config.n_samples, config.mini_bs, config.resolved_max_running_prompts())
        )

    monkeypatch.setattr("areno.cli.auto_tune.auto_tune_config", fake_auto_tune)
    monkeypatch.setattr(train_cli, "run", fake_run)

    result = CliRunner().invoke(
        train_cli.train_command,
        [
            "--algo",
            "gspo",
            "--ckpt",
            "actor",
            "--dataset-path",
            "dataset",
            "--reward-ckpt",
            "reward-model",
            "--world-size",
            "2",
            "--tp-size",
            "1",
            "--tune-params",
            "--mem-frac",
            "0.82",
            "--tune-max-samples",
            "64",
        ],
    )

    assert result.exit_code == 0, result.output
    output = unstyle(result.output)
    assert "AReno parameter tune selected: tp_size=2, batch_size=1, n_samples=4, mini_bs=2" in output
    assert "adam_8bit=True, drop_rollout_state=True" in output
    assert "tp_size       2" in output
    assert "batch_size           1" in output
    assert "n_samples            4" in output
    assert "max_running_prompts  4" in output
    assert events == [("tune", 32, 8, 16, 0.82, 64), ("run", 1, 4, 2, 4)]


EXPECTED_HELP_SECTIONS = [
    "Basic:",
    "Rollout:",
    "Train:",
    "Checkpoint:",
    "Observability:",
]


def _help_output() -> str:
    result = CliRunner().invoke(train_cli.train_command, ["--help"])
    assert result.exit_code == 0, result.output
    return unstyle(result.output)


def test_train_help_groups_sections_in_intent_order():
    output = _help_output()

    positions = []
    for section in EXPECTED_HELP_SECTIONS:
        assert section in output, f"missing help section: {section}"
        positions.append(output.index(section))
    assert positions == sorted(positions), "help sections out of intent order"


def test_train_help_places_epochs_under_basic_not_checkpointing():
    output = _help_output()

    basic = output.index("Basic:")
    next_section = output.index("Rollout:", basic)
    epochs = output.index("--epochs", basic)
    checkpoint = output.index("Checkpoint:")

    # --epochs is a run-setup flag that belongs in Basic, not with the
    # save flags in the Checkpoint group.
    assert basic < epochs < next_section
    assert "--epochs" not in output[checkpoint:]


def test_train_help_remains_complete_and_groups_every_declared_option():
    ctx = train_cli.click.Context(train_cli.train_command)
    declared = {
        param.name for param in train_cli.train_command.get_params(ctx) if param.get_help_record(ctx) is not None
    }
    grouped = [name for _, names in TRAIN_OPTION_GROUPS for name in names]

    assert len(grouped) == len(set(grouped)), "an option is listed in more than one group"
    # Every declared option is grouped except the auto-added --help, which the
    # renderer keeps under a trailing catch-all so help output stays complete.
    assert set(grouped) == declared - {"help"}

    output = _help_output()
    for param in train_cli.train_command.get_params(ctx):
        record = param.get_help_record(ctx)
        if record is not None:
            assert record[0].split()[0].rstrip(",") in output, f"option dropped from help: {param.name}"


def _options(**overrides):
    defaults = dict(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        dataset_loader_fn=None,
        reward_fn_path=None,
        save_path="save",
        save_interval=10,
        tune_params=False,
        mem_frac=0.9,
        tune_max_samples=256,
        epochs=2,
        tp_size=1,
        world_size=1,
        batch_size=2,
        n_samples=2,
        mini_bs=1,
        gradient_accumulation_steps=None,
        max_prompt_tokens=128,
        max_new_tokens=16,
        max_context_len=None,
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
        attn_backend="flash",
        disable_thinking=False,
        metrics_log_dir=None,
        agent_fn=None,
        agent_timeout_s=300.0,
        train_tool_results=False,
        gspo_clip_eps=3.0e-4,
        grpo_clip_eps=0.2,
        ref_ckpt=None,
        dpo_beta=0.1,
        reward_ckpt="reward-model",
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
