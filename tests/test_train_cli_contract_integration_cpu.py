"""CPU tests for ``--validate-data-contract`` integration in ``areno train``.

These tests verify:
- The ``--validate-data-contract`` CLI option appears in help and produces
  the ``validate_data_contract`` field on the resolved config.
- ``_contract_mode_for_config`` maps each algo/config to the correct mode.
- ``_validate_data_contract_or_raise`` raises on invalid data and succeeds on
  valid data (using click echo + ClickException).
- The default for ``validate_data_contract`` is ``False`` (backward compat).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import click
from click.testing import CliRunner

from areno.api.trainer_config import TrainerConfig
from areno.cli import train as train_cli


def _train_args(**overrides) -> SimpleNamespace:
    """Build a minimal SimpleNamespace resembling Click args."""
    base = dict(
        algo="sft",
        ckpt="unused",
        dataset_path="unused",
        model_hub="hf",
        dataset_loader_fn=None,
        reward_fn_path=None,
        ref_ckpt=None,
        reward_ckpt=None,
        critic_ckpt=None,
        save_path=None,
        save_interval=100,
        metrics_log_dir="/tmp/unused",
        epochs=1,
        max_steps=None,
        tune_params=False,
        mem_frac=0.9,
        tune_max_samples=256,
        smoke_infer=False,
        smoke_train=False,
        tp_size=1,
        world_size=1,
        batch_size=1,
        n_samples=1,
        mini_bs=1,
        score_micro_bs=1,
        gradient_accumulation_steps=None,
        max_prompt_tokens=128,
        max_new_tokens=128,
        max_context_len=None,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        greedy=False,
        max_running_prompts=None,
        lr=1e-6,
        min_lr=1e-7,
        lr_decay_steps=1000,
        lr_decay_style="cosine",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_8bit=False,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        activation_checkpointing=True,
        drop_rollout_state=False,
        attn_backend="native",
        eager_decode=False,
        keep_rollout_state=True,
        agent_fn=None,
        agent_timeout_s=300.0,
        train_tool_results=False,
        disable_thinking=False,
        chat_template_enable_thinking=None,
        validate_data_contract=False,
        # DPO specific
        dpo_beta=0.1,
        # RL specific
        gspo_clip_eps=3e-4,
        grpo_clip_eps=0.2,
        # PPO specific
        critic_lr=1e-6,
        critic_warmup_steps=0,
        kl_coef=0.0,
        use_kl_loss=False,
        kl_loss_coef=0.0,
        kl_loss_type="k1",
        clip_eps=0.2,
        clip_ratio_c=3.0,
        value_clip_eps=0.2,
        value_loss_coef=1.0,
        gamma=1.0,
        lam=1.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class ContractModeMappingTest(unittest.TestCase):
    """_contract_mode_for_config should map algo/agent_fn to the right mode."""

    def test_sft_maps_to_sft(self):
        cfg = SimpleNamespace(algo="sft", agent_fn=None)
        self.assertEqual(train_cli._contract_mode_for_config(cfg), "sft")

    def test_dpo_maps_to_dpo(self):
        cfg = SimpleNamespace(algo="dpo", agent_fn=None)
        self.assertEqual(train_cli._contract_mode_for_config(cfg), "dpo")

    def test_gspo_maps_to_online_rl(self):
        cfg = SimpleNamespace(algo="gspo", agent_fn=None)
        self.assertEqual(train_cli._contract_mode_for_config(cfg), "online_rl")

    def test_grpo_maps_to_online_rl(self):
        cfg = SimpleNamespace(algo="grpo", agent_fn=None)
        self.assertEqual(train_cli._contract_mode_for_config(cfg), "online_rl")

    def test_ppo_maps_to_online_rl(self):
        cfg = SimpleNamespace(algo="ppo", agent_fn=None)
        self.assertEqual(train_cli._contract_mode_for_config(cfg), "online_rl")

    def test_agent_fn_overrides_to_agentic(self):
        cfg = SimpleNamespace(algo="gspo", agent_fn="/path/to/agent.py")
        self.assertEqual(train_cli._contract_mode_for_config(cfg), "agentic")


class ValidateDataContractOrRaiseTest(unittest.TestCase):
    """_validate_data_contract_or_raise should succeed on valid and fail on invalid data."""

    def test_valid_dataset_passes(self):
        """A valid SFT dataset should not raise."""
        dataset = [{"prompt": "Hello", "response": "Hi"}]
        cfg = TrainerConfig(
            algo="sft",
            ckpt="unused",
            dataset_path="unused",
            world_size=1,
            tp_size=1,
        )
        # Should not raise
        train_cli._validate_data_contract_or_raise(dataset, cfg)

    def test_invalid_dataset_raises_click_exception(self):
        """An invalid SFT dataset should raise a ClickException."""
        dataset = [{"prompt": "Missing response"}]
        cfg = TrainerConfig(
            algo="sft",
            ckpt="unused",
            dataset_path="unused",
            world_size=1,
            tp_size=1,
        )
        with self.assertRaises(click.ClickException) as exc:
            train_cli._validate_data_contract_or_raise(dataset, cfg)
        self.assertIn("data contract validation failed", str(exc.exception))

    def test_online_rl_mode_for_invalid_prompt(self):
        """Invalid online RL data should raise with the right mode."""
        dataset = [{"prompt": None}]
        cfg = TrainerConfig(
            algo="gspo",
            ckpt="unused",
            dataset_path="unused",
            world_size=1,
            tp_size=1,
        )
        with self.assertRaises(click.ClickException):
            train_cli._validate_data_contract_or_raise(dataset, cfg)

    def test_agentic_mode_for_invalid_messages(self):
        """Invalid agentic messages should raise."""
        dataset = [{"messages": [{"role": "narrator", "content": "bad"}]}]
        cfg = TrainerConfig(
            algo="gspo",
            ckpt="unused",
            dataset_path="unused",
            world_size=1,
            tp_size=1,
            agent_fn="/path/to/agent.py",
        )
        with self.assertRaises(click.ClickException):
            train_cli._validate_data_contract_or_raise(dataset, cfg)


class ValidateDataContractConfigFieldTest(unittest.TestCase):
    """The validate_data_contract field should default to False and propagate."""

    def test_default_is_false(self):
        """TrainerConfig should default validate_data_contract to False."""
        cfg = TrainerConfig(algo="sft", ckpt="unused", dataset_path="unused")
        self.assertFalse(cfg.validate_data_contract)

    def test_can_be_enabled(self):
        """TrainerConfig should accept validate_data_contract=True."""
        cfg = TrainerConfig(
            algo="sft",
            ckpt="unused",
            dataset_path="unused",
            validate_data_contract=True,
        )
        self.assertTrue(cfg.validate_data_contract)

    def test_sft_config_from_args_propagates_flag(self):
        """_trainer_config_from_args should pass validate_data_contract to SFT config."""
        args = _train_args(algo="sft", validate_data_contract=True)
        cfg = train_cli._trainer_config_from_args(args)
        self.assertTrue(cfg.validate_data_contract)

    def test_dpo_config_from_args_propagates_flag(self):
        """_trainer_config_from_args should pass validate_data_contract to DPO config."""
        args = _train_args(algo="dpo", validate_data_contract=True)
        cfg = train_cli._trainer_config_from_args(args)
        self.assertTrue(cfg.validate_data_contract)

    def test_gspo_config_from_args_propagates_flag(self):
        """_trainer_config_from_args should pass validate_data_contract to GSPO config."""
        args = _train_args(algo="gspo", validate_data_contract=True)
        cfg = train_cli._trainer_config_from_args(args)
        self.assertTrue(cfg.validate_data_contract)

    def test_config_from_args_default_false(self):
        """Without --validate-data-contract the config field should be False."""
        args = _train_args(algo="sft")
        cfg = train_cli._trainer_config_from_args(args)
        self.assertFalse(cfg.validate_data_contract)


class TrainCliHelpTest(unittest.TestCase):
    """The train CLI help should list --validate-data-contract."""

    def test_help_lists_option(self):
        result = CliRunner().invoke(train_cli.train_command, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--validate-data-contract", result.output)
        self.assertIn("--no-validate-data-contract", result.output)


if __name__ == "__main__":
    unittest.main()
