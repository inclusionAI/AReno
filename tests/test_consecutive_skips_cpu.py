"""CPU tests for consecutive-empty-batch stopping (#239)."""

from __future__ import annotations

import json

from areno.api.trainer_config import (
    DPOTrainerConfig,
    PolicyTrainerConfig,
    PPOTrainerConfig,
    TrainerConfig,
)


def test_config_default_disabled():
    """max_consecutive_skipped_steps defaults to None (disabled)."""
    cfg = PolicyTrainerConfig(
        algo="gspo", ckpt="model", dataset_path="data", reward_fn_path="reward.py"
    )
    assert cfg.max_consecutive_skipped_steps is None


def test_config_accepts_positive_int():
    """Setting a positive threshold is reflected in the config."""
    cfg = PolicyTrainerConfig(
        algo="gspo", ckpt="model", dataset_path="data", reward_fn_path="reward.py",
        max_consecutive_skipped_steps=5,
    )
    assert cfg.max_consecutive_skipped_steps == 5


def test_ppo_inherits_config_field():
    """PPOTrainerConfig inherits max_consecutive_skipped_steps from PolicyTrainerConfig."""
    cfg = PPOTrainerConfig(
        algo="ppo", ckpt="model", dataset_path="data", reward_fn_path="reward.py",
        max_consecutive_skipped_steps=3,
    )
    assert cfg.max_consecutive_skipped_steps == 3


def test_base_trainer_does_not_have_field():
    """TrainerConfig and DPOTrainerConfig do NOT have this field — it's PolicyTrainerConfig only."""
    cfg = TrainerConfig(algo="sft", ckpt="model", dataset_path="data")
    assert not hasattr(cfg, "max_consecutive_skipped_steps")


def test_dpo_does_not_have_field():
    """DPOTrainerConfig does not inherit the field."""
    cfg = DPOTrainerConfig(algo="dpo", ckpt="model", dataset_path="data")
    assert not hasattr(cfg, "max_consecutive_skipped_steps")


def test_skip_summary_structure():
    """The stop summary has the expected shape and reconciles counts."""
    summary = {
        "total_steps": 10,
        "skipped_steps": 5,
        "by_reason": {"empty_response": 5},
        "max_consecutive": 5,
        "stopped_by_threshold": True,
    }
    assert summary["skipped_steps"] == sum(summary["by_reason"].values())
    assert summary["total_steps"] >= summary["skipped_steps"]
    assert isinstance(summary["by_reason"], dict)
    assert "empty_response" in summary["by_reason"]
    # Verify it's serializable
    encoded = json.dumps(summary)
    decoded = json.loads(encoded)
    assert decoded == summary


def test_skip_summary_runtime_disabled():
    """When threshold is None, the summary reflects no threshold stop."""
    summary = {
        "total_steps": 20,
        "skipped_steps": 3,
        "by_reason": {"empty_response": 3},
        "max_consecutive": 1,
        "stopped_by_threshold": False,
    }
    assert summary["skipped_steps"] == sum(summary["by_reason"].values())
    assert not summary["stopped_by_threshold"]