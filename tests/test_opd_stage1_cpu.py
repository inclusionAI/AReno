"""CPU-only checks for the Stage 1 OPD registry, config and CLI integration."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from areno.api.algorithms import get_algorithm
from areno.api.trainer_config import OPDTrainerConfig, RolloutTrainerConfig
from areno.api.trainer_factory import build_trainer
from areno.api.trainers.opd import OPDTrainer
from areno.cli import model_refs, train as cli


def test_opd_registered_with_no_reward_required():
    spec = get_algorithm("opd")
    assert spec.requires_rollout is True
    assert spec.requires_reward is False
    assert spec.experimental is True
    assert spec.resolve_trainer_cls() is OPDTrainer


def test_config_inherits_rollout_without_reward_fields():
    cfg = OPDTrainerConfig(algo="opd", ckpt="student", dataset_path="prompts", teacher_ckpt="teacher")
    assert isinstance(cfg, RolloutTrainerConfig)
    assert cfg.opd_coef == 1.0
    assert cfg.score_micro_bs == 8
    assert not hasattr(cfg, "reward_fn_path")
    assert not hasattr(cfg, "reward_ckpt")
    assert cfg.n_samples == 8


@pytest.mark.parametrize("teacher", [None, "", "  "])
def test_missing_teacher_is_rejected(teacher):
    with pytest.raises(ValueError, match="--teacher-ckpt is required"):
        OPDTrainerConfig(algo="opd", ckpt="student", dataset_path="prompts", teacher_ckpt=teacher)


@pytest.mark.parametrize("coef", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_loss_coef_is_rejected(coef):
    with pytest.raises(ValueError, match="--opd-coef must be a finite positive"):
        OPDTrainerConfig(algo="opd", ckpt="student", dataset_path="prompts",
                         teacher_ckpt="teacher", opd_coef=coef)


def test_cli_parses_opd_without_reward(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "run", lambda cfg: seen.append(cfg))
    monkeypatch.setattr(cli, "_model_config_for_summary", lambda cfg: None)
    monkeypatch.setattr("areno.cli.dashboard_registry.register_dashboard_job", lambda **kwargs: None)
    result = CliRunner().invoke(cli.train_command, [
        "--algo", "opd", "--ckpt", "student", "--dataset-path", "prompts",
        "--teacher-ckpt", "teacher", "--opd-coef", "0.7", "--score-micro-bs", "2",
        "--n-samples", "1", "--tp-size", "1", "--world-size", "1",
    ])
    assert result.exit_code == 0, result.output
    assert len(seen) == 1
    cfg = seen[0]
    assert isinstance(cfg, OPDTrainerConfig)
    assert (cfg.teacher_ckpt, cfg.opd_coef, cfg.score_micro_bs, cfg.n_samples) == ("teacher", 0.7, 2, 1)


def test_cli_rejects_missing_teacher():
    result = CliRunner().invoke(cli.train_command, [
        "--algo", "opd", "--ckpt", "student", "--dataset-path", "prompts",
        "--tp-size", "1", "--world-size", "1",
    ])
    assert result.exit_code != 0
    assert "--teacher-ckpt is required" in result.output


def test_existing_reward_requirement_unchanged():
    result = CliRunner().invoke(cli.train_command, [
        "--algo", "gspo", "--ckpt", "student", "--dataset-path", "prompts",
        "--tp-size", "1", "--world-size", "1",
    ])
    assert result.exit_code != 0
    assert "--reward-fn-path or --reward-ckpt is required" in result.output


def test_cli_help_includes_opd_flags():
    result = CliRunner().invoke(cli.train_command, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--teacher-ckpt" in result.output
    assert "--opd-coef" in result.output


def test_teacher_model_ref_resolved(monkeypatch):
    seen = []
    def download(ref, *, model_hub):
        seen.append((ref, model_hub))
        return "/cache/" + ref
    monkeypatch.setattr(model_refs, "_snapshot_download", download)
    cfg = OPDTrainerConfig(algo="opd", ckpt="student", dataset_path="prompts",
                           teacher_ckpt="teacher", model_hub="hf")
    result = model_refs.resolve_model_refs_for_config(cfg)
    assert (result.ckpt, result.teacher_ckpt) == ("/cache/student", "/cache/teacher")
    assert seen == [("student", "hf"), ("teacher", "hf")]


def test_trainer_factory_resolves_opd_and_loss_is_bound():
    cfg = OPDTrainerConfig(algo="opd", ckpt="student", dataset_path="prompts", teacher_ckpt="teacher")
    loss_fn = get_algorithm("opd").make_loss_fn(cfg)
    assert loss_fn.keywords == {"opd_coef": 1.0}
    trainer = build_trainer(cfg, instance=object(), dataset=[], reward_fn=None, loss_fn=loss_fn)
    assert isinstance(trainer, OPDTrainer)
    # Stage 3 wires fit() into the existing loop. The Stage 4 loss is still
    # a deliberate stub, so model training is not part of this Stage 1 test.
