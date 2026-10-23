from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from areno.api.trainer_config import TrainerConfig
from areno.cli import train as train_cli


@pytest.fixture
def trainer_classes():
    pytest.importorskip("torch")
    from areno.api import trainers

    return trainers


@pytest.mark.parametrize("algo", ["sft", "dpo", "gspo", "grpo", "ppo"])
def test_cli_passes_export_options_to_all_algorithms(monkeypatch, tmp_path, algo):
    configs = []
    monkeypatch.setattr(train_cli, "run", configs.append)
    monkeypatch.setattr(train_cli, "_model_config_for_summary", lambda config: None)
    from areno.cli import dashboard_registry

    monkeypatch.setattr(dashboard_registry, "register_dashboard_job", lambda **kwargs: None)
    args = [
        "--algo",
        algo,
        "--backend",
        "cuda",
        "--ckpt",
        "actor",
        "--dataset-path",
        "dataset",
        "--world-size",
        "1",
        "--tp-size",
        "1",
        "--max-steps",
        "1",
        "--attn-backend",
        "native",
        "--export-gguf",
        str(tmp_path / "export"),
        "--gguf-file-type",
        "q8_0",
    ]
    if algo == "sft":
        args += ["--dataset-loader-fn", "examples/sft/alpaca/dataset_loader.py"]
    elif algo in {"gspo", "grpo", "ppo"}:
        reward = tmp_path / "reward.py"
        reward.write_text("def reward_fn(record):\n    return 1.0\n")
        args += ["--reward-fn-path", str(reward)]
    result = CliRunner().invoke(train_cli.train_command, args)
    assert result.exit_code == 0, result.output
    assert len(configs) == 1
    assert configs[0].export_gguf_path == str(tmp_path / "export")
    assert configs[0].gguf_file_type == "q8_0"
    assert configs[0].max_steps == 1


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--backend", "mlx"], "requires --backend cuda"),
        (["--qlora"], "does not support packed QLoRA"),
        (["--smoke-train"], "cannot be combined"),
        (["--gguf-file-type", "q4_0"], "Invalid value"),
    ],
)
def test_cli_rejects_unsupported_export_options(tmp_path, extra, message):
    result = CliRunner().invoke(
        train_cli.train_command,
        [
            "--algo",
            "gspo",
            "--backend",
            "cuda",
            "--ckpt",
            "actor",
            "--dataset-path",
            "dataset",
            "--reward-fn-path",
            "reward.py",
            "--export-gguf",
            str(tmp_path / "export"),
            *extra,
        ],
    )
    assert result.exit_code == 2
    assert message in result.output


def test_cli_rejects_existing_output_before_training(tmp_path):
    result = CliRunner().invoke(
        train_cli.train_command,
        [
            "--algo",
            "gspo",
            "--backend",
            "cuda",
            "--ckpt",
            "actor",
            "--dataset-path",
            "dataset",
            "--reward-fn-path",
            "reward.py",
            "--export-gguf",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "new output directory" in result.output


@pytest.mark.parametrize("trainer_name", ["SFTTrainer", "DPOTrainer", "PolicyOnlyTrainer", "PPOTrainer"])
@pytest.mark.parametrize("mode", ["enabled", "disabled", "train_failure", "export_failure"])
def test_export_lifecycle_and_error_cleanup(trainer_classes, trainer_name, mode):
    events = []

    def train():
        events.append("train")
        if mode == "train_failure":
            raise RuntimeError("training failed")

    def export(path, *, file_type):
        events.append("export")
        assert path == "export" and file_type == "f16"
        if mode == "export_failure":
            raise RuntimeError("export failed")
        return "export/model.gguf"

    cls = getattr(trainer_classes, trainer_name)
    trainer = cls.__new__(cls)
    trainer.config = SimpleNamespace(export_gguf_path=None if mode == "disabled" else "export", gguf_file_type="f16")
    trainer.logger = logging.getLogger(__name__)
    trainer.areno = SimpleNamespace(
        init=lambda: events.append("init"), close=lambda: events.append("close"), export_gguf=export
    )
    trainer._fit_initialized = train
    trainer._ensure_roles = lambda: None
    if mode in {"train_failure", "export_failure"}:
        with pytest.raises(RuntimeError):
            trainer.fit()
    else:
        trainer.fit()
    expected = ["init", "train"]
    if mode in {"enabled", "export_failure"}:
        expected.append("export")
    assert events == expected + ["close"]


def test_sft_max_steps_exports_after_one_update(trainer_classes):
    events = []
    backend = SimpleNamespace(
        init=lambda: events.append("init"),
        close=lambda: events.append("close"),
        get_tokenizer=lambda: None,
        get_processor=lambda: None,
        train=lambda *args, **kwargs: events.append("train") or {"loss": 1.0},
        export_gguf=lambda *args, **kwargs: events.append("export") or "export/model.gguf",
    )
    config = TrainerConfig(
        algo="sft",
        ckpt="actor",
        dataset_path="dataset",
        backend="cuda",
        max_steps=1,
        epochs=2,
        export_gguf_path="export",
    )
    trainer = trainer_classes.SFTTrainer(config, instance=backend, dataset=[], reward_fn=None, loss_fn=lambda: None)
    trainer._iter_train_batches = lambda *args, **kwargs: iter([[object()], [object()]])
    trainer.fit()
    assert events == ["init", "train", "export", "close"]
