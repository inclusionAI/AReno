import shlex

from areno.dashboard.launcher import preview_launcher
from areno.dashboard.server import build_serve_command, build_train_command


def _train_config(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt": "2 + 2"}\n', encoding="utf-8")
    return {
        "algo": "sft",
        "ckpt": str(model),
        "dataset_path": str(dataset),
        "dataset_loader_fn": "examples/sft/alpaca/dataset_loader.py",
        "world_size": 2,
        "tp_size": 1,
        "epochs": 1,
        "batch_size": 2,
        "mini_bs": 1,
        "score_micro_bs": 1,
        "max_prompt_tokens": 32,
        "max_new_tokens": 16,
        "save_path": str(tmp_path / "output path"),
        "extra_args": "--metrics-log-dir 'metrics with spaces'",
    }


def test_train_preview_is_shell_safe_and_matches_submitted_arguments(tmp_path):
    config = _train_config(tmp_path)

    result = preview_launcher("train", config, build_train_command)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["warnings"] == []
    assert result["command"] == build_train_command(result["resolved_args"])
    assert shlex.split(result["shell_command"]) == result["command"]
    assert "'metrics with spaces'" in result["shell_command"]


def test_preview_reports_field_errors_without_building_or_modifying_inputs(tmp_path):
    config = _train_config(tmp_path)
    config.update(
        {
            "ckpt": str(tmp_path / "missing"),
            "world_size": 3,
            "tp_size": 2,
            "batch_size": 0,
        }
    )
    original = dict(config)

    result = preview_launcher("train", config, build_train_command)

    assert result["ok"] is False
    assert result["shell_command"] == ""
    assert config == original
    assert {item["field"] for item in result["errors"]} == {
        "ckpt",
        "tp_size",
        "batch_size",
    }


def test_explicit_gpu_count_is_validated_and_rendered(tmp_path):
    config = _train_config(tmp_path)
    config["train_devices"] = [0, 1]

    valid = preview_launcher("train", config, build_train_command)
    config["world_size"] = 1
    invalid = preview_launcher("train", config, build_train_command)

    assert "--train-devices" in valid["command"]
    assert valid["command"][valid["command"].index("--train-devices") + 1] == "0,1"
    assert {item["field"] for item in invalid["errors"]} == {"world_size"}


def test_remote_reference_warnings_require_explicit_acknowledgement():
    config = {
        "model_path": "Qwen/Qwen3-0.6B",
        "model_hub": "modelscope",
        "host": "127.0.0.1",
        "port": 8000,
        "world_size": 1,
        "tp_size": 1,
        "max_running_prompts": 1,
        "default_max_tokens": 1,
    }

    blocked = preview_launcher("serve", config, build_serve_command)
    accepted = preview_launcher(
        "serve", config, build_serve_command, acknowledge_warnings=True
    )

    assert blocked["errors"] == []
    assert blocked["requires_acknowledgement"] is True
    assert blocked["ok"] is False
    assert accepted["ok"] is True
    assert accepted["command"] == blocked["command"]
    assert accepted["resolved_args"] == blocked["resolved_args"]


def test_serve_preview_validates_port_boundary_and_malformed_extra_args(tmp_path):
    config = {
        "model_path": str(tmp_path),
        "host": "127.0.0.1",
        "port": 65536,
        "world_size": 1,
        "tp_size": 1,
        "max_running_prompts": 1,
        "default_max_tokens": 1,
        "extra_args": "--served-model-name 'unterminated",
    }

    result = preview_launcher("serve", config, build_serve_command)

    assert {item["field"] for item in result["errors"]} == {"port", "extra_args"}
    assert result["command"] == []
