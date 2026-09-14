"""Repository metadata and algorithm-specific argument contract tests."""

import ast
import json
import shutil

import pytest

from arenoflow.catalog import ROOT, arguments, catalog, cli_schema
from arenoflow.workflows import plan


@pytest.fixture(scope="module")
def metadata():
    return catalog()


@pytest.fixture
def request_config():
    return {
        "model": {"adapter": "qwen3", "checkpoint": "Qwen/Qwen3-0.6B"},
        "stages": [{"algo": "sft", "params": {"dataset_path": "/artifacts/data.jsonl"}}],
    }


def test_every_cli_option_is_exposed(metadata):
    for kind in ("train", "serve"):
        tree = ast.parse((ROOT / f"areno/cli/{kind}.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == f"{kind}_command")
        count = sum(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "option"
            for n in function.decorator_list
        )
        assert len(metadata[kind]) == count
        assert len({option["name"] for option in metadata[kind]}) == count
        assert all(option["group"] for option in metadata[kind])


@pytest.mark.parametrize("algo", ["sft", "dpo"])
def test_offline_algorithms_have_no_rewards(metadata, algo):
    names = {p["name"] for p in metadata["train"] if algo in p["algorithms"]}
    assert not names & {"reward_fn_path", "reward_ckpt", "n_samples", "temperature", "critic_lr", "kl_loss_coef"}
    assert {"lr", "dataset_path", "max_new_tokens", "save_path", "lora_rank"} <= names


@pytest.mark.parametrize(
    "algo,own,other", [("gspo", "gspo_clip_eps", "grpo_clip_eps"), ("grpo", "grpo_clip_eps", "gspo_clip_eps")]
)
def test_objectives_are_separate(metadata, algo, own, other):
    names = {p["name"] for p in metadata["train"] if algo in p["algorithms"]}
    assert own in names and other not in names
    assert "reward_fn_path" in names
    assert "critic_ckpt" not in names


def test_new_parameters_are_discovered(tmp_path):
    shutil.copytree(ROOT / "areno/cli", tmp_path / "areno/cli")
    file = tmp_path / "areno/cli/train.py"
    file.write_text(
        file.read_text().replace(
            "def train_command(**options)",
            '@click.option("--future-control", type=int, default=7, help="Future control")\ndef train_command(**options)',
        )
    )
    item = next(p for p in cli_schema("train", tmp_path) if p["name"] == "future_control")
    assert item["default"] == 7 and item["type"] == "int"
    assert set(item["algorithms"]) == {"sft", "dpo", "gspo", "grpo", "ppo"}


def test_registered_models_only(metadata):
    ids = {m["id"] for m in metadata["models"]}
    assert {"qwen3", "qwen3_moe", "qwen3_5_vl", "gemma4"} <= ids
    for model in metadata["models"]:
        assert (ROOT / model["source"]).is_file()


def test_pipeline_passes_checkpoint_forward(metadata, request_config):
    request_config["stages"].append({"algo": "gspo", "params": {"dataset_path": "gsm8k:main"}})
    output = plan(request_config, metadata, "test-job")
    stages = output["manifest"]["stages"]
    assert stages[1]["params"]["ckpt"] == "__previous__"
    assert stages[0]["save_path"] != stages[1]["save_path"]
    assert stages[0]["save_path"].startswith("/artifacts/runs/test-job/")
    assert "reward_fn_path" not in stages[0]["params"]
    assert "estimate" not in json.dumps(output)


def test_shell_metacharacters_are_literal_arguments(metadata, request_config):
    dangerous = "/artifacts/$(touch pwned); data.jsonl"
    request_config["stages"][0]["params"]["dataset_path"] = dangerous
    output = plan(request_config, metadata)
    args = output["manifest"]["stages"][0]["args"]
    assert args[args.index("--dataset-path") + 1] == dangerous


@pytest.mark.parametrize(
    "key,value", [("reward_fn_path", "x.py"), ("n_samples", 4), ("critic_lr", 0.01), ("gspo_clip_eps", 0.2)]
)
def test_reject_inapplicable_parameters(metadata, request_config, key, value):
    request_config["stages"][0]["params"][key] = value
    with pytest.raises(ValueError, match="SFT does not use"):
        plan(request_config, metadata)


@pytest.mark.parametrize(
    "key,value", [("unknown", 1), ("lr", float("nan")), ("max_steps", 1.5), ("activation_checkpointing", "yes")]
)
def test_invalid_option_values(metadata, request_config, key, value):
    request_config["stages"][0]["params"][key] = value
    with pytest.raises(ValueError):
        plan(request_config, metadata)


def test_reject_missing_dataset_and_invalid_resources(metadata, request_config):
    request_config["stages"][0]["params"] = {}
    with pytest.raises(ValueError, match="dataset"):
        plan(request_config, metadata)
    request_config["resources"] = {"count": -1}
    with pytest.raises(ValueError, match="GPU count"):
        plan(request_config, metadata)


def test_paired_boolean_false_is_preserved(metadata):
    args = arguments("train", {"sequence_parallel": False, "activation_checkpointing": True}, metadata["train"])
    assert "--no-sequence-parallel" in args and "--activation-checkpointing" in args


def test_deployment_internal_port_cannot_be_exposed(metadata, request_config):
    request_config.update(kind="deployment", serve={"host": "0.0.0.0", "port": 9999})
    args = plan(request_config, metadata)["manifest"]["serve_args"]
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "8000"


def test_persistent_output_path_validation(metadata, request_config):
    request_config["stages"][0]["params"]["save_path"] = "/artifacts/../tmp/lost"
    with pytest.raises(ValueError, match="inside /artifacts"):
        plan(request_config, metadata)


def test_integer_inputs_are_normalized_for_click(metadata):
    assert arguments("train", {"max_steps": "1.0"}, metadata["train"]) == ["--max-steps", "1"]
