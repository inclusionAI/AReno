"""Named hooks, immutable training snapshots and multimodal attachment contracts."""

import base64
import json
from pathlib import Path

import pytest

from arenoflow.assets import referenced_uploads, save_upload
from arenoflow.catalog import catalog
from arenoflow.datasets import materialize_dataset, resolve_request, save_dataset, save_function
from arenoflow.store import Store
from arenoflow.workflows import plan

LOADER = (
    "def load_training_dataset(dataset_path, *, default_loader, **kwargs):\n    return default_loader(dataset_path)\n"
)
REWARD = "def reward_fn(record):\n    return 1.0\n"
AGENT = "async def run_agent(ctx, batch):\n    return []\n"


def upload(store, name, content):
    return save_upload(store.directory, name, base64.b64encode(content).decode())


def test_function_lifecycle_and_references_survive_restart(tmp_path):
    store = Store(tmp_path)
    function = save_function(store, {"name": "Normalizer", "kind": "dataset_loader", "source": LOADER})
    dataset = save_dataset(store, {"name": "Training data", "source": "org/data", "loader_id": function["id"]})
    assert dataset["model_hub"] == "hf"
    assert Store(tmp_path).datasets()[0] == dataset
    assert "path" not in function and "loader_source" not in dataset
    with pytest.raises(ValueError, match="assigned"):
        store.delete_function(function["id"])
    store.delete_dataset(dataset["id"])
    store.delete_function(function["id"])
    assert not store.functions() and not store.datasets()


@pytest.mark.parametrize("kind,source", [("dataset_loader", LOADER), ("reward", REWARD), ("agentic", AGENT)])
def test_code_validation_never_executes_imports(tmp_path, kind, source):
    store = Store(tmp_path)
    record = save_function(
        store, {"name": kind, "kind": kind, "source": 'raise RuntimeError("must not execute locally")\n' + source}
    )
    assert record["source"].startswith("raise")


@pytest.mark.parametrize(
    "kind,source",
    [
        ("reward", "def other(record): return 1"),
        ("reward", "def reward_fn(a,b): return 1"),
        ("dataset_loader", "def load_training_dataset(path): return []"),
        ("dataset_loader", "async def load_training_dataset(path, **kwargs): return []"),
        ("agentic", "def run_agent(ctx): return []"),
        ("reward", "def reward_fn(:"),
    ],
)
def test_invalid_hook_signature_rejected(tmp_path, kind, source):
    with pytest.raises(ValueError):
        save_function(Store(tmp_path), {"name": "Invalid", "kind": kind, "source": source})


def test_stage_hook_types_and_code_snapshot(tmp_path):
    store = Store(tmp_path)
    loader = save_function(store, {"name": "Loader", "kind": "dataset_loader", "source": LOADER})
    reward = save_function(store, {"name": "Reward", "kind": "reward", "source": REWARD})
    agent = save_function(store, {"name": "Agent", "kind": "agentic", "source": AGENT})
    dataset = save_dataset(store, {"name": "Data", "source": "org/data", "loader_id": loader["id"]})
    request = {
        "model": {"adapter": "qwen3", "checkpoint": "Qwen/Qwen3-0.6B"},
        "stages": [
            {"algo": "sft", "dataset_id": dataset["id"]},
            {
                "algo": "gspo",
                "dataset_id": dataset["id"],
                "reward_function_id": reward["id"],
                "agentic_function_id": agent["id"],
            },
        ],
    }
    resolved = resolve_request(request, store)
    prepared = plan(resolved, catalog())
    files = referenced_uploads(prepared["manifest"], store.directory)
    assert len(files) == 3
    assert "reward_fn_path" not in prepared["manifest"]["stages"][0]["params"]
    assert "--agent-fn" in prepared["commands"][1]
    old = resolved["stages"][1]["params"]["reward_fn_path"]
    save_function(store, {**reward, "source": REWARD.replace("1.0", "2.0")})
    assert resolve_request(request, store)["stages"][1]["params"]["reward_fn_path"] != old
    assert (store.directory / "uploads" / Path(old).name).read_text() == REWARD
    assert "params" not in request["stages"][0]
    request["stages"][0]["reward_function_id"] = reward["id"]
    with pytest.raises(ValueError, match="SFT does not use"):
        resolve_request(request, store)
    with pytest.raises(ValueError, match="Dataset Loader"):
        save_dataset(store, {"name": "Bad", "source": "org/data", "loader_id": reward["id"]})


def test_multimodal_manifest_resolves_image_audio_video_and_stages_all_assets(tmp_path):
    store = Store(tmp_path)
    media = [upload(store, name, b"test media " + name.encode()) for name in ("scene.png", "speech.wav", "clip.mp4")]
    manifest = upload(
        store,
        "data.jsonl",
        json.dumps(
            {
                "prompt": "Describe",
                "images": ["scene.png"],
                "audio": {"path": "speech.wav"},
                "messages": [{"content": [{"type": "video", "video": "clip.mp4"}]}],
            }
        ).encode(),
    )
    dataset = save_dataset(
        store,
        {
            "name": "Mixed media",
            "source_type": "upload",
            "source": manifest["path"],
            "source_name": "data.jsonl",
            "media": media,
            "modalities": ["image", "audio", "video"],
        },
    )
    resolved = resolve_request(
        {
            "stages": [{"algo": "sft", "dataset_id": dataset["id"]}],
            "model": {"adapter": "qwen3_5_vl", "checkpoint": "Qwen/Qwen3.5-0.8B"},
        },
        store,
    )
    prepared = plan(resolved, catalog())
    files = referenced_uploads(prepared["manifest"], store.directory)
    assert len(files) == 4
    normalized = json.loads(
        (store.directory / "uploads" / Path(resolved["stages"][0]["params"]["dataset_path"]).name).read_text()
    )
    assert normalized["images"][0] == media[0]["path"]
    assert normalized["audio"]["path"] == media[1]["path"]
    assert normalized["messages"][0]["content"][0]["video"] == media[2]["path"]


def test_csv_media_manifest(tmp_path):
    store = Store(tmp_path)
    media = upload(store, "image.png", b"image")
    manifest = upload(store, "data.csv", b"prompt,image\ncaption,image.png\n")
    dataset = save_dataset(
        store, {"name": "CSV", "source_type": "upload", "source": manifest["path"], "media": [media]}
    )
    normalized = materialize_dataset(dataset, store)
    assert media["path"] in (store.directory / "uploads" / Path(normalized).name).read_text()


def test_model_catalog_uses_huggingface_and_omits_bailing_linear():
    metadata = catalog()
    assert "bailing_moe_linear_v2" not in {m["id"] for m in metadata["models"]}
    assert all(m["checkpoint"] and "/" in m["checkpoint"] for m in metadata["models"])
    assert all(preset["model_hub"] == "hf" for preset in metadata["presets"].values())


def test_loader_is_selected_per_training_stage(tmp_path):
    store = Store(tmp_path)
    one = save_function(store, {"name": "Loader one", "kind": "dataset_loader", "source": LOADER})
    two = save_function(store, {"name": "Loader two", "kind": "dataset_loader", "source": LOADER + "\n# alternative\n"})
    dataset = save_dataset(store, {"name": "Shared", "source": "org/data"})
    resolved = resolve_request(
        {
            "stages": [
                {"algo": "sft", "dataset_id": dataset["id"], "dataset_loader_id": one["id"]},
                {"algo": "grpo", "dataset_id": dataset["id"], "dataset_loader_id": two["id"]},
                {"algo": "dpo", "dataset_id": dataset["id"], "dataset_loader_id": ""},
            ]
        },
        store,
    )
    loaders = [s["params"]["dataset_loader_fn"] for s in resolved["stages"]]
    assert loaders[0] != loaders[1]
    assert loaders[0].endswith(":load_training_dataset")
    assert loaders[2] is None
    assert store.datasets()[0]["loader_id"] is None


def test_stage_rejects_reward_script_as_loader(tmp_path):
    store = Store(tmp_path)
    reward = save_function(store, {"name": "Reward", "kind": "reward", "source": REWARD})
    with pytest.raises(ValueError, match="dataset_loader"):
        resolve_request({"stages": [{"algo": "sft", "dataset_loader_id": reward["id"]}]}, store)
