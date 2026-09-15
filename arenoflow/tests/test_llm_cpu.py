"""Script generation validates real entrypoints without importing generated code."""

import base64
import io
import json

import pytest

from arenoflow.assets import save_upload
from arenoflow.datasets import save_dataset, save_function
from arenoflow.llm import ScriptGenerator, dataset_sample
from arenoflow.server import Application


@pytest.fixture
def setup(tmp_path):
    app = Application(tmp_path)
    asset = save_upload(tmp_path, "sample.jsonl", base64.b64encode(b'{"prompt":"test","answer":"ok"}\n').decode())
    dataset = save_dataset(app.controller.store, {"name": "sample", "source_type": "upload", "source": asset["path"]})
    app.llm.configure({"base_url": "http://localhost:9999/v1", "model": "test-model", "api_key": "test-secret"})
    return app, dataset


def test_settings_hide_key_and_clear_on_provider_change():
    llm = ScriptGenerator()
    settings = llm.configure({"base_url": "https://example.com/v1", "model": "test", "api_key": "secret"})
    assert settings["has_api_key"] and "secret" not in json.dumps(settings)
    llm.configure({"base_url": "https://other.example/v1", "model": "test"})
    assert not llm.settings()["has_api_key"]


def test_sample_reads_records_without_running_loader(setup):
    app, dataset = setup
    sample = dataset_sample(app.controller.store, dataset["id"])
    assert json.loads(sample["sample"])[0]["answer"] == "ok"


def test_generation_passes_context_and_returns_complete_script(monkeypatch, setup):
    app, dataset = setup
    source = 'def helper(value):\n    return float(value == "ok")\n\ndef reward_fn(record):\n    return helper(record.completion)\n'

    class Client:
        def open(self, request, timeout):
            assert request.full_url == "http://localhost:9999/v1/chat/completions"
            assert request.get_header("Authorization") == "Bearer test-secret"
            payload = json.loads(request.data)
            context = json.loads(payload["messages"][1]["content"])
            assert context["algorithm"] == "grpo"
            assert context["dataset"]["name"] == "sample"
            assert context["dataset_sample"] == '{"answer":"ok"}'
            assert "test-secret" not in request.data.decode()
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": source}}]}).encode())

    monkeypatch.setattr("arenoflow.llm.urllib.request.build_opener", lambda *args: Client())
    result = app.llm.generate(
        {
            "kind": "reward",
            "algorithm": "grpo",
            "dataset_id": dataset["id"],
            "sample": '{"answer":"ok"}',
            "prompt": "Exact match",
        },
        app.controller.store,
        app.catalog,
    )
    assert "def helper" in result["source"]
    assert not app.controller.store.functions()
    saved = save_function(app.controller.store, {**result, "name": "Reward module"})
    assert saved["dataset_id"] == dataset["id"] and saved["algorithm"] == "grpo"


@pytest.mark.parametrize("algo", ["sft", "dpo", "missing"])
def test_incompatible_algorithms_fail_before_provider_request(setup, algo):
    app, dataset = setup
    with pytest.raises(ValueError, match="compatible"):
        app.llm.generate({"kind": "reward", "algorithm": algo}, app.controller.store, app.catalog)


def test_invalid_generated_entrypoint_not_saved(monkeypatch, setup):
    app, dataset = setup

    class Client:
        def open(self, *args, **kwargs):
            return io.BytesIO(b'{"choices":[{"message":{"content":"raise RuntimeError()"}}]}')

    monkeypatch.setattr("arenoflow.llm.urllib.request.build_opener", lambda *args: Client())
    with pytest.raises(ValueError, match="entrypoint"):
        app.llm.generate(
            {"kind": "reward", "algorithm": "grpo", "dataset_id": dataset["id"], "sample": "{}", "prompt": "test"},
            app.controller.store,
            app.catalog,
        )
    assert not app.controller.store.functions()


def test_batch_generation_uses_one_request_for_all_scripts(monkeypatch, setup):
    app, dataset = setup
    scripts = [
        {
            "kind": "dataset_loader",
            "name": "Loader",
            "source": "def load_training_dataset(path, **kwargs):\n    return []\n",
        },
        {"kind": "reward", "name": "Reward", "source": "def reward_fn(record):\n    return 1.0\n"},
        {"kind": "agentic", "name": "Agent", "source": "async def run_agent(ctx, batch):\n    return []\n"},
    ]
    calls = []

    class Client:
        def open(self, request, timeout):
            calls.append(request)
            context = json.loads(json.loads(request.data)["messages"][1]["content"])
            assert {s["kind"] for s in context["scripts"]} == {s["kind"] for s in scripts}
            return io.BytesIO(
                json.dumps({"choices": [{"message": {"content": json.dumps({"scripts": scripts})}}]}).encode()
            )

    monkeypatch.setattr("arenoflow.llm.urllib.request.build_opener", lambda *args: Client())
    result = app.llm.generate(
        {
            "kinds": [s["kind"] for s in scripts],
            "algorithm": "grpo",
            "dataset_id": dataset["id"],
            "sample": "{}",
            "prompt": "Generate matching scripts",
        },
        app.controller.store,
        app.catalog,
    )
    assert len(calls) == 1
    assert len(result["scripts"]) == 3
    assert all(s["name"] == "sample - " + s["kind"] for s in result["scripts"])
    assert not app.controller.store.functions()
    saved = app.post("/api/scripts/batch", result)
    assert len(saved["scripts"]) == 3
    assert len(app.controller.store.functions()) == 3


def test_batch_save_validates_every_script_before_writing(setup):
    app, _ = setup
    with pytest.raises(ValueError):
        app.post(
            "/api/scripts/batch",
            {
                "scripts": [
                    {"kind": "reward", "name": "Valid", "source": "def reward_fn(record):\n    return 1.0\n"},
                    {"kind": "agentic", "name": "Invalid", "source": "invalid python !"},
                ]
            },
        )
    assert not app.controller.store.functions()


@pytest.mark.parametrize("kinds", [[], ["reward", "reward"], ["unknown"], ["dataset_loader", "reward"]])
def test_invalid_batch_selection_rejected(setup, kinds):
    app, _ = setup
    with pytest.raises(ValueError):
        app.llm.generate({"kinds": kinds, "algorithm": "sft"}, app.controller.store, app.catalog)


def test_incomplete_llm_batch_rejected(monkeypatch, setup):
    app, dataset = setup

    class Client:
        def open(self, request, timeout):
            scripts = [{"kind": "reward", "name": "Reward", "source": "def reward_fn(record):\n    return 0\n"}]
            return io.BytesIO(
                json.dumps({"choices": [{"message": {"content": json.dumps({"scripts": scripts})}}]}).encode()
            )

    monkeypatch.setattr("arenoflow.llm.urllib.request.build_opener", lambda *args: Client())
    with pytest.raises(ValueError, match="entrypoint"):
        app.llm.generate(
            {
                "kinds": ["reward", "agentic"],
                "algorithm": "grpo",
                "dataset_id": dataset["id"],
                "sample": "{}",
                "prompt": "test",
            },
            app.controller.store,
            app.catalog,
        )
    assert not app.controller.store.functions()


def test_demonstrations_include_complete_repository_sources_and_valid_board():
    import ast
    import runpy

    from arenoflow.catalog import ROOT
    from arenoflow.script_context import demonstration_context

    demos = demonstration_context()
    assert [demo["name"] for demo in demos] == ["math", "tictactoe", "sft"]
    for demo in demos:
        for file in demo["files"]:
            assert file["source"] == (ROOT / file["reference"]).read_text()
            ast.parse(file["source"])
    game = runpy.run_path(str(ROOT / "examples/agentic/tictactoe/game.py"))
    board = game["normalize_board"](demos[1]["sample"]["board"])
    assert game["next_player"](board) == "X" and not game["is_terminal"](board)


def test_sft_generation_includes_three_demos_without_requesting_reward(monkeypatch, setup):
    app, dataset = setup
    calls = []
    source = "def load_training_dataset(path, **kwargs):\n    return kwargs['default_loader'](path)\n"

    class Client:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            calls.append(payload)
            context = json.loads(payload["messages"][1]["content"])
            assert context["demonstrations"] == ["math", "tictactoe", "sft"]
            assert context["scripts"] == [{"kind": "dataset_loader", "entrypoint": "load_training_dataset"}]
            system = payload["messages"][0]["content"]
            assert "SFT must not acquire reward or agent behavior" in system
            assert "inline any required helpers" in system
            for filename in (
                "examples/math/dataset_loader.py",
                "examples/agentic/tictactoe/run_agent.py",
                "examples/sft/alpaca/dataset_loader.py",
            ):
                assert filename in system
            return io.BytesIO(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {"scripts": [{"kind": "dataset_loader", "name": "Loader", "source": source}]}
                                    )
                                }
                            }
                        ]
                    }
                ).encode()
            )

    monkeypatch.setattr("arenoflow.llm.urllib.request.build_opener", lambda *args: Client())
    app.llm.generate(
        {
            "kinds": ["dataset_loader"],
            "algorithm": "sft",
            "dataset_id": dataset["id"],
            "sample": "{}",
            "prompt": "Normalize instruction/input/output",
        },
        app.controller.store,
        app.catalog,
    )
    assert len(calls) == 1
