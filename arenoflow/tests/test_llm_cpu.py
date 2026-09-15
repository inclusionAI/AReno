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
