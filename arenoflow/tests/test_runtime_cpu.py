"""Cloud-free lifecycle, persistence, billing and remote orchestration tests."""

import datetime as dt
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace

import pytest

from arenoflow import remote
from arenoflow.billing import fetch_billing
from arenoflow.catalog import catalog
from arenoflow.controller import Controller
from arenoflow.store import Store


class FakeSandbox:
    object_id = "sb-test"
    stdout = [
        remote.PREFIX + json.dumps({"type": "metric", "tag": "train/loss", "value": 0.8, "step": 1, "time": 1}) + "\n",
        remote.PREFIX + json.dumps({"type": "checkpoint", "index": 0, "path": "/artifacts/final", "time": 2}) + "\n",
    ]
    stderr = []
    terminated = False

    def poll(self):
        return 0

    def terminate(self):
        self.terminated = True


class FakeProvider:
    def __init__(self, *_args):
        self.sandbox = FakeSandbox()

    def check(self):
        pass

    def start(self, manifest, resources, endpoint_key, on_phase=None):
        if on_phase:
            on_phase("building_image")
            on_phase("starting_sandbox")
        self.manifest = manifest
        self.endpoint_key = endpoint_key
        return self.sandbox

    def attach(self, _id):
        return self.sandbox


def test_job_completion_and_persistence(tmp_path):
    store = Store(tmp_path)
    controller = Controller(store, catalog(), FakeProvider, lambda x: x + "@resolved")
    controller.connect("fake-id", "fake-secret")
    record = controller.submit(
        {
            "model": {"adapter": "qwen3", "checkpoint": "Qwen/Qwen3-0.6B"},
            "stages": [{"algo": "sft", "params": {"dataset_path": "data"}}],
        }
    )
    deadline = time.time() + 5
    while store.get(record["id"])["status"] not in ("succeeded", "failed") and time.time() < deadline:
        time.sleep(0.01)
    final = store.get(record["id"])
    assert final["status"] == "succeeded"
    assert [e["phase"] for e in store.events(record["id"]) if e["type"] == "phase"][:3] == [
        "resolving_image",
        "building_image",
        "starting_sandbox",
    ]
    assert final["checkpoint"] == "/artifacts/final"
    assert final["sandbox_id"] == "sb-test"
    assert any(event.get("tag") == "train/loss" for event in store.events(record["id"]))
    assert Store(tmp_path).get(record["id"]) == final
    assert "fake-secret" not in json.dumps(store.jobs())


def test_queued_cancellation_does_not_create_cloud_resources(tmp_path):
    controller = Controller(Store(tmp_path), catalog(), FakeProvider, lambda x: x)
    controller.connect("id", "secret")
    controller._thread = lambda *_args: None
    job = controller.submit(
        {
            "model": {"adapter": "qwen3", "checkpoint": "model"},
            "stages": [{"algo": "sft", "params": {"dataset_path": "data"}}],
        }
    )
    controller.stop(job["id"])
    controller._start(job["id"], controller.provider, "")
    assert controller.store.get(job["id"])["status"] == "cancelled"
    assert not hasattr(controller.provider, "manifest")


def test_secrets_redacted_from_logs(tmp_path):
    controller = Controller(Store(tmp_path), catalog(), FakeProvider)
    controller.secrets = ["secret-value"]
    controller.store.put({"id": "job"})
    controller._read("job", ["error secret-value"])
    assert "secret-value" not in json.dumps(controller.store.events("job"))


def test_endpoint_key_not_persisted(tmp_path):
    controller = Controller(Store(tmp_path), catalog(), FakeProvider)
    controller.connect("id", "secret")
    controller._thread = lambda *_args: None
    key = "private-endpoint-key-that-is-long"
    job = controller.submit(
        {"kind": "deployment", "model": {"adapter": "qwen3", "checkpoint": "model"}, "endpoint_key": key}
    )
    assert key not in json.dumps(controller.store.get(job["id"]))
    assert "endpoint_key" not in json.dumps(job)


def test_billing_preserves_actual_decimal_data_and_denied_report():
    @dataclass
    class Summary:
        metered_cost: Decimal = Decimal("3.123456789")
        billed_cost: Decimal = Decimal("0.50")

    class Billing:
        def summary(self):
            return Summary()

        def report(self, **kwargs):
            assert kwargs["resolution"] == "h"
            assert kwargs["start"].tzinfo == dt.timezone.utc
            raise PermissionError("Team plan required")

    workspace = SimpleNamespace(billing=Billing())
    provider = SimpleNamespace(
        modal=SimpleNamespace(Workspace=SimpleNamespace(from_context=lambda **kw: workspace)), client=object()
    )
    result = fetch_billing(provider)
    assert result["summary"]["metered_cost"] == "3.123456789"
    assert result["report"] is None
    assert "Team plan required" in result["errors"]["report"]
    assert result["scope"] == "workspace"


def test_checkpoint_prefers_final_and_accepts_peft(tmp_path):
    for name in ("step_000100", "final"):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").touch()
    assert remote.latest_checkpoint(tmp_path) == str(tmp_path / "final")
    (tmp_path / "final/config.json").unlink()
    (tmp_path / "final/adapter_config.json").write_text("{}")
    assert remote.latest_checkpoint(tmp_path) == str(tmp_path / "final")


def test_pipeline_stops_on_failure_and_passes_real_artifact(tmp_path, monkeypatch):
    calls = []
    manifest = {
        "kind": "training",
        "image": "test-image",
        "revision": "test",
        "stages": [
            {"algo": "sft", "args": ["--ckpt", "base"], "save_path": str(tmp_path / "a")},
            {"algo": "gspo", "args": ["--ckpt", "__previous__"], "save_path": str(tmp_path / "b")},
        ],
    }

    def popen(args, **kwargs):
        stage = json.loads(args[-1])
        calls.append(stage)
        path = tmp_path / "a/final"
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").touch()
        return SimpleNamespace(wait=lambda: 0 if len(calls) == 1 else 17)

    monkeypatch.setattr(remote.subprocess, "Popen", popen)
    with pytest.raises(SystemExit) as exc:
        remote.main(manifest)
    assert exc.value.code == 17
    assert calls[1]["args"] == ["--ckpt", str(tmp_path / "a/final")]


def test_final_checkpoint_saved_only_on_success(tmp_path, monkeypatch):
    """Exercise the adapter around AReno's public Trainer without CUDA imports."""
    import sys
    from types import ModuleType

    modules = {
        name: ModuleType(name) for name in ["areno", "areno.api", "areno.api.metrics", "areno.cli", "areno.cli.train"]
    }
    calls = []

    class Trainer:
        def init(self):
            pass

        def save_checkpoint(self, path):
            calls.append(("save", path))

        def close(self):
            calls.append(("close",))

    modules["areno"].Trainer = Trainer
    modules["areno"].api = modules["areno.api"]
    modules["areno.api"].metrics = modules["areno.api.metrics"]
    modules["areno.api"].Trainer = Trainer
    modules["areno.api.metrics"].create_tensorboard_writer = lambda directory: None

    def main(**kwargs):
        trainer = modules["areno.api"].Trainer()
        trainer.init()
        trainer.close()
        trainer = modules["areno.api"].Trainer()
        trainer.init()
        try:
            raise RuntimeError("training failed")
        except RuntimeError:
            trainer.close()

    modules["areno.cli.train"].train_command = SimpleNamespace(main=main)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    remote.stage_main({"args": [], "save_path": str(tmp_path), "index": 0, "algo": "sft"})
    assert calls == [("save", str(tmp_path / "final")), ("close",), ("close",)]


def test_chunked_metric_stream_and_repeated_lines(tmp_path):
    from arenoflow.events import lines, parse_line

    message = remote.PREFIX + json.dumps({"type": "metric", "tag": "loss", "value": 0.1, "step": 1})
    result = [parse_line(s) for s in lines(["hello\n" + message[:8], message[8:] + "\nlast line"])]
    assert len(result) == 3
    assert result[1]["type"] == "metric" and result[1]["value"] == 0.1
    assert result[-1]["message"] == "last line"
    assert parse_line(remote.PREFIX + "[1, 2]")["type"] == "log"
    assert parse_line(remote.PREFIX + '{"type":"metric","tag":"loss","value":NaN}')["type"] == "log"


def test_reconnection_deduplicates_metrics_without_merging_stages(tmp_path):
    store = Store(tmp_path)
    metric = {"type": "metric", "tag": "train/loss", "value": 0.8, "step": 1, "time": 12, "index": 0}
    store.event("run", metric)
    store.event("run", metric)
    store.event("run", {**metric, "index": 1})
    assert len(store.events("run")) == 2


def test_same_credentials_can_reconcile_active_jobs(tmp_path):
    controller = Controller(Store(tmp_path), catalog(), FakeProvider)
    controller._thread = lambda *_args: None
    controller.connect("token-id", "token-secret")
    controller.store.put({"id": "run", "status": "unknown", "sandbox_id": "sb-test", "created_at": 1})
    assert controller.connect("token-id", "token-secret") == {"connected": True}
    with pytest.raises(ValueError, match="switching"):
        controller.connect("another-id", "another-secret")


def test_original_model_snapshots_use_volume_cache_and_reuse_refs(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from arenoflow.remote import cache_model_refs

    calls = []

    def download(reference, cache_dir):
        calls.append((reference, cache_dir))
        return str(tmp_path / "snapshot")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setenv("HF_HUB_CACHE", "/artifacts/cache/hf/hub")
    result = cache_model_refs(["--ckpt", "org/model", "--ref-ckpt", "org/model", "--model-hub", "hf"])
    assert calls == [("org/model", "/artifacts/cache/hf/hub")]
    assert result[1] == result[3] == str(tmp_path / "snapshot")
    assert cache_model_refs(["--ckpt", str(tmp_path)]) == ["--ckpt", str(tmp_path)]


def test_modelscope_original_weights_use_volume(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from arenoflow.remote import cache_model_refs

    calls = []

    def download(reference, cache_dir):
        calls.append((reference, cache_dir))
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=download))
    monkeypatch.setenv("MODELSCOPE_CACHE", "/artifacts/cache/modelscope")
    cache_model_refs(["--model-path", "org/model", "--model-hub", "modelscope"])
    assert calls == [("org/model", "/artifacts/cache/modelscope")]


@pytest.mark.parametrize("kind", ["image_build", "model_download"])
def test_preparation_lifecycle_without_training_inputs(tmp_path, kind):
    class PreparationProvider(FakeProvider):
        def start(self, manifest, resources, endpoint_key, on_phase=None):
            assert resources["gpu"] is None and resources["count"] == 0
            assert manifest["stages"] == []
            on_phase("building_image")
            if manifest["kind"] == "image_build":
                return None
            sandbox = FakeSandbox()
            sandbox.stdout = []
            return sandbox

    controller = Controller(Store(tmp_path), catalog(), PreparationProvider, lambda x: x)
    controller.connect("id", "secret")
    controller._thread = lambda *_args: None
    record = controller.submit({"kind": kind, "model": {"adapter": "qwen3", "checkpoint": "Qwen/Qwen3-0.6B"}})
    controller._start(record["id"], controller.provider, "")
    job = controller.store.get(record["id"])
    assert job["status"] == "succeeded"
    assert bool(job["sandbox_id"]) == (kind == "model_download")
    assert job["finished_at"] >= job["started_at"]


def test_pre_download_only_resolves_original_model(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "cache_model_refs", lambda args: calls.append(args))
    remote.main(
        {
            "kind": "model_download",
            "image": "image",
            "revision": "rev",
            "model": {"checkpoint": "Qwen/Qwen3-0.6B"},
            "model_hub": "hf",
        }
    )
    assert calls == [["--model-path", "Qwen/Qwen3-0.6B", "--model-hub", "hf"]]


@pytest.mark.parametrize("kind", ["image_build", "model_download", "training"])
def test_provider_preparation_build_and_gpu_reservation(kind):
    from unittest.mock import MagicMock

    from arenoflow.provider import ModalProvider

    provider = ModalProvider.__new__(ModalProvider)
    provider.modal = MagicMock()
    provider.client = object()
    image = provider.modal.Image.from_registry.return_value.add_local_file.return_value
    phases = []
    provider.start(
        {"kind": kind, "image": "image"},
        {
            "gpu": None if kind == "model_download" else "H100",
            "count": 1,
            "cpu": 2,
            "memory_gib": 8,
            "timeout_seconds": 120,
        },
        on_phase=phases.append,
    )
    image.build.assert_called_once_with(provider.modal.App.lookup.return_value)
    if kind == "image_build":
        provider.modal.Sandbox.create.assert_not_called()
    else:
        options = provider.modal.Sandbox.create.call_args.kwargs
        assert options["gpu"] == (None if kind == "model_download" else "H100:1")
        assert options["env"]["HF_HUB_CACHE"] == "/artifacts/cache/hf/hub"
        assert "/artifacts" in options["volumes"]
    assert "building_image" in phases
