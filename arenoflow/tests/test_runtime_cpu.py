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

    def start(self, manifest, resources, endpoint_key):
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
    assert final["checkpoint"] == "/artifacts/final"
    assert final["sandbox_id"] == "sb-test"
    assert store.events(record["id"])[0]["tag"] == "train/loss"
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
    remote.stage_main({"args": [], "save_path": str(tmp_path)})
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
