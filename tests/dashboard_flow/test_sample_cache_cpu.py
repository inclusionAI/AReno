"""Dataset registration prefetches samples; later reads reuse the disk cache."""

import threading

from areno.dashboard.flow.sample_cache import SampleCache
from areno.dashboard.flow.server import Application


def test_register_dataset_downloads_once_and_survives_restart(tmp_path, monkeypatch):
    calls = []
    entered, release = threading.Event(), threading.Event()

    def download(store, identifier):
        calls.append(identifier)
        entered.set()
        assert release.wait(3)
        return {"sample": '[{"answer":42}]', "row_count": 1, "manual": False}

    monkeypatch.setattr("areno.dashboard.flow.sample_cache.dataset_sample", download)
    app = Application(tmp_path)
    record = app.post("/api/datasets", {"name": "Cached", "source": "owner/data", "model_hub": "hf"})
    assert entered.wait(3)
    assert record["sample_status"] == "downloading"
    assert app.post("/api/scripts/sample", {"dataset_id": record["id"]})["status"] == "downloading"
    release.set()
    app.samples.executor.shutdown(wait=True)
    assert app.post("/api/scripts/sample", {"dataset_id": record["id"]})["sample"] == '[{"answer":42}]'
    assert len(calls) == 1
    restarted = SampleCache(app.controller.store)
    assert restarted.enqueue(record)["status"] == "ready"
    assert len(calls) == 1
    restarted.executor.shutdown()


def test_changed_source_uses_a_different_cache_key(tmp_path):
    app = Application(tmp_path)
    first = {"source": "owner/data:main:train", "source_type": "repository", "model_hub": "hf"}
    assert app.samples.key(first) != app.samples.key({**first, "source": "owner/data:main:test"})
    assert app.samples.key(first) != app.samples.key({**first, "model_hub": "modelscope"})
    app.samples.executor.shutdown()


def test_failure_retains_dataset_and_reports_retry_state(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("provider detail")

    monkeypatch.setattr("areno.dashboard.flow.sample_cache.dataset_sample", fail)
    app = Application(tmp_path)
    record = app.post("/api/datasets", {"name": "Unavailable", "source": "owner/data"})
    app.samples.executor.shutdown(wait=True)
    assert app.get("/api/datasets", {})[0]["sample_status"] == "failed"
    assert app.post("/api/scripts/sample", {"dataset_id": record["id"]})["status"] == "failed"
