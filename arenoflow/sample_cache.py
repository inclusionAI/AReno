"""Persistent sample cache populated when datasets are registered."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from arenoflow.samples import dataset_sample


class SampleCache:
    def __init__(self, store):
        self.store = store
        self.directory = store.directory / "dataset_samples"
        self.directory.mkdir(exist_ok=True, mode=0o700)
        self.lock = threading.Lock()
        self.pending = set()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dataset-sample")

    def key(self, dataset):
        identity = {k: dataset.get(k) for k in ("source", "source_type", "model_hub")}
        identity["cache_version"] = 2
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def status(self, dataset):
        key = self.key(dataset)
        with self.lock:
            if key in self.pending:
                return {"status": "downloading"}
            file = self.directory / (key + ".json")
            if file.exists():
                return json.loads(file.read_text())
        return {"status": "not_cached"}

    def enqueue(self, dataset, retry=False):
        key = self.key(dataset)
        with self.lock:
            file = self.directory / (key + ".json")
            if key in self.pending:
                return {"status": "downloading"}
            if file.exists() and not retry:
                return json.loads(file.read_text())
            self.pending.add(key)
        self.executor.submit(self._download, dataset, key)
        return {"status": "downloading"}

    def _download(self, dataset, key):
        result = None
        try:
            result = {"status": "ready", **dataset_sample(self.store, dataset["id"])}
            current = next((d for d in self.store.datasets() if d["id"] == dataset["id"]), None)
            if current is None or self.key(current) != key:
                result = None
        except Exception:
            result = {"status": "failed", "error": "Sample download failed; retry from Dataset Manager"}
        finally:
            with self.lock:
                try:
                    if result is not None:
                        target = self.directory / (key + ".json")
                        temporary = target.with_suffix(".tmp")
                        temporary.write_text(json.dumps(result, ensure_ascii=False))
                        temporary.replace(target)
                finally:
                    self.pending.discard(key)
