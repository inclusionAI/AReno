"""Background job lifecycle, credential session, monitoring and cancellation."""

from __future__ import annotations

import os
import threading
import time
import uuid

from arenoflow.assets import referenced_uploads
from arenoflow.datasets import resolve_request
from arenoflow.events import lines, parse_line
from arenoflow.provider import ModalProvider, resolve_image
from arenoflow.workflows import plan

TERMINAL = {"succeeded", "failed", "cancelled"}


class Controller:
    def __init__(self, store, catalog, provider_factory=ModalProvider, image_resolver=resolve_image):
        self.store, self.catalog = store, catalog
        self.provider_factory, self.image_resolver = provider_factory, image_resolver
        self.provider = None
        self.credentials = None
        self.cost_estimator = None
        self.secrets = []
        self.lock = threading.RLock()
        self.watching = set()
        self.env_credentials = (os.environ.get("MODAL_TOKEN_ID", ""), os.environ.get("MODAL_TOKEN_SECRET", ""))

    def connect(self, token_id="", token_secret=""):
        if not token_id and not token_secret:
            token_id, token_secret = self.env_credentials
        if not token_id or not token_secret:
            raise ValueError("Modal requires a Token ID and Token Secret")
        with self.lock:
            if (
                self.provider
                and self.credentials != (token_id, token_secret)
                and any(j["status"] not in TERMINAL for j in self.store.jobs())
            ):
                raise ValueError("Stop active jobs before switching Modal credentials")
            self.secrets.extend([token_id, token_secret])
            provider = self.provider_factory(token_id, token_secret)
            provider.check()
            self.provider = provider
            self.credentials = (token_id, token_secret)
        for job in self.store.jobs():
            if job["status"] not in TERMINAL:
                if job.get("sandbox_id"):
                    self._thread(self._reattach, job["id"], provider)
                else:
                    self.store.update(
                        job["id"],
                        status="failed",
                        error="Local service stopped before a sandbox ID was saved",
                        finished_at=time.time(),
                    )
        return {"connected": True}

    def redact(self, text):
        for secret in self.secrets:
            if secret:
                text = text.replace(secret, "[redacted]")
        return text

    @staticmethod
    def _thread(target, *args):
        threading.Thread(target=target, args=args, daemon=True).start()

    def submit(self, request):
        if not self.provider:
            raise ValueError("Connect your Modal workspace before launching")
        job_id = uuid.uuid4().hex[:16]
        prepared = plan(resolve_request(request, self.store), self.catalog, job_id)
        referenced_uploads(prepared["manifest"], self.store.directory)
        endpoint_key = request.get("endpoint_key", "")
        if prepared["manifest"]["kind"] == "deployment" and len(endpoint_key) < 24:
            raise ValueError("Choose an endpoint API key of at least 24 characters")
        if endpoint_key:
            self.secrets.append(endpoint_key)
        # Credentials and endpoint keys must never reach persistent records.
        record = dict(
            id=job_id,
            name=str(request.get("name") or "Untitled workflow")[:120],
            kind=prepared["manifest"]["kind"],
            status="queued",
            created_at=time.time(),
            updated_at=time.time(),
            stage=0,
            sandbox_id=None,
            **prepared,
        )
        if self.cost_estimator and record["kind"] in ("training", "deployment"):
            try:
                record["estimate"] = self.cost_estimator(
                    record["resources"], request.get("estimate_hours", record["resources"]["timeout_seconds"] / 3600)
                )
            except Exception as exc:
                record["estimate_error"] = self.redact(str(exc))
        self.store.put(record)
        self._thread(self._start, job_id, self.provider, endpoint_key)
        return record

    def _phase(self, job_id, phase):
        with self.store.lock:
            job = self.store.get(job_id)
            if job.get("cancel_requested"):
                raise InterruptedError("Job cancellation requested")
            if job["status"] in TERMINAL:
                return
            stamp = time.time()
            self.store.update(job_id, phase=phase, phase_started_at=stamp)
            self.store.event(job_id, {"type": "phase", "phase": phase, "time": stamp})

    def _start(self, job_id, provider, endpoint_key):
        sandbox = None
        try:
            self.store.update(job_id, status="starting", started_at=time.time())
            job = self.store.get(job_id)
            manifest = job["manifest"]
            self._phase(job_id, "resolving_image")
            manifest["image"] = self.image_resolver(manifest["image"])
            self.store.update(job_id, manifest=manifest)
            if self.store.get(job_id).get("cancel_requested"):
                self.store.update(job_id, status="cancelled", finished_at=time.time())
                return
            uploads = referenced_uploads(manifest, self.store.directory)
            if uploads:
                self._phase(job_id, "uploading_data")
                provider.upload(uploads)
            sandbox = provider.start(
                manifest, job["resources"], endpoint_key, on_phase=lambda phase: self._phase(job_id, phase)
            )
            if manifest["kind"] == "image_build":
                cancelled = self.store.get(job_id).get("cancel_requested")
                self.store.update(job_id, status="cancelled" if cancelled else "succeeded", finished_at=time.time())
                self.store.event(job_id, {"type": "log", "message": "Container build finished", "time": time.time()})
                return
            self.store.update(
                job_id,
                sandbox_id=sandbox.object_id,
                status="stopping" if self.store.get(job_id).get("cancel_requested") else "running",
                phase="starting_runtime",
                phase_started_at=time.time(),
                started_at=time.time(),
            )
            if self.store.get(job_id).get("cancel_requested"):
                sandbox.terminate()
            self._watch(job_id, sandbox)
        except Exception as exc:
            if sandbox is not None:
                try:
                    sandbox.terminate()
                except Exception:
                    self.store.update(
                        job_id, status="unknown", error="Monitoring failed; reconnect to reconcile the sandbox"
                    )
                    return
            self.store.update(
                job_id,
                status="cancelled" if self.store.get(job_id).get("cancel_requested") else "failed",
                error=self.redact(str(exc)),
                finished_at=time.time(),
            )

    def _reattach(self, job_id, provider):
        try:
            sandbox = provider.attach(self.store.get(job_id)["sandbox_id"])
            if self.store.get(job_id).get("cancel_requested"):
                sandbox.terminate()
            self._watch(job_id, sandbox)
        except Exception as exc:
            self.store.update(job_id, status="unknown", error=self.redact(str(exc)))

    def _watch(self, job_id, sandbox):
        with self.lock:
            if job_id in self.watching:
                return
            self.watching.add(job_id)
        readers = []
        try:
            for stream in (sandbox.stdout, sandbox.stderr):
                thread = threading.Thread(target=self._read, args=(job_id, stream), daemon=True)
                thread.start()
                readers.append(thread)
            while sandbox.poll() is None:
                if self.store.get(job_id).get("cancel_requested"):
                    sandbox.terminate()
                if self.store.get(job_id).get("ready") and not self.store.get(job_id).get("endpoint"):
                    self.store.update(job_id, endpoint=sandbox.tunnels()[8080].url + "/v1", status="ready")
                time.sleep(2)
            for thread in readers:
                thread.join(timeout=5)
            job = self.store.get(job_id)
            code = sandbox.poll()
            status = "cancelled" if job.get("cancel_requested") else "succeeded" if code == 0 else "failed"
            self.store.update(job_id, status=status, exit_code=code, finished_at=time.time())
        finally:
            with self.lock:
                self.watching.discard(job_id)

    def _read(self, job_id, stream):
        try:
            for line in lines(stream):
                event = parse_line(self.redact(line))
                self.store.event(job_id, event)
                if event.get("type") == "phase":
                    with self.store.lock:
                        job = self.store.get(job_id)
                        if (
                            job["status"] not in TERMINAL
                            and not job.get("cancel_requested")
                            and event.get("time", 0) >= job.get("remote_phase_time", 0)
                        ):
                            self.store.update(
                                job_id,
                                phase=event["phase"],
                                phase_started_at=event["time"],
                                remote_phase_time=event["time"],
                            )
                elif event.get("type") == "stage":
                    self.store.update(job_id, stage=event["index"])
                elif event.get("type") == "checkpoint":
                    self.store.update(job_id, checkpoint=event["path"], adapter_only=event.get("adapter_only", False))
                elif event.get("type") == "ready":
                    self.store.update(job_id, ready=True)
                elif event.get("type") == "error":
                    self.store.update(job_id, error=event.get("message"))
        except Exception as exc:
            self.store.event(
                job_id,
                {"type": "log", "message": f"Log stream interrupted: {self.redact(str(exc))}", "time": time.time()},
            )

    def stop(self, job_id):
        job = self.store.get(job_id)
        if job["status"] in TERMINAL:
            return job
        if not self.provider:
            raise ValueError("Reconnect Modal to stop this remote job")
        self.store.update(job_id, cancel_requested=True, status="stopping")
        if job.get("sandbox_id"):
            self.provider.attach(job["sandbox_id"]).terminate()
        return self.store.get(job_id)
