"""Adapt Areno Flow records to the dashboard without a second HTTP service."""

from __future__ import annotations

import base64
import datetime as dt
import http.client
import ipaddress
import json
import secrets
import socket
import ssl
import threading
import time
from decimal import Decimal
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from areno.dashboard.flow.assets import DATA_SUFFIXES, MAX_UPLOAD_BYTES
from areno.dashboard.flow.datasets import resolve_request
from areno.dashboard.flow.server import Application


def timestamp(value):
    return dt.datetime.fromtimestamp(value, dt.timezone.utc).isoformat() if value else None


class ModalFlow:
    def __init__(self, directory, application=None):
        self.app = application or Application(directory)
        self.lock = threading.RLock()
        self.plans = {}

    def import_url(self, body):
        url = str(body.get("url", "")).strip()
        parsed = urlsplit(url)
        parts = parsed.path.strip("/").split("/")
        if (
            parsed.scheme == "https"
            and parsed.hostname in ("huggingface.co", "modelscope.cn")
            and len(parts) == 3
            and parts[0] == "datasets"
        ):
            return self.app.post(
                "/api/datasets",
                {
                    "name": body.get("name") or "/".join(parts[1:]),
                    "source": "/".join(parts[1:]),
                    "source_type": "repository",
                    "model_hub": "hf" if parsed.hostname == "huggingface.co" else "modelscope",
                },
            )
        filename, content = download_dataset(url)
        asset = self.app.post("/api/uploads", {"name": filename, "content": base64.b64encode(content).decode()})
        return self.app.post(
            "/api/datasets",
            {
                "name": body.get("name") or filename,
                "source": asset["path"],
                "source_type": "upload",
                "source_name": filename,
            },
        )

    def preview(self, request):
        request = resolve_request(request, self.app.controller.store)
        for stage in request.get("stages", []):
            for key in ("dataset_id", "dataset_loader_id", "reward_function_id", "agentic_function_id"):
                stage.pop(key, None)
        prepared = self.app.post("/api/preview", request)
        # Keep endpoint keys on the server, never in chat history or browser storage.
        identifier = secrets.token_urlsafe(24)
        with self.lock:
            self.plans = {key: value for key, value in self.plans.items() if value[0] > time.time()}
            self.plans[identifier] = (time.time() + 1800, json.loads(json.dumps(request)))
        return {
            "id": identifier,
            "status": "proposed",
            "tool": "start_modal",
            "objective": request.get("name") or "Run AReno on Modal",
            "summary": "Review the GPU reservation and commands. Execution starts a billable Modal sandbox.",
            "parameters": {"plan_id": identifier},
            "resources": prepared["resources"],
            "steps": [
                {"id": i + 1, "title": command, "status": "pending"} for i, command in enumerate(prepared["commands"])
            ],
            "command": "\n".join(prepared["commands"]),
        }

    def execute(self, identifier):
        with self.lock:
            item = self.plans.get(identifier)
            if not item or item[0] <= time.time():
                raise ValueError("Plan expired or already executed; preview the task again")
            record = self.app.post("/api/jobs", item[1])
            del self.plans[identifier]
        return {"job_id": "modal-" + record["id"], "endpoint_key": item[1].get("endpoint_key")}

    def jobs(self, job_class):
        return [self.job(record["id"], job_class, detail=False) for record in self.app.controller.store.jobs()]

    def job(self, identifier, job_class, detail=True):
        record = self.app.controller.store.get(identifier.removeprefix("modal-"))
        manifest = record["manifest"]
        job = job_class(
            kind="serve" if record["kind"] == "deployment" else "train",
            name=record["name"],
            command=record.get("commands", []),
            config=manifest,
            metrics_dir=None,
        )
        job.id = "modal-" + record["id"]
        job.provider = "modal"
        job.status = {"ready": "running", "cancelled": "stopped"}.get(record["status"], record["status"])
        job.stage = record.get("phase") or str(record.get("stage", "queued"))
        job.created_at = timestamp(record["created_at"])
        job.updated_at = timestamp(record["updated_at"])
        job.returncode = record.get("exit_code")
        job.modal = {
            "sandbox_id": record.get("sandbox_id"),
            "endpoint": record.get("endpoint"),
            "resources": record["resources"],
            "remote_status": record["status"],
        }
        started = record.get("started_at") if record.get("sandbox_id") else None
        elapsed = max(0, (record.get("finished_at") or time.time()) - started) if started else 0
        quote = record.get("estimate")
        job.usage = {
            "currency": "USD",
            "seconds": elapsed,
            "estimated": True,
            "accrued_cost": str(Decimal(quote["hourly_cost"]) * Decimal(str(elapsed)) / 3600) if quote else None,
            "source": quote.get("rates", {}).get("source") if quote else None,
            "error": record.get("estimate_error") if not quote else None,
        }
        if detail:
            cursor = 0
            while True:
                events = self.app.controller.store.events(record["id"], cursor)
                if not events:
                    break
                for event in events:
                    cursor = event["cursor"]
                    if event["type"] != "metric":
                        job.logs.append(event.get("message") or json.dumps(event, ensure_ascii=False))
                job.logs = job.logs[-300:]
            if record.get("error"):
                job.logs.append(record["error"])
        for event in (
            self.app.controller.store.metrics(record["id"])
            if detail
            else self.app.controller.store.latest_metrics(record["id"])
        ):
            name = event["tag"]
            if len(manifest.get("stages", [])) > 1:
                name = f"stage-{event.get('index', 0)}/{name}"
            job.metrics.append(
                {
                    "name": name,
                    "value": float(event["value"]),
                    "step": event.get("step", 0),
                    "time": timestamp(event.get("time")),
                }
            )
            job.perf[name] = float(event["value"])
            job.step = max(job.step, int(event.get("step", 0)))
        return job


def download_dataset(url):
    """Fetch a bounded public HTTPS file, pinning validated DNS at each redirect."""
    original_name = Path(unquote(urlsplit(url).path)).name
    for _ in range(6):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 443)
        ):
            raise ValueError("Use a public HTTPS dataset file URL or dataset repository URL")
        addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
            raise ValueError("Dataset URLs must resolve to public internet addresses")
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=20)
        try:
            sock = socket.create_connection((addresses[0][4][0], 443), timeout=20)
            try:
                connection.sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parsed.hostname)
            except Exception:
                sock.close()
                raise
            connection.request(
                "GET",
                parsed.path + ("?" + parsed.query if parsed.query else ""),
                headers={"User-Agent": "AReno-Dashboard"},
            )
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise ValueError("Dataset redirect has no destination")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f"Dataset download returned HTTP {response.status}")
            filename = (
                original_name
                if Path(original_name).suffix.lower() in DATA_SUFFIXES
                else Path(unquote(parsed.path)).name
            )
            if Path(filename).suffix.lower() not in DATA_SUFFIXES:
                raise ValueError("The URL must point to a JSON, JSONL, CSV, TSV, Parquet or Arrow file")
            content = response.read(MAX_UPLOAD_BYTES + 1)
            if not content or len(content) > MAX_UPLOAD_BYTES:
                raise ValueError("Dataset files must be non-empty and at most 16 MiB")
            return filename, content
        finally:
            connection.close()
    raise ValueError("Too many dataset URL redirects")
