"""Adapt Areno Flow records to the dashboard without a second HTTP service."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import secrets
import threading
import time
from decimal import Decimal
from urllib.parse import unquote, urlsplit

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
        host = (parsed.hostname or "").removeprefix("www.")
        if (
            parsed.scheme != "https"
            or host not in ("huggingface.co", "modelscope.cn")
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 443)
        ):
            raise ValueError("Paste a Hugging Face or ModelScope dataset repository URL")
        parts = unquote(parsed.path).strip("/").split("/")
        # ModelScope's dataset overview links commonly end in /summary.
        if host == "modelscope.cn" and len(parts) == 4 and parts[-1] == "summary":
            parts.pop()
        valid_length = len(parts) in (2, 3) if host == "huggingface.co" else len(parts) == 3
        if (
            not valid_length
            or parts[0] != "datasets"
            or any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", part) for part in parts[1:])
        ):
            raise ValueError(
                "Use the dataset repository page: https://huggingface.co/datasets/owner/name or https://modelscope.cn/datasets/owner/name"
            )
        return self.app.post(
            "/api/datasets",
            {
                "name": body.get("name") or "/".join(parts[1:]),
                "source": "/".join(parts[1:]),
                "source_type": "repository",
                "model_hub": "hf" if host == "huggingface.co" else "modelscope",
            },
        )

    def preview(self, request):
        request = json.loads(json.dumps(request))
        groups = [(stage.setdefault("params", {}), self.app.catalog["train"]) for stage in request.get("stages", [])]
        if request.get("kind") == "deployment":
            groups.append((request.setdefault("serve", {}), self.app.catalog["serve"]))
        for params, schema in groups:
            for option in schema:
                key = option["name"]
                value = params.get(key)
                if not isinstance(value, str):
                    continue
                if value == "":
                    params.pop(key, None)
                elif option["type"] == "bool":
                    if value.lower() not in ("true", "false"):
                        raise ValueError(f"{key} must be true or false")
                    params[key] = value.lower() == "true"
                elif option["type"] in ("int", "float") and not option.get("multiple"):
                    try:
                        number = float(value)
                        if not math.isfinite(number) or (option["type"] == "int" and not number.is_integer()):
                            raise ValueError()
                        params[key] = int(number) if option["type"] == "int" else number
                    except ValueError:
                        raise ValueError(f"{key} must be a finite {option['type']}") from None
                elif option.get("multiple") and value.lstrip().startswith("["):
                    params[key] = json.loads(value)
        request = resolve_request(request, self.app.controller.store)
        for stage in request.get("stages", []):
            params = stage.setdefault("params", {})
            params.setdefault("attn_backend", "flash")
            if "adam_4bit" not in params and not params.get("adam_8bit"):
                params["adam_4bit"] = True
        if request.get("kind") == "deployment":
            request.setdefault("serve", {}).setdefault("attn_backend", "flash")
        for stage in request.get("stages", []):
            for key in ("dataset_id", "dataset_loader_id", "reward_function_id", "agentic_function_id"):
                stage.pop(key, None)
        prepared = self.app.post("/api/preview", request)
        request["image"] = self.app.controller.image_resolver(
            request.get("image") or "ghcr.io/inclusionai/areno:latest"
        )
        request["resources"] = prepared["resources"]
        request["estimate_hours"] = prepared["resources"]["timeout_seconds"] / 3600
        estimate, estimate_error = None, None
        try:
            estimate = self.app.pricing.quote(prepared["resources"], request["estimate_hours"])
        except Exception as exc:
            estimate_error = self.app.controller.redact(str(exc))
        # Keep endpoint keys on the server, never in chat history or browser storage.
        identifier = secrets.token_urlsafe(24)
        with self.lock:
            self.plans = {key: value for key, value in self.plans.items() if value[0] > time.time()}
            self.plans[identifier] = (time.time() + 1800, json.loads(json.dumps(request)))
            self.app.controller.store.save_plan(identifier, request, self.plans[identifier][0])
        return {
            "id": identifier,
            "status": "proposed",
            "tool": "start_modal",
            "objective": request.get("name") or "Run AReno on Modal",
            "summary": "Review the GPU reservation and commands. Execution starts a billable Modal sandbox.",
            "parameters": {"plan_id": identifier},
            "resources": prepared["resources"],
            "workflow": {key: value for key, value in request.items() if key != "endpoint_key"},
            "image": request["image"],
            "estimate": estimate,
            "estimate_error": estimate_error,
            "steps": [
                {"id": i + 1, "title": command, "status": "pending"} for i, command in enumerate(prepared["commands"])
            ],
            "command": "\n".join(prepared["commands"]),
        }

    def revise(self, identifier, workflow):
        if not isinstance(workflow, dict):
            raise ValueError("Workflow must be an object")
        with self.lock:
            saved = self.app.controller.store.get_plan(identifier)
            if saved and saved["status"] != "proposed":
                raise ValueError("Plan already executed; prepare another plan")
            item = self.plans.get(identifier) or ((saved["expires"], saved["request"]) if saved else None)
            if not item:
                # Legacy chat plans predate durable storage; editing revalidates the full workflow.
                item = (0, {})
            request = json.loads(json.dumps(workflow))
            request.pop("endpoint_key", None)
            if request.get("kind") == "deployment":
                request["endpoint_key"] = item[1].get("endpoint_key") or secrets.token_urlsafe(32)
            revised = self.preview(request)
            replacement_id = revised["id"]
            self.plans[identifier] = self.plans.pop(replacement_id)
            self.app.controller.store.save_plan(replacement_id, {}, 0, "superseded")
            self.app.controller.store.save_plan(identifier, self.plans[identifier][1], self.plans[identifier][0])
            revised.update(id=identifier, parameters={"plan_id": identifier})
            return revised

    def execute(self, identifier):
        with self.lock:
            saved = self.app.controller.store.get_plan(identifier)
            item = self.plans.get(identifier) or ((saved["expires"], saved["request"]) if saved else None)
            if not item or item[0] <= time.time() or (saved and saved["status"] != "proposed"):
                raise ValueError("Plan expired or already executed; edit and save the plan to revalidate it")
            request = json.loads(json.dumps(item[1]))
            if request.get("kind") == "deployment":
                request.setdefault("endpoint_key", secrets.token_urlsafe(32))
            self.app.controller.store.save_plan(identifier, request, item[0], "executing")
            try:
                record = self.app.post("/api/jobs", request)
            except Exception:
                self.app.controller.store.save_plan(identifier, request, item[0])
                raise
            self.app.controller.store.save_plan(identifier, request, item[0], "started")
            self.plans.pop(identifier, None)
        return {"job_id": "modal-" + record["id"], "endpoint_key": request.get("endpoint_key")}

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
                    if event["type"] == "rollout_sample" and isinstance(event.get("sample"), dict):
                        job.samples.append(
                            {
                                **event["sample"],
                                "stage_index": event.get("index", 0),
                                "time": timestamp(event.get("time")),
                            }
                        )
                        job.samples = job.samples[-50:]
                    elif event["type"] != "metric":
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
