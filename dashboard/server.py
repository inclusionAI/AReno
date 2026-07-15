#!/usr/bin/env python3
"""Low-intrusion AReno dashboard API and static server."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_context import agent_system_prompt  # noqa: E402
from agent_files import AgentFileBrowser  # noqa: E402

from areno.cli.dashboard_registry import GLOBAL_REGISTRY_FILE  # noqa: E402
from areno.cli.diagnostics import collect_env, run_checks  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "dist"
STATE_FILE = ROOT / ".areno-dashboard-state.json"
DEFAULT_METRICS_LOG_DIR = "/tmp/areno/tfevent"
TIME_SEGMENT_ORDER = [
    "rollout",
    "make_sample",
    "reward",
    "old policy log probs",
    "actor log probs",
    "ref log probs",
    "value",
    "advantages",
    "sync weight",
    "train",
    "save",
    "other",
]
ENV_REPORT_CACHE: dict[str, Any] | None = None
ENV_CHECKS_CACHE: list[Any] | None = None
ENV_CHECK_COUNTS_CACHE: dict[str, int] | None = None
ENV_CACHE_LOCK = threading.Lock()
FILE_BROWSER = AgentFileBrowser(ROOT)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


class Job:
    def __init__(
        self,
        *,
        kind: str,
        name: str,
        command: list[str],
        config: dict[str, Any],
        metrics_dir: str | None,
        pid: int | None = None,
    ):
        self.id = uuid4().hex[:12]
        self.kind = kind
        self.name = name
        self.command = command
        self.config = config
        self.metrics_dir = metrics_dir
        self.status = "created"
        self.stage = "created"
        self.role = ""
        self.step = 0
        self.created_at = now()
        self.updated_at = self.created_at
        self.logs: list[str] = []
        self.metrics: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self.timeperf: list[dict[str, Any]] = []
        self.perf: dict[str, float] = {}
        self.launch_config: dict[str, Any] = dict(config)
        self.config_text = ""
        self.process: subprocess.Popen[str] | None = None
        self.pid = pid
        self.returncode: int | None = None
        self._metric_keys: set[tuple[str, int, float]] = set()
        self._timeperf_keys: set[int] = set()
        self._sample_keys: set[tuple[int, int, int]] = set()

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> Job:
        job = cls(
            kind=item.get("kind", "train"),
            name=item.get("name", "restored job"),
            command=list(item.get("command") or []),
            config=dict(item.get("launch") or item.get("config") or {}),
            metrics_dir=item.get("metrics_dir"),
            pid=item.get("pid"),
        )
        job.id = item.get("id") or job.id
        job.status = item.get("status", "unknown")
        job.stage = item.get("stage", "restored")
        job.role = item.get("role") or ""
        job.step = int(item.get("step") or 0)
        job.created_at = item.get("created_at") or job.created_at
        job.updated_at = item.get("updated_at") or job.updated_at
        job.returncode = item.get("returncode")
        job.logs = list(item.get("logs") or [])
        job.config = dict(item.get("config") or {})
        job.config_text = item.get("config_text") or ""
        job.launch_config = dict(item.get("launch") or {})
        return job

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "command": self.command,
            "config": self.config,
            "config_text": self.config_text,
            "launch": self.launch_config,
            "metrics_dir": self.metrics_dir,
            "status": self.status,
            "stage": self.stage,
            "role": self.role,
            "step": self.step,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "returncode": self.returncode,
            "pid": self.pid,
            "logs": self.logs[-300:],
            "metrics_count": len(self.metrics),
            "samples": self.samples[-50:],
            "timeperf": self.timeperf[-80:],
            "perf": self.perf,
        }

    def to_summary_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "metrics_dir": self.metrics_dir,
            "status": self.status,
            "stage": self.stage,
            "role": self.role,
            "step": self.step,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "returncode": self.returncode,
            "pid": self.pid,
            "perf": self.perf,
        }


class DashboardState:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self._load_state()

    def start(self, job: Job) -> Job:
        with self.lock:
            self.jobs[job.id] = job
            job.launch_config = dict(job.config)
            job.config = {}
            job.logs.append("$ " + " ".join(job.command))
            job.updated_at = now()
            self._save_state()
        try:
            job.process = subprocess.Popen(
                job.command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            with self.lock:
                job.status = "failed"
                job.logs.append(f"failed to start command: {exc}")
                job.updated_at = now()
                self._save_state()
            return job
        job.pid = job.process.pid
        job.status = "running"
        job.updated_at = now()
        job.logs.append(f"process started pid={job.pid}; metrics_dir={job.metrics_dir or 'disabled'}")
        self._save_state()
        threading.Thread(target=self._capture_output, args=(job,), daemon=True).start()
        threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return job

    def _append_log(self, job: Job, line: str) -> None:
        with self.lock:
            job.logs.append(line.rstrip("\n"))
            if len(job.logs) > 300:
                job.logs = job.logs[-300:]
            job.updated_at = now()

    def _capture_output(self, job: Job) -> None:
        process = job.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self._append_log(job, line)
        except Exception as exc:
            self._append_log(job, f"log capture failed: {exc}")

    def _watch(self, job: Job) -> None:
        assert job.process is not None
        code = job.process.wait()
        with self.lock:
            job.returncode = code
            if job.status == "running":
                job.status = "succeeded" if code == 0 else "failed"
            job.updated_at = now()
            self._load_metric_files(job)
            self._save_state()

    def _append_timeperf_row(
        self, job: Job, *, step: int, total: float, segments: dict[str, float], source: str = "metrics"
    ) -> None:
        if total <= 0:
            return
        if step in job._timeperf_keys:
            return
        job._timeperf_keys.add(step)
        ordered = sorted(
            [{"name": name, "seconds": max(value, 0.0)} for name, value in segments.items() if value > 0],
            key=lambda item: (
                TIME_SEGMENT_ORDER.index(item["name"])
                if item["name"] in TIME_SEGMENT_ORDER
                else len(TIME_SEGMENT_ORDER)
            ),
        )
        accounted = sum(item["seconds"] for item in ordered)
        if total > accounted:
            ordered.append({"name": "other", "seconds": total - accounted})
        rollout = sum(item["seconds"] for item in ordered if item["name"] == "rollout")
        train = sum(item["seconds"] for item in ordered if item["name"] == "train")
        other = max(total - rollout - train, 0.0)
        job.timeperf.append(
            {
                "step": step,
                "segments": ordered,
                "rollout_s": rollout,
                "train_s": train,
                "other_s": other,
                "total_s": total,
                "time": now(),
                "source": source,
            }
        )

    def _load_metric_files(self, job: Job) -> None:
        if not job.metrics_dir:
            return
        path = (ROOT / job.metrics_dir).resolve()
        if not path.exists() or not path.is_dir():
            return
        self._load_dashboard_state(job, path)
        self._load_tensorboard_scalars(job, path)
        self._load_rollout_samples(job, path)
        self._load_run_config(job, path)
        for file in sorted(path.glob("*.jsonl"))[-4:]:
            if file.name.startswith("rollout_samples"):
                continue
            try:
                for line in file.read_text(encoding="utf-8").splitlines()[-200:]:
                    item = json.loads(line)
                    if "name" in item and "value" in item:
                        self._add_metric(job, str(item["name"]), float(item["value"]), int(item.get("step", job.step)))
            except Exception:
                continue

    def _load_dashboard_state(self, job: Job, path: Path) -> None:
        state_file = dashboard_state_source(path, job_pid(job))
        if state_file is None:
            return
        try:
            payload = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        if job.pid is not None and int(payload.get("pid", job.pid)) != job.pid:
            return
        stage = payload.get("stage")
        if isinstance(stage, str) and stage:
            job.stage = stage
        role = payload.get("role")
        if isinstance(role, str):
            job.role = role
        try:
            job.step = max(job.step, int(payload.get("step", job.step)))
        except (TypeError, ValueError):
            pass
        status = payload.get("status")
        if isinstance(status, str) and job.status not in {"stopped", "failed", "succeeded", "exited"}:
            job.status = status
        job.updated_at = now()

    def _load_tensorboard_scalars(self, job: Job, path: Path) -> None:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except Exception:
            return
        job.timeperf = [row for row in job.timeperf if row.get("source") != "metrics"]
        job._timeperf_keys = {int(row.get("step", -1)) for row in job.timeperf}
        by_step: dict[int, dict[str, float]] = {}
        for accumulator_path in tensorboard_event_sources(path, job_pid(job)):
            try:
                accumulator = EventAccumulator(str(accumulator_path), size_guidance={"scalars": 10000})
                accumulator.Reload()
                tags = accumulator.Tags().get("scalars", [])
            except Exception:
                continue
            for tag in tags:
                try:
                    events = accumulator.Scalars(tag)[-500:]
                except Exception:
                    continue
                for event in events:
                    step = int(event.step)
                    value = float(event.value)
                    if math.isnan(value):
                        continue
                    self._add_metric(job, tag, value, step)
                    time_name = tensorboard_time_segment_name(tag)
                    if time_name:
                        by_step.setdefault(step, {})[time_name] = value
                    if tag in {"train/step_e2e_time_s", "time/total", "time/e2e"}:
                        by_step.setdefault(step, {})["total"] = value
                    elif tag == "train/step_rollout_time_s":
                        by_step.setdefault(step, {})["rollout"] = value
                    elif tag in {"train/step_train_time_s", "train/policy_train_wall_time_s"}:
                        by_step.setdefault(step, {})["train"] = value
        for step, values in sorted(by_step.items()):
            total = values.pop("total", None)
            if total is None:
                total = sum(value for value in values.values() if value > 0)
            self._append_timeperf_row(job, step=step, total=total, segments=values, source="metrics")

    def _load_rollout_samples(self, job: Job, path: Path) -> None:
        sample_files = rollout_sample_sources(path, job_pid(job))
        if not sample_files:
            return
        for sample_file in sample_files:
            try:
                lines = sample_file.read_text(encoding="utf-8").splitlines()[-100:]
            except Exception:
                continue
            for line in lines:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if job.pid is not None and item.get("pid") not in (None, job.pid):
                    continue
                step = int(item.get("step", 0))
                prompt_idx = int(item.get("prompt_idx", -1))
                sample_idx = int(item.get("sample_idx", -1))
                key = (step, prompt_idx, sample_idx)
                if key in job._sample_keys:
                    continue
                job._sample_keys.add(key)
                item["time"] = now()
                job.samples.append(item)
                job.step = max(job.step, step)

    def _load_run_config(self, job: Job, path: Path) -> None:
        text_file, json_file = run_config_sources(path, job_pid(job))
        if text_file.exists():
            try:
                job.config_text = text_file.read_text(encoding="utf-8")
            except Exception:
                pass
        if not json_file.exists():
            return
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(payload, dict):
            settings = payload.get("settings")
            if isinstance(settings, dict):
                job.config = settings
            if not job.config_text and isinstance(payload.get("summary_text"), str):
                job.config_text = payload["summary_text"]

    def _add_metric(self, job: Job, name: str, value: float, step: int) -> None:
        key = (name, step, value)
        if key in job._metric_keys:
            return
        job._metric_keys.add(key)
        job.metrics.append({"name": name, "value": value, "step": step, "time": now()})
        job.step = max(job.step, step)

    def stop(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            pid = job.pid
            process = job.process
            job.status = "stopped"
            job.updated_at = now()
            self._save_state()
        if process is not None and process.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
        elif pid is not None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                return False
        self._save_state()
        return True

    def list_jobs(self) -> list[dict[str, Any]]:
        self.scan_registered_jobs()
        with self.lock:
            for job in self.jobs.values():
                self._refresh_job_status(job)
            self._save_state()
            return [
                job.to_summary_json()
                for job in sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)
            ]

    def _refresh_job_status(self, job: Job) -> None:
        if job.process is not None:
            code = job.process.poll()
            if job.status == "stopped":
                if code is not None:
                    job.returncode = code
                return
            if code is None:
                job.status = "running"
            elif job.status == "running":
                job.returncode = code
                job.status = "succeeded" if code == 0 else "failed"
                job.updated_at = now()
            return
        # Registry jobs are refreshed in scan_registered_jobs.
        if job.pid is not None:
            return

    def get_job(self, job_id: str | None) -> Job | None:
        with self.lock:
            job = self.jobs.get(job_id) if job_id else next(iter(self.jobs.values()), None)
            if job is not None:
                self._refresh_job_status(job)
                self._load_metric_files(job)
                self._save_state()
            return job

    def metric_summaries(self, job_id: str | None) -> list[dict[str, Any]]:
        job = self.get_job(job_id)
        if job is None:
            return []
        grouped: dict[str, dict[str, Any]] = {}
        for point in job.metrics:
            name = str(point.get("name") or "")
            if not name:
                continue
            step = int(point.get("step") or 0)
            value = point.get("value")
            current = grouped.get(name)
            if current is None:
                grouped[name] = {"name": name, "count": 1, "latest_step": step, "latest_value": value}
                continue
            current["count"] += 1
            if step >= int(current.get("latest_step") or 0):
                current["latest_step"] = step
                current["latest_value"] = value
        return sorted(grouped.values(), key=lambda item: item["name"])

    def metric_series(self, job_id: str | None, metric_name: str, *, limit: int = 500) -> list[dict[str, Any]]:
        job = self.get_job(job_id)
        if job is None or not metric_name:
            return []
        points = [
            {
                "name": point.get("name"),
                "value": point.get("value"),
                "step": int(point.get("step") or 0),
                "time": point.get("time"),
            }
            for point in job.metrics
            if point.get("name") == metric_name and number_like(point.get("value"))
        ]
        points.sort(key=lambda point: int(point.get("step") or 0))
        return points[-max(1, min(limit, 5000)) :]

    def scan_registered_jobs(self) -> None:
        registry_jobs = registered_job_items()
        if not registry_jobs:
            return
        with self.lock:
            known_by_pid = {job.pid: job for job in self.jobs.values() if job.pid is not None}
            for item in registry_jobs:
                if not isinstance(item, dict):
                    continue
                try:
                    pid = int(item.get("pid"))
                except (TypeError, ValueError):
                    continue
                if pid == os.getpid():
                    continue
                command_parts = item.get("command") if isinstance(item.get("command"), list) else []
                command = " ".join(str(part) for part in command_parts)
                kind = str(item.get("kind") or detect_areno_job_kind(command) or "train")
                is_running = pid_is_running(pid)
                job = known_by_pid.get(pid)
                if job is None:
                    job = Job(
                        kind=kind,
                        name=str(item.get("name") or f"{kind} pid {pid}"),
                        command=command_parts or split_command(command),
                        config=dict(item.get("config") or {}),
                        metrics_dir=item.get("metrics_dir")
                        or parse_command_option(command, "--metrics-log-dir")
                        or DEFAULT_METRICS_LOG_DIR,
                        pid=pid,
                    )
                    job.launch_config = dict(job.config)
                    job.config = {}
                    job.status = "running" if is_running else "exited"
                    job.stage = "registered" if is_running else "exited"
                    job.logs.append("registered AReno command: " + command)
                    self.jobs[job.id] = job
                    known_by_pid[pid] = job
                else:
                    job.command = command_parts or job.command
                    job.metrics_dir = item.get("metrics_dir") or job.metrics_dir
                    if item.get("config") and not job.config:
                        job.config = dict(item.get("config") or {})
                    if item.get("config") and not job.launch_config:
                        job.launch_config = dict(item.get("config") or {})
                    if job.status != "stopped":
                        job.status = "running" if is_running else "exited"
                    job.updated_at = now()
            self._save_state()

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        for item in data.get("jobs", []):
            try:
                job = Job.from_json(item)
            except Exception:
                continue
            self.jobs[job.id] = job

    def _save_state(self) -> None:
        try:
            payload = {"jobs": [job.to_json() for job in self.jobs.values()]}
            tmp_file = STATE_FILE.with_suffix(".json.tmp")
            tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_file.replace(STATE_FILE)
        except Exception:
            pass


STATE = DashboardState()


def build_train_command(config: dict[str, Any]) -> list[str]:
    command = ["areno", "train"]
    pairs = {
        "--algo": config.get("algo"),
        "--ckpt": config.get("ckpt"),
        "--model-hub": config.get("model_hub"),
        "--dataset-path": config.get("dataset_path"),
        "--dataset-loader-fn": config.get("dataset_loader_fn"),
        "--reward-fn-path": config.get("reward_fn_path"),
        "--ref-ckpt": config.get("ref_ckpt"),
        "--reward-ckpt": config.get("reward_ckpt"),
        "--critic-ckpt": config.get("critic_ckpt"),
        "--agent-fn": config.get("agent_fn"),
        "--world-size": config.get("world_size"),
        "--tp-size": config.get("tp_size"),
        "--batch-size": config.get("batch_size"),
        "--n-samples": config.get("n_samples"),
        "--mini-bs": config.get("mini_bs"),
        "--score-micro-bs": config.get("score_micro_bs"),
        "--gradient-accumulation-steps": config.get("gradient_accumulation_steps"),
        "--epochs": config.get("epochs"),
        "--max-steps": config.get("max_steps"),
        "--max-prompt-tokens": config.get("max_prompt_tokens"),
        "--max-context-len": config.get("max_context_len"),
        "--max-new-tokens": config.get("max_new_tokens"),
        "--max-running-prompts": config.get("max_running_prompts"),
        "--temperature": config.get("temperature"),
        "--top-k": config.get("top_k"),
        "--top-p": config.get("top_p"),
        "--lr": config.get("lr"),
        "--min-lr": config.get("min_lr"),
        "--lr-decay-steps": config.get("lr_decay_steps"),
        "--lr-decay-style": config.get("lr_decay_style"),
        "--adam-beta1": config.get("adam_beta1"),
        "--adam-beta2": config.get("adam_beta2"),
        "--weight-decay": config.get("weight_decay"),
        "--grad-clip-norm": config.get("grad_clip_norm"),
        "--attn-backend": config.get("attn_backend"),
        "--agent-timeout-s": config.get("agent_timeout_s"),
        "--gspo-clip-eps": config.get("gspo_clip_eps"),
        "--grpo-clip-eps": config.get("grpo_clip_eps"),
        "--dpo-beta": config.get("dpo_beta"),
        "--critic-warmup-steps": config.get("critic_warmup_steps"),
        "--critic-lr": config.get("critic_lr"),
        "--kl-loss-coef": config.get("kl_loss_coef"),
        "--kl-loss-type": config.get("kl_loss_type"),
        "--clip-eps": config.get("clip_eps"),
        "--clip-ratio-c": config.get("clip_ratio_c"),
        "--value-clip-eps": config.get("value_clip_eps"),
        "--value-loss-coef": config.get("value_loss_coef"),
        "--gamma": config.get("gamma"),
        "--lam": config.get("lam"),
        "--save-path": config.get("save_path") or config.get("save_dir"),
        "--save-interval": config.get("save_interval"),
        "--metrics-log-dir": config.get("metrics_dir"),
        "--mem-frac": config.get("mem_frac"),
        "--tune-max-samples": config.get("tune_max_samples"),
    }
    for key, value in pairs.items():
        if value not in (None, ""):
            command.extend([key, str(value)])
    flags = {
        "--tune-params": config.get("tune_params"),
        "--greedy": config.get("greedy"),
        "--adam-8bit": config.get("adam_8bit"),
        "--drop-rollout-state": config.get("drop_rollout_state"),
        "--eager-decode": config.get("eager_decode"),
        "--disable-thinking": config.get("disable_thinking"),
        "--train-tool-results": config.get("train_tool_results"),
    }
    activation_checkpointing = config.get("activation_checkpointing")
    if activation_checkpointing not in (None, ""):
        command.append(
            "--activation-checkpointing" if bool_like(activation_checkpointing) else "--no-activation-checkpointing"
        )
    use_kl_loss = config.get("use_kl_loss")
    if use_kl_loss not in (None, ""):
        command.append("--use-kl-loss" if bool_like(use_kl_loss) else "--no-use-kl-loss")
    for key, value in flags.items():
        if bool_like(value):
            command.append(key)
    command.extend(str(config.get("extra_args") or "").split())
    return command


def build_smoke_train_command(config: dict[str, Any]) -> list[str]:
    command = build_train_command(config)
    if "--smoke-train" not in command:
        command.append("--smoke-train")
    return command


def build_smoke_infer_command(config: dict[str, Any]) -> list[str]:
    command = build_train_command(config)
    if "--smoke-infer" not in command:
        command.append("--smoke-infer")
    return command


def bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_serve_command(config: dict[str, Any]) -> list[str]:
    command = ["areno", "serve"]
    model_path = config.get("model_path") or config.get("ckpt")
    pairs = {
        "--model-path": model_path,
        "--model-hub": config.get("model_hub"),
        "--host": config.get("host"),
        "--port": config.get("port"),
        "--world-size": config.get("world_size"),
        "--tp-size": config.get("tp_size"),
        "--max-running-prompts": config.get("max_running_prompts"),
        "--default-max-tokens": config.get("default_max_tokens"),
        "--decode-progress-interval-s": config.get("decode_progress_interval_s"),
        "--attn-backend": config.get("attn_backend"),
    }
    for key, value in pairs.items():
        if value not in (None, ""):
            command.extend([key, str(value)])
    if bool_like(config.get("eager_decode")):
        command.append("--eager-decode")
    if bool_like(config.get("disable_thinking")):
        command.append("--disable-thinking")
    command.extend(str(config.get("extra_args") or "").split())
    return command


def detect_areno_job_kind(command: str) -> str | None:
    """Detect train/serve AReno commands started outside the dashboard server."""

    parts = split_command(command)
    if not parts:
        return None
    names = [Path(part).name for part in parts]
    if "dashboard" in parts or "dashboard" in names:
        return None

    for index, name in enumerate(names[:-1]):
        if name == "areno" and parts[index + 1] in {"train", "serve"}:
            return parts[index + 1]

    for index, part in enumerate(parts[:-2]):
        if part == "-m" and parts[index + 1] == "areno.cli.main" and parts[index + 2] in {"train", "serve"}:
            return parts[index + 2]

    normalized = " ".join(command.split())
    # Last-resort fallback for shell wrapper commands that are hard to split
    # faithfully, e.g. `sh -c 'areno train ...'`.
    if re.search(r"(^|[/\s])areno\s+train(\s|$)", normalized):
        return "train"
    if re.search(r"(^|[/\s])areno\s+serve(\s|$)", normalized):
        return "serve"
    return None


def split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def parse_command_option(command: str, option: str) -> str | None:
    parts = split_command(command)
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(option + "="):
            return part.split("=", 1)[1]
    return None


def registered_job_items() -> list[dict[str, Any]]:
    items_by_pid: dict[int, dict[str, Any]] = {}
    try:
        data = json.loads(GLOBAL_REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, list):
        return []
    for item in jobs:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("pid"))
        except (TypeError, ValueError):
            continue
        old = items_by_pid.get(pid)
        if old is None or float(item.get("updated_at") or 0) >= float(old.get("updated_at") or 0):
            items_by_pid[pid] = item
    return sorted(items_by_pid.values(), key=lambda item: float(item.get("created_at") or 0), reverse=True)


def job_pid(job: Job) -> int | None:
    return job.pid


def number_like(value: Any) -> bool:
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def tensorboard_event_sources(path: Path, pid: int | None) -> list[Path]:
    event_files = sorted(path.rglob("events.out.tfevents.*"), key=lambda item: item.stat().st_mtime)
    if not event_files:
        return [path]
    if pid is None:
        return event_files
    pid_marker = f".{pid}."
    matched = [file for file in event_files if pid_marker in file.name or file.parent.name == f"pid-{pid}"]
    return matched


def rollout_sample_sources(path: Path, pid: int | None) -> list[Path]:
    if pid is not None:
        candidates = [
            path / f"rollout_samples.{pid}.jsonl",
            path / f"pid-{pid}" / "rollout_samples.jsonl",
        ]
        return [file for file in candidates if file.exists()]
    legacy = path / "rollout_samples.jsonl"
    return [legacy] if legacy.exists() else sorted(path.glob("rollout_samples.*.jsonl"))


def dashboard_state_source(path: Path, pid: int | None) -> Path | None:
    if pid is not None:
        state_file = path / f"dashboard_state.{pid}.json"
        return state_file if state_file.exists() else None
    candidates = sorted(path.glob("dashboard_state.*.json"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_config_sources(path: Path, pid: int | None) -> tuple[Path, Path]:
    if pid is not None:
        text_file = path / f"areno_run_config.{pid}.txt"
        json_file = path / f"areno_run_config.{pid}.json"
        return text_file, json_file
    return path / "areno_run_config.txt", path / "areno_run_config.json"


def tensorboard_time_segment_name(tag: str) -> str | None:
    if tag.startswith("time/"):
        return normalize_time_segment_name(tag.split("/", 1)[1])
    if tag.startswith("train/"):
        leaf = tag.split("/", 1)[1]
        if leaf in {"step_e2e_time_s", "step_rollout_time_s", "step_train_time_s", "policy_train_wall_time_s"}:
            return normalize_time_segment_name(leaf)
        if leaf.endswith("_time_s"):
            return normalize_time_segment_name(leaf)
    return None


def normalize_time_segment_name(name: str) -> str | None:
    normalized = name.replace("_time_s", "").replace("step_", "").replace("_", " ").replace("-", " ").lower()
    if normalized in {"e2e", "total", "wall"}:
        return None
    if normalized in {"policy train wall", "policy train wall time"}:
        return "train"
    if normalized in {"train"}:
        return "train"
    if normalized in {"rollout"}:
        return "rollout"
    if normalized in {"reward", "calc reward", "calculate reward"}:
        return "reward"
    if normalized in {"advantage", "advantages"}:
        return "advantages"
    if normalized in {"sync weight", "sync weights"}:
        return "sync weight"
    if normalized in {"make sample", "sample"}:
        return "make_sample"
    if normalized in {"value", "critic", "value model"}:
        return "value"
    if normalized in {"save", "checkpoint"}:
        return "save"
    if "ref" in normalized and "logprob" in normalized:
        return "ref log probs"
    if "actor" in normalized and "logprob" in normalized:
        return "actor log probs"
    if "old" in normalized and "logprob" in normalized:
        return "old policy log probs"
    if "logprob" in normalized:
        return normalized.replace("logprob", "log probs")
    return normalized or None


def runtime_env() -> dict[str, Any]:
    repo = {
        "branch": run_text(["git", "branch", "--show-current"]).strip(),
        "commit": run_text(["git", "rev-parse", "--short", "HEAD"]).strip(),
    }
    report, check_items, check_counts = cached_env_checks()
    gpus = []
    try:
        output = run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        for line in output.splitlines():
            name, used, total, util = [part.strip() for part in line.split(",")]
            gpus.append(
                {"name": name, "memory_used_mb": int(used), "memory_total_mb": int(total), "utilization": int(util)}
            )
    except Exception:
        pass
    if gpus:
        gpu_summary = " / ".join(
            f"{gpu['name']} {gpu['memory_used_mb']}/{gpu['memory_total_mb']}MB" for gpu in gpus[:2]
        )
    else:
        gpu_summary = "not detected"
    return {
        "repo": repo,
        "report": report,
        "checks": check_items,
        "check_counts": check_counts,
        "ready": check_counts["fail"] == 0,
        "gpus": gpus,
        "gpu_summary": gpu_summary,
        "python": sys.version.split()[0],
        "cwd": str(ROOT),
    }


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return ""


def cached_env_checks() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    global ENV_REPORT_CACHE, ENV_CHECKS_CACHE, ENV_CHECK_COUNTS_CACHE
    with ENV_CACHE_LOCK:
        if ENV_REPORT_CACHE is None or ENV_CHECKS_CACHE is None or ENV_CHECK_COUNTS_CACHE is None:
            report = collect_env()
            checks = run_checks(report)
            ENV_REPORT_CACHE = report
            ENV_CHECKS_CACHE = [
                {"status": item.status, "name": item.name, "detail": item.detail, "next_step": item.next_step}
                for item in checks
            ]
            ENV_CHECK_COUNTS_CACHE = {
                "ok": sum(1 for item in checks if item.status == "OK"),
                "warn": sum(1 for item in checks if item.status == "WARN"),
                "fail": sum(1 for item in checks if item.status == "FAIL"),
            }
        return ENV_REPORT_CACHE, ENV_CHECKS_CACHE, ENV_CHECK_COUNTS_CACHE


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def build_agent_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    job = STATE.get_job(payload.get("job_id"))
    context = job.to_summary_json() if job else {}
    messages: list[dict[str, Any]] = [{"role": "system", "content": agent_system_prompt()}]
    for item in normalize_agent_history(payload.get("history")):
        messages.append(item)
    messages.append(
        {
            "role": "user",
            "content": json.dumps({"prompt": payload.get("prompt", ""), "job": context}, ensure_ascii=False),
        }
    )
    return messages


def normalize_agent_history(raw_history: Any) -> list[dict[str, str]]:
    if not isinstance(raw_history, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw_history[-10:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content[:8000]})
    return normalized


def agent_response(payload: dict[str, Any]) -> dict[str, Any]:
    provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    base_url = str(provider.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
    api_key = str(provider.get("api_key") or os.environ.get("OPENAI_API_KEY", ""))
    model = str(provider.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    if not base_url or not api_key:
        return {
            "content": "Configure base URL, API key, and model before asking the agent.",
            "tool_calls": [],
        }
    messages = build_agent_messages(payload)
    all_tool_calls: list[dict[str, Any]] = []
    all_tool_results: list[dict[str, Any]] = []
    try:
        for _ in range(6):
            body = {"model": model, "messages": messages, "tools": agent_tool_schemas(), "tool_choice": "auto"}
            message = post_chat_completion(base_url, api_key, body)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return {
                    "content": message.get("content") or "",
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                }
            assistant_message = {
                "role": "assistant",
                "content": message.get("content") or None,
                "tool_calls": tool_calls,
            }
            if message.get("reasoning_content"):
                assistant_message["reasoning_content"] = message.get("reasoning_content")
            messages.append(assistant_message)
            all_tool_calls.extend(tool_calls)
            for tool_call in tool_calls:
                result = execute_agent_tool(tool_call)
                all_tool_results.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": (tool_call.get("function") or {}).get("name"),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        return {
            "content": "Agent stopped after too many tool calls. Inspect the tool results and retry with a narrower request.",
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
        }
    except urllib.error.HTTPError as exc:
        return {
            "content": f"Agent request failed: HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}",
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
        }
    except Exception as exc:
        return {
            "content": f"Agent request failed: {exc}",
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
        }


def agent_event_stream(payload: dict[str, Any]):
    provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    base_url = str(provider.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
    api_key = str(provider.get("api_key") or os.environ.get("OPENAI_API_KEY", ""))
    model = str(provider.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    if not base_url or not api_key:
        yield {"type": "error", "content": "Configure base URL, API key, and model before asking the agent."}
        yield {"type": "done"}
        return
    messages = build_agent_messages(payload)
    try:
        for round_index in range(6):
            body = {
                "model": model,
                "messages": messages,
                "tools": agent_tool_schemas(),
                "tool_choice": "auto",
                "stream": True,
            }
            message = yield from stream_chat_completion(base_url, api_key, body, round_index=round_index)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                yield {"type": "done", "content": message.get("content") or ""}
                return
            assistant_message = {
                "role": "assistant",
                "content": message.get("content") or None,
                "tool_calls": tool_calls,
            }
            if message.get("reasoning_content"):
                assistant_message["reasoning_content"] = message.get("reasoning_content")
            messages.append(assistant_message)
            yield {"type": "tool_calls", "tool_calls": tool_calls}
            for tool_call in tool_calls:
                result = execute_agent_tool(tool_call)
                yield {"type": "tool_result", "tool_result": result}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": (tool_call.get("function") or {}).get("name"),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        yield {
            "type": "error",
            "content": "Agent stopped after too many tool calls. Retry with a narrower request.",
        }
        yield {"type": "done"}
    except urllib.error.HTTPError as exc:
        yield {"type": "error", "content": f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"}
        yield {"type": "done"}
    except Exception as exc:
        yield {"type": "error", "content": str(exc)}
        yield {"type": "done"}


def stream_chat_completion(base_url: str, api_key: str, body: dict[str, Any], *, round_index: int = 0):
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_text = line.removeprefix("data:").strip()
            if data_text == "[DONE]":
                break
            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                continue
            delta = (data.get("choices") or [{}])[0].get("delta") or {}
            message = (data.get("choices") or [{}])[0].get("message") or {}
            content_delta = delta.get("content")
            if content_delta:
                content_parts.append(content_delta)
                yield {"type": "content_delta", "content": content_delta}
            reasoning_delta = delta.get("reasoning_content")
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
                yield {"type": "reasoning_delta", "content": reasoning_delta}
            for chunk in delta.get("tool_calls") or []:
                index = int(chunk.get("index", 0))
                current = tool_calls_by_index.setdefault(
                    index,
                    {
                        "index": index,
                        "round": round_index,
                        "id": chunk.get("id"),
                        "type": chunk.get("type") or "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if chunk.get("id"):
                    current["id"] = chunk["id"]
                if chunk.get("type"):
                    current["type"] = chunk["type"]
                fn = chunk.get("function") or {}
                if fn.get("name"):
                    current["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    current["function"]["arguments"] += fn["arguments"]
                yield {"type": "tool_call_delta", "tool_call": current}
            for chunk in message.get("tool_calls") or []:
                index = int(chunk.get("index", len(tool_calls_by_index)))
                current = {
                    "index": index,
                    "round": round_index,
                    "id": chunk.get("id"),
                    "type": chunk.get("type") or "function",
                    "function": chunk.get("function") or {"name": "", "arguments": ""},
                }
                tool_calls_by_index[index] = current
                yield {"type": "tool_call_delta", "tool_call": current}
    tool_calls = [tool_calls_by_index[index] for index in sorted(tool_calls_by_index)]
    return {"content": "".join(content_parts), "reasoning_content": "".join(reasoning_parts), "tool_calls": tool_calls}


def post_chat_completion(base_url: str, api_key: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]


def agent_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_folder",
                "description": "List files and folders under the current repository. Read-only.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cd",
                "description": "Change the agent's repository-relative working directory. Read-only navigation.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file from the repository with optional line window. Read-only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "max_lines": {"type": "integer", "minimum": 1, "maximum": 500},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rg",
                "description": "Search repository text with ripgrep. Read-only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "max_matches": {"type": "integer", "minimum": 1, "maximum": 500},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_areno_path",
                "description": "Return the current AReno repository root, agent cwd, package path, and examples path.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_jobs",
                "description": "List registered AReno train and serve jobs.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_job",
                "description": "Get detailed job info, rollout samples, config, logs, and metric names. Use fetch_metric for scalar series.",
                "parameters": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_metric",
                "description": "Fetch one scalar metric series for a job by metric name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "metric": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
                    },
                    "required": ["job_id", "metric"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stop_job",
                "description": "Stop a running AReno job by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "smoke_train",
                "description": "Start an AReno train smoke job using --smoke-train with launcher-style config.",
                "parameters": {
                    "type": "object",
                    "properties": {"config": {"type": "object", "additionalProperties": True}},
                    "required": ["config"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "smoke_infer",
                "description": "Start an AReno inference smoke job using --smoke-infer with launcher-style train config.",
                "parameters": {
                    "type": "object",
                    "properties": {"config": {"type": "object", "additionalProperties": True}},
                    "required": ["config"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_train",
                "description": "Start an AReno train job with the same config fields as the launcher form.",
                "parameters": {
                    "type": "object",
                    "properties": {"config": {"type": "object", "additionalProperties": True}},
                    "required": ["config"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_serve",
                "description": "Start an AReno serve job. The serve config uses model_path, not ckpt.",
                "parameters": {
                    "type": "object",
                    "properties": {"config": {"type": "object", "additionalProperties": True}},
                    "required": ["config"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_runtime_env",
                "description": "Inspect runtime env, checks, repo, and GPU utilization.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]


def execute_agent_tool(tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function") or {}
    name = str(fn.get("name") or "")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    try:
        if name == "list_folder":
            return {"name": name, "ok": True, **FILE_BROWSER.list_folder(args.get("path"))}
        if name == "cd":
            return {"name": name, "ok": True, **FILE_BROWSER.cd(str(args.get("path") or ""))}
        if name == "read_file":
            return {
                "name": name,
                "ok": True,
                **FILE_BROWSER.read_file(
                    str(args.get("path") or ""),
                    start_line=int(args.get("start_line") or 1),
                    max_lines=int(args.get("max_lines") or 200),
                ),
            }
        if name == "rg":
            return {
                "name": name,
                "ok": True,
                **FILE_BROWSER.rg(
                    str(args.get("pattern") or ""),
                    path=args.get("path"),
                    max_matches=int(args.get("max_matches") or 100),
                ),
            }
        if name == "get_areno_path":
            return {"name": name, "ok": True, **FILE_BROWSER.path_info()}
        if name == "list_jobs":
            return {"name": name, "ok": True, "jobs": STATE.list_jobs()}
        if name == "get_job":
            job = STATE.get_job(args.get("job_id"))
            return {
                "name": name,
                "ok": job is not None,
                "job": job.to_json() if job else None,
                "metrics": STATE.metric_summaries(args.get("job_id")) if job else [],
            }
        if name == "fetch_metric":
            metric_name = str(args.get("metric") or "")
            limit = int(args.get("limit") or 500)
            return {
                "name": name,
                "ok": True,
                "metric": metric_name,
                "points": STATE.metric_series(args.get("job_id"), metric_name, limit=limit),
            }
        if name == "stop_job":
            return {"name": name, "ok": STATE.stop(str(args.get("job_id") or ""))}
        if name == "start_train":
            config = args.get("config") if isinstance(args.get("config"), dict) else {}
            job = Job(
                kind="train",
                name=f"train {config.get('algo', 'sft')} {config.get('ckpt', '')}",
                command=build_train_command(config),
                config=config,
                metrics_dir=config.get("metrics_dir"),
            )
            return {"name": name, "ok": True, "job": STATE.start(job).to_summary_json()}
        if name == "smoke_train":
            config = args.get("config") if isinstance(args.get("config"), dict) else {}
            job = Job(
                kind="train",
                name=f"smoke train {config.get('algo', 'sft')} {config.get('ckpt', '')}",
                command=build_smoke_train_command(config),
                config=config,
                metrics_dir=config.get("metrics_dir"),
            )
            return {"name": name, "ok": True, "job": STATE.start(job).to_summary_json()}
        if name == "smoke_infer":
            config = args.get("config") if isinstance(args.get("config"), dict) else {}
            job = Job(
                kind="train",
                name=f"smoke infer {config.get('algo', 'sft')} {config.get('ckpt', '')}",
                command=build_smoke_infer_command(config),
                config=config,
                metrics_dir=config.get("metrics_dir"),
            )
            return {"name": name, "ok": True, "job": STATE.start(job).to_summary_json()}
        if name == "start_serve":
            config = args.get("config") if isinstance(args.get("config"), dict) else {}
            model_path = config.get("model_path") or config.get("ckpt", "")
            job = Job(
                kind="serve",
                name=f"serve {model_path}",
                command=build_serve_command(config),
                config=config,
                metrics_dir=None,
            )
            return {"name": name, "ok": True, "job": STATE.start(job).to_summary_json()}
        if name == "get_runtime_env":
            return {"name": name, "ok": True, "env": runtime_env()}
        return {"name": name, "ok": False, "error": "unknown tool"}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"{self.log_date_time_string()} - {fmt % args}\n")

    def do_GET(self) -> None:
        try:
            path = self.route_path()
            if path == "/api/env":
                self.json(runtime_env())
            elif path == "/api/jobs":
                self.json({"jobs": STATE.list_jobs()})
            elif path.startswith("/api/jobs/") and path.endswith("/metrics"):
                job_id = path.split("/")[-2]
                self.json({"metrics": STATE.metric_summaries(job_id)})
            elif path.startswith("/api/jobs/") and path.endswith("/metric"):
                job_id = path.split("/")[-2]
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                metric_name = query.get("name", [""])[0]
                limit = int(query.get("limit", ["500"])[0] or 500)
                self.json({"metric": metric_name, "points": STATE.metric_series(job_id, metric_name, limit=limit)})
            elif path.startswith("/api/jobs/"):
                job = STATE.get_job(path.split("/")[-1])
                if not job:
                    self.error("job not found", HTTPStatus.NOT_FOUND)
                else:
                    self.json({"job": job.to_json()})
            else:
                self.static(path)
        except Exception as exc:
            self.error(str(exc))

    def do_POST(self) -> None:
        try:
            path = self.route_path()
            payload = self.read_json()
            if path == "/api/jobs/train":
                job = Job(
                    kind="train",
                    name=f"train {payload.get('algo', 'sft')} {payload.get('ckpt', '')}",
                    command=build_train_command(payload),
                    config=payload,
                    metrics_dir=payload.get("metrics_dir"),
                )
                self.json({"job": STATE.start(job).to_json()})
            elif path == "/api/jobs/serve":
                model_path = payload.get("model_path") or payload.get("ckpt", "")
                job = Job(
                    kind="serve",
                    name=f"serve {model_path}",
                    command=build_serve_command(payload),
                    config=payload,
                    metrics_dir=None,
                )
                self.json({"job": STATE.start(job).to_json()})
            elif path.startswith("/api/jobs/") and path.endswith("/stop"):
                self.json({"stopped": STATE.stop(path.split("/")[-2])})
            elif path == "/api/agent/stream":
                self.stream_jsonl(agent_event_stream(payload))
            elif path == "/api/agent":
                self.json({"response": agent_response(payload)})
            else:
                self.error("not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.error(str(exc))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def route_path(self) -> str:
        path = self.path.split("?", 1)[0]
        marker = "/api/"
        if marker in path:
            return path[path.index(marker) :]
        if path.endswith("/api"):
            return "/api"
        return path

    def json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def stream_jsonl(self, events) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for event in events:
            chunk = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            self.wfile.write(chunk)
            self.wfile.flush()

    def error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self.json({"error": message}, status)

    def static(self, path: str) -> None:
        if "/assets/" in path:
            path = path[path.index("/assets/") :]
        elif path in {"", "/"} or not Path(path).suffix:
            path = "/index.html"
        target = (STATIC_DIR / path.lstrip("/")).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists():
            target = STATIC_DIR / "index.html"
        if not target.exists():
            self.error("dashboard static build not found; run pnpm --dir dashboard build", HTTPStatus.NOT_FOUND)
            return
        data = target.read_bytes()
        content_type = "text/html" if target.suffix == ".html" else "application/octet-stream"
        if target.suffix == ".js":
            content_type = "text/javascript"
        elif target.suffix == ".css":
            content_type = "text/css"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AReno dashboard server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AReno dashboard API listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
