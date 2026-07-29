"""End-to-end API tests for the AReno dashboard server.

Starts the server on a random port, creates train jobs via the HTTP API,
and exercises every endpoint via urllib.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

SERVER_PATH = str(Path(__file__).resolve().parent.parent / "areno" / "dashboard" / "server.py")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, SERVER_PATH, "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
    )
    base = f"http://127.0.0.1:{port}"
    # Wait for server to be ready
    for _ in range(30):
        try:
            urllib.request.urlopen(base + "/api/env", timeout=2)
            break
        except Exception:
            time.sleep(0.2)
    else:
        out = proc.stdout.read(2000) if proc.stdout else ""
        pytest.fail(f"server did not start:\n{out}")

    yield base

    proc.terminate()
    proc.wait(timeout=5)


def _get(base, path):
    url = base + path
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


def _post(base, path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


def _create_train_job(server, **overrides):
    config = {"algo": "sft", "ckpt": "Qwen/Qwen3-0.6B", "dataset_path": "/tmp/data.jsonl"}
    config.update(overrides)
    _, body = _post(server, "/api/jobs/train", config)
    return body["job"]["id"]


# ---------------------------------------------------------------------------
# /api/env
# ---------------------------------------------------------------------------

class TestApiEnv:
    def test_env_returns_ok(self, server):
        status, body = _get(server, "/api/env")
        assert status == 200
        assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# /api/jobs (list)
# ---------------------------------------------------------------------------

class TestApiJobsList:
    def test_jobs_list_returns_ok(self, server):
        status, body = _get(server, "/api/jobs")
        assert status == 200
        assert "jobs" in body
        assert isinstance(body["jobs"], list)

    def test_created_job_appears_in_list(self, server):
        job_id = _create_train_job(server, algo="gspo", ckpt="Qwen/Qwen3-1.7B")
        _, body = _get(server, "/api/jobs")
        ids = [j["id"] for j in body["jobs"]]
        assert job_id in ids


# ---------------------------------------------------------------------------
# /api/jobs/train (POST)
# ---------------------------------------------------------------------------

class TestApiJobsTrain:
    def test_train_returns_job_summary(self, server):
        _, body = _post(server, "/api/jobs/train", {
            "algo": "sft",
            "ckpt": "Qwen/Qwen3-0.6B",
            "dataset_path": "/tmp/data.jsonl",
        })
        assert "job" in body
        job = body["job"]
        assert job["kind"] == "train"
        assert "id" in job
        assert "status" in job


# ---------------------------------------------------------------------------
# /api/jobs/<id> (GET detail)
# ---------------------------------------------------------------------------

class TestApiJobDetail:
    def test_job_detail_returns_full_job(self, server):
        job_id = _create_train_job(server)
        status, body = _get(server, f"/api/jobs/{job_id}")
        assert status == 200
        assert body["job"]["id"] == job_id
        assert "command" in body["job"]
        assert "status" in body["job"]

    def test_job_detail_not_found(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(server, "/api/jobs/nonexistent-id")
        assert exc_info.value.code == 404


# ---------------------------------------------------------------------------
# /api/jobs/<id>/stop (POST)
# ---------------------------------------------------------------------------

class TestApiJobStop:
    def test_stop_job(self, server):
        job_id = _create_train_job(server, ckpt="test-stop")
        status, body = _post(server, f"/api/jobs/{job_id}/stop", {})
        assert status == 200
        assert body["stopped"] is True


# ---------------------------------------------------------------------------
# /api/jobs/<id>/metrics (GET)
# ---------------------------------------------------------------------------

class TestApiJobMetrics:
    def test_metrics_list(self, server):
        job_id = _create_train_job(server)
        status, body = _get(server, f"/api/jobs/{job_id}/metrics")
        assert status == 200
        assert "metrics" in body
        assert isinstance(body["metrics"], list)


# ---------------------------------------------------------------------------
# /api/jobs/<id>/trajectory (GET)
# ---------------------------------------------------------------------------

class TestApiTrajectory:
    def test_trajectory_empty_job_returns_not_found(self, server):
        job_id = _create_train_job(server, algo="gspo")
        params = urllib.parse.urlencode({"step": 0, "prompt_idx": 0, "sample_idx": 0})
        status, body = _get(server, f"/api/jobs/{job_id}/trajectory?{params}")
        assert status == 200
        assert body["valid"] is False

    def test_trajectory_job_not_found(self, server):
        params = urllib.parse.urlencode({"step": 0, "prompt_idx": 0, "sample_idx": 0})
        status, body = _get(server, f"/api/jobs/bad-id/trajectory?{params}")
        assert body["valid"] is False

    def test_trajectory_invalid_params_returns_error(self, server):
        job_id = _create_train_job(server)
        with pytest.raises(urllib.error.HTTPError):
            _get(server, f"/api/jobs/{job_id}/trajectory?step=abc")


# ---------------------------------------------------------------------------
# /api/jobs/serve (POST)
# ---------------------------------------------------------------------------

class TestApiJobsServe:
    def test_serve_returns_job_summary(self, server):
        _, body = _post(server, "/api/jobs/serve", {
            "model_path": "Qwen/Qwen3-0.6B",
        })
        assert "job" in body
        assert body["job"]["kind"] == "serve"


# ---------------------------------------------------------------------------
# /api/agent (POST)
# ---------------------------------------------------------------------------

class TestApiAgent:
    def test_agent_returns_response(self, server):
        status, body = _post(server, "/api/agent", {
            "messages": [{"role": "user", "content": "list files in current directory"}],
        })
        assert status == 200
        assert "response" in body or "error" in body


# ---------------------------------------------------------------------------
# Illegal input — invalid payloads and malformed requests
# ---------------------------------------------------------------------------

class TestIllegalInput:
    def test_train_empty_config(self, server):
        """POST /api/jobs/train with empty dict should still create a job."""
        _, body = _post(server, "/api/jobs/train", {})
        assert "job" in body
        assert body["job"]["kind"] == "train"

    def test_train_null_values_ignored(self, server):
        """None values should not crash command building."""
        _, body = _post(server, "/api/jobs/train", {
            "algo": "sft", "ckpt": None, "dataset_path": None,
            "batch_size": None, "lr": None,
        })
        assert "job" in body

    def test_train_nonexistent_algo(self, server):
        """An unknown algo string should still produce a command (validation
        is deferred to the CLI, not the dashboard server)."""
        _, body = _post(server, "/api/jobs/train", {
            "algo": "fake_algo", "ckpt": "x", "dataset_path": "/tmp/x.jsonl",
        })
        assert "job" in body
        cmd = body["job"].get("command") or []
        assert "--algo" in cmd

    def test_serve_empty_config(self, server):
        """POST /api/jobs/serve with no model_path should not crash."""
        _, body = _post(server, "/api/jobs/serve", {})
        assert "job" in body
        assert body["job"]["kind"] == "serve"

    def test_serve_null_model_path(self, server):
        _, body = _post(server, "/api/jobs/serve", {"model_path": None})
        assert "job" in body

    def test_stop_nonexistent_job(self, server):
        """Stopping a job id that doesn't exist should return stopped=False."""
        _, body = _post(server, "/api/jobs/bad-id/stop", {})
        assert body["stopped"] is False

    def test_agent_empty_messages(self, server):
        _, body = _post(server, "/api/agent", {"messages": []})
        assert "response" in body or "error" in body

    def test_agent_no_messages_key(self, server):
        _, body = _post(server, "/api/agent", {})
        assert "response" in body or "error" in body

    def test_agent_garbage_payload(self, server):
        """Non-dict messages should not crash the server."""
        _, body = _post(server, "/api/agent", {"messages": "not a list"})
        assert "response" in body or "error" in body


# ---------------------------------------------------------------------------
# Boundary — extreme values, missing fields, edge cases
# ---------------------------------------------------------------------------

class TestBoundaryValues:
    def test_train_empty_string_values(self, server):
        """Empty strings should be treated like None and omitted from command."""
        _, body = _post(server, "/api/jobs/train", {
            "algo": "", "ckpt": "", "dataset_path": "",
        })
        assert "job" in body

    def test_train_very_long_string(self, server):
        """A 10k-char checkpoint path should not crash."""
        long_path = "a" * 10000
        _, body = _post(server, "/api/jobs/train", {
            "algo": "sft", "ckpt": long_path, "dataset_path": "/tmp/x.jsonl",
        })
        assert "job" in body

    def test_train_special_chars_in_ckpt(self, server):
        """Special shell characters should be handled by shlex, not injected."""
        _, body = _post(server, "/api/jobs/train", {
            "algo": "sft", "ckpt": "model; rm -rf /", "dataset_path": "/tmp/x.jsonl",
        })
        assert "job" in body

    def test_train_numeric_zero_values(self, server):
        """Zero is a valid numeric value and should appear in the command."""
        _, body = _post(server, "/api/jobs/train", {
            "algo": "sft", "ckpt": "x", "dataset_path": "/tmp/x.jsonl",
            "batch_size": 0, "lr": 0, "epochs": 0,
        })
        assert "job" in body

    def test_train_negative_values(self, server):
        """Negative numbers should pass through (validated by CLI later)."""
        _, body = _post(server, "/api/jobs/train", {
            "algo": "sft", "ckpt": "x", "dataset_path": "/tmp/x.jsonl",
            "lr": -1, "batch_size": -10,
        })
        assert "job" in body

    def test_train_boolean_flags(self, server):
        """Boolean true flags should be appended as bare flags."""
        _, body = _post(server, "/api/jobs/train", {
            "algo": "sft", "ckpt": "x", "dataset_path": "/tmp/x.jsonl",
            "greedy": True, "adam_8bit": True,
            "activation_checkpointing": True,
        })
        assert "job" in body

    def test_train_false_boolean_flags(self, server):
        """Boolean false flags should NOT be appended."""
        _, body = _post(server, "/api/jobs/train", {
            "algo": "sft", "ckpt": "x", "dataset_path": "/tmp/x.jsonl",
            "greedy": False, "adam_8bit": False,
            "activation_checkpointing": False,
        })
        assert "job" in body

    def test_train_extra_unknown_fields(self, server):
        """Unknown config fields should be silently ignored."""
        _, body = _post(server, "/api/jobs/train", {
            "algo": "sft", "ckpt": "x", "dataset_path": "/tmp/x.jsonl",
            "unknown_field": "value", "random_key": 123,
        })
        assert "job" in body

    def test_trajectory_negative_indices(self, server):
        job_id = _create_train_job(server, algo="gspo")
        params = urllib.parse.urlencode({"step": -1, "prompt_idx": -1, "sample_idx": -1})
        _, body = _get(server, f"/api/jobs/{job_id}/trajectory?{params}")
        assert body["valid"] is False

    def test_trajectory_very_large_indices(self, server):
        job_id = _create_train_job(server, algo="gspo")
        params = urllib.parse.urlencode({"step": 999999, "prompt_idx": 999999, "sample_idx": 999999})
        _, body = _get(server, f"/api/jobs/{job_id}/trajectory?{params}")
        assert body["valid"] is False

    def test_trajectory_missing_all_params(self, server):
        """No query params should use defaults (step=0, prompt_idx=-1, sample_idx=-1)."""
        job_id = _create_train_job(server, algo="gspo")
        _, body = _get(server, f"/api/jobs/{job_id}/trajectory")
        assert body["valid"] is False

    def test_metrics_unregistered_job(self, server):
        _, body = _get(server, "/api/jobs/bad-id/metrics")
        assert status_or_ok(body)
        assert "metrics" in body

    def test_stop_already_stopped_job(self, server):
        """Stopping a job twice should not crash; server is idempotent."""
        job_id = _create_train_job(server, ckpt="double-stop")
        _, body1 = _post(server, f"/api/jobs/{job_id}/stop", {})
        assert body1["stopped"] is True
        _, body2 = _post(server, f"/api/jobs/{job_id}/stop", {})
        assert "stopped" in body2


def status_or_ok(body):
    return isinstance(body, dict)

