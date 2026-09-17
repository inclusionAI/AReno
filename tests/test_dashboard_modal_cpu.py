"""Dashboard Modal boundary tests: no credentials, network, or GPU allocation."""

import base64
import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from areno.dashboard import server
from areno.dashboard.flow.store import Store
from areno.dashboard.modal_flow import ModalFlow


@pytest.fixture
def flow(tmp_path, monkeypatch):
    value = ModalFlow(tmp_path)
    value.app.samples.enqueue = Mock(return_value={"status": "ready"})
    value.app.controller.provider = object()
    value.app.controller.image_resolver = lambda _: "ghcr.io/inclusionai/areno@sha256:" + "a" * 64
    value.app.pricing.quote = Mock(return_value={"planned_cost": "14.4", "hourly_cost": "3.6"})
    value.app.controller._thread = Mock()
    value.app.controller.cost_estimator = lambda *_: {"hourly_cost": "3.6", "rates": {"source": "public list rates"}}
    monkeypatch.setattr(server, "MODAL_FLOW", value)
    monkeypatch.setattr(server, "STATE_FILE", tmp_path / "state.json")
    return value


def training():
    return {
        "name": "test training",
        "kind": "training",
        "model": {"adapter": "qwen3", "checkpoint": "org/model"},
        "stages": [{"algo": "sft", "params": {"dataset_path": "org/data"}}],
        "resources": {"gpu": "L4", "count": 1},
    }


def test_plan_does_not_launch_and_executes_once(flow):
    request = training()
    preview = flow.preview(request)
    assert not flow.app.controller.store.jobs()
    request["model"]["checkpoint"] = "changed"
    result = flow.execute(preview["id"])
    job = flow.job(result["job_id"], server.Job)
    assert job.config["model"]["checkpoint"] == "org/model"
    assert job.provider == "modal"
    assert flow.app.controller._thread.call_count == 1
    with pytest.raises(ValueError, match="expired or already"):
        flow.execute(preview["id"])


def test_metrics_logs_summary_and_frozen_terminal_cost(flow):
    identifier = flow.execute(flow.preview(training())["id"])["job_id"]
    raw = identifier.removeprefix("modal-")
    store = flow.app.controller.store
    store.update(raw, status="succeeded", sandbox_id="sb-test", started_at=100, finished_at=160)
    for i in range(2105):
        store.event(raw, {"type": "metric", "tag": "train/loss", "step": i, "value": i / 10, "time": i + 100})
    store.event(raw, {"type": "log", "message": "remote stdout", "time": 100})
    job = server.STATE.get_job(identifier)
    assert len(server.STATE.metric_series(identifier, "train/loss")) == 2105
    assert server.STATE.metric_summaries(identifier)[0]["count"] == 2105
    assert "remote stdout" in job.logs
    assert float(job.usage["accrued_cost"]) == pytest.approx(0.06)
    summary = flow.jobs(server.Job)[0].to_summary_json()
    assert summary["step"] == 2104 and summary["provider"] == "modal"
    assert summary["perf"]["train/loss"] == 210.4
    # Simulate bounded event retention; metrics survive both pruning and restart.
    with store.lock, store.db:
        store.db.execute("DELETE FROM events WHERE job=?", (raw,))
    reopened = Store(store.directory)
    assert len(reopened.metrics(raw)) == 2105
    assert len(flow.job(identifier, server.Job).metrics) == 2105
    reopened.db.close()


def test_stages_keep_distinct_metric_names(flow):
    request = training()
    request["stages"].append({"algo": "sft", "params": {"dataset_path": "org/data"}})
    identifier = flow.execute(flow.preview(request)["id"])["job_id"]
    for index in (0, 1):
        flow.app.controller.store.event(
            identifier[6:], {"type": "metric", "tag": "train/loss", "index": index, "step": 1, "value": 0.1}
        )
    assert {point["name"] for point in flow.job(identifier, server.Job).metrics} == {
        "stage-0/train/loss",
        "stage-1/train/loss",
    }


def test_serve_key_never_persisted_and_stop_routes_remote(flow):
    request = {**training(), "kind": "deployment", "endpoint_key": "x" * 32}
    preview = flow.preview(request)
    assert "x" * 32 not in json.dumps(preview)
    result = flow.execute(preview["id"])
    assert result["endpoint_key"] == "x" * 32
    assert "x" * 32 not in json.dumps(flow.app.controller.store.jobs())
    flow.app.controller.stop = Mock()
    assert server.STATE.stop(result["job_id"])
    flow.app.controller.stop.assert_called_once_with(result["job_id"][6:])


def test_uploaded_dataset_snapshot_survives_manager_edits(flow):
    asset = flow.app.post(
        "/api/uploads", {"name": "data.jsonl", "content": base64.b64encode(b'{"text":"test"}\n').decode()}
    )
    dataset = flow.app.post("/api/datasets", {"name": "My data", "source": asset["path"], "source_type": "upload"})
    request = training()
    request["stages"][0]["dataset_id"] = dataset["id"]
    preview = flow.preview(request)
    flow.app.controller.store.delete_dataset(dataset["id"])
    result = flow.execute(preview["id"])
    assert asset["path"] in flow.job(result["job_id"], server.Job).config["stages"][0]["args"]


@pytest.mark.parametrize(
    "url,hub,source",
    [
        ("https://huggingface.co/datasets/owner/data", "hf", "owner/data"),
        ("https://huggingface.co/datasets/gsm8k/", "hf", "gsm8k"),
        ("https://modelscope.cn/datasets/owner/data", "modelscope", "owner/data"),
        ("https://www.modelscope.cn/datasets/owner/data/summary", "modelscope", "owner/data"),
    ],
)
def test_repository_url_import(flow, url, hub, source):
    dataset = flow.import_url({"url": url, "name": "My dataset"})
    assert dataset["name"] == "My dataset"
    assert dataset["source_type"] == "repository"
    assert dataset["source"] == source
    assert dataset["model_hub"] == hub
    flow.app.samples.enqueue.assert_called_once()


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/data.csv",
        "https://127.0.0.1/data.csv",
        "https://huggingface.co/models/owner/model",
        "https://modelscope.cn/datasets",
        "https://huggingface.co/datasets/owner/data/resolve/main/data.csv",
        "https://huggingface.co@evil.example/datasets/owner/data",
        "https://huggingface.co/datasets/owner/%2e%2e",
    ],
)
def test_repository_url_rejects_other_sources(flow, url):
    with pytest.raises(ValueError):
        flow.import_url({"url": url})
    assert flow.app.controller.store.datasets() == []


def test_http_preview_csrf_and_dataset_routes(flow):
    instance = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()

    def call(path, body=None, csrf=False):
        conn = http.client.HTTPConnection("127.0.0.1", instance.server_port)
        conn.request(
            "GET" if body is None else "POST",
            path,
            body=json.dumps(body) if body is not None else None,
            headers={"Content-Type": "application/json", "X-Arenoflow-CSRF": flow.app.csrf if csrf else ""},
        )
        response = conn.getresponse()
        status, data = response.status, json.loads(response.read())
        conn.close()
        return status, data

    try:
        assert call("/api/modal/bootstrap")[0] == 200
        assert call("/api/modal/preview", training())[0] == 403
        status, response = call("/api/modal/preview", training(), True)
        assert status == 200 and response["plan"]["tool"] == "start_modal"
        assert call("/api/modal/datasets")[1] == []
        status, result = call("/api/modal/execute", response["plan"]["parameters"], True)
        assert status == 200 and result["job"]["provider"] == "modal"
        assert call("/api/jobs/" + result["job"]["id"])[0] == 200
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join()


def test_agent_can_prepare_but_cannot_directly_launch_modal(flow):
    schemas = {tool["function"]["name"] for tool in server.agent_tool_schemas()}
    assert "prepare_modal_plan" in schemas and "start_modal" not in schemas
    result = server.execute_agent_tool(
        {"function": {"name": "prepare_modal_plan", "arguments": json.dumps({"request": training()})}}
    )
    assert result["ok"] and result["plan"]["tool"] == "start_modal"
    assert not flow.app.controller.store.jobs()


def test_plan_defaults_image_fee_and_edit_add_remove(flow):
    preview = flow.preview(training())
    assert preview["image"].startswith("ghcr.io/inclusionai/areno@sha256:")
    assert preview["estimate"]["planned_cost"] == "14.4"
    params = preview["workflow"]["stages"][0]["params"]
    assert params["adam_4bit"] is True and params["attn_backend"] == "flash"
    params["lr"] = 0.0002
    params["max_steps"] = 25
    revised = flow.revise(preview["id"], preview["workflow"])
    assert "--lr 0.0002" in revised["command"]
    del revised["workflow"]["stages"][0]["params"]["lr"]
    revised = flow.revise(preview["id"], revised["workflow"])
    assert "lr" not in revised["workflow"]["stages"][0]["params"]
    identifier = flow.execute(revised["id"])["job_id"]
    assert flow.job(identifier, server.Job).config["stages"][0]["params"]["max_steps"] == 25


def test_invalid_plan_edit_preserves_previous_plan(flow):
    preview = flow.preview(training())
    preview["workflow"]["resources"]["gpu"] = "invalid"
    with pytest.raises(ValueError):
        flow.revise(preview["id"], preview["workflow"])
    assert flow.plans[preview["id"]][1]["resources"]["gpu"] == "L4"


def test_editor_keeps_decimal_and_boolean_text_until_validated(flow):
    preview = flow.preview(training())
    params = preview["workflow"]["stages"][0]["params"]
    params.update(lr="0.000025", adam_4bit="false", max_steps="35", max_new_tokens="1e3")
    revised = flow.revise(preview["id"], preview["workflow"])
    params = revised["workflow"]["stages"][0]["params"]
    assert params["lr"] == 0.000025
    assert params["adam_4bit"] is False
    assert params["max_new_tokens"] == 1000
    assert "--max-steps 35" in revised["command"]


def test_plan_survives_restart_and_cannot_execute_twice(flow):
    preview = flow.preview(training())
    restored = ModalFlow(flow.app.controller.store.directory, application=flow.app)
    restored.revise(preview["id"], preview["workflow"])
    restored.execute(preview["id"])
    restarted = ModalFlow(flow.app.controller.store.directory, application=flow.app)
    with pytest.raises(ValueError, match="already executed"):
        restarted.execute(preview["id"])
    with pytest.raises(ValueError, match="already executed"):
        restarted.revise(preview["id"], preview["workflow"])


def test_old_chat_plan_can_be_revalidated_after_restart(flow):
    preview = flow.preview(training())
    old_identifier = "old-chat-plan"
    revised = flow.revise(old_identifier, preview["workflow"])
    assert revised["id"] == old_identifier
    assert flow.app.controller.store.get_plan(old_identifier)["status"] == "proposed"


def test_plan_storage_excludes_endpoint_secret(flow):
    request = {**training(), "kind": "deployment", "endpoint_key": "z" * 32}
    plan = flow.preview(request)
    stored = flow.app.controller.store.get_plan(plan["id"])
    assert "z" * 32 not in json.dumps(stored)
    restored = ModalFlow(flow.app.controller.store.directory, application=flow.app)
    result = restored.execute(plan["id"])
    assert len(result["endpoint_key"]) >= 24


def test_modal_rollout_samples_reach_job_details_and_survive_reload(flow):
    identifier = flow.execute(flow.preview(training())["id"])["job_id"]
    store = flow.app.controller.store
    for index in range(55):
        event = {
            "type": "rollout_sample",
            "sample": {"step": index, "prompt": "question", "completion": "answer"},
            "index": 1,
            "time": 100 + index,
        }
        store.event(identifier[6:], event)
        store.event(identifier[6:], event)  # Reattached stdout must not duplicate samples.
    for _ in range(2):
        samples = flow.job(identifier, server.Job).samples
        assert len(samples) == 50
        assert samples[0]["step"] == 5
        assert samples[-1]["completion"] == "answer"
        assert samples[-1]["stage_index"] == 1


def test_explicit_loader_survives_plan_revision_and_execution(flow):
    from areno.dashboard.flow.assets import save_upload
    from areno.dashboard.flow.datasets import save_dataset

    upload = save_upload(flow.app.controller.store.directory, "data.jsonl", base64.b64encode(b"{}\n").decode())
    dataset = save_dataset(
        flow.app.controller.store,
        {"name": "Data", "source_type": "upload", "source": upload["path"]},
    )
    request = training()
    request["stages"][0].update(dataset_id=dataset["id"], dataset_loader_id=None)
    loader = "examples/sft/alpaca/dataset_loader.py"
    request["stages"][0]["params"]["dataset_loader_fn"] = loader
    plan = flow.preview(request)
    assert plan["workflow"]["stages"][0]["params"]["dataset_loader_fn"] == loader
    assert "--dataset-loader-fn" in plan["command"]
    revised = flow.revise(plan["id"], plan["workflow"])
    assert revised["workflow"]["stages"][0]["params"]["dataset_loader_fn"] == loader
    identifier = flow.execute(plan["id"])["job_id"]
    record = flow.app.controller.store.get(identifier[6:])
    assert record["manifest"]["stages"][0]["params"]["dataset_loader_fn"] == loader


def test_modal_timeperf_groups_stages_and_survives_event_pruning(flow):
    request = training()
    request["stages"].append(request["stages"][0].copy())
    identifier = flow.execute(flow.preview(request)["id"])["job_id"]
    store = flow.app.controller.store
    for stage in (0, 1):
        for tag, value in {
            "time/rollout": 3,
            "train/step_rollout_time_s": 3,
            "time/train": 4,
            "time/reward": 1,
            "train/step_e2e_time_s": 10,
        }.items():
            store.event(identifier[6:], {"type": "metric", "tag": tag, "value": value, "step": 1, "index": stage})
    with store.lock, store.db:
        store.db.execute("DELETE FROM events WHERE job=?", (identifier[6:],))
    rows = flow.job(identifier, server.Job).timeperf
    assert len(rows) == 2
    assert [row["stage_index"] for row in rows] == [0, 1]
    assert all(row["total_s"] == 10 and row["rollout_s"] == 3 and row["train_s"] == 4 for row in rows)
    assert {s["name"]: s["seconds"] for s in rows[0]["segments"]} == {"rollout": 3, "train": 4, "reward": 1, "other": 2}


def test_modal_trainer_state_from_older_logs_and_new_events(flow):
    identifier = flow.execute(flow.preview(training())["id"])["job_id"]
    store = flow.app.controller.store
    store.update(identifier[6:], status="running", phase="training")
    store.event(
        identifier[6:], {"type": "log", "message": "epoch=0 step=3 role=actor stage=rollout_start", "time": 100}
    )
    job = flow.job(identifier, server.Job)
    assert (job.stage, job.step, job.role) == ("rollout_start", 3, "actor")
    store.event(identifier[6:], {"type": "dashboard_state", "state": {"stage": "train_start", "step": 3}, "time": 101})
    assert flow.job(identifier, server.Job).stage == "train_start"
    store.update(identifier[6:], status="succeeded", finished_at=102)
    job = flow.job(identifier, server.Job)
    assert job.stage == "done"
    assert job.finished_at == "1970-01-01T00:01:42+00:00"
    store.update(identifier[6:], updated_at=200)
    assert flow.job(identifier, server.Job).finished_at == job.finished_at
