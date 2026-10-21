"""Visual reward contracts, including partial credit and unrenderable-page gates."""

import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.agentic.pi.frontend_dataset import FrontendDataset
from examples.agentic.pi.visual_reward import RubricDimension, VLMJudge

PNG = b"\x89PNG\r\n\x1a\nfixture"


@contextmanager
def endpoint(response, *, status=200):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            assert self.headers["Authorization"] == "Bearer fixture-secret"
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            result = response(len(requests)) if callable(response) else response
            body = json.dumps({"choices": [{"message": {"content": json.dumps(result)}}]}).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        yield VLMJudge(f"http://127.0.0.1:{server.server_address[1]}/v1", "fixture-vision", "fixture-secret"), requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def grade(judge, **kwargs):
    return asyncio.run(
        judge.score(
            instruction="Build the specified interface",
            images={"Candidate desktop": PNG},
            dimensions={"one": RubricDimension("criterion one"), "two": RubricDimension("criterion two", 3)},
            validity_checks={"works": "The page is usable"},
            **kwargs,
        )
    )


def response(*, passed=True, first=51.25):
    return {
        "dimensions": {
            "one": {"score": first, "reason": "visible evidence"},
            "two": {"score": 80, "reason": "visible evidence"},
        },
        "checks": {"works": {"passed": passed, "reason": "browser evidence"}},
    }


def test_continuous_multidimensional_score_and_environment_credentials(monkeypatch):
    with endpoint(response()) as (judge, requests):
        monkeypatch.setenv("CUSTOM_BASE_URL", judge.base_url)
        monkeypatch.setenv("CUSTOM_MODEL", judge.model)
        monkeypatch.setenv("CUSTOM_API_KEY", judge.api_key)
        result = grade(VLMJudge.from_env("CUSTOM_"))
    assert result["reward"] == pytest.approx(0.728125)
    assert result["dimensions"]["one"]["score"] == 51.25
    assert requests[0]["temperature"] == 0
    assert "only the supplied viewports" in requests[0]["messages"][0]["content"]
    assert requests[0]["messages"][1]["content"][2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "fixture-secret" not in repr(judge) + json.dumps(result) + json.dumps(requests)


def test_failed_visual_validity_check_forces_zero_despite_high_scores():
    with endpoint(response(passed=False, first=100)) as (judge, requests):
        result = grade(judge)
    assert result["visual_score"] > 0 and result["reward"] == 0
    assert len(requests) == 1 and result["judge_format_attempts"] == 1


def test_schema_retry_requires_named_hard_checks_and_accepts_the_first_valid_grade():
    malformed = response(first=100)
    malformed["checks"] = {}
    with endpoint(lambda attempt: malformed if attempt == 1 else response(first=0)) as (judge, requests):
        result = grade(judge)
    assert len(requests) == 2 and result["judge_format_attempts"] == 2
    assert result["reward"] == pytest.approx(0.6)
    schema = requests[0]["response_format"]["json_schema"]
    assert schema["strict"]
    properties = schema["schema"]["properties"]
    assert properties["dimensions"]["required"] == ["one", "two"]
    assert properties["checks"]["required"] == ["works"]
    assert properties["checks"]["additionalProperties"] is False


def test_permanent_schema_failure_is_bounded_and_never_receives_a_fallback_reward():
    with endpoint({"dimensions": {}, "checks": []}) as (judge, requests):
        with pytest.raises(ValueError, match="rubric dimensions"):
            grade(judge)
    assert len(requests) == 2


@pytest.mark.parametrize("score", [float("nan"), -1, 101, True, "80"])
def test_invalid_judge_response_is_not_a_zero_reward_training_sample(score):
    with endpoint(response(first=score)) as (judge, _), pytest.raises(ValueError, match="scores"):
        grade(judge)


def test_transport_failure_does_not_expose_credentials():
    with endpoint(response(), status=503) as (judge, _), pytest.raises(RuntimeError) as raised:
        grade(judge)
    assert str(raised.value) == "VLM judge HTTP 503"


def test_provider_generation_options_are_loaded_from_environment(monkeypatch):
    with endpoint(response()) as (judge, requests):
        monkeypatch.setenv("CUSTOM_BASE_URL", judge.base_url)
        monkeypatch.setenv("CUSTOM_MODEL", judge.model)
        monkeypatch.setenv("CUSTOM_API_KEY", judge.api_key)
        monkeypatch.setenv("CUSTOM_REQUEST_OPTIONS", '{"enable_thinking":false}')
        grade(VLMJudge.from_env("CUSTOM_"))
    assert requests[0]["enable_thinking"] is False
    assert requests[0]["max_tokens"] == 4096


def test_frontend_generator_preserves_source_and_mixes_supported_briefs(tmp_path):
    from examples.agentic.pi.generate_dataset import convert_row, generate, select_tasks

    rows = []
    for category in ["documentation", "restaurant", "portfolio"]:
        for index in range(3):
            rows.append(
                {
                    "task_id": f"{category}-{index}",
                    "instruction": f"Create a {category} website for project {index}. "
                    + "clear layout and sample content " * 20,
                }
            )
    unsupported = {
        "task_id": "framework",
        "instruction": "Create a React website. " + "clear layout and sample content " * 20,
    }
    selected = select_tasks([*rows, unsupported], seed=0)
    assert len({row["task_id"].split("-")[0] for row in selected[:3]}) == 3
    assert unsupported not in selected
    converted = convert_row(rows[0], source={"dataset": "custom", "revision": "pinned"})
    assert converted["source"]["dataset"] == "custom"
    assert converted["serve_dir"] == "dist" and converted["max_turns"] == 20
    assert "preview.mjs" in converted["files"]
    preview = json.loads(converted["files"]["preview.json"])
    assert preview["viewports"] == {"desktop": {"width": 1280, "height": 960}}
    task = FrontendDataset().visual_task(SimpleNamespace(item=SimpleNamespace(record=converted)))
    assert task.viewports == preview["viewports"]
    assert "mobile" not in task.dimensions["responsive"].description
    assert task.validity_checks == {}
    custom = dict(converted, viewports={"custom": {"width": 1440, "height": 900}})
    assert (
        FrontendDataset().visual_task(SimpleNamespace(item=SimpleNamespace(record=custom))).viewports
        == custom["viewports"]
    )
    assert rows[0]["instruction"].strip() in converted["prompt"]
    wrapped = convert_row(
        {
            **rows[0],
            "instruction": "Vendor-specific build instructions.\nBuild a website for the following request:\n"
            + rows[0]["instruction"],
        },
        source={"dataset": "custom"},
    )
    assert "Vendor-specific" not in wrapped["prompt"]
    assert wrapped["grading_instruction"] == rows[0]["instruction"].strip()
    adapter = FrontendDataset()
    normalized = adapter.normalize(wrapped)
    assert (
        adapter.visual_task(SimpleNamespace(item=SimpleNamespace(record=normalized))).instruction
        == rows[0]["instruction"].strip()
    )
    with pytest.raises(ValueError, match="grading_instruction"):
        adapter.normalize(dict(wrapped, grading_instruction=""))
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = generate(metadata, tmp_path / "dataset.jsonl", limit=4, max_turns=20)
    assert report["compatible_rows"] == 9 and len(report["selected_task_ids"]) == 4


def test_short_design_subset_deduplicates_and_preserves_original_requirements(tmp_path):
    from examples.agentic.pi.generate_dataset import generate, select_tasks

    rows = [
        {
            "task_id": "store",
            "source_id": "original-store",
            "instruction": "Create a minimalist European product showcase with images and text.",
        },
        {
            "task_id": "portfolio",
            "instruction": "Design a calm editorial portfolio with refined typography and generous whitespace.",
        },
    ]
    duplicate = {**rows[0], "task_id": "duplicate", "instruction": rows[0]["instruction"].upper() + "  "}
    unsupported = [
        {"task_id": "framework", "instruction": "Build a polished landing page using Tailwind and React."},
        {"task_id": "interaction", "instruction": "Build a card editor with drag-and-drop interactions."},
        {"task_id": "video", "instruction": "Create a luxurious landing page with a video background."},
    ]
    selected = select_tasks([*rows, duplicate, *unsupported], seed=0)
    assert {row["task_id"] for row in selected} == {"store", "portfolio"}
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("".join(json.dumps(row) + "\n" for row in [*rows, duplicate, *unsupported]))
    output = tmp_path / "dataset.jsonl"
    report = generate(metadata, output, limit=2)
    converted = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["compatible_rows"] == 2
    assert {row["grading_instruction"] for row in converted} == {row["instruction"] for row in rows}
    assert {row["source"]["task_id"] for row in converted} == {"store", "portfolio"}
    with pytest.raises(ValueError, match="not enough compatible"):
        generate(metadata, output, limit=200)


@pytest.mark.parametrize("fault", ["blank", "javascript", "resource", "image", "interaction", "timeout", "no_views"])
def test_visible_pages_receive_partial_credit_despite_page_faults(tmp_path, fault):
    report = {
        "views": {"desktop": {"visible_content": True, "broken_images": 0}},
        "errors": [],
        "resource_failures": [],
        "interactions": [],
    }
    if fault == "blank":
        report["views"]["desktop"]["visible_content"] = False
    if fault == "javascript":
        report["errors"] = ["ReferenceError"]
    if fault == "resource":
        report["resource_failures"] = ["404: style.css"]
    if fault == "image":
        report["views"]["desktop"]["broken_images"] = 1
    if fault == "interaction":
        report["interactions"] = [{"action": {"action": "click"}, "passed": False}]
    if fault == "timeout":
        report["capture_error"] = "unresponsive page"
    if fault == "no_views":
        report["views"] = {}

    class Renderer:
        async def capture(self, *args, **kwargs):
            return report, {"Candidate desktop": PNG}

    class Judge:
        called = False

        async def score(self, **kwargs):
            self.called = True
            assert kwargs["evidence"] == report
            assert kwargs["validity_checks"] == {}
            return {"reward": 0.42, "visual_score": 0.42, "checks": {}}

    async def download(source, target):
        Path(target).mkdir(parents=True)

    async def exists(*args, **kwargs):
        return SimpleNamespace(return_code=0)

    attempt = SimpleNamespace(
        root=tmp_path,
        item=SimpleNamespace(record={"prompt": "build", "files": {}}),
        environment=SimpleNamespace(
            workspace="/workspace", environment=SimpleNamespace(download_dir=download, exec=exists)
        ),
    )
    judge = Judge()
    result = asyncio.run(FrontendDataset(renderer=Renderer(), judge=judge).verify(attempt))
    if fault in {"blank", "timeout", "no_views"}:
        assert not judge.called
        assert result["reward"] == 0 and not result["page_valid"] and result["zero_reasons"]
    else:
        assert judge.called
        assert result["reward"] == 0.42 and result["page_valid"] and not result["zero_reasons"]
        assert result["page_issues"]
    assert (tmp_path / "artifacts/grading.json").exists()


def test_missing_deliverable_is_zero_instead_of_invalid_trajectory(tmp_path):
    async def missing(*args, **kwargs):
        assert kwargs["cwd"] == "/"
        return SimpleNamespace(return_code=1)

    attempt = SimpleNamespace(
        root=tmp_path,
        verify_timeout=60,
        item=SimpleNamespace(record={"prompt": "build", "files": {}, "serve_dir": "dist"}),
        environment=SimpleNamespace(workspace="/workspace", environment=SimpleNamespace(exec=missing)),
    )
    result = asyncio.run(FrontendDataset().verify(attempt))
    assert result["reward"] == 0 and result["grading_complete"] and not result["page_valid"]
    assert "missing delivered entrypoint" in result["zero_reasons"]
