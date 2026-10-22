"""Visual reward contracts, including partial credit and unrenderable-page gates."""

import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_transport_failure_does_not_expose_credentials(monkeypatch):
    monkeypatch.setattr("examples.agentic.pi.visual_reward.time.sleep", lambda _: None)
    with endpoint(response(), status=503) as (judge, _), pytest.raises(RuntimeError) as raised:
        grade(judge)
    assert str(raised.value) == "VLM judge HTTP 503"


def test_rate_limit_retries_preserve_real_grade(monkeypatch):
    import io
    from urllib.error import HTTPError

    calls = []
    waits = []

    def request(req, **kwargs):
        calls.append(req.data)
        if len(calls) < 3:
            raise HTTPError(req.full_url, 429, "limited", {"Retry-After": "3"}, None)
        return io.StringIO(json.dumps({"choices": [{"message": {"content": json.dumps(response())}}]}))

    monkeypatch.setattr("examples.agentic.pi.visual_reward.urlopen", request)
    monkeypatch.setattr("examples.agentic.pi.visual_reward.time.sleep", waits.append)
    result = grade(VLMJudge("https://judge.invalid/v1", "fixture", "fixture-secret"))
    assert waits == [3, 3] and len(set(calls)) == 1
    assert result["reward"] == pytest.approx(0.728125)


def test_grading_concurrency_is_independent_of_rollout_capacity():
    pytest.importorskip("harbor", reason="Harbor integration requires Python 3.12+ and Harbor 0.24")
    from examples.agentic.pi.svg_dataset import SVGIconDataset

    active = peak = 0

    class Grader(SVGIconDataset):
        async def _verify(self, attempt):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return {"reward": 0.5}
            finally:
                active -= 1

    async def check():
        results = await asyncio.gather(*(Grader(grading_concurrency=2).verify(None) for _ in range(16)))
        assert peak == 2 and active == 0
        assert all(result["reward"] == 0.5 for result in results)

    asyncio.run(check())
    # A later rollout can run on a different event loop without stale bindings.
    asyncio.run(check())


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_environment_judge_timeout_is_rejected(monkeypatch, timeout):
    for name in ("BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.setenv("CUSTOM_" + name, "fixture")
    monkeypatch.setenv("CUSTOM_TIMEOUT", str(timeout))
    with pytest.raises(ValueError, match="TIMEOUT must be finite and positive"):
        VLMJudge.from_env("CUSTOM_")


def test_environment_judge_timeout_reaches_transport(monkeypatch):
    import io

    observed = []

    def request(req, *, timeout):
        observed.append(timeout)
        return io.StringIO(json.dumps({"choices": [{"message": {"content": json.dumps(response())}}]}))

    for name in ("BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.setenv("CUSTOM_" + name, "https://fixture.invalid")
    monkeypatch.setenv("CUSTOM_TIMEOUT", "300")
    monkeypatch.setattr("examples.agentic.pi.visual_reward.urlopen", request)
    result = grade(VLMJudge.from_env("CUSTOM_"))
    assert observed == [300] and result["reward"] == pytest.approx(0.728125)


def test_provider_generation_options_are_loaded_from_environment(monkeypatch):
    with endpoint(response()) as (judge, requests):
        monkeypatch.setenv("CUSTOM_BASE_URL", judge.base_url)
        monkeypatch.setenv("CUSTOM_MODEL", judge.model)
        monkeypatch.setenv("CUSTOM_API_KEY", judge.api_key)
        monkeypatch.setenv("CUSTOM_REQUEST_OPTIONS", '{"enable_thinking":false}')
        grade(VLMJudge.from_env("CUSTOM_"))
    assert requests[0]["enable_thinking"] is False
    assert requests[0]["max_tokens"] == 4096


@pytest.mark.parametrize("fault", ["blank", "javascript", "resource", "image", "interaction", "timeout", "no_views"])
def test_visible_pages_receive_partial_credit_despite_page_faults(tmp_path, fault):
    pytest.importorskip("harbor", reason="Harbor integration requires Python 3.12+ and Harbor 0.24")
    from examples.agentic.pi.svg_dataset import SVGIconDataset

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
    result = asyncio.run(SVGIconDataset(renderer=Renderer(), judge=judge).verify(attempt))
    if fault in {"blank", "timeout", "no_views"}:
        assert not judge.called
        assert result["reward"] == 0 and not result["page_valid"] and result["zero_reasons"]
    else:
        assert judge.called
        assert result["reward"] == 0.42 and result["page_valid"] and not result["zero_reasons"]
        assert result["page_issues"]
    assert (tmp_path / "artifacts/grading.json").exists()


def test_missing_deliverable_is_zero_instead_of_invalid_trajectory(tmp_path):
    pytest.importorskip("harbor", reason="Harbor integration requires Python 3.12+ and Harbor 0.24")
    from examples.agentic.pi.svg_dataset import SVGIconDataset

    async def missing(*args, **kwargs):
        assert kwargs["cwd"] == "/"
        return SimpleNamespace(return_code=1)

    attempt = SimpleNamespace(
        root=tmp_path,
        verify_timeout=60,
        item=SimpleNamespace(record={"prompt": "build", "files": {}, "entrypoint": "icon.svg"}),
        environment=SimpleNamespace(workspace="/workspace", environment=SimpleNamespace(exec=missing)),
    )
    result = asyncio.run(SVGIconDataset().verify(attempt))
    assert result["reward"] == 0 and result["grading_complete"] and not result["page_valid"]
    assert "missing delivered SVG" in result["zero_reasons"]
