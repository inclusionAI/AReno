"""Typed API validation, calibrated outputs, and serialized scorer lifecycle."""

import math
import threading

import pytest
from fastapi.testclient import TestClient

from areno.api.jev.scorer import answer_from_logits, render_state, validate_question
from areno.api.jev.serve import create_decision_app


def test_decision_output_types_and_calibration():
    question = {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}
    validate_question("q", question)
    raw = answer_from_logits(question, [0.0, 2.0], 1.0)
    calibrated = answer_from_logits(question, [0.0, 2.0], 2.0)
    assert raw["choice"] == calibrated["choice"] == "b"
    assert sum(raw["probabilities"].values()) == pytest.approx(1.0)
    assert calibrated["probabilities"]["b"] < raw["probabilities"]["b"]
    assert 0 <= calibrated["confidence"] <= 1
    assert answer_from_logits({"type": "noul"}, [0.0, 0.0], 1.0)["noul"] == 0.5
    score = answer_from_logits({"type": "score", "criteria": ["low", "middle", "high"]}, [0.0, 0.0, 0.0], 1.0)
    assert score["score"] == pytest.approx(1.0)
    with pytest.raises(FloatingPointError):
        answer_from_logits(question, [math.nan, 0.0], 1.0)
    with pytest.raises(ValueError):
        answer_from_logits(question, [0.0, 0.0], 0.0)
    assert render_state({"z": 2, "a": 1}) == render_state('{"a":1,"z":2}')
    with pytest.raises(ValueError):
        validate_question("q", {**question, "criteria": {"a": "only one"}})


def test_http_aliases_validate_requests_and_close_on_the_owning_thread():
    events = []

    class FakeScorer:
        def __init__(self, model, checkpoint):
            self.thread = threading.get_ident()
            events.append("init")

        def decide(self, state, questions):
            assert threading.get_ident() == self.thread
            for qid, question in questions.items():
                validate_question(qid, question)
            events.append("decide")
            return {"q": {"type": "noul", "noul": 0.75}}, 10

        def close(self):
            assert threading.get_ident() == self.thread
            events.append("close")

    app = create_decision_app("base", "checkpoint", scorer_factory=FakeScorer)
    with TestClient(app) as client:
        assert client.get("/health").json()["ready"]
        assert client.get("/v1/models").json()["data"][0]["id"] == "checkpoint"
        payload = {"state": {"query": "test"}, "questions": {"q": {"type": "noul", "instructions": "Is this a test?"}}}
        for route in ("/api/alpha/decisions", "/v1/systemone"):
            response = client.post(route, json=payload)
            assert response.status_code == 200
            assert response.json()["answers"]["q"]["noul"] == 0.75
            assert response.json()["usage"] == {"input_tokens": 10, "output_tokens": 0}
        assert client.post("/v1/systemone", json={**payload, "model": "unknown"}).status_code == 404
        assert client.post("/v1/systemone", json={**payload, "extra": 1}).status_code == 422
        assert (
            client.post("/v1/systemone", json={**payload, "questions": {"q": {"type": "invalid"}}}).status_code == 422
        )
    assert events == ["init", "decide", "decide", "close"]


def test_create_app_dispatches_decision_checkpoint_without_loading_chat_runtime(monkeypatch):
    import areno.api.jev.serve as decision_serve
    from areno.api.models import BackendType
    from areno.cli.serve import create_app

    expected = object()
    monkeypatch.setattr(decision_serve, "create_decision_app", lambda model, path, **options: expected)
    kwargs = dict(
        model_path="base",
        tp_size=1,
        world_size=1,
        max_running_prompts=1,
        default_max_tokens=10,
        decode_progress_interval_s=0.0,
        decision_checkpoint="checkpoint",
    )
    assert create_app(**kwargs, backend_type=BackendType.MLX) is expected
    assert create_app(**kwargs, backend_type=BackendType.CUDA) is expected


def test_concurrent_requests_keep_independent_scores_and_serialize_model_access():
    import time
    from concurrent.futures import ThreadPoolExecutor

    calls = []

    class Scorer:
        def __init__(self, model, checkpoint):
            self.thread = threading.get_ident()

        def decide(self, state, questions):
            assert threading.get_ident() == self.thread
            calls.append(state)
            time.sleep(0.002)
            if state == "bad":
                raise ValueError("invalid state")
            return {"q": {"type": "noul", "noul": int(state) / 20}}, 1

        def decide_many(self, requests):
            raise AssertionError("independent requests must not be combined")

        def close(self):
            assert threading.get_ident() == self.thread

    app = create_decision_app("base", "checkpoint", scorer_factory=Scorer)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=8) as executor:
        states = [str(index) for index in range(20)] + ["bad"]

        def request(state):
            return client.post(
                "/api/alpha/decisions",
                json={
                    "state": state,
                    "questions": {"q": {"type": "noul", "instructions": "Check"}},
                },
            )

        responses = list(executor.map(request, states))
        for index, response in enumerate(responses[:-1]):
            assert response.status_code == 200
            assert response.json()["answers"]["q"]["noul"] == index / 20
        assert responses[-1].status_code == 422
    assert sorted(calls) == sorted(states)
