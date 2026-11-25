import math

import pytest
from fastapi.testclient import TestClient

from areno.api.jev.labels.serve import LabelDecisionScorer, create_label_decision_app
from areno.api.models import BackendType


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[-1]["content"] + "\nASSISTANT:"

    def encode(self, text, **kwargs):
        for label, token in (("yes", 1001), ("no", 1002)):
            if text.endswith(label):
                return [ord(c) for c in text[: -len(label)]] + [token]
        return [ord(c) for c in text]


class Scorer(LabelDecisionScorer):
    closed = False

    def __init__(self, *args, **kwargs):
        self.tokenizer = Tokenizer()
        self.temperature = 2.0
        self.max_seq_len = 4096

    def label_logprobs(self, encoded):
        return [math.log(0.2), math.log(0.3)]

    def close(self):
        type(self).closed = True


@pytest.mark.parametrize("backend", [BackendType.CUDA, BackendType.MLX])
def test_label_jev_routes_share_prompt_and_temperature_cpu(backend):
    app = create_label_decision_app("merged", backend_type=backend, scorer_factory=Scorer)
    with TestClient(app) as client:
        system = client.post(
            "/v1/systemone",
            json={
                "state": "State",
                "questions": {
                    "q": {"type": "choice", "instructions": "Choose", "criteria": {"x": "first", "y": "second"}}
                },
                "return_prompt_token_ids": True,
            },
        )
        native = client.post(
            "/v1/decisions",
            json={
                "input": "State",
                "questions": [
                    {
                        "id": "q",
                        "type": "choice",
                        "question": "Choose",
                        "options": [{"name": "x", "description": "first"}, {"name": "y", "description": "second"}],
                    }
                ],
                "return_prompt_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        assert system.status_code == native.status_code == 200
        a, b = system.json()["answers"]["q"], native.json()["answers"]["q"]
        assert a == b
        assert a["choice"] == "y"
        assert a["label_mass"] == pytest.approx(0.5)
        expected = math.sqrt(0.2) / (math.sqrt(0.2) + math.sqrt(0.3))
        assert a["probabilities"]["x"] == pytest.approx(expected)
        assert a["label_token_ids"] == [ord("A"), ord("B")]
        assert client.get("/health").json()["ready"]
        assert client.post("/v1/systemone", json={"state": "", "questions": {}, "model": "unknown"}).status_code == 404
        assert (
            client.post(
                "/v1/decisions", json={"input": "", "questions": [], "chat_template_kwargs": {"enable_thinking": True}}
            ).status_code
            == 422
        )
        assert (
            client.post("/v1/decisions", json={"input": "", "questions": [], "prompt_format_version": 2}).status_code
            == 422
        )
        assert client.post("/v1/systemone", json={"state": "", "questions": {}, "temperature": 0}).status_code == 422
    assert Scorer.closed


def test_label_jev_noul_order_cpu():
    scorer = Scorer()
    typed, _ = scorer.decide("S", {"q": {"type": "noul", "instructions": "True?"}})
    native, _ = scorer.decide("S", {"q": {"type": "noul", "instructions": "True?"}}, native=True)
    assert typed["q"]["noul"] == typed["q"]["probabilities"]["true"]
    assert native["q"]["probabilities"]["yes"] == typed["q"]["noul"]


def test_serve_sgl_jev_flag_dispatch_cpu(monkeypatch):
    import areno.api.jev.labels.serve as labels
    from areno.cli.serve import create_app

    captured = {}

    def factory(model_path, **kwargs):
        captured.update(kwargs)
        return "label-app"

    monkeypatch.setattr(labels, "create_label_decision_app", factory)
    common = dict(
        model_path="merged",
        tp_size=1,
        world_size=1,
        max_running_prompts=1,
        default_max_tokens=1,
        decode_progress_interval_s=0,
        backend_type=BackendType.CUDA,
        sgl_jev=True,
    )
    assert create_app(**common) == "label-app"
    assert captured["backend_type"] == BackendType.CUDA
    with pytest.raises(ValueError, match="decision-checkpoint"):
        create_app(**common, decision_checkpoint="scalar-head")


def test_serve_click_forwards_sgl_jev_cpu(monkeypatch):
    import uvicorn
    from click.testing import CliRunner

    import areno.cli.dashboard_registry as registry
    import areno.cli.serve as serve

    captured = {}
    monkeypatch.setattr(serve, "resolve_model_ref", lambda model_path, **kwargs: model_path)
    monkeypatch.setattr(registry, "register_dashboard_job", lambda **kwargs: None)

    def create_app(**kwargs):
        captured.update(kwargs)
        return "app"

    monkeypatch.setattr(serve, "create_app", create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    result = CliRunner().invoke(serve.serve_command, ["--sgl-jev", "--model-path", "merged", "--backend", "cuda"])
    assert result.exit_code == 0, result.output
    assert captured["sgl_jev"] is True
