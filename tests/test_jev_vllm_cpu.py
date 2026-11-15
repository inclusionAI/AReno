from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
import urllib.error
from pathlib import Path

import pytest

from areno.api.jev_vllm import (
    VLLMHTTPError,
    VLLMProtocolError,
    VLLMSystemOneAdapter,
    validate_systemone_response,
)


def load_native_reference():
    path = Path(__file__).resolve().parents[1] / "examples" / "jev_label_lora" / "vllm_native_reference.py"
    spec = importlib.util.spec_from_file_location("_jev_vllm_native_reference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


native_reference = load_native_reference()

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team handles this?",
        "criteria": {"billing": "money", "shipping": "delivery", "security": "access"},
    },
    "urgent": {
        "type": "noul",
        "instructions": "Is this urgent?",
        "criteria": {"true": "money is moving", "false": "it can wait"},
    },
    "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}


def native_response(model="base"):
    return {
        "id": "decision-test",
        "object": "structured_decision",
        "created": 123,
        "model": model,
        "answers": {
            "team": {
                "type": "choice",
                "choice": "shipping",
                "probabilities": {"billing": 0.25, "shipping": 0.5, "security": 0.25},
                "confidence": 0.2,
            },
            "urgent": {
                "type": "noul",
                "noul": 0.2,
                "probabilities": {"yes": 0.2, "no": 0.8},
                "confidence": 0.4,
            },
            "tone": {
                "type": "score",
                "score": 1.5,
                "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
                "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                "confidence": 0.15,
            },
        },
        "usage": {"input_tokens": 100, "output_tokens": 3},
        "diagnostics": {
            "team": {"label_mass": 0.4, "argmax_is_label": True},
            "urgent": {"label_mass": 0.5, "argmax_is_label": False},
            "tone": {"label_mass": 0.25, "argmax_is_label": True},
        },
    }


class FakeResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(json.dumps(payload, allow_nan=True).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeOpener:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result)


def adapter_for(results):
    opener = FakeOpener(results)
    return VLLMSystemOneAdapter("http://127.0.0.1:8000", "base", opener=opener, timeout=7.0), opener


def request_json(opener, index=-1):
    request, timeout = opener.calls[index]
    assert timeout == 7.0
    return json.loads(request.data)


def score_response(read, raw_logprobs, model="base", sampled_token_id=999):
    top_logprobs = {"token_id:999": -0.125}
    for token_id, value in reversed(list(zip(read["label_token_ids"], raw_logprobs))):
        top_logprobs[f"token_id:{token_id}"] = value
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 123,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": "?",
                "logprobs": {
                    "text_offset": [0],
                    "token_logprobs": [-0.125],
                    "tokens": ["token_id:999"],
                    "top_logprobs": [top_logprobs],
                },
                "finish_reason": "length",
                "token_ids": [sampled_token_id],
                "prompt_token_ids": read["prompt_token_ids"],
            }
        ],
        "usage": {"prompt_tokens": len(read["prompt_token_ids"]), "completion_tokens": 1, "total_tokens": 5},
    }


def encoded_reads():
    return [
        {
            "question_id": "team",
            "prompt_token_ids": [1, 2, 3, 90],
            "labels": ["A", "B", "C"],
            "label_token_ids": [32, 33, 34],
        },
        {
            "question_id": "urgent",
            "prompt_token_ids": [1, 4, 5, 90],
            "labels": ["yes", "no"],
            "label_token_ids": [9454, 2753],
        },
        {
            "question_id": "tone",
            "prompt_token_ids": [1, 6, 7, 90],
            "labels": ["0", "1", "2"],
            "label_token_ids": [15, 16, 17],
        },
    ]


def test_decision_request_and_three_question_response():
    adapter, opener = adapter_for([native_response("trained")])
    response = adapter.decide(
        {"ticket": "charged twice"},
        QUESTIONS,
        instructions="Route accurately.",
        adapter_name="trained",
    )

    assert response == native_response("trained")
    assert opener.calls[0][0].full_url == "http://127.0.0.1:8000/v1/systemone"
    assert request_json(opener) == {
        "model": "trained",
        "state": {"ticket": "charged twice"},
        "questions": QUESTIONS,
        "chat_template_kwargs": {"enable_thinking": False},
        "instructions": "Route accurately.",
    }
    assert "raw_logprobs" not in response
    assert tuple(response["answers"]["team"]["probabilities"]) == ("billing", "shipping", "security")
    assert tuple(response["answers"]["urgent"]["probabilities"]) == ("yes", "no")
    assert tuple(response["answers"]["tone"]["probabilities"]) == ("0", "1", "2")


def test_base_adapter_base_switching_does_not_leak_selection():
    responses = [native_response("base"), native_response("trained"), native_response("base")]
    adapter, opener = adapter_for(responses)

    adapter.decide("state", QUESTIONS)
    adapter.decide("state", QUESTIONS, adapter_name="trained")
    adapter.decide("state", QUESTIONS)

    assert [request_json(opener, index)["model"] for index in range(3)] == ["base", "trained", "base"]


@pytest.mark.parametrize("status", [400, 404])
def test_server_rejects_bad_base_or_adapter_with_real_error_body(status):
    body = json.dumps({"error": {"message": "model or adapter mismatch", "code": status}})
    error = urllib.error.HTTPError("http://localhost/v1/systemone", status, "bad", {}, io.BytesIO(body.encode()))
    adapter, _ = adapter_for([error])

    with pytest.raises(VLLMHTTPError) as caught:
        adapter.decide("state", QUESTIONS, adapter_name="wrong-base-adapter")

    assert caught.value.status == status
    assert caught.value.body == body


@pytest.mark.parametrize(
    "case",
    [
        "object",
        "model",
        "question_order",
        "negative_mass",
        "large_mass",
        "nan_mass",
        "probability_order",
        "probability_sum",
        "nan_probability",
        "choice",
        "noul",
        "score",
        "legend",
        "confidence",
        "argmax",
        "usage",
    ],
)
def test_invalid_native_response_is_rejected(case):
    response = native_response()
    if case == "object":
        response["object"] = "chat.completion"
    elif case == "model":
        response["model"] = "other"
    elif case == "question_order":
        response["answers"] = dict(reversed(list(response["answers"].items())))
    elif case == "negative_mass":
        response["diagnostics"]["team"]["label_mass"] = -0.1
    elif case == "large_mass":
        response["diagnostics"]["team"]["label_mass"] = 1.1
    elif case == "nan_mass":
        response["diagnostics"]["team"]["label_mass"] = math.nan
    elif case == "probability_order":
        probabilities = response["answers"]["team"]["probabilities"]
        response["answers"]["team"]["probabilities"] = {
            "shipping": probabilities["shipping"],
            "billing": probabilities["billing"],
            "security": probabilities["security"],
        }
    elif case == "probability_sum":
        response["answers"]["team"]["probabilities"]["billing"] = 0.2
    elif case == "nan_probability":
        response["answers"]["team"]["probabilities"]["billing"] = math.nan
    elif case == "choice":
        response["answers"]["team"]["choice"] = "billing"
    elif case == "noul":
        response["answers"]["urgent"]["noul"] = 0.8
    elif case == "score":
        response["answers"]["tone"]["score"] = 2.0
    elif case == "legend":
        response["answers"]["tone"]["legend"]["0"] = "very calm"
    elif case == "confidence":
        response["answers"]["team"]["confidence"] = 0.5
    elif case == "argmax":
        response["diagnostics"]["team"]["argmax_is_label"] = 1
    elif case == "usage":
        response["usage"]["input_tokens"] = -1

    adapter = VLLMSystemOneAdapter("http://localhost", "base")
    payload = adapter.build_decision_payload("state", QUESTIONS)
    with pytest.raises(VLLMProtocolError):
        validate_systemone_response(response, payload)


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"x": {"type": "choice", "criteria": {"only": None}}},
        {"x": {"type": "score", "criteria": ["only"]}},
        {"x": {"type": "noul", "criteria": {"unknown": "bad"}}},
        {"x": {"type": "other", "criteria": None}},
        {"x": {"type": "choice", "criteria": {"a": None, "b": None}, "unknown": 1}},
    ],
)
def test_invalid_request_schema_is_rejected_before_http(questions):
    adapter = VLLMSystemOneAdapter("http://localhost", "base")
    with pytest.raises(VLLMProtocolError):
        adapter.build_decision_payload("state", questions)


def test_raw_label_logprobs_preserve_requested_ids_and_order():
    reads = encoded_reads()
    raw = [[-2.0, -0.5, -3.0], [-1.25, -0.75], [-4.0, -2.0, -1.0]]
    responses = [score_response(read, values, model="trained") for read, values in zip(reads, raw)]
    adapter, opener = adapter_for(responses)

    scores = adapter.score_labels(reads, adapter_name="trained")

    assert [score.question_id for score in scores] == ["team", "urgent", "tone"]
    for index, (score, read, expected_raw) in enumerate(zip(scores, reads, raw)):
        assert score.labels == tuple(read["labels"])
        assert score.label_token_ids == tuple(read["label_token_ids"])
        assert score.raw_logprobs == tuple(expected_raw)
        assert score.prompt_token_ids == tuple(read["prompt_token_ids"])
        assert score.sampled_token_id == 999
        assert not score.argmax_is_label
        top = max(expected_raw)
        weights = [math.exp(value - top) for value in expected_raw]
        expected_probs = [value / sum(weights) for value in weights]
        assert score.restricted_probabilities == pytest.approx(expected_probs, abs=1e-12)
        assert score.label_mass == pytest.approx(sum(math.exp(value) for value in expected_raw), abs=1e-12)
        payload = request_json(opener, index)
        assert payload == {
            "model": "trained",
            "prompt": read["prompt_token_ids"],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": len(read["label_token_ids"]),
            "logprob_token_ids": read["label_token_ids"],
            "add_special_tokens": False,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        }
        assert opener.calls[index][0].full_url == "http://127.0.0.1:8000/v1/completions"


def test_raw_score_base_adapter_base_switching():
    read = encoded_reads()[0]
    responses = [score_response(read, [-1.0, -2.0, -3.0], model=model) for model in ["base", "trained", "base"]]
    adapter, opener = adapter_for(responses)

    adapter.score_labels([read])
    adapter.score_labels([read], adapter_name="trained")
    adapter.score_labels([read])

    assert [request_json(opener, index)["model"] for index in range(3)] == ["base", "trained", "base"]


@pytest.mark.parametrize("case", ["missing_label", "prompt_changed", "nan", "positive", "two_tokens", "two_choices"])
def test_invalid_raw_score_response_is_rejected(case):
    read = encoded_reads()[0]
    response = score_response(read, [-1.0, -2.0, -3.0])
    if case == "missing_label":
        del response["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:33"]
    elif case == "prompt_changed":
        response["choices"][0]["prompt_token_ids"] = [1, 2, 3]
    elif case == "nan":
        response["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:33"] = math.nan
    elif case == "positive":
        response["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:33"] = 0.1
    elif case == "two_tokens":
        response["choices"][0]["token_ids"] = [999, 32]
    elif case == "two_choices":
        response["choices"].append(copy.deepcopy(response["choices"][0]))
    adapter, _ = adapter_for([response])

    with pytest.raises(VLLMProtocolError):
        adapter.score_labels([read])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda read: read.update(label_token_ids=[32, 32, 34]),
        lambda read: read.update(labels=["A", "A", "C"]),
        lambda read: read.update(label_token_ids=[32, 33]),
        lambda read: read.update(prompt_token_ids=[]),
        lambda read: read.update(prompt_token_ids=[1, -2, 90]),
    ],
)
def test_invalid_encoded_boundary_contract_is_rejected_before_http(mutation):
    read = encoded_reads()[0]
    mutation(read)
    adapter = VLLMSystemOneAdapter("http://localhost", "base")
    with pytest.raises(VLLMProtocolError):
        adapter.build_score_payload(read)


def contract_encoded():
    return {
        "request": {"state": "state", "questions": QUESTIONS, "instructions": "instructions"},
        "reads": encoded_reads(),
        "metadata": {"enable_thinking": False, "answer_boundary": "first_reply_token"},
    }


def test_encoded_contract_compares_every_prompt_and_label_token():
    result = native_reference.compare(contract_encoded(), contract_encoded())

    assert result["status"] == "PASS"
    assert result["request_match"]
    assert result["question_order_match"]
    assert all(item["match"] for item in result["questions"])


@pytest.mark.parametrize(
    "field,value", [("prompt_token_ids", [1, 2, 3, 91]), ("labels", ["A", "C", "B"]), ("label_token_ids", [32, 34, 33])]
)
def test_encoded_contract_rejects_single_token_and_order_mismatch(field, value):
    areno = contract_encoded()
    native = contract_encoded()
    areno["reads"][0][field] = value

    result = native_reference.compare(areno, native)

    assert result["status"] == "FAIL"
    assert not result["questions"][0][field]["match"]
