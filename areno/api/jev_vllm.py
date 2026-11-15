from __future__ import annotations

import json
import math
import string
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


class VLLMAdapterError(RuntimeError):
    pass


class VLLMProtocolError(VLLMAdapterError):
    pass


class VLLMHTTPError(VLLMAdapterError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"vLLM request failed with HTTP {status}: {body}")


@dataclass(frozen=True)
class LabelScore:
    question_id: str
    labels: tuple[str, ...]
    label_token_ids: tuple[int, ...]
    raw_logprobs: tuple[float, ...]
    prompt_token_ids: tuple[int, ...]
    sampled_token_id: int
    http_response: dict[str, Any]

    @property
    def restricted_probabilities(self) -> tuple[float, ...]:
        top = max(self.raw_logprobs)
        weights = [math.exp(value - top) for value in self.raw_logprobs]
        total = sum(weights)
        return tuple(value / total for value in weights)

    @property
    def label_mass(self) -> float:
        return sum(math.exp(value) for value in self.raw_logprobs)

    @property
    def argmax_is_label(self) -> bool:
        return self.sampled_token_id in self.label_token_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "labels": list(self.labels),
            "label_token_ids": list(self.label_token_ids),
            "raw_logprobs": list(self.raw_logprobs),
            "restricted_probabilities": list(self.restricted_probabilities),
            "label_mass": self.label_mass,
            "prompt_token_ids": list(self.prompt_token_ids),
            "sampled_token_id": self.sampled_token_id,
            "argmax_is_label": self.argmax_is_label,
            "http_response": self.http_response,
        }


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VLLMProtocolError(f"{where} must be an object")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise VLLMProtocolError(f"{where} must be a finite number")
    return float(value)


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VLLMProtocolError(f"{where} must be an integer >= {minimum}")
    return value


def _question_spec(question_id: str, spec: Any) -> tuple[Mapping[str, Any], tuple[str, ...], tuple[str, ...]]:
    if not isinstance(question_id, str) or not question_id:
        raise VLLMProtocolError("question ids must be non-empty strings")
    spec = _mapping(spec, f"question {question_id!r}")
    if set(spec) - {"type", "instructions", "criteria"}:
        raise VLLMProtocolError(f"question {question_id!r} has unknown fields")
    question_type = spec.get("type")
    criteria = spec.get("criteria")
    if question_type == "choice":
        if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= 26:
            raise VLLMProtocolError(f"question {question_id!r} needs 2-26 choice options")
        names = tuple(str(name) for name in criteria)
        if len(set(names)) != len(names):
            raise VLLMProtocolError(f"question {question_id!r} has duplicate option names")
        return spec, tuple(string.ascii_uppercase[: len(names)]), names
    if question_type == "noul":
        if criteria is not None and (not isinstance(criteria, Mapping) or not set(criteria) <= {"true", "false"}):
            raise VLLMProtocolError(f"question {question_id!r} has invalid noul criteria")
        return spec, ("yes", "no"), ("yes", "no")
    if question_type == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise VLLMProtocolError(f"question {question_id!r} needs 2-10 score levels")
        labels = tuple(string.digits[: len(criteria)])
        return spec, labels, labels
    raise VLLMProtocolError(f"question {question_id!r} has unsupported type {question_type!r}")


def _probabilities(answer: Mapping[str, Any], keys: tuple[str, ...], where: str) -> tuple[float, ...]:
    probabilities = _mapping(answer.get("probabilities"), f"{where}.probabilities")
    if tuple(probabilities) != keys:
        raise VLLMProtocolError(f"{where}.probabilities has incorrect labels or order")
    values = tuple(_number(probabilities[key], f"{where}.probabilities[{key!r}]") for key in keys)
    if any(not 0.0 <= value <= 1.0 for value in values) or not math.isclose(
        sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5
    ):
        raise VLLMProtocolError(f"{where}.probabilities is not a normalized distribution")
    return values


def _validate_confidence(answer: Mapping[str, Any], expected: float, where: str) -> None:
    confidence = _number(answer.get("confidence"), f"{where}.confidence")
    if not 0.0 <= confidence <= 1.0 or not math.isclose(confidence, expected, rel_tol=1e-5, abs_tol=1e-6):
        raise VLLMProtocolError(f"{where}.confidence is inconsistent with label mass")


def validate_systemone_response(response: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    response = _mapping(response, "systemone response")
    if response.get("object") != "structured_decision":
        raise VLLMProtocolError("systemone response has an invalid object")
    if not isinstance(response.get("id"), str) or not response["id"]:
        raise VLLMProtocolError("systemone response id must be a non-empty string")
    _integer(response.get("created"), "systemone response.created")
    if response.get("model") != payload["model"]:
        raise VLLMProtocolError(
            f"systemone response model {response.get('model')!r} does not match requested {payload['model']!r}"
        )

    questions = _mapping(payload.get("questions"), "request.questions")
    question_ids = tuple(questions)
    answers = _mapping(response.get("answers"), "systemone response.answers")
    diagnostics = _mapping(response.get("diagnostics"), "systemone response.diagnostics")
    if tuple(answers) != question_ids or tuple(diagnostics) != question_ids:
        raise VLLMProtocolError("systemone response question ids or order do not match the request")

    usage = _mapping(response.get("usage"), "systemone response.usage")
    _integer(usage.get("input_tokens"), "systemone response.usage.input_tokens")
    _integer(usage.get("output_tokens"), "systemone response.usage.output_tokens")

    for question_id in question_ids:
        spec, labels, answer_keys = _question_spec(question_id, questions[question_id])
        where = f"systemone response.answers[{question_id!r}]"
        answer = _mapping(answers[question_id], where)
        if answer.get("type") != spec["type"]:
            raise VLLMProtocolError(f"{where}.type does not match the request")
        diagnostic = _mapping(diagnostics[question_id], f"systemone response.diagnostics[{question_id!r}]")
        label_mass = _number(diagnostic.get("label_mass"), f"{where} label_mass")
        if not 0.0 <= label_mass <= 1.0:
            raise VLLMProtocolError(f"{where} label_mass must be in [0, 1]")
        if not isinstance(diagnostic.get("argmax_is_label"), bool):
            raise VLLMProtocolError(f"{where} argmax_is_label must be boolean")
        values = _probabilities(answer, answer_keys, where)
        if spec["type"] == "choice":
            top = max(range(len(values)), key=values.__getitem__)
            if answer.get("choice") != answer_keys[top]:
                raise VLLMProtocolError(f"{where}.choice is inconsistent with probabilities")
        elif spec["type"] == "noul":
            noul = _number(answer.get("noul"), f"{where}.noul")
            if not math.isclose(noul, values[0], rel_tol=0.0, abs_tol=1e-6):
                raise VLLMProtocolError(f"{where}.noul is inconsistent with probabilities")
        else:
            score = _number(answer.get("score"), f"{where}.score")
            expected_score = sum(index * value for index, value in enumerate(values))
            if not math.isclose(score, expected_score, rel_tol=1e-5, abs_tol=1e-6):
                raise VLLMProtocolError(f"{where}.score is inconsistent with probabilities")
            legend = _mapping(answer.get("legend"), f"{where}.legend")
            if tuple(legend) != labels or any(
                legend[label] != str(level) for label, level in zip(labels, spec["criteria"])
            ):
                raise VLLMProtocolError(f"{where}.legend is inconsistent with the request")
        _validate_confidence(answer, max(values) * label_mass, where)
    return dict(response)


class VLLMSystemOneAdapter:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 120.0,
        api_key: str | None = None,
        opener: Any = None,
    ):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) server URL without query or fragment")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        root = base_url.rstrip("/")
        self._v1_url = root if root.endswith("/v1") else f"{root}/v1"
        self.model = model
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._opener = opener if opener is not None else urllib.request.build_opener()

    def _model(self, adapter_name: str | None) -> str:
        if adapter_name is None:
            return self.model
        if not isinstance(adapter_name, str) or not adapter_name:
            raise ValueError("adapter_name must be a non-empty string or None")
        return adapter_name

    def build_decision_payload(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        instructions: str | None = None,
        adapter_name: str | None = None,
    ) -> dict[str, Any]:
        questions = _mapping(questions, "request.questions")
        if not 1 <= len(questions) <= 64:
            raise VLLMProtocolError("request.questions needs 1-64 questions")
        normalized_questions = {}
        for question_id, spec in questions.items():
            normalized, _, _ = _question_spec(question_id, spec)
            normalized_questions[question_id] = dict(normalized)
        if instructions is not None and not isinstance(instructions, str):
            raise VLLMProtocolError("request.instructions must be a string or null")
        payload = {
            "model": self._model(adapter_name),
            "state": state,
            "questions": normalized_questions,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if instructions is not None:
            payload["instructions"] = instructions
        return payload

    def build_score_payload(self, read: Mapping[str, Any], *, adapter_name: str | None = None) -> dict[str, Any]:
        read = _mapping(read, "encoded read")
        question_id = read.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise VLLMProtocolError("encoded read question_id must be a non-empty string")
        prompt_token_ids = read.get("prompt_token_ids")
        labels = read.get("labels")
        label_token_ids = read.get("label_token_ids")
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise VLLMProtocolError(f"encoded read {question_id!r} needs prompt token IDs")
        prompt_token_ids = [
            _integer(value, f"encoded read {question_id!r} prompt token ID") for value in prompt_token_ids
        ]
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise VLLMProtocolError(f"encoded read {question_id!r} needs non-empty string labels")
        if len(set(labels)) != len(labels):
            raise VLLMProtocolError(f"encoded read {question_id!r} has duplicate labels")
        if not isinstance(label_token_ids, list) or len(label_token_ids) != len(labels) or not label_token_ids:
            raise VLLMProtocolError(f"encoded read {question_id!r} label IDs do not match labels")
        label_token_ids = [_integer(value, f"encoded read {question_id!r} label token ID") for value in label_token_ids]
        if len(set(label_token_ids)) != len(label_token_ids):
            raise VLLMProtocolError(f"encoded read {question_id!r} label token IDs must be distinct")
        return {
            "model": self._model(adapter_name),
            "prompt": prompt_token_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": len(label_token_ids),
            "logprob_token_ids": label_token_ids,
            "add_special_tokens": False,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        }

    def _post(self, path: str, payload: Mapping[str, Any]) -> Any:
        request = urllib.request.Request(
            f"{self._v1_url}/{path}",
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            body = exc.read(64 * 1024).decode("utf-8", errors="replace")
            raise VLLMHTTPError(exc.code, body) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise VLLMAdapterError(f"vLLM request failed: {exc}") from exc
        if len(raw) > 16 * 1024 * 1024:
            raise VLLMProtocolError("vLLM response is too large")
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise VLLMProtocolError("vLLM returned invalid JSON") from None

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        instructions: str | None = None,
        adapter_name: str | None = None,
    ) -> dict[str, Any]:
        payload = self.build_decision_payload(state, questions, instructions=instructions, adapter_name=adapter_name)
        response = self._post("systemone", payload)
        return validate_systemone_response(response, payload)

    def score_labels(self, reads: Sequence[Mapping[str, Any]], *, adapter_name: str | None = None) -> list[LabelScore]:
        if isinstance(reads, (str, bytes)) or not isinstance(reads, Sequence):
            raise VLLMProtocolError("encoded reads must be a sequence")
        scores = []
        for read in reads:
            payload = self.build_score_payload(read, adapter_name=adapter_name)
            question_id = read["question_id"]
            response = _mapping(self._post("completions", payload), f"score response {question_id!r}")
            if response.get("object") != "text_completion" or response.get("model") != payload["model"]:
                raise VLLMProtocolError(f"score response {question_id!r} has invalid object or model")
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise VLLMProtocolError(f"score response {question_id!r} must contain one choice")
            choice = _mapping(choices[0], f"score response {question_id!r} choice")
            if choice.get("index") != 0 or choice.get("finish_reason") == "error":
                raise VLLMProtocolError(f"score response {question_id!r} has an invalid choice")
            prompt_token_ids = choice.get("prompt_token_ids")
            if prompt_token_ids != payload["prompt"]:
                raise VLLMProtocolError(f"score response {question_id!r} changed the prompt token IDs")
            token_ids = choice.get("token_ids")
            if not isinstance(token_ids, list) or len(token_ids) != 1:
                raise VLLMProtocolError(f"score response {question_id!r} must return exactly one sampled token")
            sampled_token_id = _integer(token_ids[0], f"score response {question_id!r} sampled token")
            logprobs = _mapping(choice.get("logprobs"), f"score response {question_id!r} logprobs")
            top_logprobs = logprobs.get("top_logprobs")
            if not isinstance(top_logprobs, list) or len(top_logprobs) != 1:
                raise VLLMProtocolError(f"score response {question_id!r} must return one logprob position")
            values = _mapping(top_logprobs[0], f"score response {question_id!r} top_logprobs")
            raw_logprobs = []
            for token_id in payload["logprob_token_ids"]:
                key = f"token_id:{token_id}"
                if key not in values:
                    raise VLLMProtocolError(
                        f"score response {question_id!r} is missing raw logprob for token {token_id}"
                    )
                value = _number(values[key], f"score response {question_id!r} token {token_id} logprob")
                if value > 0.0:
                    raise VLLMProtocolError(f"score response {question_id!r} has a positive logprob")
                raw_logprobs.append(value)
            scores.append(
                LabelScore(
                    question_id=question_id,
                    labels=tuple(read["labels"]),
                    label_token_ids=tuple(payload["logprob_token_ids"]),
                    raw_logprobs=tuple(raw_logprobs),
                    prompt_token_ids=tuple(prompt_token_ids),
                    sampled_token_id=sampled_token_id,
                    http_response=dict(response),
                )
            )
        return scores
