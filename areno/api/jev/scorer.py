"""Jev-compatible typed decisions using the same provider and head as training."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from areno.api.jev.schema import ANSWER_MARK, candidate_ids, question_prefix, render_candidate


def validate_question(qid: str, question: dict[str, Any]) -> None:
    if not isinstance(qid, str) or not qid.strip() or not isinstance(question, dict):
        raise ValueError("question ids and bodies must be nonempty strings and objects")
    if set(question) - {"type", "instructions", "criteria"}:
        raise ValueError(f"question {qid}: unsupported fields")
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        raise ValueError(f"question {qid}: instructions must be nonempty text")
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError(f"question {qid}: choice requires 2-255 candidates")
        if any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
            for k, v in criteria.items()
        ):
            raise ValueError(f"question {qid}: candidate ids and descriptions must be nonempty text")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError(f"question {qid}: score requires 2-10 ordered levels")
        if any(not isinstance(v, str) or not v.strip() for v in criteria):
            raise ValueError(f"question {qid}: score levels must be nonempty text")
    elif kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict)
            or set(criteria) - {"true", "false"}
            or any(not isinstance(v, str) or not v.strip() for v in criteria.values())
        ):
            raise ValueError(f"question {qid}: noul criteria may contain only true/false descriptions")
    else:
        raise ValueError(f"question {qid}: type must be choice, score, or noul")


def render_state(state: Any) -> str:
    if isinstance(state, str) and state.strip():
        # Match training's canonical serialization when the state is JSON text.
        try:
            return json.dumps(json.loads(state), ensure_ascii=False, sort_keys=True)
        except ValueError:
            return state
    if isinstance(state, dict | list) and state:
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    raise ValueError("state must be nonempty text, object, or list")


def answer_from_logits(question: dict[str, Any], logits: Any, temperature: float) -> dict[str, Any]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scores = np.asarray(logits, dtype=np.float64) / temperature
    if not np.isfinite(scores).all():
        raise FloatingPointError("decision head produced nonfinite scores")
    probabilities = np.exp(scores - np.logaddexp.reduce(scores)).tolist()
    ids = candidate_ids(question)
    kind = question["type"]
    if kind == "noul":
        return {"type": kind, "noul": probabilities[1]}
    confidence = max(0.0, min(1.0, (max(probabilities) - 1 / len(ids)) * len(ids) / (len(ids) - 1)))
    answer = {"type": kind, "probabilities": dict(zip(ids, probabilities, strict=True)), "confidence": confidence}
    if kind == "choice":
        answer["choice"] = ids[int(np.argmax(probabilities))]
    else:
        answer["score"] = math.fsum(i * p for i, p in enumerate(probabilities))
        answer["legend"] = dict(zip(ids, question["criteria"], strict=True))
    return answer


class DecisionScorer:
    """Load an AReno LoRA+head checkpoint; no vocabulary projection or decoding."""

    def __init__(self, model_path, checkpoint, *, max_length=None, quantization_bits=None, quantization_group_size=64):
        import mlx.core as mx

        from areno import LoraConfig, Trainer
        from areno.api.jev.config import JevMlxConfig

        config = json.loads((Path(checkpoint) / "jev_config.json").read_text())
        self.temperature = float(config.get("temperature", 1.0))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("checkpoint temperature must be finite and positive")
        self.max_length = max_length or int(config.get("max_seq_len", 1024))
        runtime = JevMlxConfig(
            model_path=str(model_path),
            lora=LoraConfig(adapter_path=str(checkpoint)),
            decision_checkpoint=str(checkpoint),
            gradient_checkpointing=False,
            quantization_bits=quantization_bits,
            quantization_group_size=quantization_group_size,
            inference_only=True,
        )
        self.trainer = Trainer(1, str(model_path), custom_config=runtime)
        try:
            self.trainer.init()
            self.provider = self.trainer._backend.provider
            self.provider.model.eval()
            self.provider.model.set_dtype(mx.float32)
            mx.eval(self.provider.model.parameters())
        except BaseException:
            self.trainer.close()
            raise
        self.calls = 0

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], int]:
        import mlx.core as mx

        from areno.api.backend.mlx.decision import question_logits
        from areno.api.jev.data import encode_question

        state = render_state(state)
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 96:
            raise ValueError("questions must contain 1-96 questions")
        for qid, question in questions.items():
            validate_question(qid, question)
        if sum(len(candidate_ids(q)) for q in questions.values()) > 512:
            raise ValueError("at most 512 candidate paths are allowed per request")
        answers, tokens = {}, 0
        tokenizer = self.trainer.get_tokenizer()
        for qid, question in questions.items():
            ids = candidate_ids(question)
            row = {
                "prompt": question_prefix(state, question),
                "question_id": qid,
                "candidates": [render_candidate(question, i) + "\n" + ANSWER_MARK for i in range(len(ids))],
                "target": [1 / len(ids)] * len(ids),
            }
            encoded = encode_question(row, tokenizer, self.max_length)
            # Training and serving share the exact same suffix-preserving encoding.
            logits = question_logits(self.provider, encoded)
            mx.eval(logits)
            answers[qid] = answer_from_logits(question, np.asarray(logits), self.temperature)
            tokens += sum(len(item.tokens) for item in encoded)
            mx.clear_cache()
        self.calls += 1
        return answers, tokens

    def close(self) -> None:
        self.trainer.close()
