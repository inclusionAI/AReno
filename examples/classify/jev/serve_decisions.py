"""Serve a classify checkpoint behind the Jev decisions API.

    POST /api/alpha/decisions  {model, state, questions} -> {model, answers, usage, latency_ms}
    POST /v1/systemone         same contract (jev-forge's path)
    GET  /v1/models, /health

The checkpoint is an AReno classify output (`step_XXXXXX/`): an HF backbone
plus `score_head.safetensors`, e.g. `~/areno-runs/ling-3.0-tiny-jev`. The
backbone is loaded with `trust_remote_code=True`, so Ling (`bailing_hybrid`)
works through its own modeling code.

Candidate paths are encoded exactly as in training (`dataset_loader.py`). All
paths of one request are scored in one right-padded batch; each path is read
at its own last token. `--batch-mode auto` first checks that padded scores
match unpadded ones and falls back to one path per forward if they do not.

Answers follow jev-forge's `Predictor.decide`:
    noul   -> {"type": "noul", "noul": P(true)}
    choice -> {"type": "choice", "choice": id, "probabilities": {...}, "confidence": c}
    score  -> {"type": "score", "score": E[level], "legend": {...}, "probabilities": {...}, "confidence": c}

    python examples/classify/jev/serve_decisions.py --checkpoint ~/areno-runs/ling-3.0-tiny-jev --port 8123
"""

# No `from __future__ import annotations`: FastAPI must resolve the
# `Request` annotation of the locally defined route at runtime.
import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import ANSWER_MARK, candidate_ids, question_prefix, render_candidate  # noqa: E402

QUESTION_TYPES = ("choice", "score", "noul")
MAX_QUESTIONS = 96
MAX_PATHS = 512
logger = logging.getLogger("jev.serve")


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_question(qid: str, question: dict) -> None:
    """Same rules as jev-forge `schema.validate_question`."""

    if not _nonempty(qid) or not isinstance(question, dict):
        raise ValueError(f"invalid question id or body: {qid!r}")
    extra = set(question) - {"type", "instructions", "criteria"}
    if extra:
        raise ValueError(f"question {qid}: unsupported fields {sorted(extra)}")
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"question {qid}: type must be one of {QUESTION_TYPES}")
    if not _nonempty(question.get("instructions")):
        raise ValueError(f"question {qid}: instructions must be a non-empty string")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError(f"question {qid}: choice criteria must map 2-255 options")
        if not all(_nonempty(k) and _nonempty(v) for k, v in criteria.items()):
            raise ValueError(f"question {qid}: choice ids and descriptions must be non-empty strings")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError(f"question {qid}: score criteria must list 2-10 ordered levels")
        if not all(_nonempty(v) for v in criteria):
            raise ValueError(f"question {qid}: score levels must be non-empty strings")
    elif "criteria" in question:
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise ValueError(f"question {qid}: noul criteria may only hold true/false")
        if not all(_nonempty(v) for v in criteria.values()):
            raise ValueError(f"question {qid}: noul criteria must be non-empty strings")


def render_state(state) -> str:
    """Text states pass through; JSON states are serialized like jev-forge."""

    if _nonempty(state):
        return state
    if isinstance(state, dict | list) and state:
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    raise ValueError("state must be non-empty text or a JSON object")


def confidence_from(probabilities: list[float]) -> float:
    """0 for a uniform distribution, 1 for one-hot (jev-forge)."""

    k = len(probabilities)
    if k < 2:
        return 1.0
    return max(0.0, min(1.0, (max(probabilities) - 1.0 / k) * k / (k - 1)))


class DecisionModel:
    """HF backbone + score head; scores candidate paths, returns Jev answers."""

    def __init__(self, checkpoint: str, *, max_length: int, temperature: float, batch_mode: str, max_batch: int):
        import torch
        from safetensors.torch import load_file
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        root = Path(checkpoint).expanduser().resolve(strict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
        pad = self.tokenizer.pad_token_id
        self.pad_id = int(pad if pad is not None else (self.tokenizer.eos_token_id or 0))
        self.backbone = (
            AutoModel.from_pretrained(str(root), trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
        )
        state = load_file(str(root / "score_head.safetensors"))
        hidden = int(state["0.weight"].shape[0])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.head.load_state_dict(state)
        self.head = self.head.cuda().float().eval()
        self.max_length = max_length
        self.temperature = temperature
        self.max_batch = max_batch
        self.calls = 0
        if batch_mode == "auto":
            self.padded = self._padding_matches_sequential()
        else:
            self.padded = batch_mode == "padded"
        logger.info("batch mode: %s", "padded (parallel)" if self.padded else "sequential")

    def _encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def encode(self, state: str, questions: dict) -> list[tuple[str, dict, list[list[int]]]]:
        """Same token layout as training: encode(prefix) + encode(candidate)."""

        encoded = []
        for qid, question in questions.items():
            prefix = self._encode(question_prefix(state, question))
            leaves = [
                prefix + self._encode(render_candidate(question, index) + "\n" + ANSWER_MARK)
                for index in range(len(candidate_ids(question)))
            ]
            longest = max(len(leaf) for leaf in leaves)
            if longest > self.max_length:
                raise ValueError(f"question {qid}: candidate path has {longest} tokens > max_length={self.max_length}")
            encoded.append((qid, question, leaves))
        return encoded

    def score(self, leaves: list[list[int]]):
        """One logit per path; padded batches when enabled, else one path per forward."""

        torch = self.torch
        with torch.inference_mode():
            if not self.padded:
                rows = []
                for leaf in leaves:
                    ids = torch.tensor([leaf], device="cuda")
                    rows.append(self.backbone(input_ids=ids).last_hidden_state[0, -1])
                return self.head(torch.stack(rows).float()).squeeze(-1)
            out = []
            for start in range(0, len(leaves), self.max_batch):
                chunk = leaves[start : start + self.max_batch]
                width = max(len(leaf) for leaf in chunk)
                ids = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(chunk), width), dtype=torch.long)
                for row, leaf in enumerate(chunk):
                    ids[row, : len(leaf)] = torch.tensor(leaf)
                    mask[row, : len(leaf)] = 1
                hidden = self.backbone(input_ids=ids.cuda(), attention_mask=mask.cuda()).last_hidden_state
                last = mask.sum(dim=1).cuda() - 1
                rows = hidden[torch.arange(len(chunk), device="cuda"), last]
                out.append(self.head(rows.float()).squeeze(-1))
            return torch.cat(out)

    def _padding_matches_sequential(self) -> bool:
        probe = {
            "q": {
                "type": "choice",
                "instructions": "Which option fits?",
                "criteria": {"a": "short", "b": "a noticeably longer option description", "c": "mid length"},
            }
        }
        leaves = self.encode("probe state for padding check", probe)[0][2]
        self.padded = False
        sequential = self.score(leaves)
        self.padded = True
        padded = self.score(leaves)
        diff = float((sequential - padded).abs().max())
        logger.info("padding self-check: max |padded - sequential| = %.4g", diff)
        return diff < 5e-2

    def decide(self, state: str, questions: dict) -> tuple[dict, int]:
        torch = self.torch
        encoded = self.encode(state, questions)
        flat = [leaf for _, _, leaves in encoded for leaf in leaves]
        logits = self.score(flat)
        self.calls += 1
        answers = {}
        offset = 0
        for qid, question, leaves in encoded:
            group = logits[offset : offset + len(leaves)]
            offset += len(leaves)
            values = torch.softmax(group / self.temperature, dim=-1).tolist()
            values = [v if math.isfinite(v) else 0.0 for v in values]
            total = math.fsum(values) or 1.0
            values = [v / total for v in values]
            ids = candidate_ids(question)
            kind = question["type"]
            if kind == "noul":
                answers[qid] = {"type": "noul", "noul": values[1]}
            elif kind == "choice":
                best = max(range(len(ids)), key=values.__getitem__)
                answers[qid] = {
                    "type": "choice",
                    "choice": ids[best],
                    "probabilities": dict(zip(ids, values, strict=True)),
                    "confidence": confidence_from(values),
                }
            else:
                answers[qid] = {
                    "type": "score",
                    "score": math.fsum(i * v for i, v in enumerate(values)),
                    "legend": {str(i): text for i, text in enumerate(question["criteria"])},
                    "probabilities": {str(i): v for i, v in enumerate(values)},
                    "confidence": confidence_from(values),
                }
        return answers, sum(len(leaf) for leaf in flat)


def build_app(model: DecisionModel, model_name: str, api_key: str | None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="ling-jev decisions")

    def error(status: int, message: str):
        return JSONResponse(status_code=status, content={"error": message})

    @app.get("/health")
    def health():
        return {"ready": True, "model": model_name, "calls": model.calls, "parallel": model.padded}

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": model_name, "owned_by": "areno", "temperature": model.temperature}]}

    async def decisions(request: Request):
        started = time.perf_counter()
        if api_key is not None and request.headers.get("authorization") != f"Bearer {api_key}":
            return error(401, "invalid or missing bearer token")
        try:
            payload = await request.json()
        except ValueError:
            return error(400, "body must be JSON")
        if not isinstance(payload, dict) or set(payload) - {"state", "model", "questions"}:
            return error(422, "body must hold state, model, questions")
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            return error(422, "questions must be a non-empty object")
        if len(questions) > MAX_QUESTIONS:
            return error(422, f"at most {MAX_QUESTIONS} questions per request")
        try:
            state = render_state(payload.get("state"))
            paths = 0
            for qid, question in questions.items():
                validate_question(qid, question)
                paths += 2 if question["type"] == "noul" else len(question["criteria"])
            if paths > MAX_PATHS:
                return error(422, f"{paths} candidate paths exceed the {MAX_PATHS} limit")
            answers, input_tokens = model.decide(state, questions)
        except ValueError as exc:
            return error(422, str(exc))
        return {
            "model": model_name,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0, "candidate_paths": paths},
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    app.add_api_route("/api/alpha/decisions", decisions, methods=["POST"])
    app.add_api_route("/v1/systemone", decisions, methods=["POST"])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="classify step dir (HF backbone + score_head.safetensors)")
    parser.add_argument("--model-name", default=None, help="defaults to the checkpoint directory name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--max-length", type=int, default=512, help="longest candidate path accepted")
    parser.add_argument("--temperature", type=float, default=1.0, help="calibration temperature")
    parser.add_argument("--batch-mode", choices=["auto", "padded", "sequential"], default="auto")
    parser.add_argument("--max-batch", type=int, default=64, help="candidate paths per forward")
    parser.add_argument("--api-key", default=None, help="require `Authorization: Bearer <key>` when set")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    model = DecisionModel(
        args.checkpoint,
        max_length=args.max_length,
        temperature=args.temperature,
        batch_mode=args.batch_mode,
        max_batch=args.max_batch,
    )
    name = args.model_name or Path(args.checkpoint).expanduser().name
    import uvicorn

    uvicorn.run(build_app(model, name, args.api_key), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
