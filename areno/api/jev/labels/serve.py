"""Native AReno serving of the same LM-label protocol used for training."""

import asyncio
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from areno.api.jev.labels.protocol import encode_question


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str | dict | list
    questions: dict[str, dict]
    model: str | None = None
    temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    return_prompt_token_ids: bool = False


class LabelDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str | dict | list
    questions: list[dict]
    model: str | None = None
    temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    return_prompt_token_ids: bool = False
    prompt_format_version: int = 1
    chat_template_kwargs: dict = Field(default_factory=dict)


def native_questions(questions):
    result = {}
    for item in questions:
        qid = item["id"]
        if not isinstance(qid, str) or not qid or qid in result:
            raise ValueError("question IDs must be nonempty and unique")
        kind = item["type"]
        criteria = None
        if kind == "choice":
            options = item["options"]
            criteria = {option["name"]: option.get("description", "") for option in options}
            if len(criteria) != len(options):
                raise ValueError("choice option names must be unique")
        elif kind == "score":
            criteria = item["levels"]
        elif kind == "yes_no":
            criteria = {"true": item.get("yes", ""), "false": item.get("no", "")}
            kind = "noul"
        else:
            raise ValueError("question type must be choice, score, or yes_no")
        result[qid] = {"type": kind, "instructions": item.get("question"), "criteria": criteria}
    return result


class LabelDecisionScorer:
    def __init__(
        self,
        model_path,
        *,
        backend_type,
        lora=None,
        quantization_bits=None,
        quantization_group_size=64,
        attn_backend="native",
        eager_decode=True,
    ):
        from areno.api.models import BackendType
        from areno.api.tokenizer import load_tokenizer

        self.engine = self.trainer = None
        self.tokenizer = load_tokenizer(model_path)
        metadata_root = Path(lora.adapter_path) if lora is not None and lora.adapter_path else Path(model_path)
        metadata_path = metadata_root / "label_jev_config.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata and (metadata.get("format") != "areno-label-jev-v1" or metadata.get("prompt_version") != 1):
            raise ValueError("unsupported LM-label JEV checkpoint protocol")
        self.temperature = float(metadata.get("temperature", 1.0))
        self.max_seq_len = int(metadata.get("max_seq_len", 4096))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("checkpoint temperature must be finite and positive")
        template_hash = hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest()
        if metadata.get("chat_template_sha256", template_hash) != template_hash:
            raise ValueError("tokenizer chat template differs from the training checkpoint")
        self.backend_type = backend_type
        if backend_type == BackendType.MLX:
            from areno import Trainer
            from areno.api.jev.labels.config import LabelJevMlxConfig

            config = LabelJevMlxConfig(
                lora=lora,
                inference_only=True,
                quantization_bits=quantization_bits,
                quantization_group_size=quantization_group_size,
            )
            self.trainer = Trainer(1, model_path, backend_type=backend_type, custom_config=config)
            self.trainer.init()
            self.model = self.trainer._backend.model
            self.model.eval()
        elif backend_type == BackendType.CUDA:
            from areno.engine.api import ArenoEngine
            from areno.engine.config import RuntimeConfig

            self.engine = ArenoEngine.from_pretrained(
                model_path,
                tp_size=1,
                devices=[0],
                role="rollout",
                lora_config=lora,
                runtime_config=RuntimeConfig(attn_backend=attn_backend, eager_decode=eager_decode, compile_model=False),
            )
        else:
            raise ValueError("label JEV serving requires CUDA or MLX")

    def label_logprobs(self, encoded):
        if self.engine is not None:
            sequences = [encoded["prompt_token_ids"] + [label] for label in encoded["label_token_ids"]]
            rows = self.engine.score_logprobs(
                "actor", sequences, pad_token_id=self.tokenizer.pad_token_id or 0, microbatch_size=1
            )
            return [row[-1] for row in rows]
        import mlx.core as mx

        logits = self.model(mx.array([encoded["prompt_token_ids"]]))
        logits = logits.logits if hasattr(logits, "logits") else logits
        logits = logits[0, -1].astype(mx.float32)
        logprobs = logits[mx.array(encoded["label_token_ids"])] - mx.logsumexp(logits)
        return logprobs.tolist()

    def decide(self, state, questions, *, temperature=None, return_prompt_token_ids=False, native=False):
        temperature = self.temperature if temperature is None else temperature
        if not math.isfinite(temperature) or temperature <= 0 or not questions:
            raise ValueError("nonempty questions and a finite positive temperature are required")
        answers, tokens = {}, 0
        for qid, question in questions.items():
            encoded = encode_question(state, question, self.tokenizer, max_seq_len=self.max_seq_len)
            logprobs = self.label_logprobs(encoded)
            if len(logprobs) != len(encoded["labels"]) or not all(math.isfinite(value) for value in logprobs):
                raise ValueError("model returned invalid label logprobs")
            largest = max(logprobs)
            weights = [math.exp((value - largest) / temperature) for value in logprobs]
            total = math.fsum(weights)
            probabilities = [value / total for value in weights]
            kind = question["type"]
            names = encoded["labels"] if native and kind == "noul" else encoded["names"]
            answer = {
                "type": "yes_no" if native and kind == "noul" else kind,
                "probabilities": dict(zip(names, probabilities, strict=True)),
                "label_mass": math.fsum(math.exp(value) for value in logprobs),
                "label_logprobs": logprobs,
            }
            if kind == "choice":
                answer["choice"] = names[probabilities.index(max(probabilities))]
            elif kind == "score":
                answer["score"] = math.fsum(i * p for i, p in enumerate(probabilities))
            elif not native:
                answer["noul"] = probabilities[0]
            if return_prompt_token_ids:
                answer.update(prompt_token_ids=encoded["prompt_token_ids"], label_token_ids=encoded["label_token_ids"])
            answers[qid] = answer
            tokens += len(encoded["prompt_token_ids"])
        return answers, tokens

    def close(self):
        if self.engine is not None:
            self.engine.close()
        if self.trainer is not None:
            self.trainer.close()


def create_label_decision_app(
    model_path,
    *,
    backend_type,
    lora=None,
    quantization_bits=None,
    quantization_group_size=64,
    attn_backend="native",
    eager_decode=True,
    scorer_factory=None,
):
    factory = scorer_factory or LabelDecisionScorer

    @asynccontextmanager
    async def lifespan(app):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="areno-label-jev")
        app.state.executor = executor
        try:
            app.state.scorer = await asyncio.get_running_loop().run_in_executor(
                executor,
                partial(
                    factory,
                    model_path,
                    backend_type=backend_type,
                    lora=lora,
                    quantization_bits=quantization_bits,
                    quantization_group_size=quantization_group_size,
                    attn_backend=attn_backend,
                    eager_decode=eager_decode,
                ),
            )
            yield
        finally:
            if getattr(app.state, "scorer", None) is not None:
                await asyncio.get_running_loop().run_in_executor(executor, app.state.scorer.close)
            executor.shutdown(wait=True)

    app = FastAPI(title="AReno LM-label JEV", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ready": getattr(app.state, "scorer", None) is not None, "model": model_path}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_path, "object": "model", "owned_by": "areno"}]}

    async def execute(payload, state, questions, *, native=False):
        if payload.model not in {None, "default", model_path}:
            raise HTTPException(404, "unknown decision model")
        try:
            answers, tokens = await asyncio.get_running_loop().run_in_executor(
                app.state.executor,
                partial(
                    app.state.scorer.decide,
                    state,
                    questions,
                    temperature=payload.temperature,
                    return_prompt_token_ids=payload.return_prompt_token_ids,
                    native=native,
                ),
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "model": model_path,
            "prompt_format_version": 1,
            "answers": answers,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens, "output_tokens": 0},
        }

    @app.post("/v1/systemone")
    @app.post("/api/alpha/decisions")
    async def systemone(payload: SystemOneRequest):
        return await execute(payload, payload.state, payload.questions)

    @app.post("/v1/decisions")
    async def decisions(payload: LabelDecisionRequest):
        if payload.prompt_format_version != 1:
            raise HTTPException(422, "only prompt_format_version=1 is supported")
        if (
            set(payload.chat_template_kwargs) - {"enable_thinking"}
            or payload.chat_template_kwargs.get("enable_thinking", False) is not False
        ):
            raise HTTPException(422, "label JEV requires thinking disabled and the saved chat template")
        try:
            questions = native_questions(payload.questions)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return await execute(payload, payload.input, questions, native=True)

    return app
