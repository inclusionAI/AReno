"""HTTP lifecycle and transport for the model-independent JEV scorer."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str | dict | list
    questions: dict[str, dict]
    model: str | None = None


def create_decision_app(
    model_path, checkpoint, *, model_name=None, scorer_factory=None, quantization_bits=None, quantization_group_size=64
):
    from areno.api.jev.scorer import DecisionScorer

    model_name = model_name or str(checkpoint)
    factory = scorer_factory or DecisionScorer

    @asynccontextmanager
    async def lifespan(app):
        # One worker owns all MLX operations, and serializes model access.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="areno-jev")
        app.state.executor = executor
        loop = asyncio.get_running_loop()
        try:
            kwargs = (
                {"quantization_bits": quantization_bits, "quantization_group_size": quantization_group_size}
                if quantization_bits is not None
                else {}
            )
            app.state.scorer = await loop.run_in_executor(executor, partial(factory, model_path, checkpoint, **kwargs))
            yield
        finally:
            scorer = getattr(app.state, "scorer", None)
            if scorer is not None:
                await loop.run_in_executor(executor, scorer.close)
                app.state.scorer = None
            executor.shutdown(wait=True)

    app = FastAPI(title="AReno JEV decisions", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ready": getattr(app.state, "scorer", None) is not None, "model": model_name}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "areno"}]}

    @app.post("/api/alpha/decisions")
    @app.post("/v1/systemone")
    async def decisions(payload: DecisionRequest):
        if payload.model is not None and payload.model != model_name:
            raise HTTPException(404, "unknown decision model")
        scorer = getattr(app.state, "scorer", None)
        if scorer is None:
            raise HTTPException(503, "decision model is not ready")
        started = time.perf_counter()
        try:
            answers, tokens = await asyncio.get_running_loop().run_in_executor(
                app.state.executor,
                scorer.decide,
                payload.state,
                payload.questions,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "model": model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    return app
