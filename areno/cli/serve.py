"""OpenAI-compatible FastAPI server fronting the areno engine.

Exposes a `/v1/chat/completions` endpoint backed by concurrent rollout calls.
HTTP disconnects resolve the client request promptly. The engine rollout keeps
running so concurrent requests can be coalesced for throughput.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import click
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from areno.cli.model_refs import resolve_model_ref
from areno.engine.data import SamplingParams
from areno.engine.data.tokenizer import load_tokenizer
from areno.engine import ArenoEngine


SERVE_COALESCE_DEADLINE_S = 5.0


def _serve_loss_fn(*_: Any) -> torch.Tensor:
    """Placeholder loss function; serving never trains, so any invocation is an error."""
    raise RuntimeError("areno serve engine does not support training")


class ChatMessage(BaseModel):
    """OpenAI chat message: role plus string or multi-part content."""

    role: Literal["system", "user", "assistant", "tool"] | str
    content: str | list[Any] | None = None


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat-completions request schema accepted by this server."""


    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None


class ChatCompletionChoice(BaseModel):
    """One generated completion within a response, indexed by `n` position."""

    index: int
    message: dict[str, str]
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """Token accounting echoed back to the caller."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response envelope."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


@dataclass(frozen=True, slots=True)
class BatchKey:
    """Hashable bundle of fields that must match for two requests to share a rollout.

    Requests with identical `BatchKey` produce bit-comparable sampling behaviour
    (same length budget, temperature/top-p/top-k, seed, stop ids, eos id) and
    can therefore be merged into one engine call.
    """

    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int | None
    stop_token_ids: tuple[int, ...]
    eos_token_id: int | None


@dataclass(slots=True)
class PendingRequest:
    """Per-request bookkeeping carried from HTTP handler through the scheduler.

    `future` is the asyncio handoff used to deliver the response back to the
    request coroutine.
    """

    request: ChatCompletionRequest
    prompt: list[int]
    key: BatchKey
    future: asyncio.Future
    created_at: float = field(default_factory=time.monotonic)
    cancelled: bool = False


@dataclass(slots=True)
class ServeState:
    """Process-wide serving state held on `app.state.areno_serve`.

    Holds the loaded engine/tokenizer and tracks in-flight request tasks.
    """

    model_path: str
    tokenizer: Any
    engine: ArenoEngine
    max_running_prompts: int
    default_max_tokens: int
    active_tasks: set[asyncio.Task] = field(default_factory=set)
    closing: bool = False


def create_app(
    *,
    model_path: str,
    tp_size: int,
    world_size: int,
    max_running_prompts: int,
    default_max_tokens: int,
    decode_progress_interval_s: float,
) -> FastAPI:
    """Construct the FastAPI app: load tokenizer/engine, install routes and lifecycle hooks."""
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if tp_size < 1:
        raise ValueError("tp_size must be >= 1")
    if world_size % tp_size != 0:
        raise ValueError("world_size must be divisible by tp_size")

    tokenizer = load_tokenizer(model_path)
    engine = ArenoEngine.from_pretrained(
        model_path,
        tp_size=tp_size,
        dp_size=world_size // tp_size,
        devices=list(range(world_size)),
        loss_fn=_serve_loss_fn,
    )
    state = ServeState(
        model_path=model_path,
        tokenizer=tokenizer,
        engine=engine,
        max_running_prompts=max_running_prompts,
        default_max_tokens=default_max_tokens,
    )
    app = FastAPI(title="areno OpenAI-compatible server")
    app.state.areno_serve = state
    app.state.decode_progress_interval_s = float(decode_progress_interval_s)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        """Signal closing, drain in-flight request tasks, then tear the engine down."""
        state.closing = True
        if state.active_tasks:
            await asyncio.gather(*state.active_tasks, return_exceptions=True)
        state.engine.close()

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        """Single-entry OpenAI-style model listing for the loaded checkpoint."""
        return {
            "object": "list",
            "data": [
                {
                    "id": state.model_path,
                    "object": "model",
                    "created": 0,
                    "owned_by": "areno",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(raw_request: Request, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Validate the request, encode the prompt, run rollout, and await the response."""
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must be non-empty")

        prompt = _encode_messages(state.tokenizer, request.messages)
        key = BatchKey(
            max_new_tokens=int(request.max_completion_tokens or request.max_tokens or state.default_max_tokens),
            temperature=float(request.temperature),
            top_p=float(request.top_p),
            top_k=int(request.top_k),
            seed=request.seed,
            stop_token_ids=_stop_token_ids(state.tokenizer),
            eos_token_id=_first_eos_token_id(state.tokenizer),
        )
        pending = PendingRequest(
            request=request,
            prompt=prompt,
            key=key,
            future=asyncio.get_running_loop().create_future(),
        )
        if state.closing:
            raise HTTPException(status_code=503, detail="server is shutting down")
        task = asyncio.create_task(_run_request_task(app, pending))
        state.active_tasks.add(task)
        task.add_done_callback(state.active_tasks.discard)
        return await _await_pending_response(state, raw_request, pending)

    return app


async def _run_request_task(app: FastAPI, item: PendingRequest) -> None:
    """Run one HTTP request as an independent concurrent rollout call."""

    try:
        response = await _run_request_rollout(app, item)
        _set_future_result(item.future, response)
    except BaseException as exc:
        if not item.future.done():
            item.future.set_exception(exc)


async def _run_request_rollout(app: FastAPI, item: PendingRequest) -> ChatCompletionResponse | None:
    """Run one request through the async engine rollout path."""

    state: ServeState = app.state.areno_serve
    key = item.key
    prompts = [item.prompt for _ in range(int(item.request.n))]
    if item.cancelled or item.future.done():
        return None

    rollout = await state.engine.generate_rollout_async(
        prompts,
        max_new_tokens=key.max_new_tokens,
        max_running_prompts=max(state.max_running_prompts, len(prompts)),
        eos_token_id=key.eos_token_id,
        sampling_params=SamplingParams(
            temperature=key.temperature,
            top_p=key.top_p,
            top_k=key.top_k,
            seed=key.seed,
            stop_token_ids=key.stop_token_ids,
        ),
        decode_progress_interval_s=app.state.decode_progress_interval_s,
        coalesce_timeout_s=SERVE_COALESCE_DEADLINE_S,
    )
    if item.future.done():
        return None
    return _build_response(state, item.request, item.prompt, rollout.response_ids, rollout.finish_reason)


def _set_future_result(future: asyncio.Future, response: ChatCompletionResponse) -> None:
    """Resolve `future` with `response` unless something else got there first."""
    if response is not None and not future.done():
        future.set_result(response)


async def _await_pending_response(state: ServeState, raw_request: Request, item: PendingRequest) -> ChatCompletionResponse:
    """Wait for `item.future`, run a disconnect watcher in parallel, and return the response.

    Uses `asyncio.shield` so a cancelled awaiter (e.g. client gone) does not
    propagate cancellation into the future itself; instead we explicitly mark
    the request cancelled and synthesise an empty response.
    """
    disconnect_task = asyncio.create_task(_watch_disconnect(state, raw_request, item))
    try:
        return await asyncio.shield(item.future)
    except asyncio.CancelledError:
        _cancel_pending_request(item)
        return _build_cancelled_response(state, item)
    finally:
        disconnect_task.cancel()


async def _watch_disconnect(state: ServeState, raw_request: Request, item: PendingRequest) -> None:
    """Poll the underlying HTTP request and flag cancellation if the client drops.

    On disconnect, resolves the future with an empty cancelled response so the
    caller's await returns promptly. The already-submitted engine rollout is
    allowed to finish so serve requests remain batchable.
    """
    while not item.future.done():
        if await raw_request.is_disconnected():
            _cancel_pending_request(item)
            if not item.future.done():
                item.future.set_result(_build_cancelled_response(state, item))
            return
        await asyncio.sleep(0.1)


def _cancel_pending_request(item: PendingRequest) -> None:
    """Mark the request cancelled."""
    item.cancelled = True


def _build_cancelled_response(state: ServeState, item: PendingRequest) -> ChatCompletionResponse:
    """Synthesise an empty-token response with stop finish reason for a cancelled request."""
    response_ids = [[] for _ in range(int(item.request.n))]
    finish_reasons = ["stop" for _ in response_ids]
    return _build_response(state, item.request, item.prompt, response_ids, finish_reasons)


def _build_response(
    state: ServeState,
    request: ChatCompletionRequest,
    prompt: list[int],
    response_ids: list[list[int]],
    finish_reasons: list[str],
) -> ChatCompletionResponse:
    """Thin shim that forwards to `_build_response_from` using state's tokenizer/model_path."""
    return _build_response_from(state.tokenizer, state.model_path, request, prompt, response_ids, finish_reasons)


def _build_response_from(
    tokenizer: Any,
    model_path: str,
    request: ChatCompletionRequest,
    prompt: list[int],
    response_ids: list[list[int]],
    finish_reasons: list[str],
) -> ChatCompletionResponse:
    """Decode token ids back to text, apply stop-string trimming, assemble the OpenAI envelope."""
    stop_strings = _normalize_stop(request.stop)
    choices: list[ChatCompletionChoice] = []
    completion_tokens = 0
    for index, token_ids in enumerate(response_ids):
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        text, stop_hit = _trim_stop_strings(text, stop_strings)
        completion_tokens += len(token_ids)
        finish_reason = "stop" if stop_hit or finish_reasons[index] == "stop" else "length"
        choices.append(
            ChatCompletionChoice(
                index=index,
                message={"role": "assistant", "content": text},
                finish_reason=finish_reason,
            )
        )
    prompt_tokens = len(prompt) * len(response_ids)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model or model_path,
        choices=choices,
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _encode_messages(tokenizer: Any, messages: list[ChatMessage]) -> list[int]:
    """Tokenise a chat history, using the tokenizer's chat template when available."""
    payload = [{"role": msg.role, "content": _message_content(msg.content)} for msg in messages]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            payload,
            tokenize=True,
            add_generation_prompt=True,
        )
    text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in payload) + "\nassistant:"
    return tokenizer.encode(text, add_special_tokens=True)


def _message_content(content: str | list[Any] | None) -> str:
    """Flatten OpenAI-style content (string or list of parts) into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        else:
            parts.append(str(item))
    return "\n".join(part for part in parts if part)


def _first_eos_token_id(tokenizer: Any) -> int | None:
    """Return the first eos id when the tokenizer reports one (handles list/int forms)."""
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        return eos
    if isinstance(eos, (list, tuple)) and eos:
        return int(eos[0])
    return None


def _stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return the tokenizer's eos id(s) as a tuple of ints for use in `BatchKey`."""
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        return (eos,)
    if isinstance(eos, (list, tuple)):
        return tuple(int(value) for value in eos)
    return ()


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    """Coerce the OpenAI `stop` field (str/list/None) into a list of non-empty strings."""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [value for value in stop if value]


def _trim_stop_strings(text: str, stop: list[str]) -> tuple[str, bool]:
    """Trim `text` at the earliest occurrence of any stop string; return (trimmed, hit?)."""
    if not stop:
        return text, False
    first = None
    for marker in stop:
        idx = text.find(marker)
        if idx >= 0 and (first is None or idx < first):
            first = idx
    if first is None:
        return text, False
    return text[:first], True


@click.command(
    name="serve",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Serve an OpenAI-compatible /v1/chat/completions API with areno.",
)
@click.option("--model-path", required=True, help="Local checkpoint/tokenizer path or Hugging Face repo ID.")
@click.option("--tp-size", type=int, default=1, show_default=True, help="Tensor parallel size.")
@click.option("--world-size", type=int, default=1, show_default=True, help="Total number of local worker ranks.")
@click.option("--host", default="0.0.0.0", show_default=True, help="HTTP bind host.")
@click.option("--port", type=int, default=8000, show_default=True, help="HTTP bind port.")
@click.option("--max-running-prompts", type=int, default=128, show_default=True, help="Maximum concurrent rollout prompts per request chunk.")
@click.option("--default-max-tokens", type=int, default=1024, show_default=True, help="Default max generated tokens.")
@click.option("--decode-progress-interval-s", type=float, default=0.0, show_default=True, help="Worker decode progress log interval.")
def serve_command(
    model_path: str,
    tp_size: int,
    world_size: int,
    host: str,
    port: int,
    max_running_prompts: int,
    default_max_tokens: int,
    decode_progress_interval_s: float,
) -> None:
    """Click entry point: build the app and hand it to uvicorn."""
    import uvicorn

    model_path = resolve_model_ref(model_path)
    app = create_app(
        model_path=model_path,
        tp_size=tp_size,
        world_size=world_size,
        max_running_prompts=max_running_prompts,
        default_max_tokens=default_max_tokens,
        decode_progress_interval_s=decode_progress_interval_s,
    )
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Console-script entrypoint for `areno serve`."""

    serve_command.main(prog_name="areno serve")


if __name__ == "__main__":
    main()
