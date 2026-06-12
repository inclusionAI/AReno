"""OpenAI-compatible FastAPI server fronting the areno engine.

Exposes a `/v1/chat/completions` endpoint backed by concurrent rollout calls.
HTTP disconnects resolve the client request promptly. The engine rollout keeps
running so worker-side continuous batching can admit later requests.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import click
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from areno.cli.model_refs import resolve_model_ref
from areno.api.openai_chat import build_chat_completion_response, messages_to_prompt_tokens
from areno.api.tool_call_parser import ToolCallParser, get_tool_call_parser, infer_tool_call_parser_name
from areno.engine.config import RuntimeConfig
from areno.engine.data import SamplingParams
from areno.engine.data.tokenizer import load_tokenizer
from areno.engine import ArenoEngine


def _serve_loss_fn(*_: Any) -> torch.Tensor:
    """Placeholder loss function; serving never trains, so any invocation is an error."""
    raise RuntimeError("areno serve engine does not support training")


class ChatMessage(BaseModel):
    """OpenAI chat message: role plus string or multi-part content."""

    role: Literal["system", "user", "assistant", "tool"] | str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat-completions request schema accepted by this server."""

    model: str | None = None
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None


class ChatCompletionChoice(BaseModel):
    """One generated completion within a response, indexed by `n` position."""

    index: int
    message: dict[str, Any]
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
    max_model_len: int
    tool_call_parser: ToolCallParser
    active_tasks: set[asyncio.Task] = field(default_factory=set)
    closing: bool = False
    rollout_session_started: bool = False


class _ToolParserTrainerShim:
    """Adapter used to infer the shared tool-call parser in serve mode."""

    def __init__(self, *, model_path: str, tokenizer: Any) -> None:
        self._model_path = model_path
        self._tokenizer = tokenizer

    def get_tokenizer(self) -> Any:
        return self._tokenizer


def create_app(
    *,
    model_path: str,
    tp_size: int,
    world_size: int,
    max_running_prompts: int,
    default_max_tokens: int,
    decode_progress_interval_s: float,
    eager_decode: bool = False,
) -> FastAPI:
    """Construct the FastAPI app: load tokenizer/engine, install routes and lifecycle hooks."""
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if tp_size < 1:
        raise ValueError("tp_size must be >= 1")
    if world_size % tp_size != 0:
        raise ValueError("world_size must be divisible by tp_size")

    tokenizer = load_tokenizer(model_path)
    parser_trainer = _ToolParserTrainerShim(model_path=model_path, tokenizer=tokenizer)
    engine = ArenoEngine.from_pretrained(
        model_path,
        tp_size=tp_size,
        dp_size=world_size // tp_size,
        devices=list(range(world_size)),
        runtime_config=RuntimeConfig(eager_decode=bool(eager_decode)),
        loss_fn=_serve_loss_fn,
    )
    state = ServeState(
        model_path=model_path,
        tokenizer=tokenizer,
        engine=engine,
        max_running_prompts=max_running_prompts,
        default_max_tokens=default_max_tokens,
        max_model_len=int(engine.config.model.max_position_embeddings),
        tool_call_parser=get_tool_call_parser(infer_tool_call_parser_name(parser_trainer)),
    )
    app = FastAPI(title="areno OpenAI-compatible server")
    app.state.areno_serve = state
    app.state.decode_progress_interval_s = float(decode_progress_interval_s)

    @app.on_event("startup")
    async def startup() -> None:
        """Open one long-lived rollout session for serving."""
        try:
            await state.engine.begin_rollout_session_async()
        except BaseException:
            state.engine.close()
            raise
        state.rollout_session_started = True

    @app.on_event("shutdown")
    async def shutdown() -> None:
        """Signal closing, drain in-flight request tasks, then tear the engine down."""
        state.closing = True
        if state.active_tasks:
            await asyncio.gather(*state.active_tasks, return_exceptions=True)
        try:
            if state.rollout_session_started:
                await state.engine.end_rollout_session_async()
        finally:
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

        prompt = _encode_messages(state.tokenizer, request.messages, tools=request.tools)
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
        max_prompt_len=max(state.max_model_len - key.max_new_tokens, len(item.prompt)),
        eos_token_id=key.eos_token_id,
        sampling_params=SamplingParams(
            temperature=key.temperature,
            top_p=key.top_p,
            top_k=key.top_k,
            seed=key.seed,
            stop_token_ids=key.stop_token_ids,
        ),
        decode_progress_interval_s=app.state.decode_progress_interval_s,
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
    return _build_response_from(state.tokenizer, state.model_path, state.tool_call_parser, request, prompt, response_ids, finish_reasons)


def _build_response_from(
    tokenizer: Any,
    model_path: str,
    tool_call_parser: ToolCallParser,
    request: ChatCompletionRequest,
    prompt: list[int],
    response_ids: list[list[int]],
    finish_reasons: list[str],
) -> ChatCompletionResponse:
    """Decode token ids, parse optional tool calls, and assemble the OpenAI envelope."""

    data = build_chat_completion_response(
        tokenizer=tokenizer,
        model=request.model or model_path,
        prompt_tokens=len(prompt) * len(response_ids),
        response_ids=response_ids,
        finish_reasons=finish_reasons,
        tools=request.tools,
        tool_choice=request.tool_choice,
        tool_call_parser=tool_call_parser,
        stop_strings=_normalize_stop(request.stop),
    )
    return ChatCompletionResponse(**data)


def _encode_messages(tokenizer: Any, messages: list[ChatMessage], *, tools: list[dict[str, Any]] | None = None) -> list[int]:
    """Tokenise a chat history, using the tokenizer's chat template when available."""
    payload = [_chat_message_payload(msg) for msg in messages]
    return messages_to_prompt_tokens(tokenizer, payload, tools=tools, fallback_prompt=_messages_fallback_text(payload))


def _chat_message_payload(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": _message_content(message.content)}
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls is not None:
        payload["tool_calls"] = message.tool_calls
    return payload


def _messages_fallback_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{msg['role']}: {msg.get('content', '')}" for msg in messages) + "\nassistant:"


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
@click.option("--eager-decode", is_flag=True, help="Run decode in eager mode instead of CUDA graph replay.")
def serve_command(
    model_path: str,
    tp_size: int,
    world_size: int,
    host: str,
    port: int,
    max_running_prompts: int,
    default_max_tokens: int,
    decode_progress_interval_s: float,
    eager_decode: bool,
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
        eager_decode=eager_decode,
    )
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Console-script entrypoint for `areno serve`."""

    serve_command.main(prog_name="areno serve")


if __name__ == "__main__":
    main()
