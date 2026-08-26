"""OpenAI-compatible FastAPI server fronting the areno engine.

Exposes a `/v1/chat/completions` endpoint backed by concurrent rollout calls.
HTTP disconnects resolve the client request promptly. The engine rollout keeps
running so worker-side continuous batching can admit later requests.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import click
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from areno.adapters import LoraConfig
from areno.api import MLX, BackendType, MlxConfig, SamplingParams, Trainer, default_backend_type
from areno.api.multimodal import (
    expand_image_tokens,
    image_token_counts_from_features,
    mrope_position_ids_from_image_grid,
)
from areno.api.openai_chat import build_chat_completion_response, messages_to_prompt_tokens
from areno.api.tokenizer import apply_chat_template_with_options, configure_chat_template_enable_thinking
from areno.api.tool_call_parser import ToolCallParser, get_tool_call_parser, infer_tool_call_parser_name
from areno.cli.model_refs import resolve_model_ref
from areno.engine.data.tokenizer import load_processor, load_tokenizer

# Kept as an injectable compatibility seam for CPU tests and embedders. The
# CUDA runtime imports the real class lazily; MLX installations never import it.
ArenoEngine = None


def config_from_hf(model_path: str):
    """Load CUDA model config lazily while preserving the injectable CLI seam."""

    from areno.models.registry import config_from_hf as load_config

    return load_config(model_path)


def flash_attention_unsupported_gpu_reason(devices):
    """Resolve CUDA capability lazily so MLX-only imports do not require Torch."""

    from areno.engine.config import flash_attention_unsupported_gpu_reason as resolve_reason

    return resolve_reason(devices)


def flash_attention_unsupported_model_reason(model_config):
    """Resolve model attention support lazily so tests can replace the probe."""

    from areno.engine.config import flash_attention_unsupported_model_reason as resolve_reason

    return resolve_reason(model_config)


def _serve_loss_fn(*_: Any) -> Any:
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
    prompt_features: dict[str, Any] | None
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
    processor: Any
    engine: Any
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


@dataclass(slots=True)
class _ServeRollout:
    response_ids: list[list[int]]
    finish_reason: list[str]


class _CudaServeRuntime:
    """Lazy CUDA adapter preserving the existing ArenoEngine serving path."""

    def __init__(
        self,
        model_path: str,
        *,
        tp_size: int,
        world_size: int,
        eager_decode: bool,
        attn_backend: str,
        lora: LoraConfig | None,
        base_model_name_or_path: str | None,
    ) -> None:
        from areno.engine.config import RuntimeConfig

        engine_cls = ArenoEngine
        if engine_cls is None:
            from areno.engine import ArenoEngine as engine_cls

        self._engine = engine_cls.from_pretrained(
            model_path,
            tp_size=tp_size,
            dp_size=world_size // tp_size,
            devices=list(range(world_size)),
            runtime_config=RuntimeConfig(eager_decode=bool(eager_decode), attn_backend=attn_backend),
            loss_fn=_serve_loss_fn,
            lora_config=lora,
            base_model_name_or_path=base_model_name_or_path,
        )
        self.max_model_len = int(self._engine.config.model.max_position_embeddings)

    async def begin_rollout_session_async(self) -> None:
        await self._engine.begin_rollout_session_async()

    async def end_rollout_session_async(self) -> None:
        await self._engine.end_rollout_session_async()

    def close(self) -> None:
        self._engine.close()

    async def generate_rollout_async(self, prompts, *, sampling_params: SamplingParams, **kwargs) -> Any:
        from areno.engine.data import SamplingParams as CudaSamplingParams

        cuda_sampling = CudaSamplingParams(
            temperature=0.0 if sampling_params.greedy else sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=max(sampling_params.top_k, 0),
            seed=getattr(sampling_params, "seed", None),
            stop_token_ids=tuple(sampling_params.stop_token_ids or ()),
        )
        return await self._engine.generate_rollout_async(prompts, sampling_params=cuda_sampling, **kwargs)


class _MlxServeRuntime:
    """Serve adapter over the same public Trainer lifecycle used by training."""

    def __init__(
        self,
        model_path: str,
        *,
        max_running_prompts: int,
        decode_progress_interval_s: float,
    ) -> None:
        config = MlxConfig(
            model_path=model_path,
            max_running_prompts=max_running_prompts,
            decode_progress_interval_s=decode_progress_interval_s,
        )
        self._trainer = Trainer(1, model_path, backend_type=MLX, custom_config=config)
        self._trainer.init()
        self.tokenizer = self._trainer.get_tokenizer()
        self.processor = self._trainer.get_processor()
        self.max_model_len = int(self._trainer.model_context_len() or 32768)

    async def begin_rollout_session_async(self) -> None:
        await self._trainer.begin_rollout_session_async()

    async def end_rollout_session_async(self) -> None:
        await self._trainer.end_rollout_session_async()

    def close(self) -> None:
        self._trainer.close()

    async def generate_rollout_async(
        self,
        prompts,
        *,
        max_new_tokens: int,
        sampling_params: SamplingParams,
        prompt_features=None,
        **kwargs,
    ) -> _ServeRollout:
        del kwargs
        params = sampling_params.model_copy(update={"max_new_tokens": int(max_new_tokens)})
        results = await self._trainer.rollout_token_batch_async(
            prompts,
            n_samples=1,
            sampling_params=params,
            prompt_features=prompt_features,
        )
        sequences = [result.sequences[0] for result in results]
        response_ids = [sequence.resp_tokens for sequence in sequences]
        finish_reason = ["length" if len(tokens) >= max_new_tokens else "stop" for tokens in response_ids]
        return _ServeRollout(response_ids=response_ids, finish_reason=finish_reason)


def _create_serve_runtime(
    *,
    model_path: str,
    backend_type: BackendType,
    tp_size: int,
    world_size: int,
    max_running_prompts: int,
    decode_progress_interval_s: float,
    eager_decode: bool,
    attn_backend: str,
    lora: LoraConfig | None,
    base_model_name_or_path: str | None,
) -> _CudaServeRuntime | _MlxServeRuntime:
    if backend_type == MLX:
        if lora is not None:
            raise ValueError("native LoRA serving is only supported by the CUDA backend")
        if world_size != 1 or tp_size != 1:
            raise ValueError("MLX serving requires --world-size 1 and --tp-size 1")
        return _MlxServeRuntime(
            model_path,
            max_running_prompts=max_running_prompts,
            decode_progress_interval_s=decode_progress_interval_s,
        )
    del max_running_prompts
    return _CudaServeRuntime(
        model_path,
        tp_size=tp_size,
        world_size=world_size,
        eager_decode=eager_decode,
        attn_backend=attn_backend,
        lora=lora,
        base_model_name_or_path=base_model_name_or_path,
    )


def create_app(
    *,
    model_path: str,
    tp_size: int,
    world_size: int,
    max_running_prompts: int,
    default_max_tokens: int,
    decode_progress_interval_s: float,
    eager_decode: bool = False,
    attn_backend: Literal["flash", "native"] = "flash",
    chat_template_enable_thinking: bool | None = None,
    lora: LoraConfig | None = None,
    base_model_name_or_path: str | None = None,
) -> FastAPI:
    """Construct the FastAPI app: load tokenizer/engine, install routes and lifecycle hooks."""
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if tp_size < 1:
        raise ValueError("tp_size must be >= 1")
    if world_size % tp_size != 0:
        raise ValueError("world_size must be divisible by tp_size")

    backend_type = default_backend_type()
    tokenizer = load_tokenizer(model_path)
    processor = load_processor(model_path)
    configure_chat_template_enable_thinking(tokenizer, chat_template_enable_thinking)
    configure_chat_template_enable_thinking(processor, chat_template_enable_thinking)
    attn_backend, attn_warning = _resolve_serve_attn_backend(
        model_path=model_path, attn_backend=attn_backend, world_size=world_size, backend_type=backend_type
    )
    if attn_warning is not None:
        warnings.warn(attn_warning, RuntimeWarning, stacklevel=2)
    parser_trainer = _ToolParserTrainerShim(model_path=model_path, tokenizer=tokenizer)
    engine = _create_serve_runtime(
        model_path=model_path,
        backend_type=backend_type,
        tp_size=tp_size,
        world_size=world_size,
        max_running_prompts=max_running_prompts,
        decode_progress_interval_s=decode_progress_interval_s,
        eager_decode=eager_decode,
        attn_backend=attn_backend,
        lora=lora,
        base_model_name_or_path=base_model_name_or_path,
    )
    if backend_type == MLX:
        tokenizer = engine.tokenizer
        processor = engine.processor
        configure_chat_template_enable_thinking(tokenizer, chat_template_enable_thinking)
        configure_chat_template_enable_thinking(processor, chat_template_enable_thinking)
    state = ServeState(
        model_path=model_path,
        tokenizer=tokenizer,
        processor=processor,
        engine=engine,
        max_running_prompts=max_running_prompts,
        default_max_tokens=default_max_tokens,
        max_model_len=int(engine.max_model_len),
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

        try:
            prompt, prompt_features = _encode_messages_with_features(
                state.tokenizer,
                state.processor,
                request.messages,
                tools=request.tools,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            prompt_features=prompt_features,
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


def _resolve_serve_attn_backend(
    *,
    model_path: str,
    attn_backend: Literal["flash", "native"],
    world_size: int,
    backend_type: BackendType = BackendType.CUDA,
) -> tuple[Literal["flash", "native"], str | None]:
    """Apply flash-attn compatibility fallback before serve starts workers."""

    if backend_type == MLX:
        return "native", None
    if attn_backend != "flash":
        return attn_backend, None
    model_config = None
    try:
        model_config = config_from_hf(model_path)
    except Exception:
        # Engine startup still owns config loading errors. This preflight can
        # still catch GPU-only fallback when model config is unavailable.
        pass
    reasons = [
        reason
        for reason in (
            flash_attention_unsupported_gpu_reason(list(range(world_size))),
            flash_attention_unsupported_model_reason(model_config) if model_config is not None else None,
        )
        if reason is not None
    ]
    if not reasons:
        return "flash", None
    reason = "; ".join(reasons)
    warning = (
        f"flash-attn does not support the detected serve runtime configuration ({reason}); "
        "AReno will use attn_backend='native'. Native attention is a compatibility path and may be slower."
    )
    return "native", warning


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
    prompt_features = [item.prompt_features for _ in prompts] if item.prompt_features is not None else None
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
        prompt_features=prompt_features,
        decode_progress_interval_s=app.state.decode_progress_interval_s,
    )
    if item.future.done():
        return None
    return _build_response(state, item.request, item.prompt, rollout.response_ids, rollout.finish_reason)


def _set_future_result(future: asyncio.Future, response: ChatCompletionResponse) -> None:
    """Resolve `future` with `response` unless something else got there first."""
    if response is not None and not future.done():
        future.set_result(response)


async def _await_pending_response(
    state: ServeState, raw_request: Request, item: PendingRequest
) -> ChatCompletionResponse:
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
    return _build_response_from(
        state.tokenizer, state.model_path, state.tool_call_parser, request, prompt, response_ids, finish_reasons
    )


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
        prompt_tokens=len(prompt),
        response_ids=response_ids,
        finish_reasons=finish_reasons,
        tools=request.tools,
        tool_choice=request.tool_choice,
        tool_call_parser=tool_call_parser,
        stop_strings=_normalize_stop(request.stop),
    )
    return ChatCompletionResponse(**data)


def _encode_messages(
    tokenizer: Any, messages: list[ChatMessage], *, tools: list[dict[str, Any]] | None = None
) -> list[int]:
    """Tokenise a chat history, using the tokenizer's chat template when available."""
    payload = [_chat_message_payload(msg) for msg in messages]
    return messages_to_prompt_tokens(tokenizer, payload, tools=tools, fallback_prompt=_messages_fallback_text(payload))


def _encode_messages_with_features(
    tokenizer: Any,
    processor: Any,
    messages: list[ChatMessage],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[list[int], dict[str, Any] | None]:
    payload = [_chat_message_payload(msg, preserve_content_parts=True) for msg in messages]
    if not _messages_have_multimodal(payload):
        return (
            messages_to_prompt_tokens(
                tokenizer, payload, tools=tools, fallback_prompt=_messages_fallback_text(payload)
            ),
            None,
        )
    if processor is None:
        raise ValueError("multimodal input requires a checkpoint processor")
    from areno.api.multimodal import encode_processor_messages

    if _messages_have_audio_or_video(payload):
        return encode_processor_messages(processor, payload, tools=tools)
    images = _load_message_images(payload)
    text = _processor_chat_text(processor, payload, tools=tools)
    encoded = _encode_text_and_images(tokenizer, processor, text, images)
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise ValueError("processor did not return input_ids for image request")
    features = {
        key: value
        for key, value in dict(encoded).items()
        if key not in {"input_ids", "attention_mask", "token_type_ids"}
    }
    image_token_id = _image_token_id(tokenizer, processor)
    if image_token_id is not None:
        features["image_token_id"] = image_token_id
    counts = image_token_counts_from_features(features)
    tokens = input_ids[0].tolist()
    if counts:
        if image_token_id is None:
            raise ValueError("image input requires an image token id from tokenizer or processor")
        tokens, _ = expand_image_tokens(tokens, image_token_id=image_token_id, image_token_counts=counts)
        mrope_position_ids = mrope_position_ids_from_image_grid(
            tokens,
            image_token_id=image_token_id,
            features=features,
        )
        if mrope_position_ids is not None:
            features["mrope_position_ids"] = mrope_position_ids
    return tokens, features or None


def _encode_text_and_images(tokenizer: Any, processor: Any, text: str, images: list[Any]) -> dict[str, Any]:
    return_tensors = getattr(processor, "_areno_return_tensors", "pt")
    try:
        return dict(processor(text=[text], images=images, return_tensors=return_tensors))
    except TypeError as exc:
        if "images" not in str(exc):
            raise
    image_processor = _image_processor_from_processor(processor)
    text_encoded = tokenizer([text], return_tensors=return_tensors)
    image_encoded = image_processor(images=images, return_tensors=return_tensors)
    encoded = dict(image_encoded)
    encoded["input_ids"] = text_encoded["input_ids"]
    if text_encoded.get("attention_mask") is not None:
        encoded["attention_mask"] = text_encoded["attention_mask"]
    return encoded


def _image_processor_from_processor(processor: Any):
    nested = getattr(processor, "image_processor", None)
    if nested is not None:
        return nested
    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:
        raise ValueError("image input requires transformers AutoImageProcessor") from exc
    name_or_path = getattr(processor, "name_or_path", None)
    if not name_or_path:
        raise ValueError("image input requires an image processor")
    return AutoImageProcessor.from_pretrained(name_or_path, trust_remote_code=True)


def _chat_message_payload(message: ChatMessage, *, preserve_content_parts: bool = False) -> dict[str, Any]:
    content = message.content if preserve_content_parts else _message_content(message.content)
    payload: dict[str, Any] = {"role": message.role, "content": content}
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


def _messages_have_images(messages: list[dict[str, Any]]) -> bool:
    return any(_content_has_image(message.get("content")) for message in messages)


def _messages_have_multimodal(messages: list[dict[str, Any]]) -> bool:
    return any(_content_has_multimodal(message.get("content")) for message in messages)


def _messages_have_audio_or_video(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = str(part.get("type", ""))
            if kind == "input_audio" or kind.removesuffix("_url") in {"audio", "video"}:
                return True
    return False


def _content_has_multimodal(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("type") in {"image", "image_url", "audio", "audio_url", "input_audio", "video", "video_url"}
        for item in content
    )


def _content_has_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(item, dict) and item.get("type") in {"image", "image_url"} for item in content)


def _load_message_images(messages: list[dict[str, Any]]) -> list[Any]:
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") not in {"image", "image_url"}:
                continue
            images.append(_load_image_part(item))
    if not images:
        raise ValueError("image request did not include any image parts")
    return images


def _load_image_part(part: dict[str, Any]) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("image input requires Pillow") from exc
    image_ref = part.get("image")
    if image_ref is None:
        image_url = part.get("image_url")
        image_ref = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not image_ref:
        raise ValueError("image part must include image or image_url.url")
    if not isinstance(image_ref, str):
        raise ValueError("image reference must be a string path, file URL, or data URL")
    if image_ref.startswith("data:"):
        _, _, payload = image_ref.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    parsed = urlparse(image_ref)
    if parsed.scheme == "file":
        return Image.open(parsed.path).convert("RGB")
    if parsed.scheme in {"http", "https"}:
        raise ValueError("HTTP image URLs are not supported yet; use a local path, file URL, or data URL")
    return Image.open(image_ref).convert("RGB")


def _processor_chat_text(
    processor: Any, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None
) -> str:
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    messages = _normalize_processor_multimodal_messages(messages)
    if callable(apply_chat_template):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        rendered = apply_chat_template_with_options(processor, messages, **kwargs)
        if isinstance(rendered, str):
            return rendered
    if tools:
        raise ValueError("image input with tools requires a processor chat template that supports tools")
    text_messages = [{**message, "content": _message_content(message.get("content"))} for message in messages]
    return _messages_fallback_text(text_messages)


def _normalize_processor_multimodal_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI image_url parts to the Transformers processor chat format."""

    normalized = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [_normalize_processor_content_part(part) for part in content]
        normalized.append(item)
    return normalized


def _normalize_processor_content_part(part: Any) -> Any:
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return part
    image_url = part.get("image_url")
    if not isinstance(image_url, dict) or "url" not in image_url:
        raise ValueError("image_url content must be an object with a url field")
    normalized = dict(part)
    normalized["type"] = "image"
    normalized["image"] = image_url["url"]
    normalized.pop("image_url", None)
    return normalized


def _image_token_id(tokenizer: Any, processor: Any) -> int | None:
    for obj in (processor, tokenizer):
        for attr in ("image_token_id", "image_token_index"):
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                return int(value)
        token = getattr(obj, "image_token", None)
        if isinstance(token, str):
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert):
                token_id = convert(token)
                if isinstance(token_id, int) and token_id >= 0:
                    return int(token_id)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        for token in ("<|image_pad|>", "<|image|>", "<image>"):
            token_id = convert(token)
            if isinstance(token_id, int) and token_id >= 0:
                return int(token_id)
    return None


def _first_eos_token_id(tokenizer: Any) -> int | None:
    """Return the first eos id when the tokenizer reports one (handles list/int forms)."""
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        return eos
    if isinstance(eos, list | tuple) and eos:
        return int(eos[0])
    return None


def _stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return the tokenizer's eos id(s) as a tuple of ints for use in `BatchKey`."""
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        return (eos,)
    if isinstance(eos, list | tuple):
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
@click.option("--model-path", required=True, help="Local checkpoint/tokenizer path or remote model repo ID.")
@click.option(
    "--model-hub",
    type=click.Choice(["hf", "modelscope"], case_sensitive=False),
    default="modelscope",
    show_default=True,
    help="Remote hub for non-local model refs. Use 'modelscope' for ModelScope or 'hf' for Hugging Face.",
)
@click.option(
    "--base-model-name-or-path",
    default=None,
    help="Stable base model reference associated with the PEFT adapter; defaults to the original model path.",
)
@click.option("--tp-size", type=int, default=1, show_default=True, help="Tensor parallel size.")
@click.option("--world-size", type=int, default=1, show_default=True, help="Total number of local worker ranks.")
@click.option("--host", default="0.0.0.0", show_default=True, help="HTTP bind host.")
@click.option("--port", type=int, default=8000, show_default=True, help="HTTP bind port.")
@click.option(
    "--max-running-prompts",
    type=int,
    default=16,
    show_default=True,
    help="Maximum concurrent rollout prompts per request chunk.",
)
@click.option("--default-max-tokens", type=int, default=1024, show_default=True, help="Default max generated tokens.")
@click.option(
    "--decode-progress-interval-s",
    type=float,
    default=0.0,
    show_default=True,
    help="Worker decode progress log interval.",
)
@click.option("--eager-decode", is_flag=True, help="Run decode in eager mode instead of CUDA graph replay.")
@click.option(
    "--attn-backend",
    type=click.Choice(["flash", "native"]),
    default="flash",
    show_default=True,
    help="Attention backend. Use native for slower areno_accel attention compatibility/logprob diagnostics.",
)
@click.option(
    "--disable-thinking",
    is_flag=True,
    help="Pass enable_thinking=False to tokenizer chat templates when supported.",
)
@click.option("--lora-rank", type=int, default=None, help="Enable native LoRA with this rank.")
@click.option("--lora-alpha", type=float, default=16.0, show_default=True, help="Native LoRA alpha.")
@click.option("--lora-dropout", type=float, default=0.0, show_default=True, help="Native LoRA dropout (must be 0).")
@click.option(
    "--lora-target-modules",
    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    show_default=True,
    help=(
        "Comma-separated native projection targets (selected Bailing V3 KDA q/k/v/f/g projections "
        "use independent canonical adapters)."
    ),
)
@click.option("--lora-adapter-path", default=None, help="Standard PEFT adapter to serve.")
def serve_command(
    model_path: str,
    model_hub: Literal["hf", "modelscope"],
    base_model_name_or_path: str | None,
    tp_size: int,
    world_size: int,
    host: str,
    port: int,
    max_running_prompts: int,
    default_max_tokens: int,
    decode_progress_interval_s: float,
    eager_decode: bool,
    attn_backend: Literal["flash", "native"],
    disable_thinking: bool,
    lora_rank: int | None,
    lora_alpha: float,
    lora_dropout: float,
    lora_target_modules: str,
    lora_adapter_path: str | None,
) -> None:
    """Click entry point: build the app and hand it to uvicorn."""
    import uvicorn

    if base_model_name_or_path is None:
        base_model_name_or_path = model_path
    model_path = resolve_model_ref(model_path, model_hub=model_hub)
    lora = None
    if lora_rank is not None or lora_adapter_path is not None:
        lora = LoraConfig(
            rank=8 if lora_rank is None else lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=tuple(item.strip() for item in lora_target_modules.split(",") if item.strip()),
            adapter_path=lora_adapter_path,
        )
    from areno.cli.dashboard_registry import register_dashboard_job

    register_dashboard_job(
        kind="serve",
        name=f"serve {model_path}",
        config={
            "ckpt": model_path,
            "model_hub": model_hub,
            "tp_size": tp_size,
            "world_size": world_size,
            "host": host,
            "port": port,
            "max_running_prompts": max_running_prompts,
            "default_max_tokens": default_max_tokens,
            "eager_decode": eager_decode,
            "attn_backend": attn_backend,
        },
        metrics_dir=None,
    )
    app = create_app(
        model_path=model_path,
        tp_size=tp_size,
        world_size=world_size,
        max_running_prompts=max_running_prompts,
        default_max_tokens=default_max_tokens,
        decode_progress_interval_s=decode_progress_interval_s,
        eager_decode=eager_decode,
        attn_backend=attn_backend,
        chat_template_enable_thinking=False if disable_thinking else None,
        lora=lora,
        base_model_name_or_path=base_model_name_or_path,
    )
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Console-script entrypoint for `areno serve`."""

    serve_command.main(prog_name="areno serve")


if __name__ == "__main__":
    main()
