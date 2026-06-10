"""OpenAI-compatible FastAPI server fronting the areno engine.

Exposes a `/v1/chat/completions` endpoint backed by a continuous-batching
scheduler that merges compatible requests (same BatchKey) into a single
rollout session. New prompts arriving mid-rollout take a fast-attach
path into the active session when its KV cache can absorb them;
otherwise a new session is started under the engine lock. Per-prompt
cancellation is signalled through a shared-memory uint8 flag tensor
that the worker observes, so HTTP disconnects abort generation
without tearing down the whole batch.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque
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

    `future` is the asyncio handoff used by the worker thread to deliver the
    response back to the request coroutine. `prompt_start`/`prompt_end` mark
    this request's contiguous slice in the session-wide `cancel_flags` tensor
    (assigned once the request is admitted into a continuous session).
    """

    request: ChatCompletionRequest
    prompt: list[int]
    key: BatchKey
    future: asyncio.Future
    created_at: float = field(default_factory=time.monotonic)
    cancelled: bool = False
    prompt_start: int = -1
    prompt_end: int = -1
    cancel_flags: torch.Tensor | None = None


@dataclass(slots=True)
class ContinuousRolloutSession:
    """State of one merged rollout session shared across many requests.

    A session owns a single engine.generate_rollout call. New compatible
    batches can fast-attach as long as `can_accept` is satisfied; partial
    callbacks fan results back out to per-request futures using
    `prompt_to_choice` to map engine prompt indices back to (request, choice).
    """

    id: int
    key: BatchKey
    loop: asyncio.AbstractEventLoop
    tokenizer: Any
    model_path: str
    max_cache_len: int = 0
    prompt_count: int = 0
    entries: list[tuple[PendingRequest, int, int]] = field(default_factory=list)
    prompt_to_choice: dict[int, tuple[PendingRequest, int]] = field(default_factory=dict)
    partial_responses: dict[int, list[list[int] | None]] = field(default_factory=dict)
    partial_reasons: dict[int, list[str | None]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def can_accept(self, batch: list[PendingRequest]) -> bool:
        """True if every prompt in `batch` fits the session's cache headroom."""
        if self.max_cache_len <= 0:
            return True
        return all(len(item.prompt) + self.key.max_new_tokens <= self.max_cache_len for item in batch)

    def add_batch(self, batch: list[PendingRequest]) -> tuple[list[list[int]], int]:
        """Append `batch` into the session, returning duplicated prompts and the starting offset.

        Each pending request contributes `n` copies of its prompt (one per
        sampled choice). Tracks per-request slot ranges in `entries` and a
        reverse map in `prompt_to_choice` so partial callbacks can be routed.
        """
        prompts: list[list[int]] = []
        with self.lock:
            offset = self.prompt_count
            for item in batch:
                start = self.prompt_count
                prompts.extend([item.prompt for _ in range(item.request.n)])
                self.prompt_count += int(item.request.n)
                end = self.prompt_count
                self.entries.append((item, start, end))
                self.partial_responses[id(item)] = [None for _ in range(item.request.n)]
                self.partial_reasons[id(item)] = [None for _ in range(item.request.n)]
                for choice_index, prompt_index in enumerate(range(start, end)):
                    self.prompt_to_choice[prompt_index] = (item, choice_index)
            return prompts, offset

    def on_partial(self, prompt_index: int, response_ids: list[int], finish_reason: str) -> None:
        """Worker callback delivering a finished prompt slot's tokens back to its request.

        Stores the slot's tokens and finish reason; once every choice for the
        owning request is filled, builds the full response and resolves the
        request future on its originating event loop.
        """
        with self.lock:
            slot = self.prompt_to_choice.get(prompt_index)
            if slot is None:
                return
            item, choice_index = slot
            responses = self.partial_responses[id(item)]
            reasons = self.partial_reasons[id(item)]
            if item.future.done() or responses[choice_index] is not None:
                return
            responses[choice_index] = response_ids
            reasons[choice_index] = finish_reason
            if not all(row is not None for row in responses):
                return
            response = _build_response_from(
                self.tokenizer,
                self.model_path,
                item.request,
                item.prompt,
                [row for row in responses if row is not None],
                [reason or "stop" for reason in reasons],
            )
            # Hop back onto the request's event loop to resolve the future safely.
            self.loop.call_soon_threadsafe(_set_future_result, item.future, response)


@dataclass(slots=True)
class ServeState:
    """Process-wide serving state held on `app.state.areno_serve`.

    Holds the loaded engine and tokenizer, scheduler tunables, and the
    queue/condition/lock primitives that gate the batcher loop, the engine
    call and the currently active continuous session.
    """

    model_path: str
    tokenizer: Any
    engine: ArenoEngine
    max_running_prompts: int
    default_max_tokens: int
    batch_wait_s: float
    max_batch_prompts: int
    queue: deque[PendingRequest] = field(default_factory=deque)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    engine_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_batch_tasks: set[asyncio.Task] = field(default_factory=set)
    active_session: ContinuousRolloutSession | None = None
    next_session_id: int = 1
    batcher_task: asyncio.Task | None = None
    closing: bool = False


def create_app(
    *,
    model_path: str,
    tp_size: int,
    world_size: int,
    max_running_prompts: int,
    default_max_tokens: int,
    decode_progress_interval_s: float,
    batch_wait_ms: float,
    max_batch_prompts: int,
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
        batch_wait_s=max(0.0, float(batch_wait_ms)) / 1000.0,
        max_batch_prompts=max(1, int(max_batch_prompts)),
    )
    app = FastAPI(title="areno OpenAI-compatible server")
    app.state.areno_serve = state
    app.state.decode_progress_interval_s = float(decode_progress_interval_s)

    @app.on_event("startup")
    async def startup() -> None:
        """Spawn the long-running batcher coroutine when the app starts."""
        state.batcher_task = asyncio.create_task(_batcher_loop(app))

    @app.on_event("shutdown")
    async def shutdown() -> None:
        """Signal closing, drain in-flight batch tasks, then tear the engine down."""
        async with state.condition:
            state.closing = True
            state.condition.notify_all()
        if state.batcher_task is not None:
            await state.batcher_task
        if state.active_batch_tasks:
            await asyncio.gather(*state.active_batch_tasks, return_exceptions=True)
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
        """Validate the request, encode the prompt, enqueue it for the batcher, and await the response.

        Streaming is unsupported. The prompt's `BatchKey` is computed up front
        so that the scheduler can match it against compatible peers without
        re-decoding the request. A watcher task is started by
        `_await_pending_response` to observe client disconnects.
        """
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
        async with state.condition:
            if state.closing:
                raise HTTPException(status_code=503, detail="server is shutting down")
            state.queue.append(pending)
            # Wake the batcher loop so it can consider this request immediately.
            state.condition.notify()
        return await _await_pending_response(state, raw_request, pending)

    return app


async def _batcher_loop(app: FastAPI) -> None:
    """Main scheduler loop: wait for queued requests, coalesce, then dispatch a batch task.

    Once the queue is non-empty it waits up to `batch_wait_s` past the head
    request's arrival for additional same-key requests to pile up (improves
    GPU utilisation), then pops a compatible batch and hands it off to
    `_run_batch_task`.
    """
    state: ServeState = app.state.areno_serve
    while True:
        async with state.condition:
            while not state.queue and not state.closing:
                await state.condition.wait()
            if state.closing and not state.queue:
                return
            first = state.queue[0]
            # Allow same-key requests to coalesce for at most `batch_wait_s`.
            deadline = first.created_at + state.batch_wait_s
            while not state.closing and _queued_prompt_count(state.queue) < state.max_batch_prompts:
                delay = deadline - time.monotonic()
                if delay <= 0:
                    break
                try:
                    # Wait either for a new request or until the coalescing deadline.
                    await asyncio.wait_for(state.condition.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    break
            batch = _pop_compatible_batch(state)
        if batch:
            task = asyncio.create_task(_run_batch_task(app, batch))
            state.active_batch_tasks.add(task)
            task.add_done_callback(state.active_batch_tasks.discard)


async def _run_batch_task(app: FastAPI, batch: list[PendingRequest]) -> None:
    """Dispatch one popped batch: fast-attach to the active session or start a new one.

    Fast-attach path (cheap, holds only the condition): if a session is live
    with a matching BatchKey and enough cache headroom, hand the prompts to
    the engine via `add_rollout_batch` and return without acquiring the
    engine lock. Slow path: acquire `engine_lock` (serialising engine
    sessions) and spin up a fresh `ContinuousRolloutSession` for this batch.
    """
    state: ServeState = app.state.areno_serve
    try:
        async with state.condition:
            session = state.active_session
            if session is not None and session.key == batch[0].key and session.can_accept(batch):
                # Fast-attach: feed prompts into the running rollout without a new engine call.
                prompts, offset = session.add_batch(batch)
                state.engine.add_rollout_batch(prompts, session_id=session.id, prompt_offset=offset)
                return
        # No compatible active session: serialise on engine_lock to launch a new one.
        async with state.engine_lock:
            async with state.condition:
                session_id = state.next_session_id
                state.next_session_id += 1
                session = ContinuousRolloutSession(
                    id=session_id,
                    key=batch[0].key,
                    loop=asyncio.get_running_loop(),
                    tokenizer=state.tokenizer,
                    model_path=state.model_path,
                    max_cache_len=max(len(item.prompt) + batch[0].key.max_new_tokens for item in batch),
                )
                state.active_session = session
            try:
                # The engine call is blocking C++/CUDA, so run it on a worker thread.
                await asyncio.to_thread(_run_continuous_session, app, session, batch)
            finally:
                async with state.condition:
                    if state.active_session is session:
                        state.active_session = None
    except BaseException as exc:
        for item in batch:
            if not item.future.done():
                item.future.set_exception(exc)


def _pop_compatible_batch(state: ServeState) -> list[PendingRequest]:
    """Pop the head request plus any same-key followers, dropping cancelled entries.

    Walks `state.queue` once, keeping the head request and any subsequent
    requests that share its `BatchKey` and fit the prompt budget. Items that
    were cancelled or already completed are silently dropped. Non-matching
    items are preserved in queue order via `kept`.
    """
    first = state.queue.popleft()
    while (first.cancelled or first.future.done()) and state.queue:
        # Skip cancelled/done requests at the head so the batch starts on a live one.
        first = state.queue.popleft()
    if first.cancelled or first.future.done():
        return []
    batch = [first]
    batch_prompts = int(first.request.n)
    prompt_budget = max(1, min(state.max_running_prompts, state.max_batch_prompts))
    kept: deque[PendingRequest] = deque()
    while state.queue and batch_prompts < prompt_budget:
        item = state.queue.popleft()
        if item.cancelled or item.future.done():
            continue
        item_prompts = int(item.request.n)
        if item.key == first.key and batch_prompts + item_prompts <= prompt_budget:
            batch.append(item)
            batch_prompts += item_prompts
        else:
            kept.append(item)
    # Re-attach the unmatched tail so it can be considered on the next loop iteration.
    kept.extend(state.queue)
    state.queue = kept
    return batch


def _queued_prompt_count(queue: deque[PendingRequest]) -> int:
    """Total number of distinct prompts (counting `n` per request) currently queued."""
    return sum(int(item.request.n) for item in queue)


def _run_continuous_session(app: FastAPI, session: ContinuousRolloutSession, batch: list[PendingRequest]) -> None:
    """Run a fresh rollout session for `batch` on the engine; called via asyncio.to_thread.

    Allocates the per-session `cancel_flags` shared-memory uint8 tensor (one
    byte per prompt slot, observable across threads and the worker process),
    assigns each request a contiguous slot range, and invokes
    `engine.generate_rollout`. Any request that was already cancelled before
    the call has its slots pre-flagged so the worker skips them.

    On return, fan the final response_ids/finish_reasons back to each
    request future. Most responses are usually already resolved via
    `session.on_partial`; the loop here covers the synchronous tail.
    """
    state: ServeState = app.state.areno_serve
    key = batch[0].key
    prompts, _offset = session.add_batch(batch)
    # uint8 + share_memory_ so the worker thread/process can observe cancellation toggles.
    cancel_flags = torch.zeros(sum(int(item.request.n) for item in batch), dtype=torch.uint8).share_memory_()
    start = 0
    for item in batch:
        # Each request gets a contiguous [prompt_start, prompt_end) slot range.
        item.prompt_start = start
        item.prompt_end = start + int(item.request.n)
        item.cancel_flags = cancel_flags
        if item.cancelled or item.future.cancelled():
            # Pre-flag already-cancelled requests so the engine never runs their slots.
            cancel_flags[item.prompt_start : item.prompt_end] = 1
        start = item.prompt_end
    if all(item.cancelled or item.future.done() for item in batch):
        return

    rollout = state.engine.generate_rollout(
        prompts,
        max_new_tokens=key.max_new_tokens,
        max_running_prompts=min(state.max_running_prompts, max(len(prompts), 1)),
        eos_token_id=key.eos_token_id,
        sampling_params=SamplingParams(
            temperature=key.temperature,
            top_p=key.top_p,
            top_k=key.top_k,
            seed=key.seed,
            stop_token_ids=key.stop_token_ids,
        ),
        decode_progress_interval_s=app.state.decode_progress_interval_s,
        partial_callback=session.on_partial,
        cancel_flags=cancel_flags,
        session_id=session.id,
    )
    # Snapshot under the lock; mid-rollout fast-attach may have appended entries.
    with session.lock:
        entries = list(session.entries)
    for item, start, end in entries:
        if item.future.done() or end > len(rollout.response_ids):
            continue
        response = _build_response(state, item.request, item.prompt, rollout.response_ids[start:end], rollout.finish_reason[start:end])
        # Resolve from the worker thread by hopping back to the request's loop.
        item.future.get_loop().call_soon_threadsafe(_set_future_result, item.future, response)


def _set_future_result(future: asyncio.Future, response: ChatCompletionResponse) -> None:
    """Resolve `future` with `response` unless something else got there first."""
    if not future.done():
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

    On disconnect, sets `cancel_flags` for this request's slots so the engine
    aborts those rollouts at the next decode step, and resolves the future
    with an empty cancelled response so the caller's await returns promptly.
    """
    while not item.future.done():
        if await raw_request.is_disconnected():
            _cancel_pending_request(item)
            if not item.future.done():
                item.future.set_result(_build_cancelled_response(state, item))
            return
        await asyncio.sleep(0.1)


def _cancel_pending_request(item: PendingRequest) -> None:
    """Mark the request cancelled and, if admitted into a session, flip its cancel_flags slots."""
    item.cancelled = True
    if item.cancel_flags is not None and item.prompt_start >= 0:
        # Write 1 to this request's contiguous slot range; the worker polls this tensor.
        item.cancel_flags[item.prompt_start : item.prompt_end] = 1


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
@click.option("--max-batch-prompts", type=int, default=128, show_default=True, help="Maximum prompts to merge into one generate call.")
@click.option("--batch-wait-ms", type=float, default=10.0, show_default=True, help="Milliseconds to wait for compatible requests before starting a new decode session.")
@click.option("--default-max-tokens", type=int, default=1024, show_default=True, help="Default max generated tokens.")
@click.option("--decode-progress-interval-s", type=float, default=0.0, show_default=True, help="Worker decode progress log interval.")
def serve_command(
    model_path: str,
    tp_size: int,
    world_size: int,
    host: str,
    port: int,
    max_running_prompts: int,
    max_batch_prompts: int,
    batch_wait_ms: float,
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
        batch_wait_ms=batch_wait_ms,
        max_batch_prompts=max_batch_prompts,
    )
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Console-script entrypoint for `areno serve`."""

    serve_command.main(prog_name="areno serve")


if __name__ == "__main__":
    main()
