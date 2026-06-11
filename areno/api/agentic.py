"""Agentic rollout session support.

This module provides the first agentic rollout path for Areno. It exposes a
small OpenAI-compatible HTTP surface that agent code can call with a standard
OpenAI client, records the generated trajectories, and converts them into the
same token/logprob rows consumed by existing trainers.

The first implementation intentionally supports non-streaming
``/v1/chat/completions``. The public data model already keeps trace, messages,
tool calls, tool results, and loss masks explicit so streaming and richer tool
coverage can be added without changing trainer boundaries.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from areno.api.tool_call_parser import get_tool_call_parser, infer_tool_call_parser_name

if TYPE_CHECKING:
    from areno.api.models import SamplingParams

logger = logging.getLogger(__name__)
AGENTIC_PROXY_COALESCE_WAIT_S = 10.0


@dataclass(slots=True)
class LossMaskPolicy:
    """Controls which agent trajectory spans contribute to policy loss."""

    assistant_text: bool = True
    assistant_tool_calls: bool = True
    tool_results: bool = False
    final_assistant_text: bool = True
    system_prompt: bool = False
    user_prompt: bool = False


@dataclass(slots=True)
class AgentItem:
    """One expanded agent task for a prompt/sample pair."""

    record: dict[str, Any]
    prompt: str
    input_tokens: list[int]
    prompt_index: int
    sample_index: int


@dataclass(slots=True)
class AgentBatch:
    """Prompt batch expanded into agent-callable samples."""

    records: list[dict[str, Any]]
    prompts: list[str]
    input_tokens: list[list[int]]
    n_samples: int

    def __len__(self) -> int:
        return len(self.records) * self.n_samples

    @classmethod
    def from_prompt_batch(cls, prompt_batch, n_samples: int) -> "AgentBatch":
        """Build an agent batch from the trainer's tokenized prompt batch."""

        return cls(
            records=[dict(item.record) for item in prompt_batch.items],
            prompts=[item.prompt for item in prompt_batch.items],
            input_tokens=[list(item.input_tokens) for item in prompt_batch.items],
            n_samples=int(n_samples),
        )

    def iter_samples(self) -> Iterator[AgentItem]:
        """Yield one item per prompt/sample pair in stable row order."""

        for prompt_index, (record, prompt, input_tokens) in enumerate(zip(self.records, self.prompts, self.input_tokens, strict=True)):
            for sample_index in range(self.n_samples):
                yield AgentItem(
                    record=record,
                    prompt=prompt,
                    input_tokens=input_tokens,
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                )


class RewardEvent(BaseModel):
    """Normalized event in an agent trajectory."""

    type: Literal["request", "assistant_text", "assistant_tool_call", "tool_result", "finish", "error"]
    text: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | str | None = None
    content: str | None = None
    messages: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardRecord(BaseModel):
    """Unified reward input for prompt and agentic rollouts."""

    prompt: str
    completion: str
    rendered_completion: str | None = None
    final_answer: str | None = None
    answer: Any | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[RewardEvent] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    tokens: list[int] = Field(default_factory=list)
    logprobs: list[float] = Field(default_factory=list)
    loss_mask: list[bool] = Field(default_factory=list)
    source_record: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class AgentTrainBatch:
    """Agentic rollout batch consumed by trainers."""

    token_rows: list[list[int]]
    response_masks: list[list[bool]]
    loss_masks: list[list[bool]]
    rollout_logprobs: list[list[float]]
    rewards: list[float] | None
    records: list[dict[str, Any]]
    reward_records: list[RewardRecord]


@dataclass(slots=True)
class _AgentSample:
    item: AgentItem
    messages: list[dict[str, Any]]
    response_text: str
    last_response_text: str
    response_tokens: list[int]
    response_logprobs: list[float]
    trace: list[RewardEvent]
    response_kind: Literal["assistant_text", "assistant_tool_call"] = "assistant_text"
    loss_mask_override: list[bool] | None = None
    token_row: list[int] = field(default_factory=list)
    response_mask_row: list[bool] = field(default_factory=list)
    loss_mask_row: list[bool] = field(default_factory=list)
    rollout_logprobs_row: list[float] = field(default_factory=list)


@dataclass(slots=True)
class _ResponseData:
    response_tokens: list[int]
    response_logprobs: list[float]


@dataclass(slots=True)
class _AgentTrainRows:
    token_rows: list[list[int]]
    response_masks: list[list[bool]]
    loss_masks: list[list[bool]]
    rollout_logprobs: list[list[float]]
    total_tokens: int


@dataclass(frozen=True, slots=True)
class _ChatBatchKey:
    greedy: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop_token_ids: tuple[int, ...]
    ignore_eos: bool
    skip_special_tokens: bool
    max_prompt_len: int | None


@dataclass(slots=True)
class _PendingChat:
    item: AgentItem
    messages: list[dict[str, Any]]
    input_tokens: list[int]
    params: Any
    key: _ChatBatchKey
    model: str
    created_at: float
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    event: threading.Event = field(default_factory=threading.Event)
    future: asyncio.Future | None = None
    response: dict[str, Any] | None = None
    error: BaseException | None = None
    prompt_index: int = -1
    sample_recorded: bool = False
    cancelled: bool = False


class _AgenticHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 2048
    max_threads = 2048

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_slots = threading.BoundedSemaphore(int(self.max_threads))

    def process_request(self, request, client_address) -> None:
        self._thread_slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._thread_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._thread_slots.release()


class RolloutSession:
    """Async context manager exposing an OpenAI-compatible rollout proxy."""

    api_key = "areno-agentic"

    def __init__(
        self,
        trainer,
        *,
        sampling_params: "SamplingParams",
        loss_mask_policy: LossMaskPolicy | None = None,
        max_running_prompts: int | None = None,
        timeout_s: float = 300.0,
        proxy: bool = True,
    ) -> None:
        self._trainer = trainer
        self._sampling_params = sampling_params
        self._loss_mask_policy = loss_mask_policy or LossMaskPolicy()
        self._tool_call_parser = get_tool_call_parser(infer_tool_call_parser_name(trainer))
        self._dp_size = max(_trainer_dp_size(trainer), 1)
        self._max_running_prompts = max(1, int(max_running_prompts)) if max_running_prompts is not None else self._dp_size
        self._local_max_running_prompts = max(_ceil_div(self._max_running_prompts, self._dp_size), 1)
        self._timeout_s = float(timeout_s)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._request_queue: asyncio.Queue[_PendingChat] | None = None
        self._batch_task: asyncio.Task | None = None
        self._deferred_pending: list[_PendingChat] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._closing = False
        self._agent_items: list[AgentItem] = []
        self._agent_items_by_prompt: dict[str, list[AgentItem]] = {}
        self._agent_reuse_cursor_by_prompt: dict[str, int] = {}
        self._next_item = 0
        self._samples: list[_AgentSample] = []
        self._filtered_items: set[tuple[int, int]] = set()
        self._errors: list[str] = []
        self._base_url = ""
        self._proxy_enabled = bool(proxy)

    @property
    def max_running_prompts(self) -> int:
        """Maximum concurrently running prompts for agentic rollout."""

        return self._max_running_prompts

    async def __aenter__(self) -> "RolloutSession":
        """Start the local proxy."""

        self._loop = asyncio.get_running_loop()
        await self._trainer.begin_rollout_session_async()
        if not self._proxy_enabled:
            return self
        self._request_queue = asyncio.Queue()
        self._batch_task = asyncio.create_task(self._batch_worker_loop(), name="areno-agentic-proxy-batcher")
        handler_cls = self._handler_cls()
        try:
            self._server = self._http_server_cls()(("127.0.0.1", 0), handler_cls)
            self._server.timeout = 0.5
            host, port = self._server.server_address[:2]
            self._base_url = f"http://{host}:{port}/v1"
            self._thread = threading.Thread(target=self._server.serve_forever, name="areno-agentic-proxy", daemon=True)
            self._thread.start()
        except BaseException:
            self._stop_batch_worker()
            await self._trainer.end_rollout_session_async()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Stop the local proxy."""

        del exc_type, exc, tb
        try:
            self._closing = True
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self._stop_batch_worker()
        finally:
            await self._trainer.end_rollout_session_async()

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL, including the ``/v1`` prefix."""

        if not self._proxy_enabled:
            raise RuntimeError("rollout session proxy is disabled")
        return self._base_url

    def get_base_url(self) -> str:
        """Return the OpenAI-compatible base URL."""

        return self.base_url

    def finish_requests(self) -> None:
        """Compatibility no-op for agents that mark request submission done."""

    def _stop_batch_worker(self) -> None:
        if self._batch_task is not None:
            self._batch_task.cancel()

    def _http_server_cls(self) -> type[_AgenticHTTPServer]:
        return type(
            "AgenticRolloutHTTPServer",
            (_AgenticHTTPServer,),
            {"request_queue_size": max(2048, self._max_running_prompts), "max_threads": 2048},
        )

    def attach_batch(self, batch: AgentBatch) -> None:
        """Attach the prompt/sample worklist expected for this session."""

        with self._lock:
            self._agent_items = list(batch.iter_samples())
            self._agent_items_by_prompt = {}
            self._agent_reuse_cursor_by_prompt = {}
            for item in self._agent_items:
                self._agent_items_by_prompt.setdefault(item.prompt, []).append(item)
            self._next_item = 0
            self._filtered_items = set()
            logger.info(
                "agentic session attached prompts=%d n_samples=%d expected_requests=%d max_running_prompts=%d local_max_running_prompts=%d dp_size=%d",
                len(batch.records),
                batch.n_samples,
                len(self._agent_items),
                self._max_running_prompts,
                self._local_max_running_prompts,
                self._dp_size,
            )

    async def get_train_batch(
        self,
        *,
        reward_fn: Callable[[RewardRecord], float] | None = None,
        normalize_rewards: bool | None = None,
        require_finished: bool = True,
    ) -> AgentTrainBatch:
        """Convert collected agent samples into trainer-facing rows."""

        del normalize_rewards
        start = time.perf_counter()
        if require_finished:
            await self._wait_for_samples()
        wait_done = time.perf_counter()
        with self._lock:
            samples = list(self._samples)
            filtered_count = len(self._filtered_items)
            errors = list(self._errors)
            expected = len(self._agent_items)
        logger.info("agentic session collected samples=%d filtered=%d expected=%d", len(samples), filtered_count, expected)
        if errors:
            raise RuntimeError("agent rollout proxy errors: " + "; ".join(errors[:3]))
        if require_finished and len(samples) + filtered_count < expected:
            raise RuntimeError(f"agent rollout produced {len(samples)} samples and filtered {filtered_count}, expected {expected}")

        reward_records = [self._reward_record(sample) for sample in samples]
        records_done = time.perf_counter()
        rewards = None
        if reward_fn is not None:
            rewards = [float(reward_fn(record)) for record in reward_records]
        rewards_done = time.perf_counter()
        rows = self._build_train_rows(samples)
        rows_done = time.perf_counter()
        tool_call_count = sum(len(record.tool_calls) for record in reward_records)
        tool_result_count = sum(len(record.tool_results) for record in reward_records)
        message_count = sum(len(record.messages) for record in reward_records)
        logger.info(
            "agentic train batch built samples=%d tokens=%d messages=%d tool_calls=%d tool_results=%d wait_s=%.3f records_s=%.3f rewards_s=%.3f rows_s=%.3f total_s=%.3f",
            len(samples),
            rows.total_tokens,
            message_count,
            tool_call_count,
            tool_result_count,
            wait_done - start,
            records_done - wait_done,
            rewards_done - records_done,
            rows_done - rewards_done,
            rows_done - start,
        )
        return AgentTrainBatch(
            token_rows=rows.token_rows,
            response_masks=rows.response_masks,
            loss_masks=rows.loss_masks,
            rollout_logprobs=rows.rollout_logprobs,
            rewards=rewards,
            records=[sample.item.record for sample in samples],
            reward_records=reward_records,
        )

    async def _wait_for_samples(self) -> None:
        deadline = time.monotonic() + self._timeout_s
        while True:
            with self._lock:
                sample_count = len(self._samples)
                filtered_count = len(self._filtered_items)
                expected = len(self._agent_items)
                errors = list(self._errors)
            if sample_count + filtered_count >= expected or errors:
                return
            if time.monotonic() >= deadline:
                return
            await asyncio.sleep(0.05)

    def _reward_record(self, sample: _AgentSample) -> RewardRecord:
        answer = sample.item.record.get("answer", sample.item.record.get("solutions"))
        messages = list(sample.messages)
        messages.append({"role": "assistant", "content": sample.last_response_text})
        tool_calls = [
            {"name": event.name, "arguments": event.arguments}
            for event in sample.trace
            if event.type == "assistant_tool_call" and event.name is not None
        ]
        tool_results = _tool_results_from_messages(messages)
        trace = _trace_with_tool_results(sample.trace, messages)
        tokenizer = self._trainer.get_tokenizer() if self._trainer is not None else None
        rendered_completion = _render_messages_for_display(tokenizer, messages)
        return RewardRecord(
            prompt=sample.item.prompt,
            completion=sample.response_text,
            rendered_completion=rendered_completion,
            final_answer=sample.last_response_text,
            answer=answer,
            messages=messages,
            trace=trace,
            tool_calls=tool_calls,
            tool_results=tool_results,
            tokens=sample.response_tokens,
            logprobs=sample.response_logprobs,
            loss_mask=self._response_loss_mask(sample),
            source_record=sample.item.record,
            metadata={"prompt_index": sample.item.prompt_index, "sample_index": sample.item.sample_index},
        )

    def _response_loss_mask(self, sample: _AgentSample) -> list[bool]:
        if sample.loss_mask_override is not None:
            return list(sample.loss_mask_override)
        return self._response_loss_mask_for_span(sample.response_kind, len(sample.response_tokens))

    def _response_loss_mask_for_span(self, response_kind: str, response_len: int) -> list[bool]:
        """Return loss-mask bits for one assistant response span."""

        if response_kind == "assistant_tool_call":
            enabled = self._loss_mask_policy.assistant_tool_calls
        else:
            enabled = self._loss_mask_policy.assistant_text
        return [bool(enabled)] * response_len

    def _build_train_rows(self, samples: list[_AgentSample]) -> _AgentTrainRows:
        token_rows: list[list[int]] = []
        response_masks: list[list[bool]] = []
        loss_masks: list[list[bool]] = []
        rollout_logprobs: list[list[float]] = []
        total_tokens = 0
        for sample in samples:
            if sample.token_row:
                token_rows.append(list(sample.token_row))
                response_masks.append(list(sample.response_mask_row))
                loss_masks.append(list(sample.loss_mask_row))
                rollout_logprobs.append(list(sample.rollout_logprobs_row))
                total_tokens += len(sample.token_row)
                continue
            prompt_len = len(sample.item.input_tokens)
            response_len = len(sample.response_tokens)
            token_row = sample.item.input_tokens + sample.response_tokens
            token_rows.append(token_row)
            response_masks.append([False] * prompt_len + [True] * response_len)
            loss_masks.append([False] * prompt_len + self._response_loss_mask(sample))
            rollout_logprobs.append([0.0] * prompt_len + list(sample.response_logprobs))
            total_tokens += len(token_row)
        return _AgentTrainRows(
            token_rows=token_rows,
            response_masks=response_masks,
            loss_masks=loss_masks,
            rollout_logprobs=rollout_logprobs,
            total_tokens=total_tokens,
        )

    def _handler_cls(self):
        session = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                session._handle_post(self)

            def log_message(self, fmt: str, *args) -> None:
                del fmt, args

        return Handler

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path not in {"/v1/chat/completions", "/chat/completions"}:
            _write_json(handler, 404, {"error": {"message": f"unsupported path: {handler.path}"}})
            return
        length = int(handler.headers.get("content-length", "0"))
        try:
            body = json.loads(handler.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            _write_json(handler, 400, {"error": {"message": f"invalid json: {exc}"}})
            return
        if body.get("stream"):
            _write_json(handler, 400, {"error": {"message": "streaming chat completions are not supported yet"}})
            return
        if self._loop is None:
            _write_json(handler, 500, {"error": {"message": "agent rollout proxy is not running"}})
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._complete_chat(body), self._loop)
            response = future.result(timeout=self._timeout_s)
            _write_json(handler, 200, response)
        except ValueError as exc:
            _write_json(handler, 400, {"error": {"message": str(exc)}})
        except Exception as exc:
            with self._lock:
                self._errors.append(str(exc))
            _write_json(handler, 500, {"error": {"message": str(exc)}})

    async def _complete_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        # Bind every HTTP turn back to one prompt/sample pair. Multi-turn agents
        # resend the same first user prompt, so this remains stable even when
        # requests arrive out of order.
        item = self._claim_item(body.get("messages") or [])
        messages = _normalize_messages(body.get("messages") or [])
        tools = list(body.get("tools") or [])
        tool_choice = body.get("tool_choice")
        params = self._sampling_params.model_copy() if hasattr(self._sampling_params, "model_copy") else self._sampling_params.copy()
        if body.get("max_tokens") is not None:
            params.max_new_tokens = int(body["max_tokens"])
        if body.get("temperature") is not None:
            params.temperature = float(body["temperature"])
        if body.get("top_p") is not None:
            params.top_p = float(body["top_p"])
        pending = _PendingChat(
            item=item,
            messages=messages,
            input_tokens=[],
            params=params,
            key=_chat_batch_key(params),
            model=body.get("model") or "policy",
            tools=tools,
            tool_choice=tool_choice,
            created_at=time.monotonic(),
        )
        if self._loop is None or self._request_queue is None:
            raise RuntimeError("agent rollout proxy is not running")
        pending.future = self._loop.create_future()
        await self._request_queue.put(pending)
        try:
            return await asyncio.wait_for(asyncio.shield(pending.future), timeout=self._timeout_s)
        except asyncio.TimeoutError as exc:
            pending.cancelled = True
            raise TimeoutError("agent rollout proxy timed out waiting for completion")

    async def _batch_worker_loop(self) -> None:
        while not self._closing:
            try:
                batch = await self._pop_chat_batch()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                with self._lock:
                    self._errors.append(str(exc))
                continue
            if not batch:
                continue
            await self._run_chat_batch(batch)

    async def _pop_chat_batch(self) -> list[_PendingChat]:
        if self._request_queue is None:
            return []
        first = self._deferred_pending.pop(0) if self._deferred_pending else await self._request_queue.get()
        if first.cancelled:
            return []
        batch = [first]
        deferred: list[_PendingChat] = []
        deadline = first.created_at + AGENTIC_PROXY_COALESCE_WAIT_S
        target = self._max_running_prompts
        while len(batch) < target:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                pending = await asyncio.wait_for(self._request_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if pending.cancelled:
                continue
            if pending.key == first.key:
                batch.append(pending)
            else:
                deferred.append(pending)
        self._deferred_pending[0:0] = deferred
        return batch

    async def _run_chat_batch(self, batch: list[_PendingChat]) -> None:
        valid: list[_PendingChat] = []
        tokenizer = self._trainer.get_tokenizer()
        for pending in batch:
            if pending.cancelled:
                continue
            try:
                pending.input_tokens = _messages_to_prompt_tokens(
                    tokenizer,
                    pending.messages,
                    tools=pending.tools,
                    fallback_prompt=pending.item.prompt,
                )
                max_sequence_len = _max_sequence_len(pending.params)
                if max_sequence_len is not None and len(pending.input_tokens) + int(pending.params.max_new_tokens) > max_sequence_len:
                    # Overlong multi-turn contexts are dropped as filtered samples so
                    # the rest of the rollout batch can keep training.
                    self._mark_item_filtered_if_unrecorded(pending.item)
                    response = _filtered_chat_response(
                        model=pending.model,
                        prompt_tokens=len(pending.input_tokens),
                        max_sequence_len=max_sequence_len,
                    )
                    self._set_pending_response(pending, response)
                    continue
                valid.append(pending)
            except BaseException as exc:
                self._set_pending_error(pending, exc)
        if not valid:
            return
        try:
            results = await self._trainer.rollout_token_batch_async([pending.input_tokens for pending in valid], 1, valid[0].params)
            for idx, pending in enumerate(valid):
                sequence = results[idx].sequences[0] if idx < len(results) and results[idx].sequences else None
                if sequence is None:
                    self._record_chat_sample(pending, [], [])
                else:
                    self._record_chat_sample(pending, sequence.resp_tokens, sequence.resp_logprobs)
        except BaseException as exc:
            for pending in valid:
                self._set_pending_error(pending, exc)

    def _record_chat_sample(self, pending: _PendingChat, response_ids: list[int], logprobs: list[float]) -> None:
        if pending.sample_recorded or pending.cancelled:
            return
        response = _ResponseData(response_tokens=list(response_ids), response_logprobs=list(logprobs))
        self._set_pending_response(pending, self._finish_pending_chat(pending, response))

    def _set_pending_response(self, pending: _PendingChat, response: dict[str, Any]) -> None:
        if pending.cancelled:
            return
        pending.response = response
        pending.event.set()
        if pending.future is not None and not pending.future.done():
            pending.future.set_result(response)

    def _set_pending_error(self, pending: _PendingChat, exc: BaseException) -> None:
        if pending.cancelled:
            return
        pending.error = exc
        pending.event.set()
        if pending.future is not None and not pending.future.done():
            pending.future.set_exception(exc)

    def _mark_item_filtered_if_unrecorded(self, item: AgentItem) -> None:
        key = (item.prompt_index, item.sample_index)
        if item.prompt_index < 0 or item.sample_index < 0:
            return
        with self._lock:
            if any((sample.item.prompt_index, sample.item.sample_index) == key for sample in self._samples):
                return
            self._filtered_items.add(key)

    def _finish_pending_chat(self, pending: _PendingChat, response: _ResponseData) -> dict[str, Any]:
        tokenizer = self._trainer.get_tokenizer()
        content = tokenizer.decode(response.response_tokens)
        tool_parse = self._tool_call_parser.parse(content, pending.tools, pending.tool_choice)
        response_kind = "assistant_tool_call" if tool_parse.tool_calls else _response_kind(content)
        events = [
            RewardEvent(type="assistant_tool_call", name=tool_call["function"]["name"], arguments=tool_call["function"]["arguments"])
            for tool_call in tool_parse.tool_calls
        ]
        if not events:
            events = [RewardEvent(type=response_kind, text=content)]
        trace = [
            RewardEvent(type="request", messages=pending.messages),
            *events,
            RewardEvent(type="finish", metadata={"finish_reason": "stop"}),
        ]
        sample = _AgentSample(
            item=pending.item,
            messages=pending.messages,
            response_text=content,
            last_response_text=content,
            response_tokens=response.response_tokens,
            response_logprobs=response.response_logprobs,
            trace=trace,
            response_kind=response_kind,
            loss_mask_override=_tool_call_loss_mask(tokenizer, response.response_tokens) if tool_parse.tool_calls else None,
        )
        # The prompt tokens are the fully rendered chat context for this turn,
        # including prior assistant/tool messages. Those context tokens are
        # needed for scoring but are masked out of policy loss.
        self._set_sample_training_row(sample, pending.input_tokens)
        with self._lock:
            if not pending.sample_recorded:
                existing = self._find_sample_for_item_locked(pending.item)
                if existing is None:
                    self._samples.append(sample)
                else:
                    self._append_sample_response(existing, sample)
                pending.sample_recorded = True
        return pending.response or self._build_pending_chat_response(
            pending,
            response.response_tokens,
            content=content,
            tool_calls=tool_parse.tool_calls,
        )

    def _find_sample_for_item_locked(self, item: AgentItem) -> _AgentSample | None:
        """Find an existing trajectory for the same prompt/sample pair."""

        if item.prompt_index < 0 or item.sample_index < 0:
            return None
        key = (item.prompt_index, item.sample_index)
        for sample in self._samples:
            if (sample.item.prompt_index, sample.item.sample_index) == key:
                return sample
        return None

    def _append_sample_response(self, existing: _AgentSample, new_sample: _AgentSample) -> None:
        """Append another model response to an existing multi-call trajectory."""

        old_response_kind = existing.response_kind
        old_response_len = len(existing.response_tokens)
        if new_sample.response_text:
            existing.response_text = f"{existing.response_text}\n{new_sample.response_text}" if existing.response_text else new_sample.response_text
            existing.last_response_text = new_sample.response_text
        existing.response_tokens.extend(new_sample.response_tokens)
        existing.response_logprobs.extend(new_sample.response_logprobs)
        existing.trace.extend(new_sample.trace)
        existing.messages = new_sample.messages
        # Each later turn is rendered as: previous messages + new assistant.
        # Append only the suffix so the training row becomes one trajectory
        # instead of duplicating the shared prefix for every tool call.
        prefix_len = _common_prefix_len(existing.token_row, new_sample.token_row)
        if prefix_len < len(new_sample.token_row):
            existing.token_row.extend(new_sample.token_row[prefix_len:])
            existing.response_mask_row.extend(new_sample.response_mask_row[prefix_len:])
            existing.loss_mask_row.extend(new_sample.loss_mask_row[prefix_len:])
            existing.rollout_logprobs_row.extend(new_sample.rollout_logprobs_row[prefix_len:])
        elif not existing.token_row:
            existing.token_row = list(existing.item.input_tokens) + list(existing.response_tokens)
            prompt_len = len(existing.item.input_tokens)
            existing.response_mask_row = [False] * prompt_len + [True] * len(existing.response_tokens)
            existing.loss_mask_row = [False] * prompt_len + self._response_loss_mask(existing)
            existing.rollout_logprobs_row = [0.0] * prompt_len + list(existing.response_logprobs)
        old_mask = existing.loss_mask_override
        if old_mask is None:
            old_mask = self._response_loss_mask_for_span(old_response_kind, old_response_len)
        new_mask = new_sample.loss_mask_override
        if new_mask is None:
            new_mask = self._response_loss_mask_for_span(new_sample.response_kind, len(new_sample.response_tokens))
        existing.loss_mask_override = list(old_mask) + list(new_mask)
        existing.response_kind = new_sample.response_kind

    def _set_sample_training_row(self, sample: _AgentSample, prompt_tokens: list[int]) -> None:
        response_mask = [True] * len(sample.response_tokens)
        loss_mask = self._response_loss_mask(sample)
        # response_mask marks generated tokens; loss_mask is stricter and can
        # suppress tool-result or other non-policy spans.
        sample.token_row = list(prompt_tokens) + list(sample.response_tokens)
        sample.response_mask_row = [False] * len(prompt_tokens) + response_mask
        sample.loss_mask_row = [False] * len(prompt_tokens) + loss_mask
        sample.rollout_logprobs_row = [0.0] * len(prompt_tokens) + list(sample.response_logprobs)

    def _build_pending_chat_response(
        self,
        pending: _PendingChat,
        response_tokens: list[int],
        *,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        content = content if content is not None else self._trainer.get_tokenizer().decode(response_tokens)
        message = {"role": "assistant", "content": content}
        finish_reason = "stop"
        if tool_calls:
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": pending.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(pending.input_tokens),
                "completion_tokens": len(response_tokens),
                "total_tokens": len(pending.input_tokens) + len(response_tokens),
            },
        }

    def _claim_item(self, messages: list[dict[str, Any]]) -> AgentItem:
        prompt = _first_user_text(messages)
        with self._lock:
            # Prefer prompt-based reuse for multi-turn conversations; fall back
            # to sequential assignment for one-shot agents and ad hoc requests.
            item = self._reuse_item_for_prompt_locked(prompt)
            if item is not None:
                return item
            if self._next_item < len(self._agent_items):
                item = self._agent_items[self._next_item]
                self._next_item += 1
                return item
            return AgentItem(record={"prompt": prompt}, prompt=prompt, input_tokens=[], prompt_index=-1, sample_index=-1)

    def _reuse_item_for_prompt_locked(self, prompt: str) -> AgentItem | None:
        if prompt and not self._agent_items_by_prompt and self._agent_items:
            for item in self._agent_items:
                self._agent_items_by_prompt.setdefault(item.prompt, []).append(item)
        items = self._agent_items_by_prompt.get(prompt)
        if not items:
            return None
        cursor = self._agent_reuse_cursor_by_prompt.get(prompt, 0)
        self._agent_reuse_cursor_by_prompt[prompt] = cursor + 1
        return items[cursor % len(items)]


def load_agent_run_fn(path: str) -> Callable[[RolloutSession, AgentBatch], Any]:
    """Load ``async def run_agent(ctx, batch)`` from a Python file."""

    module_path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load agent function from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_agent = getattr(module, "run_agent", None)
    if not callable(run_agent):
        raise ValueError(f"{module_path} must define callable run_agent(ctx, batch)")
    return run_agent


def _messages_to_prompt_tokens(tokenizer, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, fallback_prompt: str) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        try:
            # Use the model's own chat template so assistant/tool messages are
            # tokenized exactly as they were shown to the model.
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
    return tokenizer.encode(_messages_to_text(messages) or fallback_prompt)


def _render_messages_for_display(tokenizer, messages: list[dict[str, Any]]) -> str:
    """Render a message trajectory with the tokenizer chat template when available."""

    if getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(rendered, str):
                return rendered
        except TypeError:
            pass
    return _messages_to_text(messages)


def _max_sequence_len(params: Any) -> int | None:
    max_prompt_len = getattr(params, "max_prompt_len", None)
    if max_prompt_len is None:
        return None
    return int(max_prompt_len) + int(getattr(params, "max_new_tokens"))


def _filtered_chat_response(*, model: str, prompt_tokens: int, max_sequence_len: int) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
            "max_sequence_len": max_sequence_len,
        },
    }


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        item = dict(message)
        # OpenAI chat-completions assistant tool-call messages commonly carry
        # content=null. Some local chat templates treat content as a string, so
        # normalize it before tokenization while preserving tool_calls.
        if item.get("content") is None:
            item["content"] = ""
        normalized.append(item)
    return normalized


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return _messages_to_text(messages)


def _response_kind(content: str) -> Literal["assistant_text", "assistant_tool_call"]:
    """Classify generated content for loss masking.

    The non-streaming proxy currently receives plain text from the local model.
    If that text is an OpenAI-style tool-call JSON object, treat the whole span
    as a tool call so the default policy can mask it.
    """

    text = content.strip()
    if not text:
        return "assistant_text"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return "assistant_text"
    if isinstance(obj, dict) and ("tool_calls" in obj or ("name" in obj and "arguments" in obj)):
        return "assistant_tool_call"
    if isinstance(obj, list) and obj and all(isinstance(item, dict) and "name" in item for item in obj):
        return "assistant_tool_call"
    return "assistant_text"


def _tool_call_loss_mask(tokenizer, response_tokens: list[int]) -> list[bool]:
    """Mask tool-result sentinels after a generated tool call."""

    mask = [True] * len(response_tokens)
    markers = ("<|tool_response>", "<tool_response>")
    try:
        text = tokenizer.decode(response_tokens)
    except Exception:
        return mask
    marker_pos = min((pos for marker in markers if (pos := text.find(marker)) >= 0), default=-1)
    if marker_pos < 0:
        return mask
    try:
        prefix_tokens = tokenizer.encode(text[:marker_pos])
    except Exception:
        return mask
    start = min(len(prefix_tokens), len(mask))
    for mask_idx in range(start, len(mask)):
        mask[mask_idx] = False
    return mask


def _tool_results_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract OpenAI tool result messages into reward-facing records."""

    results = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        results.append(
            {
                "name": message.get("name"),
                "tool_call_id": message.get("tool_call_id"),
                "content": message.get("content"),
            }
        )
    return results


def _trace_with_tool_results(trace: list[RewardEvent], messages: list[dict[str, Any]]) -> list[RewardEvent]:
    """Return trace augmented with explicit tool-result events from messages."""

    tool_result_events = [
        RewardEvent(
            type="tool_result",
            name=result.get("name"),
            content=result.get("content"),
            metadata={"tool_call_id": result.get("tool_call_id")} if result.get("tool_call_id") is not None else {},
        )
        for result in _tool_results_from_messages(messages)
    ]
    if not tool_result_events:
        return trace
    augmented = []
    inserted = False
    for event in trace:
        if not inserted and event.type == "finish":
            augmented.extend(tool_result_events)
            inserted = True
        augmented.append(event)
    if not inserted:
        augmented.extend(tool_result_events)
    return augmented


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    """Return the shared token prefix length for incremental multi-turn rows."""

    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    return limit


def _chat_batch_key(params: Any) -> _ChatBatchKey:
    return _ChatBatchKey(
        greedy=bool(getattr(params, "greedy", False)),
        max_new_tokens=int(getattr(params, "max_new_tokens")),
        temperature=float(getattr(params, "temperature")),
        top_p=float(getattr(params, "top_p")),
        top_k=int(getattr(params, "top_k", -1)),
        stop_token_ids=tuple(int(item) for item in (getattr(params, "stop_token_ids", None) or ())),
        ignore_eos=bool(getattr(params, "ignore_eos", False)),
        skip_special_tokens=bool(getattr(params, "skip_special_tokens", True)),
        max_prompt_len=getattr(params, "max_prompt_len", None),
    )


def _trainer_dp_size(trainer: Any) -> int:
    config = getattr(trainer, "config", None)
    world_size = int(getattr(config, "world_size", 1) or 1)
    tp_size = int(getattr(config, "tp_size", 1) or 1)
    if tp_size <= 0:
        return 1
    return max(world_size // tp_size, 1)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


async def maybe_await(value):
    """Await coroutine return values while accepting sync agent functions."""

    if asyncio.iscoroutine(value):
        return await value
    return value


__all__ = [
    "AgentBatch",
    "AgentItem",
    "AgentTrainBatch",
    "LossMaskPolicy",
    "RewardEvent",
    "RewardRecord",
    "RolloutSession",
    "load_agent_run_fn",
    "maybe_await",
]
