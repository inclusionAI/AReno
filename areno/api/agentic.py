"""Agentic rollout session support.

This module provides the agentic rollout path for Areno. It exposes a small
OpenAI-compatible HTTP surface that agent code can call with a standard OpenAI
client. Agent code returns explicit trajectories, which are converted into the
same token/logprob rows consumed by existing trainers.

The first implementation intentionally supports non-streaming
``/v1/chat/completions``. The public data model already keeps trace, messages,
tool calls, tool results, and loss masks explicit so streaming and richer tool
coverage can be added without changing trainer boundaries.
"""

from __future__ import annotations

import asyncio
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

from areno.api.openai_chat import (
    build_chat_completion_response,
    first_user_text,
    messages_to_prompt_tokens,
    messages_to_text,
    normalize_messages,
)
from areno.api.rewards import RewardEvent, RewardRecord
from areno.api.tool_call_parser import get_tool_call_parser, infer_tool_call_parser_name

if TYPE_CHECKING:
    from areno.api.models import SamplingParams

logger = logging.getLogger(__name__)


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
    def from_prompt_batch(cls, prompt_batch, n_samples: int) -> AgentBatch:
        """Build an agent batch from the trainer's tokenized prompt batch."""

        return cls(
            records=[dict(item.record) for item in prompt_batch.items],
            prompts=[item.prompt for item in prompt_batch.items],
            input_tokens=[list(item.input_tokens) for item in prompt_batch.items],
            n_samples=int(n_samples),
        )

    def iter_samples(self) -> Iterator[AgentItem]:
        """Yield one item per prompt/sample pair in stable row order."""

        for prompt_index, (record, prompt, input_tokens) in enumerate(
            zip(self.records, self.prompts, self.input_tokens, strict=True)
        ):
            for sample_index in range(self.n_samples):
                yield AgentItem(
                    record=record,
                    prompt=prompt,
                    input_tokens=input_tokens,
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                )


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
class AgentTrajectoryTurn:
    """One explicit agent turn returned by ``run_agent``."""

    item: AgentItem
    messages: list[dict[str, Any]]
    response: Any | None = None
    response_tokens: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    parsed_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = "policy"
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None

    def __post_init__(self) -> None:
        if self.response is None:
            return
        metadata = _chat_response_agentic_metadata(self.response)
        self.response_tokens = list(metadata["response_tokens"])
        self.response_logprobs = [float(value) for value in metadata["response_logprobs"]]
        self.parsed_tool_calls = _chat_response_message_tool_calls(self.response)


@dataclass(slots=True)
class AgentTrajectory:
    """Explicit trajectories returned by ``run_agent`` for one rollout batch."""

    turns: list[AgentTrajectoryTurn] = field(default_factory=list)


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
    last_tool_calls: list[dict[str, Any]] = field(default_factory=list)
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
    item: AgentItem | None
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
        sampling_params: SamplingParams,
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
        self._max_running_prompts = (
            max(1, int(max_running_prompts)) if max_running_prompts is not None else self._dp_size
        )
        self._local_max_running_prompts = max(_ceil_div(self._max_running_prompts, self._dp_size), 1)
        self._timeout_s = float(timeout_s)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing = False
        self._base_url = ""
        self._proxy_enabled = bool(proxy)

    @property
    def max_running_prompts(self) -> int:
        """Maximum concurrently running prompts for agentic rollout."""

        return self._max_running_prompts

    async def __aenter__(self) -> RolloutSession:
        """Start the local proxy."""

        self._loop = asyncio.get_running_loop()
        await self._trainer.begin_rollout_session_async()
        if not self._proxy_enabled:
            return self
        handler_cls = self._handler_cls()
        try:
            self._server = self._http_server_cls()(("127.0.0.1", 0), handler_cls)
            self._server.timeout = 0.5
            host, port = self._server.server_address[:2]
            self._base_url = f"http://{host}:{port}/v1"
            self._thread = threading.Thread(target=self._server.serve_forever, name="areno-agentic-proxy", daemon=True)
            self._thread.start()
        except BaseException:
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

    async def sync_rollout_session_async(self) -> None:
        """Synchronize backend rollout workers before the agent submits requests."""

        await self._trainer.sync_rollout_session_async()

    def _http_server_cls(self) -> type[_AgenticHTTPServer]:
        return type(
            "AgenticRolloutHTTPServer",
            (_AgenticHTTPServer,),
            {"request_queue_size": max(2048, self._max_running_prompts), "max_threads": 2048},
        )

    def reward_record(self, sample: _AgentSample) -> RewardRecord:
        answer = sample.item.record.get("answer", sample.item.record.get("solutions"))
        messages = list(sample.messages)
        if sample.last_tool_calls:
            messages.append({"role": "assistant", "content": "", "tool_calls": list(sample.last_tool_calls)})
        else:
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

    def _train_rows_from_samples(self, samples: list[_AgentSample]) -> _AgentTrainRows:
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
            _write_json(handler, 500, {"error": {"message": str(exc)}})

    async def _complete_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = _normalize_messages(body.get("messages") or [])
        tools = list(body.get("tools") or [])
        tool_choice = body.get("tool_choice")
        params = (
            self._sampling_params.model_copy()
            if hasattr(self._sampling_params, "model_copy")
            else self._sampling_params.copy()
        )
        if body.get("max_tokens") is not None:
            params.max_new_tokens = int(body["max_tokens"])
        if body.get("temperature") is not None:
            params.temperature = float(body["temperature"])
        if body.get("top_p") is not None:
            params.top_p = float(body["top_p"])
        pending = _PendingChat(
            item=None,
            messages=messages,
            input_tokens=[],
            params=params,
            key=_chat_batch_key(params),
            model=body.get("model") or "policy",
            tools=tools,
            tool_choice=tool_choice,
            created_at=time.monotonic(),
        )
        if self._loop is None:
            raise RuntimeError("agent rollout proxy is not running")
        try:
            await asyncio.wait_for(asyncio.shield(self._run_chat_request(pending)), timeout=self._timeout_s)
        except asyncio.TimeoutError:
            pending.cancelled = True
            raise TimeoutError("agent rollout proxy timed out waiting for completion")
        if pending.error is not None:
            raise pending.error
        if pending.response is None:
            raise RuntimeError("agent rollout proxy produced no response")
        return pending.response

    async def _run_chat_request(self, pending: _PendingChat) -> None:
        tokenizer = self._trainer.get_tokenizer()
        if pending.cancelled:
            return
        try:
            pending.input_tokens = _messages_to_prompt_tokens(
                tokenizer,
                pending.messages,
                tools=pending.tools,
                fallback_prompt=_first_user_text(pending.messages),
            )
            max_context_len = _max_context_len(pending.params)
            if max_context_len is not None and len(pending.input_tokens) > max_context_len:
                response = _filtered_chat_response(
                    model=pending.model,
                    prompt_tokens=len(pending.input_tokens),
                    max_sequence_len=max_context_len,
                )
                self._set_pending_response(pending, response)
                return
            results = await self._trainer.rollout_token_batch_async([pending.input_tokens], 1, pending.params)
            sequence = results[0].sequences[0] if results and results[0].sequences else None
            if sequence is None:
                self._set_pending_response(pending, self._build_chat_response(pending, _ResponseData([], [])))
            else:
                self._set_pending_response(
                    pending,
                    self._build_chat_response(
                        pending,
                        _ResponseData(response_tokens=sequence.resp_tokens, response_logprobs=sequence.resp_logprobs),
                    ),
                )
        except BaseException as exc:
            self._set_pending_error(pending, exc)

    def _sample_from_trajectory_turn(self, turn: AgentTrajectoryTurn) -> _AgentSample:
        tokenizer = self._trainer.get_tokenizer()
        input_tokens = _messages_to_prompt_tokens(
            tokenizer,
            turn.messages,
            tools=turn.tools,
            fallback_prompt=turn.item.prompt,
        )
        pending = _PendingChat(
            item=turn.item,
            messages=_normalize_messages(turn.messages),
            input_tokens=input_tokens,
            params=self._sampling_params,
            key=_chat_batch_key(self._sampling_params),
            model=turn.model,
            tools=list(turn.tools),
            tool_choice=turn.tool_choice,
            created_at=time.monotonic(),
        )
        return self._sample_from_pending_chat(
            pending,
            _ResponseData(response_tokens=list(turn.response_tokens), response_logprobs=list(turn.response_logprobs)),
            tool_calls=turn.parsed_tool_calls,
        )

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

    def _sample_from_pending_chat(
        self,
        pending: _PendingChat,
        response: _ResponseData,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> _AgentSample:
        if pending.item is None:
            raise ValueError("explicit agent trajectory turn requires an AgentItem")
        tokenizer = self._trainer.get_tokenizer()
        content = tokenizer.decode(response.response_tokens)
        tool_calls = list(tool_calls or [])
        response_kind = "assistant_tool_call" if tool_calls else "assistant_text"
        events = [
            RewardEvent(
                type="assistant_tool_call",
                name=tool_call["function"]["name"],
                arguments=tool_call["function"]["arguments"],
            )
            for tool_call in tool_calls
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
            last_tool_calls=tool_calls,
            response_tokens=response.response_tokens,
            response_logprobs=response.response_logprobs,
            trace=trace,
            response_kind=response_kind,
            loss_mask_override=_tool_call_loss_mask(tokenizer, response.response_tokens) if tool_calls else None,
        )
        # The prompt tokens are the fully rendered chat context for this turn,
        # including prior assistant/tool messages. Those context tokens are
        # needed for scoring but are masked out of policy loss.
        self._set_sample_training_row(sample, pending.input_tokens)
        return sample

    def _build_chat_response(self, pending: _PendingChat, response: _ResponseData) -> dict[str, Any]:
        tokenizer = self._trainer.get_tokenizer()
        content = tokenizer.decode(response.response_tokens)
        tool_parse = self._tool_call_parser.parse(content, pending.tools, pending.tool_choice)
        return self._build_pending_chat_response(
            pending,
            response.response_tokens,
            response_logprobs=response.response_logprobs,
            content=content,
            tool_calls=tool_parse.tool_calls,
        )

    def _append_sample_response(self, existing: _AgentSample, new_sample: _AgentSample) -> None:
        """Append another model response to an existing multi-call trajectory."""

        old_response_kind = existing.response_kind
        old_response_len = len(existing.response_tokens)
        if new_sample.response_text:
            existing.response_text = (
                f"{existing.response_text}\n{new_sample.response_text}"
                if existing.response_text
                else new_sample.response_text
            )
            existing.last_response_text = new_sample.response_text
            existing.last_tool_calls = list(new_sample.last_tool_calls)
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
        response_logprobs: list[float] | None = None,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del content
        finish_reason = "tool_calls" if tool_calls else "stop"
        return build_chat_completion_response(
            tokenizer=self._trainer.get_tokenizer(),
            model=pending.model,
            prompt_tokens=len(pending.input_tokens),
            response_ids=[list(response_tokens)],
            finish_reasons=[finish_reason],
            tools=pending.tools,
            tool_choice=pending.tool_choice,
            tool_call_parser=self._tool_call_parser,
            parsed_tool_calls=[list(tool_calls or [])],
            response_logprobs=[list(response_logprobs or [])],
            include_areno_metadata=True,
            input_tokens=pending.input_tokens,
        )


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


def _messages_to_prompt_tokens(
    tokenizer, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, fallback_prompt: str
) -> list[int]:
    return messages_to_prompt_tokens(tokenizer, messages, tools=tools, fallback_prompt=fallback_prompt)


def _chat_response_agentic_metadata(response: Any) -> dict[str, Any]:
    """Extract Areno token/logprob metadata from a proxy chat response."""

    metadata = None
    if isinstance(response, dict):
        metadata = response.get("areno")
    else:
        model_extra = getattr(response, "model_extra", None)
        if isinstance(model_extra, dict):
            metadata = model_extra.get("areno")
        if metadata is None:
            metadata = getattr(response, "areno", None)
    if not isinstance(metadata, dict):
        raise ValueError("chat response does not include Areno trajectory metadata")
    response_tokens = metadata.get("response_tokens")
    response_logprobs = metadata.get("response_logprobs")
    if not isinstance(response_tokens, list) or not isinstance(response_logprobs, list):
        raise ValueError("Areno trajectory metadata must include response_tokens and response_logprobs lists")
    if len(response_tokens) != len(response_logprobs):
        raise ValueError("Areno trajectory metadata response_tokens/response_logprobs length mismatch")
    return metadata


def _chat_response_message_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract proxy-parsed OpenAI tool calls from a chat response.

    Explicit agent trajectories must trust the OpenAI-compatible response
    surface. Tool-call parsing belongs in the proxy response builder, not in
    the agent loop or trajectory ingestion path.
    """

    choices = _response_get(response, "choices")
    if not isinstance(choices, list) or not choices:
        return []
    first_choice = choices[0]
    message = _response_get(first_choice, "message") or {}
    raw_calls = _response_get(message, "tool_calls")
    if not isinstance(raw_calls, list):
        return []
    calls = []
    for raw_call in raw_calls:
        function = _response_get(raw_call, "function") or {}
        name = _response_get(function, "name")
        if not name:
            continue
        arguments = _response_get(function, "arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False, sort_keys=True)
        calls.append(
            {
                "id": _response_get(raw_call, "id") or f"call_{uuid.uuid4().hex}",
                "type": _response_get(raw_call, "type") or "function",
                "function": {"name": str(name), "arguments": arguments},
            }
        )
    return calls


def _response_get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _render_messages_for_display(tokenizer, messages: list[dict[str, Any]]) -> str:
    """Render a message trajectory with the tokenizer chat template when available."""

    messages = _normalize_messages(messages)
    if getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(rendered, str):
                return rendered
        except Exception:
            pass
    return _messages_to_text(messages)


def _max_context_len(params: Any) -> int | None:
    max_prompt_len = getattr(params, "max_prompt_len", None)
    if max_prompt_len is None:
        return None
    return int(max_prompt_len)


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
        "areno": {
            "input_tokens": [],
            "response_tokens": [],
            "response_logprobs": [],
        },
    }


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return normalize_messages(messages)


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    return messages_to_text(messages)


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    return first_user_text(messages)


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
    if trainer is None:
        return 1
    try:
        return max(int(trainer.dp_size()), 1)
    except (AttributeError, RuntimeError):
        config = trainer.config
        world_size = int(config.world_size or 1)
        tp_size = int(config.tp_size or 1)
        if tp_size <= 0:
            return 1
        return max(world_size // tp_size, 1)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("content-type", "application/json")
        handler.send_header("content-length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
        logger.debug("agentic proxy client disconnected before response write")


async def maybe_await(value):
    """Await coroutine return values while accepting sync agent functions."""

    if asyncio.iscoroutine(value):
        return await value
    return value


__all__ = [
    "AgentBatch",
    "AgentItem",
    "AgentTrainBatch",
    "AgentTrajectory",
    "AgentTrajectoryTurn",
    "LossMaskPolicy",
    "RewardEvent",
    "RewardRecord",
    "RolloutSession",
    "load_agent_run_fn",
    "maybe_await",
]
