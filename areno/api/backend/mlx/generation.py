"""Provider-neutral continuous-batch rollout helpers for MLX."""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import queue
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any

from areno.api.backend.common import expand_prompt_features, expand_prompts, group_rollout_sequences
from areno.api.backend.mlx.numerics import float32_logits_processor
from areno.api.backend.mlx.provider import MlxModelProvider
from areno.api.models import RolloutResult, RolloutSequence, SamplingParams

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GenerationConfig:
    max_running_prompts: int
    completion_batch_size: int
    prefill_batch_size: int
    prefill_step_size: int
    max_kv_size: int | None
    decode_progress_interval_s: float


@dataclass(slots=True)
class _Request:
    prompts: list[list[int]]
    n_samples: int
    sampling: SamplingParams
    features: list[dict | None] | None
    future: Future[list[RolloutResult]] = field(default_factory=Future)
    handles: list[tuple[object, int]] = field(default_factory=list)
    tokens: dict[tuple[object, int], list[int]] = field(default_factory=dict)
    logprobs: dict[tuple[object, int], list[float]] = field(default_factory=dict)
    finished: set[tuple[object, int]] = field(default_factory=set)
    expanded_prompts: list[list[int]] = field(default_factory=list)
    expanded_features: list[dict | None] = field(default_factory=list)
    next_insert: int = 0


@dataclass(slots=True)
class _DropState:
    future: Future[None] = field(default_factory=Future)


class _TextBatchGenerator:
    discard_on_state_drop = False

    def __init__(self, provider: MlxModelProvider, config: GenerationConfig) -> None:
        from mlx_lm.generate import BatchGenerator

        self._tokenizer = provider.tokenizer
        self._generator = BatchGenerator(
            provider.generation_model,
            completion_batch_size=max(int(config.completion_batch_size), 1),
            prefill_batch_size=max(int(config.prefill_batch_size), 1),
            prefill_step_size=max(int(config.prefill_step_size), 1),
            max_kv_size=config.max_kv_size,
        )

    def insert(
        self,
        prompts: list[list[int]],
        prompt_features: list[dict | None],
        params: SamplingParams,
    ) -> list[int]:
        from mlx_lm.generate import StopSequenceMatcher
        from mlx_lm.sample_utils import make_sampler

        if any(feature is not None for feature in prompt_features):
            raise ValueError("text-only MLX checkpoints cannot consume multimodal prompt features")
        sampler = make_sampler(
            temp=0.0 if params.greedy else float(params.temperature),
            top_p=float(params.top_p),
            top_k=max(int(params.top_k), 0),
        )
        stop_sequences = _stop_sequences(self._tokenizer, params)
        return self._generator.insert(
            prompts,
            max_tokens=[int(params.max_new_tokens)] * len(prompts),
            samplers=[sampler] * len(prompts),
            logits_processors=[[float32_logits_processor] for _ in prompts],
            stop_matchers=[StopSequenceMatcher(stop_sequences or None) for _ in prompts],
        )

    def next(self) -> list[Any]:
        return self._generator.next_generated()

    def token_logprob(self, response: Any) -> float:
        return float(response.logprobs[response.token].item())

    def close(self) -> None:
        self._generator.close()

    def drop_state(self) -> None:
        if self._generator.prompt_cache_nbytes:
            raise RuntimeError("cannot drop MLX rollout state while text generation is active")


class _VlmBatchGenerator:
    discard_on_state_drop = False

    def __init__(
        self,
        provider: MlxModelProvider,
        config: GenerationConfig,
        params: SamplingParams,
    ) -> None:
        from mlx_lm.sample_utils import make_sampler
        from mlx_vlm.generate import BatchGenerator

        self._provider = provider
        self._params = params
        sampler = make_sampler(
            temp=0.0 if params.greedy else float(params.temperature),
            top_p=float(params.top_p),
            top_k=max(int(params.top_k), 0),
        )
        stop_tokens = {sequence[0] for sequence in _stop_sequences(provider.tokenizer, params) if len(sequence) == 1}
        self._generator = BatchGenerator(
            provider.generation_model,
            provider.processor,
            sampler=sampler,
            stop_tokens=stop_tokens or None,
            completion_batch_size=max(int(config.completion_batch_size), 1),
            prefill_batch_size=max(int(config.prefill_batch_size), 1),
            prefill_step_size=max(int(config.prefill_step_size), 1),
            compute_logprobs=True,
            logits_processors=[float32_logits_processor],
        )

    def insert(
        self,
        prompts: list[list[int]],
        prompt_features: list[dict | None],
        params: SamplingParams,
    ) -> list[int]:
        if _sampling_key(params) != _sampling_key(self._params):
            raise ValueError("MLX-VLM batch generator sampling mismatch")
        prompt_kwargs = [
            self._provider.prepare_generation_prompt(tokens, features)
            for tokens, features in zip(prompts, prompt_features, strict=True)
        ]
        return self._generator.insert(
            prompts,
            max_tokens=[int(params.max_new_tokens)] * len(prompts),
            prompt_kwargs=prompt_kwargs,
            logits_processors=[[float32_logits_processor] for _ in prompts],
        )

    def next(self) -> list[Any]:
        _, responses = self._generator.next()
        return responses

    @staticmethod
    def token_logprob(response: Any) -> float:
        return float(response.token_logprob)

    def close(self) -> None:
        self._generator.close()

    def drop_state(self) -> None:
        if self._generator.has_work:
            raise RuntimeError("cannot drop MLX rollout state while multimodal generation is active")
        self._generator._generation_batch.filter([])
        self._generator._prompt_batch = None
        self._generator._unprocessed_sequences.clear()


class ContinuousBatchScheduler:
    """Own backend-native batch generators for one rollout session."""

    def __init__(self, provider: MlxModelProvider, config: GenerationConfig) -> None:
        self._provider = provider
        self._config = config
        self._generators: dict[object, _TextBatchGenerator | _VlmBatchGenerator] = {}
        if not provider.is_multimodal:
            self._generators[None] = _TextBatchGenerator(provider, config)
        self._commands: queue.Queue[_Request | _DropState | None] = queue.Queue()
        self._requests_by_handle: dict[tuple[object, int], _Request] = {}
        self._pending_requests: deque[_Request] = deque()
        self._closed = False
        self._failure: BaseException | None = None
        self._decode_progress_next_time = 0.0
        self._decode_progress_window_start = 0.0
        self._decode_progress_window_tokens = 0
        self._shutdown_complete = Event()
        self._park_on_shutdown = _needs_compile_cache_thread_workaround()
        self._park = Event()
        self._thread = Thread(target=self._run, name="areno-mlx-rollout", daemon=True)
        self._thread.start()

    def submit(
        self,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
        prompt_features: list[dict | None] | None = None,
    ) -> Future[list[RolloutResult]]:
        if self._closed:
            raise RuntimeError("MLX rollout scheduler is closed")
        if self._failure is not None:
            raise RuntimeError("MLX rollout scheduler failed") from self._failure
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if prompt_features is not None and len(prompt_features) != len(prompt_tokens):
            raise ValueError("prompt_features must align with prompt_tokens")
        if not prompt_tokens:
            completed: Future[list[RolloutResult]] = Future()
            completed.set_result([])
            return completed
        request = _Request(prompt_tokens, n_samples, sampling_params, prompt_features)
        self._commands.put(request)
        return request.future

    async def submit_async(
        self,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
        prompt_features: list[dict | None] | None = None,
    ) -> list[RolloutResult]:
        return await asyncio.wrap_future(self.submit(prompt_tokens, n_samples, sampling_params, prompt_features))

    def drop_state(self) -> None:
        """Release completed KV and allocator caches without replacing the scheduler."""

        if self._closed:
            raise RuntimeError("MLX rollout scheduler is closed")
        if self._failure is not None:
            raise RuntimeError("MLX rollout scheduler failed") from self._failure
        command = _DropState()
        self._commands.put(command)
        command.future.result()

    def close(self) -> None:
        if self._closed:
            return
        import mlx.core as mx

        self._closed = True
        self._commands.put(None)
        self._shutdown_complete.wait()
        if not self._park_on_shutdown:
            self._thread.join()
        mx.clear_cache()

    def _run(self) -> None:
        closing = False
        try:
            while True:
                if not self._requests_by_handle and not self._pending_requests:
                    command = self._commands.get()
                    if command is None:
                        break
                    self._process(command)
                closing = self._drain_commands() or closing
                self._admit_pending()
                for key, generator in tuple(self._generators.items()):
                    responses = generator.next()
                    token_delta = sum(response.finish_reason != "stop" for response in responses)
                    for response in responses:
                        self._record_response(key, generator, response)
                    self._record_decode_progress(token_delta)
                self._admit_pending()
                if closing and not self._requests_by_handle and not self._pending_requests:
                    break
        except BaseException as exc:
            self._failure = exc
            logger.exception("MLX rollout scheduler failed")
            self._fail_all(exc)
        finally:
            for generator in self._generators.values():
                generator.close()
            self._generators.clear()
            self._provider = None
            self._shutdown_complete.set()
            if self._park_on_shutdown:
                # MLX <= 0.32.1 can segfault while destroying a thread-local
                # CompileCache. Upstream PR #4248 fixes clear_streams(), but no
                # Python cache-clear API exists in affected releases.
                self._park.wait()

    def _drain_commands(self) -> bool:
        closing = False
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return closing
            if command is None:
                closing = True
            else:
                self._process(command)

    def _process(self, command: _Request | _DropState) -> None:
        if isinstance(command, _DropState):
            try:
                self._drop_state(command)
            except BaseException as exc:
                if not command.future.done():
                    command.future.set_exception(exc)
                raise
        else:
            self._insert_or_fail(command)

    def _drop_state(self, command: _DropState) -> None:
        if self._requests_by_handle or self._pending_requests:
            raise RuntimeError("cannot drop MLX rollout state while requests are active")
        import mlx.core as mx

        discarded = []
        for key, generator in self._generators.items():
            generator.drop_state()
            if generator.discard_on_state_drop:
                generator.close()
                discarded.append(key)
        for key in discarded:
            self._generators.pop(key)
        mx.synchronize()
        mx.clear_cache()
        command.future.set_result(None)

    def _insert_or_fail(self, request: _Request) -> None:
        try:
            request.expanded_prompts = expand_prompts(request.prompts, request.n_samples)
            features = expand_prompt_features(request.features, len(request.prompts), request.n_samples)
            request.expanded_features = features if features is not None else [None] * len(request.expanded_prompts)
            self._pending_requests.append(request)
            self._admit_pending()
        except BaseException as exc:
            if not request.future.done():
                request.future.set_exception(exc)
            raise

    def _admit_pending(self) -> None:
        capacity = max(int(self._config.max_running_prompts), 1) - len(self._requests_by_handle)
        while capacity > 0 and self._pending_requests:
            request = self._pending_requests.popleft()
            end = min(request.next_insert + capacity, len(request.expanded_prompts))
            self._insert(request, request.next_insert, end)
            admitted = end - request.next_insert
            request.next_insert = end
            capacity -= admitted
            if request.next_insert < len(request.expanded_prompts):
                self._pending_requests.append(request)

    def _insert(self, request: _Request, start: int, end: int) -> None:
        key: object = _sampling_key(request.sampling) if self._provider.is_multimodal else None
        generator = self._generators.get(key)
        if generator is None:
            generator = _VlmBatchGenerator(self._provider, self._config, request.sampling)
            self._generators[key] = generator
        uids = generator.insert(
            request.expanded_prompts[start:end],
            request.expanded_features[start:end],
            request.sampling,
        )
        handles = [(key, uid) for uid in uids]
        request.handles.extend(handles)
        request.tokens.update((handle, []) for handle in handles)
        request.logprobs.update((handle, []) for handle in handles)
        for handle in handles:
            self._requests_by_handle[handle] = request

    def _record_response(self, key: object, generator: Any, response: Any) -> None:
        handle = (key, int(response.uid))
        request = self._requests_by_handle[handle]
        if response.finish_reason != "stop":
            request.tokens[handle].append(int(response.token))
            request.logprobs[handle].append(generator.token_logprob(response))
        if response.finish_reason is None:
            return
        request.finished.add(handle)
        self._requests_by_handle.pop(handle, None)
        if len(request.finished) == len(request.expanded_prompts):
            request.future.set_result(_request_result(request))

    def _record_decode_progress(self, token_delta: int) -> None:
        """Emit the same throttled decode progress line as the CUDA backend."""

        interval_s = float(self._config.decode_progress_interval_s)
        if interval_s <= 0:
            return
        now = time.perf_counter()
        if self._decode_progress_next_time <= 0.0:
            if token_delta <= 0:
                return
            self._decode_progress_window_start = now
            self._decode_progress_next_time = now + interval_s
            return
        self._decode_progress_window_tokens += int(token_delta)
        if now < self._decode_progress_next_time:
            return
        elapsed = max(now - self._decode_progress_window_start, 1e-9)
        tokens = self._decode_progress_window_tokens
        self._decode_progress_window_start = now
        self._decode_progress_next_time = now + interval_s
        self._decode_progress_window_tokens = 0
        logger.info(
            "rollout decode progress: dp=0/1 active=%d cuda_graph=False tokens_per_second=%.1f",
            len(self._requests_by_handle),
            tokens / elapsed,
        )

    def _fail_all(self, exc: BaseException) -> None:
        requests = {id(request): request for request in self._requests_by_handle.values()}
        requests.update((id(request), request) for request in self._pending_requests)
        self._pending_requests.clear()
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            if isinstance(command, _Request):
                requests[id(command)] = command
            elif isinstance(command, _DropState) and not command.future.done():
                command.future.set_exception(exc)
        for request in requests.values():
            if not request.future.done():
                request.future.set_exception(exc)


def _request_result(request: _Request) -> list[RolloutResult]:
    flat = [
        RolloutSequence(resp_tokens=request.tokens[handle], resp_logprobs=request.logprobs[handle])
        for handle in request.handles
    ]
    return group_rollout_sequences(flat, len(request.prompts), request.n_samples)


def _sampling_key(params: SamplingParams) -> tuple[Any, ...]:
    return (
        bool(params.greedy),
        float(params.temperature),
        float(params.top_p),
        int(params.top_k),
        int(params.max_new_tokens),
        bool(params.ignore_eos),
        tuple(params.stop_token_ids or ()),
        tuple(params.stop or ()),
    )


def _needs_compile_cache_thread_workaround() -> bool:
    try:
        release = importlib.metadata.version("mlx").split("+", 1)[0]
        parts = tuple(int(part) for part in release.split(".")[:3])
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return False
    return parts <= (0, 32, 1)


def _stop_sequences(tokenizer: Any, params: SamplingParams) -> list[list[int]]:
    stop_sequences: list[list[int]] = []
    if not params.ignore_eos:
        eos_ids = getattr(tokenizer, "eos_token_ids", None)
        if eos_ids is None:
            eos_ids = getattr(tokenizer, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = []
        elif isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        stop_sequences.extend([[int(token)] for token in eos_ids])
    if params.stop_token_ids:
        stop_sequences.extend([[int(token)] for token in params.stop_token_ids])
    for value in params.stop or []:
        tokens = tokenizer.encode(value, add_special_tokens=False)
        if tokens:
            stop_sequences.append([int(token) for token in tokens])
    unique = dict.fromkeys(tuple(sequence) for sequence in stop_sequences)
    return [list(sequence) for sequence in unique]


__all__ = ["ContinuousBatchScheduler", "GenerationConfig"]
