"""MLX-LM continuous-batch rollout helpers."""

from __future__ import annotations

import asyncio
import queue
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

from areno.api.backend.common import expand_prompts, group_rollout_sequences
from areno.api.backend.mlx.numerics import float32_logits_processor
from areno.api.models import RolloutResult, RolloutSequence, SamplingParams


@dataclass(slots=True)
class GenerationConfig:
    completion_batch_size: int
    prefill_batch_size: int
    prefill_step_size: int
    max_kv_size: int | None


@dataclass(slots=True)
class _Request:
    prompts: list[list[int]]
    n_samples: int
    sampling: SamplingParams
    future: Future[list[RolloutResult]] = field(default_factory=Future)
    uids: list[int] = field(default_factory=list)
    tokens: dict[int, list[int]] = field(default_factory=dict)
    logprobs: dict[int, list[float]] = field(default_factory=dict)
    finished: set[int] = field(default_factory=set)


class ContinuousBatchScheduler:
    """Own one ``BatchGenerator`` for the duration of a rollout session."""

    def __init__(self, model: Any, tokenizer: Any, config: GenerationConfig) -> None:
        from mlx_lm.generate import BatchGenerator

        self._tokenizer = tokenizer
        self._generator = BatchGenerator(
            model,
            completion_batch_size=max(int(config.completion_batch_size), 1),
            prefill_batch_size=max(int(config.prefill_batch_size), 1),
            prefill_step_size=max(int(config.prefill_step_size), 1),
            max_kv_size=config.max_kv_size,
        )
        self._commands: queue.Queue[_Request | None] = queue.Queue()
        self._requests_by_uid: dict[int, _Request] = {}
        self._closed = False
        self._thread = Thread(target=self._run, name="areno-mlx-rollout", daemon=True)
        self._thread.start()

    def submit(
        self,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
    ) -> Future[list[RolloutResult]]:
        if self._closed:
            raise RuntimeError("MLX rollout scheduler is closed")
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if not prompt_tokens:
            completed: Future[list[RolloutResult]] = Future()
            completed.set_result([])
            return completed
        request = _Request(prompt_tokens, n_samples, sampling_params)
        self._commands.put(request)
        return request.future

    async def submit_async(
        self,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
    ) -> list[RolloutResult]:
        return await asyncio.wrap_future(self.submit(prompt_tokens, n_samples, sampling_params))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._commands.put(None)
        self._thread.join()
        self._generator.close()

    def _run(self) -> None:
        closing = False
        try:
            while True:
                if not self._requests_by_uid:
                    command = self._commands.get()
                    if command is None:
                        break
                    self._insert(command)
                closing = self._drain_commands() or closing
                responses = self._generator.next_generated()
                for response in responses:
                    self._record_response(response)
                if closing and not self._requests_by_uid:
                    break
        except BaseException as exc:
            self._fail_all(exc)

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
                self._insert(command)

    def _insert(self, request: _Request) -> None:
        from mlx_lm.generate import StopSequenceMatcher
        from mlx_lm.sample_utils import make_sampler

        params = request.sampling
        sampler = make_sampler(
            temp=0.0 if params.greedy else float(params.temperature),
            top_p=float(params.top_p),
            top_k=max(int(params.top_k), 0),
        )
        expanded = expand_prompts(request.prompts, request.n_samples)
        stop_sequences = _stop_sequences(self._tokenizer, params)
        request.uids = self._generator.insert(
            expanded,
            max_tokens=[int(params.max_new_tokens)] * len(expanded),
            samplers=[sampler] * len(expanded),
            logits_processors=[[float32_logits_processor] for _ in expanded],
            stop_matchers=[StopSequenceMatcher(stop_sequences or None) for _ in expanded],
        )
        request.tokens = {uid: [] for uid in request.uids}
        request.logprobs = {uid: [] for uid in request.uids}
        for uid in request.uids:
            self._requests_by_uid[uid] = request

    def _record_response(self, response: Any) -> None:
        request = self._requests_by_uid[response.uid]
        if response.finish_reason != "stop":
            request.tokens[response.uid].append(int(response.token))
            request.logprobs[response.uid].append(float(response.logprobs[response.token].item()))
        if response.finish_reason is None:
            return
        request.finished.add(response.uid)
        self._requests_by_uid.pop(response.uid, None)
        if len(request.finished) == len(request.uids):
            request.future.set_result(_request_result(request))

    def _fail_all(self, exc: BaseException) -> None:
        requests = {id(request): request for request in self._requests_by_uid.values()}
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            if command is not None:
                requests[id(command)] = command
        for request in requests.values():
            if not request.future.done():
                request.future.set_exception(exc)


def _request_result(request: _Request) -> list[RolloutResult]:
    flat = [
        RolloutSequence(resp_tokens=request.tokens[uid], resp_logprobs=request.logprobs[uid]) for uid in request.uids
    ]
    return group_rollout_sequences(flat, len(request.prompts), request.n_samples)


def _stop_sequences(tokenizer: Any, params: SamplingParams) -> list[list[int]]:
    stop_sequences: list[list[int]] = []
    if not params.ignore_eos:
        eos_ids = getattr(tokenizer, "eos_token_ids", None)
        if eos_ids is None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            eos_ids = [] if eos_id is None else [eos_id]
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
