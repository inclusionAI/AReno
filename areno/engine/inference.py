"""Rollout generation and decode graph management."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from areno.engine.data import RolloutOutput, SamplingParams
from areno.engine.data.rollout_state import InferenceBatchState, payload_to_infer_meta
from areno.engine.data.sampling import (
    _make_sample_generator,
    _policy_token_logprobs,
    _sample_full_vocab,
    _sample_greedy_sharded,
    _stop_token_ids,
    _tokens_match_any,
    _truncate_generated,
)
from areno.engine.protocol import RolloutPayload
from areno.engine.parallel.collectives import broadcast_object, broadcast_tensor
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.common import _check_token_ids, _device_long
from areno.engine.runtime.decode_graph import DecodeGraph, bucket_for, has_graph_capture_memory, sync_before_graph_capture
from areno.engine.runtime.metadata import InferMeta
from areno.engine.runtime.rollout import _empty_rollout


logger = logging.getLogger(__name__)

FinishedRowsCallback = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, tuple[int, ...]], None]
PartialTailReadyCallback = Callable[[int], bool]


def _cancel_stop_token(stop_token_ids: list[int], eos_token_id: int | tuple[int, ...] | None) -> int:
    """Choose a token id to write when a row is cancelled mid-decode."""

    if stop_token_ids:
        return int(stop_token_ids[0])
    if isinstance(eos_token_id, tuple) and eos_token_id:
        return int(eos_token_id[0])
    if eos_token_id is not None:
        return int(eos_token_id)
    return 0


@dataclass(slots=True)
class InferCacheSpec:
    """Runtime cache sizing derived from a rollout payload."""

    max_running_seqs: int
    max_cache_len: int
    num_blocks: int
    block_size: int
    max_blocks_per_seq: int


@dataclass(slots=True)
class PrefillPayload:
    """Typed wrapper around `InferenceBatchState.build_prefill_payload()`."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    sample_indices: torch.Tensor
    block_table: torch.Tensor
    sampling_params: SamplingParams
    sample_step: int
    eos_token_id: int | tuple[int, ...] | None
    sample_generator: torch.Generator | None
    return_logprobs: bool
    infer_meta: object | None
    raw: dict

    @classmethod
    def from_state_payload(
        cls,
        raw: dict,
        *,
        sampling_params: SamplingParams,
        sample_step: int,
        eos_token_id: int | tuple[int, ...] | None,
        sample_generator: torch.Generator | None,
        return_logprobs: bool,
    ) -> "PrefillPayload":
        """Attach sampling fields to the runtime prefill tensor bundle."""

        return cls(
            input_ids=raw["input_ids"],
            position_ids=raw["position_ids"],
            sample_indices=raw["sample_indices"],
            block_table=raw["block_table"],
            sampling_params=sampling_params,
            sample_step=sample_step,
            eos_token_id=eos_token_id,
            sample_generator=sample_generator,
            return_logprobs=return_logprobs,
            infer_meta=raw.get("infer_meta"),
            raw=raw,
        )


class InferenceManager:
    """Own rollout generation and decode graph capture."""

    def __init__(self, worker):
        object.__setattr__(self, "worker", worker)
        if not hasattr(worker, "_decode_progress_lock"):
            worker._decode_progress_lock = threading.Lock()
            worker._decode_progress_next_time = 0.0
            worker._decode_progress_window_start = time.perf_counter()
            worker._decode_progress_window_tokens = 0
            worker._decode_progress_active: dict[int, int] = {}

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def __setattr__(self, name, value):
        if name == "worker":
            object.__setattr__(self, name, value)
        else:
            setattr(self.worker, name, value)

    def _init_infer_cache(self, spec: InferCacheSpec) -> None:
        """Prepare rollout-only state without rebuilding stable CUDA graph buffers.

        The cache allocation is tied to the engine lifetime. Later rollouts reset
        KV contents and refresh inference weights, but reuse the same cache and
        graph objects so capture cost is not paid every RL step.
        """
        max_running_seqs = int(spec.max_running_seqs)
        num_blocks = int(spec.num_blocks)
        block_size = int(spec.block_size)
        max_cache_len = int(spec.max_cache_len)
        max_blocks_per_seq = int(spec.max_blocks_per_seq)
        self._prepare_actor_onloaded()
        if self._infer_cache_spec is not None:
            # Reuse path: the existing cache is large enough along every
            # dimension. We must match block_size exactly (it's baked into
            # the kernel layout), and every other quantity may shrink.
            if (
                block_size == self._infer_cache_spec[2]
                and max_running_seqs <= self._infer_batch_size
                and num_blocks <= self._infer_cache_blocks - 1
                and max_cache_len <= self._max_cache_len
                and max_blocks_per_seq <= self._max_blocks_per_seq
            ):
                onload_kv = getattr(self.model, "onload_kv_caches", None)
                if onload_kv is not None:
                    onload_kv(self.device)
                self.model.reset_kv_caches()
                self.model.onload_train_weights(self.device)
                self.model.prepare_infer_weights()
                self._train_state_ready = False
                self.model.offload_train_weights()
                if self.device.type == "cuda":
                    self._init_decode_graphs()
                return
            # Reallocation: prior CUDA graphs were captured against the old
            # cache pointers and are no longer valid.
            self._decode_graphs.clear()
            self._decode_graph_skipped_buckets.clear()
            self._decode_graph_init_attempted = False
        self._infer_batch_size = max_running_seqs
        # Allocate one extra block past `num_blocks` to use as a fixed scratch
        # block for padded rows during graph-shape decode (see _init_decode_graphs).
        self._infer_cache_blocks = num_blocks + 1
        self._scratch_block = num_blocks
        self._max_cache_len = max_cache_len
        self._max_blocks_per_seq = max_blocks_per_seq
        self._decode_graphs.clear()
        self._decode_graph_skipped_buckets.clear()
        self._decode_graph_init_attempted = False
        self._infer_cache_spec = (
            max_running_seqs,
            num_blocks,
            block_size,
            self._max_cache_len,
            self._max_blocks_per_seq,
        )
        caches = self.model.allocate_kv_caches(self._infer_cache_blocks, block_size, self.device)
        self.model.set_kv_caches(caches)
        self._train_state_ready = False
        # Materialise infer weights from train weights (e.g. dequantize / fuse),
        # then drop the train copies for the rollout's duration.
        self.model.onload_train_weights(self.device)
        self.model.prepare_infer_weights()
        self.model.offload_train_weights()
        if self.device.type == "cuda":
            self._init_decode_graphs()

    @torch.inference_mode()
    def infer_rollout(
        self,
        payload: RolloutPayload,
        finished_callback: FinishedRowsCallback | None = None,
        partial_tail_ready_callback: PartialTailReadyCallback | None = None,
    ) -> RolloutOutput | None:
        """Top-level rollout entry: prepare cache, generate, return on rank 0.

        Empty-input shards (e.g. idle DP rank) return an empty RolloutOutput
        on rank 0 / `None` elsewhere without touching the model.
        """
        ctx = get_tp_context()
        was_training = self.model.training
        self.model.eval()
        try:
            prompts = payload.prompts_by_dp[ctx.dp_rank]
            prompt_indices = payload.prompt_indices_by_dp[ctx.dp_rank]
            # Idle-DP early return: this rank received no prompts this step.
            if not prompts:
                return _empty_rollout() if ctx.is_rank0 else None
            max_new_tokens = int(payload.max_new_tokens)
            eos_token_id = payload.eos_token_id
            max_cache_len = int(payload.max_cache_len)
            state = InferenceBatchState(
                prompts,
                max_new_tokens,
                max_running_seqs=int(payload.max_running_seqs),
                max_cache_len=max_cache_len,
                max_prefill_tokens=int(payload.max_prefill_tokens),
                kv_block_size=int(payload.block_size),
                num_cache_blocks=int(payload.num_blocks),
            )
            self._init_infer_cache(
                InferCacheSpec(
                    max_running_seqs=state.batch_size,
                    max_cache_len=state.max_cache_len,
                    num_blocks=state.num_cache_blocks,
                    block_size=state.kv_block_size,
                    max_blocks_per_seq=state.max_blocks_per_seq,
                )
            )
            sampling_params = payload.sampling_params
            cancel_indices_by_dp = payload.cancel_indices_by_dp
            cooldown_until_by_dp = payload.partial_tail_cooldown_until_s_by_dp
            self._generate_rollout_tokens_no_sync(
                state,
                sampling_params,
                eos_token_id,
                decode_progress_interval_s=float(payload.decode_progress_interval_s),
                cancel_flags=payload.cancel_flags,
                cancel_indices=cancel_indices_by_dp[ctx.dp_rank] if cancel_indices_by_dp is not None else None,
                prompt_indices=prompt_indices,
                finished_callback=finished_callback,
                partial_tail_threshold=int(payload.partial_tail_threshold),
                partial_tail_min_tokens=int(payload.partial_tail_min_tokens),
                partial_tail_ready_callback=partial_tail_ready_callback,
                partial_tail_cooldown_until_s=cooldown_until_by_dp[ctx.dp_rank] if cooldown_until_by_dp is not None else None,
            )
            if ctx.is_rank0:
                return state.to_rollout()
            return None
        finally:
            if was_training:
                self.model.train()
            if not self.config.runtime.keep_rollout_state:
                self._drop_rollout_hbm()

    @torch.inference_mode()
    def _generate_rollout_tokens_no_sync(
        self,
        state: InferenceBatchState,
        sampling_params: SamplingParams,
        eos_token_id: int | tuple[int, ...] | None,
        *,
        decode_progress_interval_s: float = 0.0,
        cancel_flags: torch.Tensor | None = None,
        cancel_indices: list[int] | None = None,
        prompt_indices: list[int] | None = None,
        finished_callback: FinishedRowsCallback | None = None,
        partial_tail_threshold: int = 0,
        partial_tail_min_tokens: int = 0,
        partial_tail_ready_callback: PartialTailReadyCallback | None = None,
        partial_tail_cooldown_until_s: list[float] | None = None,
    ) -> None:
        """Prefill all prompts then decode up to `max_new_tokens` without DP-sync.

        Drives the rollout loop in-place on `state`:
          * one prefill kernel for the initial batch produces the first token;
          * each decode step samples one token per active row, evicts finished
            or cancelled rows from the active set, and admits pending rows
            from the same rollout chunk.

        `cancel_flags` is a shared-memory bool tensor written by the engine
        driver; we re-read it every step to support remote cancellation.
        """
        ctx = get_tp_context()
        prompt_count = len(state.prompts)
        progress_enabled = decode_progress_interval_s > 0 and ctx.is_rank0
        progress_key = id(state)
        # -------- prefill --------
        prefill_payload = state.build_prefill_payload()
        if prefill_payload is None:
            raise RuntimeError("no-sync rollout could not prefill all prompts; reduce batch size or increase token budget")
        sample_generator = _make_sample_generator(sampling_params, self.device)
        prefill = PrefillPayload.from_state_payload(
            prefill_payload,
            sampling_params=sampling_params,
            sample_step=0,
            eos_token_id=eos_token_id,
            sample_generator=sample_generator,
            return_logprobs=True,
        )
        next_tokens, next_logprobs = self._infer_next_token_tensor(prefill)
        stop_token_ids = _stop_token_ids(sampling_params, eos_token_id)
        # Convert per-DP cancel-index list into a tensor on CPU so the engine
        # can mutate the underlying shared memory between decode steps.
        cancel_indices_tensor = torch.tensor(cancel_indices, dtype=torch.long) if cancel_flags is not None and cancel_indices is not None else None
        cancel_token = _cancel_stop_token(stop_token_ids, eos_token_id)
        # When stop tokens exist, cancellation injects one of them; otherwise
        # we still need *something* recognisable downstream as "stop".
        truncate_stop_token_ids = stop_token_ids if stop_token_ids else ([cancel_token] if cancel_flags is not None else [])
        prompt_indices_list = list(prompt_indices) if prompt_indices is not None else list(range(prompt_count))
        partial_rows = torch.zeros(prompt_count, device=self.device, dtype=torch.bool)
        cooldown_until = torch.tensor(
            partial_tail_cooldown_until_s if partial_tail_cooldown_until_s is not None else [0.0 for _ in range(prompt_count)],
            device=self.device,
            dtype=torch.float64,
        )

        # generated/logprobs shape: (prompt_count, max_new_tokens), with only
        # the prefix [0:response_lens[i]] valid for row i.
        generated = torch.empty(prompt_count, state.max_new_tokens, device=self.device, dtype=torch.long)
        logprobs = torch.empty(prompt_count, state.max_new_tokens, device=self.device, dtype=torch.float32)
        response_lens = torch.zeros(prompt_count, device=self.device, dtype=torch.long)
        initial_rows = torch.tensor(state._last_active_ids, device=self.device, dtype=torch.long)
        generated[initial_rows, 0] = next_tokens
        logprobs[initial_rows, 0] = next_logprobs
        response_lens[initial_rows] = 1
        # block_table shape: (active_count, max_blocks_per_seq) of int32.
        block_table = prefill.block_table.to(self.device, non_blocking=True).int()
        cache_seqlens = torch.tensor([len(state.prompts[int(row)]) for row in initial_rows.tolist()], device=self.device, dtype=torch.int32)
        position_ids = cache_seqlens.to(torch.long)
        # active_rows[k] = the row index in `generated` of the k-th active seq.
        active_rows = initial_rows
        active_count = int(initial_rows.numel())
        remove = torch.zeros(active_count, device=self.device, dtype=torch.bool)
        should_filter = False
        # Apply stop-token / cancel filters to the prefill output before
        # entering the decode loop, in case a prompt was already complete.
        if stop_token_ids:
            finished = _tokens_match_any(next_tokens, stop_token_ids)
            self._mark_rollout_finished_rows(
                active_rows[finished],
                generated,
                logprobs,
                response_lens,
                "stop",
                prompt_indices_list,
                finished_callback,
                tuple(truncate_stop_token_ids),
            )
            remove |= finished
            should_filter = True
        full_length = response_lens[active_rows] >= state.max_new_tokens
        if bool(full_length.any().item()):
            self._mark_rollout_finished_rows(
                active_rows[full_length],
                generated,
                logprobs,
                response_lens,
                "length",
                prompt_indices_list,
                finished_callback,
                tuple(truncate_stop_token_ids),
            )
            remove |= full_length
            should_filter = True
        cancelled = self._cancel_mask_for_active_rows(active_rows, cancel_flags, cancel_indices_tensor)
        if cancelled is not None:
            generated[active_rows[cancelled], 0] = cancel_token
            logprobs[active_rows[cancelled], 0] = 0.0
            response_lens[active_rows[cancelled]] = 1
            remove |= cancelled
            should_filter = True
        if should_filter:
            self._free_rollout_rows(state, active_rows[remove])
            keep = ~remove
            active_rows = active_rows[keep]
            next_tokens = next_tokens[keep]
            cache_seqlens = cache_seqlens[keep]
            position_ids = position_ids[keep]
            block_table = block_table[keep]
            active_count = int(active_rows.numel())
        # -------- decode loop --------
        self._record_decode_progress(
            enabled=progress_enabled,
            interval_s=decode_progress_interval_s,
            rollout_key=progress_key,
            active_count=active_count,
            token_delta=0,
            cache_tokens=int(cache_seqlens.max().item()) if active_count else 0,
            sample_step=1,
            max_new_tokens=state.max_new_tokens,
        )
        decoded_tokens = 0
        sample_step = 1
        while True:
            if active_count == 0:
                admitted = self._admit_pending_rollout_rows(
                    state,
                    generated,
                    logprobs,
                    response_lens,
                    next_tokens,
                    cache_seqlens,
                    position_ids,
                    block_table,
                    active_rows,
                    prompt_indices_list,
                    sampling_params,
                    sample_generator,
                    eos_token_id,
                    sample_step,
                    stop_token_ids,
                    finished_callback,
                    tuple(truncate_stop_token_ids),
                )
                if admitted is None:
                    break
                generated, logprobs, response_lens, next_tokens, cache_seqlens, position_ids, block_table, active_rows, active_count = admitted
            next_tokens, next_logprobs = self._infer_decode_next_token_tensor(
                next_tokens,
                position_ids,
                cache_seqlens,
                block_table,
                active_count,
                sampling_params,
                sample_generator,
                sample_step=sample_step,
                eos_token_id=eos_token_id,
            )
            sample_step += 1
            # Write the new tokens into the per-row response buffer using
            # advanced indexing: write_pos[k] is the next free slot for row k.
            write_pos = response_lens[active_rows]
            generated[active_rows, write_pos] = next_tokens
            logprobs[active_rows, write_pos] = next_logprobs
            response_lens[active_rows] = write_pos + 1
            decoded_tokens += active_count
            self._record_decode_progress(
                enabled=progress_enabled,
                interval_s=decode_progress_interval_s,
                rollout_key=progress_key,
                active_count=active_count,
                token_delta=active_count,
                cache_tokens=int(cache_seqlens.max().item()) if active_count else 0,
                sample_step=sample_step,
                max_new_tokens=state.max_new_tokens,
            )
            cache_seqlens.add_(1)
            position_ids.add_(1)
            remove = torch.zeros(active_count, device=self.device, dtype=torch.bool)
            should_filter = False
            # EOS / stop-token filter.
            if stop_token_ids:
                finished = _tokens_match_any(next_tokens, stop_token_ids)
                self._mark_rollout_finished_rows(
                    active_rows[finished],
                    generated,
                    logprobs,
                    response_lens,
                    "stop",
                    prompt_indices_list,
                    finished_callback,
                    tuple(truncate_stop_token_ids),
                )
                remove |= finished
                should_filter = True
            full_length = response_lens[active_rows] >= state.max_new_tokens
            if bool(full_length.any().item()):
                self._mark_rollout_finished_rows(
                    active_rows[full_length],
                    generated,
                    logprobs,
                    response_lens,
                    "length",
                    prompt_indices_list,
                    finished_callback,
                    tuple(truncate_stop_token_ids),
                )
                remove |= full_length
                should_filter = True
            # Cancellation filter (overrides the just-written token with the
            # cancel sentinel so downstream sees a clean stop).
            cancelled = self._cancel_mask_for_active_rows(active_rows, cancel_flags, cancel_indices_tensor)
            if cancelled is not None:
                cancel_rows = active_rows[cancelled]
                cancel_pos = response_lens[cancel_rows].clamp_max(state.max_new_tokens - 1)
                generated[cancel_rows, cancel_pos] = cancel_token
                logprobs[cancel_rows, cancel_pos] = 0.0
                response_lens[cancel_rows] = cancel_pos + 1
                remove |= cancelled
                should_filter = True
            if should_filter:
                self._free_rollout_rows(state, active_rows[remove])
                keep = ~remove
                active_rows = active_rows[keep]
                next_tokens = next_tokens[keep]
                cache_seqlens = cache_seqlens[keep]
                position_ids = position_ids[keep]
                block_table = block_table[keep]
                active_count = int(active_rows.numel())
            if self._should_partial_tail(
                state,
                active_rows,
                response_lens,
                active_count,
                partial_tail_threshold,
                partial_tail_min_tokens,
                partial_tail_ready_callback,
                cooldown_until,
            ):
                if progress_enabled:
                    min_len = int(response_lens[active_rows].min().item()) if active_count else 0
                    logger.info(
                        "rollout partial tail: dp=%d/%d active=%d threshold=%d min_response_len=%d max_new_tokens=%d",
                        ctx.dp_rank,
                        ctx.dp_size,
                        active_count,
                        partial_tail_threshold,
                        min_len,
                        state.max_new_tokens,
                    )
                partial_rows[active_rows] = True
                self._free_rollout_rows(state, active_rows)
                break
        # Any rows still active at this point hit the length cap.
        if active_count > 0 and not bool(partial_rows[active_rows].all().item()):
            self._mark_rollout_finished_rows(
                active_rows[~partial_rows[active_rows]],
                generated,
                logprobs,
                response_lens,
                "length",
                prompt_indices_list,
                finished_callback,
                tuple(truncate_stop_token_ids),
            )
        state.metrics["decode_scheduled_tokens"] = float(decoded_tokens)
        self._record_decode_progress(
            enabled=progress_enabled,
            interval_s=decode_progress_interval_s,
            rollout_key=progress_key,
            active_count=0,
            token_delta=0,
            cache_tokens=0,
            sample_step=sample_step,
            max_new_tokens=state.max_new_tokens,
        )
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError as exc:
                raise RuntimeError("CUDA failure detected at rollout decode completion") from exc

        # Move generated tokens to CPU on rank 0 then broadcast to the rest of
        # the TP group so every rank sees the same final state.
        generated_obj = None
        logprobs_obj = None
        finish_reason_obj = None
        if ctx.is_rank0:
            response_lengths = response_lens.detach().cpu().tolist()
            generated_rows = [row[: int(length)] for row, length in zip(generated.cpu().tolist(), response_lengths, strict=True)]
            generated_obj, finish_reason_obj = _truncate_generated(generated_rows, truncate_stop_token_ids)
            partial_flags = partial_rows.detach().cpu().tolist()
            finish_reason_obj = ["partial" if partial else reason for partial, reason in zip(partial_flags, finish_reason_obj, strict=True)]
            logprobs_rows = logprobs.cpu().tolist()
            logprobs_obj = [row[: len(generated_row)] for row, generated_row in zip(logprobs_rows, generated_obj, strict=True)]
        # broadcast_object src=0 of the TP group: rank 0 holds the canonical
        # rollout output; other TP ranks adopt the same lists so state is
        # consistent at the engine boundary.
        generated_obj = broadcast_object(generated_obj, src=0)
        logprobs_obj = broadcast_object(logprobs_obj, src=0)
        finish_reason_obj = broadcast_object(finish_reason_obj, src=0)
        state.generated = generated_obj
        state.logprobs = logprobs_obj
        state.finished = [True for _ in state.generated]
        state.finish_reason = finish_reason_obj

    def _should_partial_tail(
        self,
        state: InferenceBatchState,
        active_rows: torch.Tensor,
        response_lens: torch.Tensor,
        active_count: int,
        partial_tail_threshold: int,
        partial_tail_min_tokens: int,
        partial_tail_ready_callback: PartialTailReadyCallback | None,
        partial_tail_cooldown_until: torch.Tensor,
    ) -> bool:
        """Return whether a small active tail should be resumed by a later batch."""

        if partial_tail_threshold <= 0 or active_count <= 0:
            return False
        if active_count >= state.max_running_seqs:
            return False
        if state._pending_seq_id < len(state.prompts):
            return False
        if partial_tail_min_tokens > 0 and int(response_lens[active_rows].min().item()) < partial_tail_min_tokens:
            return False
        now = time.monotonic()
        if bool((partial_tail_cooldown_until[active_rows] > now).any().item()):
            return False
        if partial_tail_ready_callback is None or not partial_tail_ready_callback(active_count):
            return False
        return active_count <= int(partial_tail_threshold)

    def _record_decode_progress(
        self,
        *,
        enabled: bool,
        interval_s: float,
        rollout_key: int,
        active_count: int,
        token_delta: int,
        cache_tokens: int,
        sample_step: int,
        max_new_tokens: int,
    ) -> None:
        """Emit one throttled decode-progress line per worker, not per rollout."""

        if not enabled:
            return
        ctx = get_tp_context()
        now = time.perf_counter()
        with self._decode_progress_lock:
            if active_count > 0:
                self._decode_progress_active[rollout_key] = active_count
            else:
                self._decode_progress_active.pop(rollout_key, None)
            self._decode_progress_window_tokens += int(token_delta)
            if self._decode_progress_next_time <= 0.0:
                self._decode_progress_window_start = now
                self._decode_progress_next_time = now + interval_s
                return
            if now < self._decode_progress_next_time:
                return
            window_elapsed = max(now - self._decode_progress_window_start, 1e-9)
            window_tokens = int(self._decode_progress_window_tokens)
            concurrent_batches = len(self._decode_progress_active)
            total_active = sum(self._decode_progress_active.values())
            self._decode_progress_window_start = now
            self._decode_progress_next_time = now + interval_s
            self._decode_progress_window_tokens = 0
        logger.info(
            "rollout decode progress: dp=%d/%d concurrent_batches=%d active=%d tokens_per_second=%.1f window_tokens=%d step=%d/%d cache_tokens=%d",
            ctx.dp_rank,
            ctx.dp_size,
            concurrent_batches,
            total_active,
            window_tokens / window_elapsed,
            window_tokens,
            sample_step,
            max_new_tokens,
            cache_tokens,
        )

    def _cancel_mask_for_active_rows(
        self,
        active_rows: torch.Tensor,
        cancel_flags: torch.Tensor | None,
        cancel_indices: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return a bool mask marking which active rows have been cancelled.

        `cancel_flags` is the engine-level shared-memory bool tensor indexed by
        global prompt id; `cancel_indices` maps this rank's local rows into
        that global table. Returns None if there is nothing to cancel.
        """
        if cancel_flags is None or cancel_indices is None or active_rows.numel() == 0:
            return None
        if not bool(cancel_flags.any()):
            return None
        local_flags = cancel_flags.index_select(0, cancel_indices).to(self.device, non_blocking=True)
        return local_flags[active_rows] != 0

    def _free_rollout_rows(self, state: InferenceBatchState, rows: torch.Tensor) -> None:
        """Return the KV blocks owned by `rows` to the free pool."""
        if rows.numel() == 0:
            return
        for row in rows.detach().cpu().tolist():
            blocks = state._seq_to_blocks.pop(int(row), None)
            if blocks:
                state._free_blocks.extend(blocks)

    def _admit_pending_rollout_rows(
        self,
        state: InferenceBatchState,
        generated: torch.Tensor,
        logprobs: torch.Tensor,
        response_lens: torch.Tensor,
        next_tokens: torch.Tensor,
        cache_seqlens: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        active_rows: torch.Tensor,
        prompt_indices: list[int],
        sampling_params: SamplingParams,
        sample_generator: torch.Generator | None,
        eos_token_id: int | tuple[int, ...] | None,
        step: int,
        stop_token_ids: tuple[int, ...],
        finished_callback: FinishedRowsCallback | None,
        truncate_stop_token_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        """Admit pending rows from the current rollout state only."""

        prefill_payload = state.build_prefill_payload()
        if prefill_payload is None:
            return None
        prefill = PrefillPayload.from_state_payload(
            prefill_payload,
            sampling_params=sampling_params,
            sample_step=step,
            eos_token_id=eos_token_id,
            sample_generator=sample_generator,
            return_logprobs=True,
        )
        new_tokens, new_logprobs = self._infer_next_token_tensor(prefill)
        new_rows = torch.tensor(state._last_active_ids, device=self.device, dtype=torch.long)
        generated[new_rows, 0] = new_tokens
        logprobs[new_rows, 0] = new_logprobs
        response_lens[new_rows] = 1
        new_cache_seqlens = torch.tensor([len(state.prompts[int(row)]) for row in new_rows.tolist()], device=self.device, dtype=torch.int32)
        new_position_ids = new_cache_seqlens.to(torch.long)
        new_block_table = prefill.block_table.to(self.device, non_blocking=True).int()
        if stop_token_ids:
            finished = _tokens_match_any(new_tokens, stop_token_ids)
            self._mark_rollout_finished_rows(
                new_rows[finished],
                generated,
                logprobs,
                response_lens,
                "stop",
                prompt_indices,
                finished_callback,
                truncate_stop_token_ids,
            )
            if bool(finished.any().item()):
                self._free_rollout_rows(state, new_rows[finished])
                keep = ~finished
                new_rows = new_rows[keep]
                new_tokens = new_tokens[keep]
                new_cache_seqlens = new_cache_seqlens[keep]
                new_position_ids = new_position_ids[keep]
                new_block_table = new_block_table[keep]
        full_length = response_lens[new_rows] >= state.max_new_tokens
        if bool(full_length.any().item()):
            self._mark_rollout_finished_rows(
                new_rows[full_length],
                generated,
                logprobs,
                response_lens,
                "length",
                prompt_indices,
                finished_callback,
                truncate_stop_token_ids,
            )
            self._free_rollout_rows(state, new_rows[full_length])
            keep = ~full_length
            new_rows = new_rows[keep]
            new_tokens = new_tokens[keep]
            new_cache_seqlens = new_cache_seqlens[keep]
            new_position_ids = new_position_ids[keep]
            new_block_table = new_block_table[keep]
        if new_rows.numel() == 0:
            return None
        return generated, logprobs, response_lens, new_tokens, new_cache_seqlens, new_position_ids, new_block_table, new_rows, int(new_rows.numel())

    def _mark_rollout_finished_rows(
        self,
        rows: torch.Tensor,
        generated: torch.Tensor,
        logprobs: torch.Tensor,
        response_lens: torch.Tensor,
        finish_reason: str,
        prompt_indices: list[int] | None = None,
        finished_callback: FinishedRowsCallback | None = None,
        truncate_stop_token_ids: tuple[int, ...] = (),
    ) -> None:
        """Finish hook; final rollout output carries completed rows."""

        del prompt_indices
        if rows.numel() == 0 or finished_callback is None or finish_reason == "partial":
            return
        finished_callback(rows, generated, logprobs, response_lens, finish_reason, truncate_stop_token_ids)

    @torch.inference_mode()
    def _infer_next_token_tensor(self, payload: PrefillPayload) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run a single prefill forward and sample one token per sequence.

        Returns either `next_tokens` or `(next_tokens, token_logprobs)`
        depending on `return_logprobs`. The sampled tokens are broadcast from
        TP rank 0 so every shard agrees on the chosen ids.
        """
        sample_indices = _device_long(payload.sample_indices, self.device)
        num_tokens = int(payload.input_ids.numel())
        # Add a leading batch dim of 1 — prefill packs all prompts into one
        # contiguous (sum_seq_lens,) tensor that the model expects as (1, T).
        input_ids = _device_long(payload.input_ids, self.device).unsqueeze(0)
        position_ids = _device_long(payload.position_ids, self.device).unsqueeze(0)
        infer_meta = payload.infer_meta
        if infer_meta is None:
            infer_meta = payload_to_infer_meta(payload.raw, self.device)
        out = self.model(input_ids=input_ids, position_ids=position_ids, infer_meta=infer_meta)
        logits_shard = out.logits_shard
        sampling_params = payload.sampling_params
        if sampling_params.temperature == 0.0:
            # Greedy across TP-sharded vocab: each rank argmaxes its shard,
            # then a cross-rank reduction picks the global argmax.
            next_tokens = _sample_greedy_sharded(
                logits_shard[0, sample_indices],
                self.config.model.vocab_size,
                self.config.tp_size,
                eos_token_id=payload.eos_token_id,
                sample_step=int(payload.sample_step),
                min_new_tokens=sampling_params.min_new_tokens,
                suppress_token_ids=sampling_params.suppress_token_ids,
            )
        else:
            # Temperature/top-k/top-p path: gathers the full vocab to rank 0
            # for sampling, since the noise injection isn't shardable.
            next_tokens = _sample_full_vocab(
                logits_shard[0, sample_indices],
                sampling_params,
                self.config.model.vocab_size,
                self.config.tp_size,
                self.device,
                generator=payload.sample_generator,
                eos_token_id=payload.eos_token_id,
                sample_step=int(payload.sample_step),
            )
        # broadcast_tensor src=0: keep the sampled ids identical across TP.
        next_tokens = broadcast_tensor(next_tokens.contiguous(), src=0)
        if payload.return_logprobs:
            _check_token_ids(next_tokens, self.config.model.vocab_size, "sampled next_tokens")
            token_logprobs = _policy_token_logprobs(
                logits_shard[0, sample_indices],
                next_tokens,
            )
            return next_tokens, token_logprobs
        _check_token_ids(next_tokens, self.config.model.vocab_size, "sampled next_tokens")
        return next_tokens

    @torch.inference_mode()
    def _infer_decode_next_token_tensor(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        active_count: int,
        sampling_params: SamplingParams,
        sample_generator: torch.Generator | None,
        *,
        sample_step: int,
        eos_token_id: int | tuple[int, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one decode step (1 token per active sequence) and sample.

        Dispatches to a captured CUDA graph for the matching bucket if one
        exists, otherwise falls back to an eager forward. Returns the sampled
        tokens and their logprobs, both length `active_count`.
        """
        graph = self._decode_graph_for_active_count(active_count)
        if graph is None:
            # Eager fallback for buckets that failed to capture (OOM) or for
            # active counts above the largest captured bucket.
            infer_meta = InferMeta(
                mode="decode",
                sample_indices=torch.arange(active_count, device=self.device, dtype=torch.long),
                cache_seqlens=cache_seqlens,
                block_table=block_table,
            )
            logits_shard = self.model(
                input_ids=input_ids[:active_count].view(1, active_count),
                position_ids=position_ids[:active_count].view(1, active_count),
                infer_meta=infer_meta,
            ).logits_shard[0, :active_count]
        else:
            # Graph replay path: copies inputs into the captured input buffers
            # and replays. Only the first `active_count` rows are meaningful;
            # the rest are padding pointed at the scratch block.
            logits_shard = graph.replay_tensors(input_ids, position_ids, cache_seqlens, block_table)[0, :active_count]

        if sampling_params.temperature == 0.0:
            next_tokens = _sample_greedy_sharded(
                logits_shard,
                self.config.model.vocab_size,
                self.config.tp_size,
                eos_token_id=eos_token_id,
                sample_step=sample_step,
                min_new_tokens=sampling_params.min_new_tokens,
                suppress_token_ids=sampling_params.suppress_token_ids,
            )
        else:
            next_tokens = _sample_full_vocab(
                logits_shard,
                sampling_params,
                self.config.model.vocab_size,
                self.config.tp_size,
                self.device,
                generator=sample_generator,
                eos_token_id=eos_token_id,
                sample_step=sample_step,
            )
        next_tokens = broadcast_tensor(next_tokens.contiguous(), src=0)
        _check_token_ids(next_tokens, self.config.model.vocab_size, "sampled next_tokens")
        token_logprobs = _policy_token_logprobs(
            logits_shard,
            next_tokens,
        )
        return next_tokens, token_logprobs

    def _decode_graph_for_active_count(self, active_count: int) -> DecodeGraph | None:
        """Resolve the smallest captured decode graph that fits `active_count`.

        Prefers an exact bucket match; otherwise falls through to the next
        larger captured bucket (padded rows use the scratch block). Returns
        None if no captured graph can cover this active count.
        """
        if self.config.runtime.eager_decode:
            return None
        bucket = bucket_for(active_count, self.config.runtime.decode_graph_buckets)
        graph = self._decode_graphs.get(bucket)
        if graph is not None:
            return graph
        for captured_bucket in sorted(self._decode_graphs):
            if captured_bucket >= active_count:
                return self._decode_graphs[captured_bucket]
        return None

    @torch.inference_mode()
    def _init_decode_graphs(self) -> None:
        """Capture decode graphs for the configured batch buckets once per worker.

        Each rank first measures the warmup peak and all ranks agree that enough
        memory exists before any rank captures. This avoids half-captured states
        when one rank is tighter on memory.

        CUDA graph invariants:
          * input_ids/position_ids/cache_seqlens/block_table buffers are stable
            allocations bound at capture time; replay copies new contents in;
          * the scratch block (last index of the KV cache) handles padded rows
            so the captured block_table shape stays fixed at `bucket` rows
            even when only `active_count < bucket` are live;
          * KV-cache pointers are baked into the graph, which is why any
            cache reallocation invalidates every captured graph.
        """
        if self.config.runtime.eager_decode:
            return
        if self._decode_graph_init_attempted:
            return
        self._decode_graph_init_attempted = True
        if self.model.training:
            self.model.eval()
        ctx = get_tp_context()
        # User-configured buckets clamped to [1, max_running_seqs], plus the
        # max so the largest active batch always has a graph.
        buckets = sorted({bucket for bucket in self.config.runtime.decode_graph_buckets if 1 <= bucket <= self._infer_batch_size})
        buckets.append(self._infer_batch_size)
        for bucket in sorted(set(buckets)):
            if bucket in self._decode_graphs or bucket in self._decode_graph_skipped_buckets:
                continue
            graph = DecodeGraph(
                self.model,
                bucket,
                self._max_blocks_per_seq,
                self._scratch_block,
                self.device,
            )
            # Warmup: run a few eager forwards at this bucket size to (a) trim
            # compiler / allocator noise and (b) measure the working-set peak
            # we need free at capture time.
            warmup_bytes = graph.warmup()
            sync_before_graph_capture(self.device, ctx.group)
            # All ranks vote on whether HBM headroom exists; any rank tight on
            # memory aborts the whole bucket so no rank is left half-captured.
            if not has_graph_capture_memory(self.device, ctx.group, warmup_bytes):
                if ctx.is_rank0:
                    free_bytes, _ = torch.cuda.mem_get_info(self.device)
                    logger.info(
                        "skipping decode CUDA graph capture: bucket=%d free_gib=%.2f warmup_peak_gib=%.2f",
                        bucket,
                        free_bytes / (1024**3),
                        warmup_bytes / (1024**3),
                    )
                sync_before_graph_capture(self.device, ctx.group)
                self._decode_graph_skipped_buckets.add(bucket)
                continue
            try:
                graph.capture()
            except torch.OutOfMemoryError:
                # Capture itself can still OOM (extra workspace allocations);
                # in that case fall back to eager for this bucket and move on.
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if ctx.is_rank0:
                    free_bytes, _ = torch.cuda.mem_get_info(self.device)
                    logger.warning(
                        "skipping decode CUDA graph capture after OOM: bucket=%d free_gib=%.2f fallback=eager",
                        bucket,
                        free_bytes / (1024**3),
                    )
                sync_before_graph_capture(self.device, ctx.group)
                self._decode_graph_skipped_buckets.add(bucket)
                continue
            self._decode_graphs[bucket] = graph
