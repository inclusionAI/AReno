"""Rollout, continuous batching, and decode graph management."""

from __future__ import annotations

import logging
import math
import queue
import time
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
from areno.engine.protocol import Op, RolloutAddPayload, RolloutPartialPayload, RolloutPayload, WorkerResult
from areno.engine.parallel.collectives import broadcast_object, broadcast_tensor
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.common import _check_token_ids, _device_long
from areno.engine.runtime.decode_graph import DecodeGraph, bucket_for, has_graph_capture_memory, sync_before_graph_capture
from areno.engine.runtime.metadata import InferMeta
from areno.engine.runtime.rollout import _empty_rollout


logger = logging.getLogger(__name__)


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
    """Own rollout generation, continuous batching, and decode graph capture."""

    def __init__(self, worker):
        object.__setattr__(self, "worker", worker)

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
    def infer_rollout(self, payload: RolloutPayload) -> RolloutOutput | None:
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
            self._generate_rollout_tokens_no_sync(
                state,
                sampling_params,
                eos_token_id,
                decode_progress_interval_s=float(payload.decode_progress_interval_s),
                cancel_flags=payload.cancel_flags,
                cancel_indices=cancel_indices_by_dp[ctx.dp_rank] if cancel_indices_by_dp is not None else None,
                session_id=payload.session_id,
                prompt_indices=prompt_indices,
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
        session_id: int | None = None,
        prompt_indices: list[int] | None = None,
    ) -> None:
        """Prefill all prompts then decode up to `max_new_tokens` without DP-sync.

        Drives the continuous-batching loop in-place on `state`:
          * one prefill kernel for the initial batch produces the first token;
          * each decode step samples one token per active row, evicts finished
            or cancelled rows from the active set, and admits any newly
            requeued rows from the engine's command queue.

        `cancel_flags` is a shared-memory bool tensor written by the engine
        driver — we re-read it every step to support remote cancellation in
        continuous-serving sessions.
        """
        ctx = get_tp_context()
        prompt_count = len(state.prompts)
        if prompt_count > state.max_running_seqs:
            raise RuntimeError("no-sync rollout requires all prompts to fit in one running batch")
        progress_enabled = decode_progress_interval_s > 0 and ctx.is_rank0
        progress_rank = f"dp={ctx.dp_rank}/{ctx.dp_size}"
        # -------- prefill --------
        prefill_payload = state.build_prefill_payload()
        if prefill_payload is None or len(state._seq_to_blocks) != prompt_count:
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

        # generated/logprobs shape: (prompt_count, max_new_tokens), with only
        # the prefix [0:response_lens[i]] valid for row i.
        generated = torch.empty(prompt_count, state.max_new_tokens, device=self.device, dtype=torch.long)
        logprobs = torch.empty(prompt_count, state.max_new_tokens, device=self.device, dtype=torch.float32)
        response_lens = torch.zeros(prompt_count, device=self.device, dtype=torch.long)
        generated[:, 0] = next_tokens
        logprobs[:, 0] = next_logprobs
        response_lens[:] = 1
        # block_table shape: (active_count, max_blocks_per_seq) of int32.
        block_table = prefill.block_table.to(self.device, non_blocking=True).int()
        cache_seqlens = torch.tensor([len(prompt) for prompt in state.prompts], device=self.device, dtype=torch.int32)
        position_ids = cache_seqlens.to(torch.long)
        # active_rows[k] = the row index in `generated` of the k-th active seq.
        active_rows = torch.arange(prompt_count, device=self.device, dtype=torch.long)
        active_count = prompt_count
        remove = torch.zeros(active_count, device=self.device, dtype=torch.bool)
        should_filter = False
        # Apply stop-token / cancel filters to the prefill output before
        # entering the decode loop, in case a prompt was already complete.
        if stop_token_ids:
            finished = _tokens_match_any(next_tokens, stop_token_ids)
            self._emit_rollout_partials(active_rows[finished], generated, response_lens, "stop", prompt_indices_list)
            remove |= finished
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
        decode_start = time.perf_counter()
        next_progress_time = decode_start + decode_progress_interval_s
        window_start = decode_start
        window_tokens = 0
        decoded_tokens = 0
        idle_deadline: float | None = None
        # Continuous-serving sessions wait up to 30s for new work before
        # giving up; one-shot rollouts (session_id is None) exit immediately.
        continuous_idle_timeout_s = 30.0 if session_id is not None else 0.0
        for step in range(1, state.max_new_tokens):
            if active_count == 0:
                # Idle gate: poll for new ADD commands until something is
                # admitted or the timeout elapses.
                while True:
                    admitted = self._admit_rollout_additions(
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
                        session_id,
                        sampling_params,
                        sample_generator,
                        eos_token_id,
                        step,
                        stop_token_ids,
                        cancel_token,
                    )
                    if admitted is not None:
                        break
                    if continuous_idle_timeout_s <= 0:
                        break
                    now = time.perf_counter()
                    if idle_deadline is None:
                        idle_deadline = now + continuous_idle_timeout_s
                    if now >= idle_deadline:
                        break
                    time.sleep(0.01)
                if admitted is None:
                    break
                generated, logprobs, response_lens, next_tokens, cache_seqlens, position_ids, block_table, active_rows, active_count = admitted
                idle_deadline = None
            next_tokens, next_logprobs = self._infer_decode_next_token_tensor(
                next_tokens,
                position_ids,
                cache_seqlens,
                block_table,
                active_count,
                sampling_params,
                sample_generator,
                sample_step=step,
                eos_token_id=eos_token_id,
            )
            # Write the new tokens into the per-row response buffer using
            # advanced indexing: write_pos[k] is the next free slot for row k.
            write_pos = response_lens[active_rows]
            generated[active_rows, write_pos] = next_tokens
            logprobs[active_rows, write_pos] = next_logprobs
            response_lens[active_rows] = write_pos + 1
            decoded_tokens += active_count
            window_tokens += active_count
            cache_seqlens.add_(1)
            position_ids.add_(1)
            remove = torch.zeros(active_count, device=self.device, dtype=torch.bool)
            should_filter = False
            # EOS / stop-token filter.
            if stop_token_ids:
                finished = _tokens_match_any(next_tokens, stop_token_ids)
                self._emit_rollout_partials(active_rows[finished], generated, response_lens, "stop", prompt_indices_list)
                remove |= finished
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
            # Opportunistically admit newly-queued rows mid-decode so we
            # keep the batch full (continuous batching).
            admitted = self._admit_rollout_additions(
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
                session_id,
                sampling_params,
                sample_generator,
                eos_token_id,
                step,
                stop_token_ids,
                cancel_token,
            )
            if admitted is not None:
                generated, logprobs, response_lens, next_tokens, cache_seqlens, position_ids, block_table, active_rows, active_count = admitted
            if progress_enabled:
                now = time.perf_counter()
                if now >= next_progress_time:
                    window_elapsed = max(now - window_start, 1e-9)
                    logger.info(
                        "rollout decode progress: %s step=%d/%d active=%d tokens_per_second=%.1f cache_tokens=%d",
                        progress_rank,
                        step,
                        state.max_new_tokens - 1,
                        active_count,
                        window_tokens / window_elapsed,
                        int(cache_seqlens.max().item()) if active_count else 0,
                    )
                    next_progress_time = now + decode_progress_interval_s
                    window_start = now
                    window_tokens = 0
        # Any rows still active at this point hit the length cap.
        if active_count > 0:
            self._emit_rollout_partials(active_rows, generated, response_lens, "length", prompt_indices_list)
        state.metrics["decode_scheduled_tokens"] = float(decoded_tokens)
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

    def _admit_rollout_additions(
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
        session_id: int | None,
        sampling_params: SamplingParams,
        sample_generator: torch.Generator | None,
        eos_token_id: int | tuple[int, ...] | None,
        step: int,
        stop_token_ids: tuple[int, ...],
        cancel_token: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        """Try to admit newly-queued prompts into the running decode batch.

        Returns the new (possibly grown) tuple of decode-state tensors, or
        None if no admissible prompts are waiting. The grown tensors must be
        re-bound by the caller to participate in the next decode step.
        """
        self._poll_rollout_add_commands(state, prompt_indices, session_id)
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
        # state._last_active_ids: row indices (within state.prompts) of the
        # prompts that were just prefilled in this call.
        new_rows = torch.tensor(state._last_active_ids, device=self.device, dtype=torch.long)
        needed_rows = len(state.prompts)
        if generated.shape[0] < needed_rows:
            # Grow the per-row buffers to accommodate the new prompts; cat is
            # cheaper than reallocating from scratch and keeps existing data.
            extra_rows = needed_rows - generated.shape[0]
            generated = torch.cat(
                [generated, torch.empty(extra_rows, state.max_new_tokens, device=self.device, dtype=generated.dtype)],
                dim=0,
            )
            logprobs = torch.cat(
                [logprobs, torch.empty(extra_rows, state.max_new_tokens, device=self.device, dtype=logprobs.dtype)],
                dim=0,
            )
            response_lens = torch.cat([response_lens, torch.zeros(extra_rows, device=self.device, dtype=response_lens.dtype)], dim=0)
        generated[new_rows, 0] = new_tokens
        logprobs[new_rows, 0] = new_logprobs
        response_lens[new_rows] = 1
        new_cache_seqlens = torch.tensor([len(state.prompts[int(row)]) for row in new_rows.tolist()], device=self.device, dtype=torch.int32)
        new_position_ids = new_cache_seqlens.to(torch.long)
        new_block_table = prefill.block_table.to(self.device, non_blocking=True).int()
        if stop_token_ids:
            # Prompts whose first sampled token is already a stop token are
            # emitted and freed before they ever enter the decode batch.
            finished = _tokens_match_any(new_tokens, stop_token_ids)
            self._emit_rollout_partials(new_rows[finished], generated, response_lens, "stop", prompt_indices)
            if bool(finished.any().item()):
                self._free_rollout_rows(state, new_rows[finished])
                keep = ~finished
                new_rows = new_rows[keep]
                new_tokens = new_tokens[keep]
                new_cache_seqlens = new_cache_seqlens[keep]
                new_position_ids = new_position_ids[keep]
                new_block_table = new_block_table[keep]
        if next_tokens.numel() == 0:
            # Batch was idle; the new prompts become the entire active set.
            active_rows = new_rows
            active_count = int(new_rows.numel())
            return generated, logprobs, response_lens, new_tokens, new_cache_seqlens, new_position_ids, new_block_table, active_rows, active_count
        # Otherwise concat the new rows onto the existing active state.
        active_rows = torch.cat([active_rows, new_rows])
        return generated, logprobs, response_lens, torch.cat([next_tokens, new_tokens]), torch.cat([cache_seqlens, new_cache_seqlens]), torch.cat([position_ids, new_position_ids]), torch.cat([block_table, new_block_table]), active_rows, int(active_rows.numel())

    def _poll_rollout_add_commands(
        self,
        state: InferenceBatchState,
        prompt_indices: list[int],
        session_id: int | None,
    ) -> None:
        """Drain ADD commands from the engine queue and append admissible ones.

        Continuous-batching gate logic:
          * commands for a different session are dropped;
          * non-ADD commands are pushed back onto `_deferred_commands` so the
            outer dispatch loop sees them once rollout finishes;
          * if the running set is already at `max_running_seqs` or there are
            not enough free KV blocks to host the new prompt at full
            `max_new_tokens`, the ADD command is requeued and we exit.
        """
        cmd_q = getattr(self, "_cmd_queue", None)
        if cmd_q is None:
            return
        while True:
            deferred = getattr(self, "_deferred_commands", [])
            if deferred:
                cmd = deferred.pop(0)
            else:
                try:
                    cmd = cmd_q.get_nowait()
                except queue.Empty:
                    return
            # Stale ADDs from a previous session are silently dropped.
            if cmd.op is Op.INFER_ROLLOUT_ADD and cmd.payload.session_id != session_id:
                continue
            if cmd.op is not Op.INFER_ROLLOUT_ADD:
                # Park non-ADD commands so the main dispatcher processes them
                # after the current rollout returns.
                getattr(self, "_deferred_commands", []).append(cmd)
                return
            ctx = get_tp_context()
            add_payload: RolloutAddPayload = cmd.payload
            prompts = add_payload.prompts_by_dp[ctx.dp_rank]
            indices = add_payload.prompt_indices_by_dp[ctx.dp_rank]
            if not prompts:
                continue
            # Running-set budget gate.
            if len(state._seq_to_blocks) >= state.max_running_seqs:
                getattr(self, "_deferred_commands", []).append(cmd)
                return
            # KV-block budget gate: worst case = prompt + max_new_tokens.
            first_prompt = prompts[0]
            needed_blocks = math.ceil((len(first_prompt) + state.max_new_tokens) / state.kv_block_size)
            if len(state._free_blocks) < needed_blocks:
                getattr(self, "_deferred_commands", []).append(cmd)
                return
            state.prompts.extend(prompts)
            state.generated.extend([] for _ in prompts)
            state.logprobs.extend([] for _ in prompts)
            state.finished.extend(False for _ in prompts)
            state.finish_reason.extend("" for _ in prompts)
            prompt_indices.extend(indices)

    def _free_rollout_rows(self, state: InferenceBatchState, rows: torch.Tensor) -> None:
        """Return the KV blocks owned by `rows` to the free pool."""
        if rows.numel() == 0:
            return
        for row in rows.detach().cpu().tolist():
            blocks = state._seq_to_blocks.pop(int(row), None)
            if blocks:
                state._free_blocks.extend(blocks)

    def _emit_rollout_partials(
        self,
        rows: torch.Tensor,
        generated: torch.Tensor,
        response_lens: torch.Tensor,
        finish_reason: str,
        prompt_indices: list[int] | None = None,
    ) -> None:
        """Send a partial-result message for finished rows to the engine driver.

        Only rank 0 publishes results (rest of TP group is a duplicate). Used
        to stream completions from continuous-serving sessions before the full
        rollout returns.
        """
        ctx = get_tp_context()
        if not ctx.is_rank0 or rows.numel() == 0:
            return
        result_queue = getattr(self, "_result_queue", None)
        rank = getattr(self, "_rank", None)
        if result_queue is None or rank is None:
            return
        row_values = rows.detach().cpu().tolist()
        lengths = response_lens[rows].detach().cpu().tolist()
        generated_rows = generated[rows].detach().cpu().tolist()
        response_ids = [row[: int(length)] for row, length in zip(generated_rows, lengths, strict=True)]
        prompt_index_values = [prompt_indices[int(row)] for row in row_values] if prompt_indices is not None else None
        result_queue.put(
            (
                rank,
                WorkerResult(
                    ok=True,
                    payload=RolloutPartialPayload(
                        dp_rank=ctx.dp_rank,
                        rows=row_values,
                        response_ids=response_ids,
                        finish_reason=[finish_reason for _ in response_ids],
                        prompt_indices=prompt_index_values,
                    ),
                    final=False,
                ),
            )
        )

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
