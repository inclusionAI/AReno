"""User-facing Python API for the areno engine.

This module exposes :class:`ArenoEngine`, the entry point that callers use to
drive a local training and inference engine. The engine itself owns no model
weights; instead it spawns a :class:`TPCluster` of worker processes (one per
TP/DP rank) and dispatches commands to them.

High-level responsibilities:

* Spawn and tear down the worker cluster.
* Translate user inputs (``SamplingParams``, ``TrainerConfig``, prompt lists,
  data packs) into the on-the-wire ``Command`` protocol consumed by
  :mod:`areno.engine.worker`.
* Split inputs across data-parallel ranks and merge rank-0 outputs back into
  the simple single-process result shape returned to the caller.
* Expose the rollout/train/score/checkpoint operations as plain Python methods
  while hiding the underlying RPC-style ``call`` / fire-and-forget
  request-demux semantics.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

from areno.engine.checkpoints.io import resolve_model_path
from areno.engine.config import EngineConfig, OptimizerConfig, RuntimeConfig
from areno.engine.data import RolloutOutput, SamplingParams, TrainStats, to_cpu
from areno.engine.protocol import (
    EnsureRolesPayload,
    Op,
    RoleSpecPayload,
    RolloutPayload,
    SaveCheckpointPayload,
    ScorePayload,
    TPCluster,
    TrainPayload,
    TrainValuesPayload,
)
from areno.engine.worker import ArenoWorker
from areno.models.registry import config_from_hf
from areno.engine.runtime.common import (
    dp_rank0_results,
    merge_metric_dicts,
    merge_train_stats,
    split_data_pack_by_dp,
    split_list_by_dp,
)
from areno.engine.runtime.decode_graph import ceil_div
from areno.engine.runtime.rollout import _build_rollout_from_rows, _merge_dp_rollouts_in_input_order, _merge_rollouts, partial_tail_threshold


PARTIAL_TAIL_MIN_TOKENS = 256
PARTIAL_TAIL_COOLDOWN_S = 10.0


def _merge_async_dp_rollouts(outputs: list[RolloutOutput | None], *, total_count: int) -> RolloutOutput:
    """Merge async DP results, allowing coalesced single-row requests to land on any DP."""

    non_empty = [output for output in outputs if output is not None and len(output.prompt_ids) > 0]
    if total_count == 1 and len(non_empty) == 1:
        return non_empty[0]
    try:
        return _merge_dp_rollouts_in_input_order(outputs, total_count=total_count)
    except RuntimeError as exc:
        row_counts = [0 if output is None else len(output.prompt_ids) for output in outputs]
        if sum(row_counts) == total_count:
            return _merge_rollouts(non_empty)
        raise RuntimeError(f"{exc}; async DP row_counts={row_counts} total_count={total_count}") from exc


class ArenoEngine:
    """User-facing coordinator for one local training/inference engine.

    The engine itself does not hold model weights. It splits user batches by DP,
    sends commands to rank workers via the :class:`TPCluster` protocol, and
    merges rank-0 DP results back into the simple API shape returned to the
    caller.

    Two request patterns are used against the cluster:

    * ``cluster.call(op, payload)`` -- blocking RPC-style; waits for every rank
      to return a result. Used for ops where the caller needs outputs
      (rollout, train step, scoring, checkpoint).
    * ``cluster.call_async(op, payload)`` -- async request-id demux over the
      same worker queues, used by request-concurrent serving and agentic
      rollout paths.
    """

    def __init__(self, config: EngineConfig):
        """Start rank workers for a validated engine config.

        Constructs the :class:`TPCluster` and starts the underlying worker
        processes; blocks until the cluster is ready to accept commands.
        """

        # Loss function is required because the engine always carries a trainer
        # path; pure-inference engines should still set a no-op loss.
        if config.train_loss_fn is None:
            raise ValueError("ArenoEngine requires train_loss_fn")
        self.config = config
        # TPCluster owns the per-rank worker processes and the IPC channels;
        # ``ArenoWorker`` is the rank-side command loop.
        self.cluster = TPCluster(config, ArenoWorker)
        self.cluster.start()

    def begin_rollout_session(self) -> None:
        """Prepare workers for one or more rollout calls."""

        self.cluster.call(Op.ROLLOUT_SESSION_BEGIN)

    async def begin_rollout_session_async(self) -> None:
        """Async variant of :meth:`begin_rollout_session`."""

        await self.cluster.call_async(Op.ROLLOUT_SESSION_BEGIN)

    def end_rollout_session(self) -> None:
        """Finalize rollout state and prepare train weights."""

        self.cluster.call(Op.ROLLOUT_SESSION_END)

    async def end_rollout_session_async(self) -> None:
        """Async variant of :meth:`end_rollout_session`."""

        await self.cluster.call_async(Op.ROLLOUT_SESSION_END)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        tp_size: int = 1,
        dp_size: int | None = None,
        devices: list[int] | None = None,
        dummy_load: bool = False,
        optimizer_config: OptimizerConfig | None = None,
        runtime_config: RuntimeConfig | None = None,
        loss_fn: Callable[[Any, torch.Tensor], torch.Tensor | tuple[torch.Tensor, dict[str, Any]]] | None = None,
    ) -> "ArenoEngine":
        """Build an engine by reading model config from a checkpoint path.

        Resolves a local path or HuggingFace repo id to a concrete checkpoint
        directory, parses the HF config into the internal model config, and
        wraps the result in an :class:`EngineConfig` before delegating to
        ``__init__``. Blocking: workers are started before the call returns.
        """

        if model is None:
            raise ValueError("from_pretrained() requires a local model path or Hugging Face model id")
        # Resolve HF repo id or local dir to an on-disk checkpoint directory.
        model_path = resolve_model_path(model)
        if model_path is None:
            raise ValueError(f"could not resolve model path: {model!r}")
        # Translate the HF config.json into the engine's internal model schema.
        model_config = config_from_hf(model_path)
        cfg = EngineConfig(
            model=model_config,
            model_path=model_path,
            train_loss_fn=loss_fn,
            tp_size=tp_size,
            dp_size=dp_size,
            devices=devices,
            dummy_load=dummy_load,
            optimizer=optimizer_config or OptimizerConfig(),
            runtime=runtime_config or RuntimeConfig(),
        )
        return cls(cfg)

    def generate_rollout(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int,
        max_running_prompts: int,
        max_prompt_len: int | None = None,
        eos_token_id: int | None = None,
        sampling_params: SamplingParams | None = None,
        decode_progress_interval_s: float = 0.0,
        coalesce_timeout_s: float = 5.0,
        cancel_flags: torch.Tensor | None = None,
    ) -> RolloutOutput:
        """Generate rollout tokens for pre-tokenized prompts.

        Dispatches ``Op.INFER_ROLLOUT`` via the blocking ``cluster.call``
        path. Inputs are token-id rows (one row per prompt); the returned
        :class:`RolloutOutput` carries response token ids and finish reasons
        merged across DP ranks in the original input order.

        Prompts are chunked by DP size and prefill-token budget so each worker
        can run bounded prefill batches while decode continues to use a fixed
        paged-KV allocation.
        """

        if not prompts:
            raise ValueError("prompts must be non-empty")
        if max_running_prompts < 1:
            raise ValueError("max_running_prompts must be >= 1")
        sampling_params = sampling_params or SamplingParams()
        outputs = []
        # Cap prompt length for KV sizing; either honour caller-provided bound
        # or fall back to the longest prompt actually seen.
        rollout_max_prompt_len = int(max_prompt_len) if max_prompt_len is not None else max(len(prompt) for prompt in prompts)
        rollout_max_cache_len = rollout_max_prompt_len + max_new_tokens
        dp_size = int(self.config.dp_size)
        local_max_running_prompts = max(ceil_div(int(max_running_prompts), dp_size), 1)
        # Worst-case per-rank prefill = every local running slot prefilling its
        # full prompt. The public max_running_prompts value is global.
        max_prefill_tokens = local_max_running_prompts * rollout_max_prompt_len
        chunks = _chunk_prompts_for_prefill_budget(
            prompts,
            max_running_prompts=max_running_prompts,
            dp_size=dp_size,
            max_prefill_tokens=max_prefill_tokens,
        )
        for chunk in chunks:
            # Global offset of this chunk inside the user's input list, used to
            # restore original indices when merging DP results.
            chunk_start = sum(len(output.prompt_ids) for output in outputs)
            # Round-robin chunk rows across DP ranks; per-rank token rows.
            prompts_by_dp = split_list_by_dp(chunk, int(self.config.dp_size))
            # Parallel split of the global prompt indices for downstream mapping.
            prompt_indices_by_dp = split_list_by_dp(list(range(chunk_start, chunk_start + len(chunk))), int(self.config.dp_size))
            # Largest per-rank queue depth for the current chunk; floor of 1
            # avoids zero-sized KV pools for trailing chunks.
            current_local_running = max(max((len(rows) for rows in prompts_by_dp), default=0), 1)
            local_max_running = current_local_running
            # Honour per-prompt cache length if any prompt+max_new exceeds the
            # global ceiling computed from rollout_max_prompt_len.
            max_cache_len = max(max(len(prompt) + max_new_tokens for prompt in chunk), rollout_max_cache_len)
            max_blocks_per_seq = ceil_div(max_cache_len, self.config.runtime.kv_block_size)
            # Blocking RPC: returns one result per rank after every prompt
            # finishes (or the cancel flag fires).
            results = self.cluster.call(
                Op.INFER_ROLLOUT,
                RolloutPayload(
                    prompts_by_dp=prompts_by_dp,
                    prompt_indices_by_dp=prompt_indices_by_dp,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                    sampling_params=sampling_params,
                    max_running_seqs=local_max_running,
                    max_cache_len=max_cache_len,
                    max_blocks_per_seq=max_blocks_per_seq,
                    max_prefill_tokens=max_prefill_tokens,
                    num_blocks=local_max_running * max_blocks_per_seq,
                    block_size=self.config.runtime.kv_block_size,
                    decode_progress_interval_s=decode_progress_interval_s,
                    cancel_flags=cancel_flags,
                    cancel_indices_by_dp=split_list_by_dp(
                        list(range(chunk_start, chunk_start + len(chunk))),
                        int(self.config.dp_size),
                    )
                    if cancel_flags is not None
                    else None,
                ),
            )
            # Drop TP duplicates (only DP rank 0 carries real results) and
            # re-order DP-shuffled rows back into chunk input order.
            dp_outputs = dp_rank0_results(results, self.config.tp_size, int(self.config.dp_size))
            total_count = len(chunk)
            outputs.append(
                _merge_dp_rollouts_in_input_order(
                    dp_outputs,
                    total_count=total_count,
                )
            )
        # Fast path for single-chunk inputs; otherwise concat per-chunk outputs.
        return outputs[0] if len(outputs) == 1 else _merge_rollouts(outputs)

    async def generate_rollout_async(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int,
        max_running_prompts: int,
        max_prompt_len: int | None = None,
        eos_token_id: int | None = None,
        sampling_params: SamplingParams | None = None,
        decode_progress_interval_s: float = 0.0,
        cancel_flags: torch.Tensor | None = None,
        coalesce_timeout_s: float = 5.0,
        partial_tail_cooldown_until_s: list[float] | None = None,
    ) -> RolloutOutput:
        """Async rollout path using TPCluster request-id demux.

        Coalesced async rollouts may cut small active tails as ``partial``.
        Those rows are transparently resumed as ``prompt + generated`` so HTTP
        and agentic callers still receive a single completed rollout row.
        """

        output = await self._generate_rollout_async_once(
            prompts,
            max_new_tokens=max_new_tokens,
            max_running_prompts=max_running_prompts,
            max_prompt_len=max_prompt_len,
            eos_token_id=eos_token_id,
            sampling_params=sampling_params,
            decode_progress_interval_s=decode_progress_interval_s,
            cancel_flags=cancel_flags,
            coalesce_timeout_s=coalesce_timeout_s,
            partial_tail_cooldown_until_s=partial_tail_cooldown_until_s,
        )
        if cancel_flags is not None:
            return output
        return await self._complete_partial_async_rollout(
            output,
            max_new_tokens=max_new_tokens,
            max_running_prompts=max_running_prompts,
            eos_token_id=eos_token_id,
            sampling_params=sampling_params or SamplingParams(),
            decode_progress_interval_s=decode_progress_interval_s,
            coalesce_timeout_s=coalesce_timeout_s,
        )

    async def _generate_rollout_async_once(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int,
        max_running_prompts: int,
        max_prompt_len: int | None = None,
        eos_token_id: int | None = None,
        sampling_params: SamplingParams | None = None,
        decode_progress_interval_s: float = 0.0,
        cancel_flags: torch.Tensor | None = None,
        coalesce_timeout_s: float = 5.0,
        partial_tail_cooldown_until_s: list[float] | None = None,
    ) -> RolloutOutput:
        """Run one async rollout pass without resolving ``partial`` rows."""

        if not prompts:
            raise ValueError("prompts must be non-empty")
        if max_running_prompts < 1:
            raise ValueError("max_running_prompts must be >= 1")
        sampling_params = sampling_params or SamplingParams()
        outputs = []
        rollout_max_prompt_len = int(max_prompt_len) if max_prompt_len is not None else max(len(prompt) for prompt in prompts)
        rollout_max_cache_len = rollout_max_prompt_len + max_new_tokens
        dp_size = int(self.config.dp_size)
        local_max_running_prompts = max(ceil_div(int(max_running_prompts), dp_size), 1)
        max_prefill_tokens = local_max_running_prompts * rollout_max_prompt_len
        chunks = _chunk_prompts_for_prefill_budget(
            prompts,
            max_running_prompts=max_running_prompts,
            dp_size=dp_size,
            max_prefill_tokens=max_prefill_tokens,
        )
        for chunk in chunks:
            chunk_start = sum(len(output.prompt_ids) for output in outputs)
            prompts_by_dp = split_list_by_dp(chunk, dp_size)
            prompt_indices_by_dp = split_list_by_dp(list(range(chunk_start, chunk_start + len(chunk))), dp_size)
            cooldown_chunk = (
                partial_tail_cooldown_until_s[chunk_start : chunk_start + len(chunk)]
                if partial_tail_cooldown_until_s is not None
                else [0.0 for _ in chunk]
            )
            current_local_running = max(max((len(rows) for rows in prompts_by_dp), default=0), 1)
            capacity_local_running = local_max_running_prompts if coalesce_timeout_s > 0.0 and cancel_flags is None else current_local_running
            max_cache_len = max(max(len(prompt) + max_new_tokens for prompt in chunk), rollout_max_cache_len)
            max_blocks_per_seq = ceil_div(max_cache_len, self.config.runtime.kv_block_size)
            payload = RolloutPayload(
                prompts_by_dp=prompts_by_dp,
                prompt_indices_by_dp=prompt_indices_by_dp,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                sampling_params=sampling_params,
                max_running_seqs=capacity_local_running,
                max_cache_len=max_cache_len,
                max_blocks_per_seq=max_blocks_per_seq,
                max_prefill_tokens=max_prefill_tokens,
                num_blocks=capacity_local_running * max_blocks_per_seq,
                block_size=self.config.runtime.kv_block_size,
                decode_progress_interval_s=decode_progress_interval_s,
                cancel_flags=cancel_flags,
                cancel_indices_by_dp=split_list_by_dp(list(range(chunk_start, chunk_start + len(chunk))), dp_size)
                if cancel_flags is not None
                else None,
                coalesce_max_running_seqs=local_max_running_prompts,
                coalesce_timeout_s=max(float(coalesce_timeout_s), 0.0),
                partial_tail_cooldown_until_s_by_dp=split_list_by_dp(cooldown_chunk, dp_size),
            )
            payload.partial_tail_threshold = partial_tail_threshold(capacity_local_running, coalesce_timeout_s)
            payload.partial_tail_min_tokens = PARTIAL_TAIL_MIN_TOKENS
            results = await self.cluster.call_async(
                Op.INFER_ROLLOUT,
                payload,
            )
            outputs.append(
                _merge_async_dp_rollouts(dp_rank0_results(results, self.config.tp_size, dp_size), total_count=len(chunk))
            )
        return outputs[0] if len(outputs) == 1 else _merge_rollouts(outputs)

    async def _complete_partial_async_rollout(
        self,
        output: RolloutOutput,
        *,
        max_new_tokens: int,
        max_running_prompts: int,
        eos_token_id: int | None,
        sampling_params: SamplingParams,
        decode_progress_interval_s: float,
        coalesce_timeout_s: float,
    ) -> RolloutOutput:
        """Resume ``partial`` rows until every async rollout row is complete."""

        prompt_rows = [list(row) for row in output.prompt_ids]
        response_rows = [list(row) for row in output.response_ids]
        finish_reason = list(output.finish_reason)
        logprob_rows = [output.logprobs[row_idx, : len(response)].detach().cpu() for row_idx, response in enumerate(response_rows)]
        now = time.monotonic()
        partial_cooldown_until = [
            now + PARTIAL_TAIL_COOLDOWN_S if reason == "partial" else 0.0
            for reason in finish_reason
        ]

        while True:
            partial_indices = [
                row_idx
                for row_idx, reason in enumerate(finish_reason)
                if reason == "partial" and len(response_rows[row_idx]) < max_new_tokens
            ]
            if not partial_indices:
                break
            groups: dict[int, list[int]] = {}
            for row_idx in partial_indices:
                remaining = max_new_tokens - len(response_rows[row_idx])
                groups.setdefault(remaining, []).append(row_idx)
            for remaining, row_indices in groups.items():
                continuation_prompts = [prompt_rows[row_idx] + response_rows[row_idx] for row_idx in row_indices]
                continuation_cooldowns = [partial_cooldown_until[row_idx] for row_idx in row_indices]
                continuation = await self._generate_rollout_async_once(
                    continuation_prompts,
                    max_new_tokens=remaining,
                    max_running_prompts=max_running_prompts,
                    max_prompt_len=None,
                    eos_token_id=eos_token_id,
                    sampling_params=sampling_params,
                    decode_progress_interval_s=decode_progress_interval_s,
                    cancel_flags=None,
                    coalesce_timeout_s=coalesce_timeout_s,
                    partial_tail_cooldown_until_s=continuation_cooldowns,
                )
                for local_idx, row_idx in enumerate(row_indices):
                    next_response = continuation.response_ids[local_idx]
                    if continuation.finish_reason[local_idx] == "partial" and not next_response:
                        finish_reason[row_idx] = "length"
                        continue
                    response_rows[row_idx].extend(next_response)
                    next_logprobs = continuation.logprobs[local_idx, : len(next_response)].detach().cpu()
                    logprob_rows[row_idx] = torch.cat([logprob_rows[row_idx], next_logprobs]) if logprob_rows[row_idx].numel() else next_logprobs
                    finish_reason[row_idx] = continuation.finish_reason[local_idx]
                    if finish_reason[row_idx] == "partial":
                        partial_cooldown_until[row_idx] = time.monotonic() + PARTIAL_TAIL_COOLDOWN_S
            if all(reason != "partial" or len(response_rows[row_idx]) >= max_new_tokens for row_idx, reason in enumerate(finish_reason)):
                break
        finish_reason = ["length" if reason == "partial" else reason for reason in finish_reason]
        return _build_rollout_from_rows(prompt_rows, response_rows, finish_reason, logprob_rows, metrics=output.metrics)

    def step(self, data_packs: list[dict[str, Any]], *, gradient_accumulation_steps: int | None = None) -> list[TrainStats]:
        """Run one or more train data packs through the DP/TP worker cluster.

        Dispatches ``Op.TRAIN`` via the blocking ``cluster.call`` path. Each
        data pack is split across DP ranks; the workers run forward + backward
        + optimizer for every pack in order and return per-step
        :class:`TrainStats`. Blocking: returns after every rank finishes all
        steps.
        """

        if not data_packs:
            return []
        data_packs_by_dp = []
        for data_pack in data_packs:
            # Per-pack DP split so each rank receives a balanced micro-batch.
            data_packs_by_dp.append(split_data_pack_by_dp(data_pack, int(self.config.dp_size)))
        # to_cpu + shared-memory copy avoids pickling large tensors across IPC.
        results = self.cluster.call(
            Op.TRAIN,
            TrainPayload(
                data_packs_by_dp=self._transport_payload(to_cpu(data_packs_by_dp)),
                gradient_accumulation_steps=gradient_accumulation_steps,
            ),
        )
        stats: list[TrainStats] = []
        # results[rank] is a list aligned with data_packs; zip transposes to
        # iterate per-step across ranks.
        for step_results in zip(*results, strict=True):
            rank0_results = [
                result
                for result in dp_rank0_results(list(step_results), self.config.tp_size, int(self.config.dp_size))
                if result is not None
            ]
            # Sum/average loss + metric fields contributed by each DP rank.
            stats.append(merge_train_stats(rank0_results))
        return stats

    def ensure_roles(self, roles: dict[str, Any]) -> None:
        """Create backend-owned auxiliary roles inside worker processes.

        Dispatches ``Op.ENSURE_ROLES`` (blocking). A role is a named secondary
        model (critic, reward model, reference policy, ...) that workers load
        alongside the primary policy. Idempotent: existing roles are left as
        is, missing ones are constructed using the provided spec.
        """

        # Flatten role specs to a transport-friendly dict; path is stringified
        # because Path objects do not always pickle cleanly across processes.
        payload = EnsureRolesPayload(
            roles={
                name: RoleSpecPayload(
                    path=str(spec.path),
                    trainable=bool(spec.trainable),
                    optimizer_lr=getattr(spec, "optimizer_lr", None),
                )
                for name, spec in roles.items()
            }
        )
        self.cluster.call(Op.ENSURE_ROLES, payload)

    def score_logprobs(self, role: str, token_rows: list[list[int]], *, pad_token_id: int) -> list[list[float]]:
        """Score fixed token rows with a model role.

        Dispatches ``Op.SCORE_LOGPROBS`` (blocking). Returns per-token
        log-probabilities for each input row in input order. Token rows are
        split across DP ranks; only rank 0's merged result is returned.
        """

        if not token_rows:
            return []
        results = self.cluster.call(
            Op.SCORE_LOGPROBS,
            ScorePayload(role=role, token_rows_by_dp=split_list_by_dp(token_rows, int(self.config.dp_size)), pad_token_id=int(pad_token_id)),
        )
        return _merge_dp_rank0_strided_results(results, self.config.tp_size, int(self.config.dp_size))

    def score_values(self, role: str, token_rows: list[list[int]], *, pad_token_id: int) -> list[list[float]]:
        """Score per-token values with a critic role.

        Dispatches ``Op.SCORE_VALUES`` (blocking). Same shape contract as
        :meth:`score_logprobs` but returns per-token critic values.
        """

        if not token_rows:
            return []
        results = self.cluster.call(
            Op.SCORE_VALUES,
            ScorePayload(role=role, token_rows_by_dp=split_list_by_dp(token_rows, int(self.config.dp_size)), pad_token_id=int(pad_token_id)),
        )
        return _merge_dp_rank0_strided_results(results, self.config.tp_size, int(self.config.dp_size))

    def score_rewards(self, role: str, token_rows: list[list[int]], *, pad_token_id: int) -> list[float]:
        """Score sequence rewards with a reward model role.

        Dispatches ``Op.SCORE_REWARDS`` (blocking). Returns one scalar reward
        per input row (not per token).
        """

        if not token_rows:
            return []
        results = self.cluster.call(
            Op.SCORE_REWARDS,
            ScorePayload(role=role, token_rows_by_dp=split_list_by_dp(token_rows, int(self.config.dp_size)), pad_token_id=int(pad_token_id)),
        )
        return _merge_dp_rank0_strided_results(results, self.config.tp_size, int(self.config.dp_size))

    def train_values(
        self,
        role: str,
        data_packs: list[dict[str, Any]],
        *,
        gradient_accumulation_steps: int | None = None,
        cliprange_value: float = 0.5,
        value_loss_coef: float = 0.5,
    ) -> dict[str, float]:
        """Train a critic role on value returns.

        Dispatches ``Op.TRAIN_VALUES`` (blocking). Each data pack is split
        across DP ranks and grouped per-rank so the worker can iterate the
        packs in input order. Returns merged scalar metrics (loss components,
        clip fraction, ...).
        """

        if not data_packs:
            return {}
        data_packs_by_dp = []
        for data_pack in data_packs:
            data_packs_by_dp.append(split_data_pack_by_dp(data_pack, int(self.config.dp_size)))
        # Transpose [pack][dp_rank] -> [dp_rank][pack] so each rank gets its own
        # sequential pack list (matches Op.TRAIN_VALUES worker contract).
        per_dp_packs = [[] for _ in range(int(self.config.dp_size))]
        for split_pack in data_packs_by_dp:
            for dp_rank, pack in enumerate(split_pack):
                per_dp_packs[dp_rank].append(pack)
        results = self.cluster.call(
            Op.TRAIN_VALUES,
            TrainValuesPayload(
                role=role,
                data_packs_by_dp=self._transport_payload(to_cpu(per_dp_packs)),
                gradient_accumulation_steps=gradient_accumulation_steps,
                cliprange_value=float(cliprange_value),
                value_loss_coef=float(value_loss_coef),
            ),
        )
        rank0_results = [result for result in dp_rank0_results(results, self.config.tp_size, int(self.config.dp_size)) if result]
        return merge_metric_dicts(rank0_results) or {}

    def save_checkpoint(self, path: str) -> str:
        """Ask workers to write a HuggingFace-compatible checkpoint.

        Dispatches ``Op.SAVE_CHECKPOINT`` (blocking). Workers cooperatively
        write shards to ``path``; only rank 0's returned path is propagated
        back to the caller.
        """

        results = self.cluster.call(Op.SAVE_CHECKPOINT, SaveCheckpointPayload(path=path))
        return results[0]["path"]

    def _transport_payload(self, payload: Any) -> Any:
        """Move tensors to CPU shared memory for zero-copy IPC to workers."""

        # share_memory=True lets the worker side mmap the underlying buffer
        # instead of receiving a pickled copy over the IPC channel.
        return to_cpu(payload, share_memory=True)

    def close(self) -> None:
        """Stop worker processes and release cluster resources.

        Blocking: waits for each rank's worker process to exit before returning.
        """

        self.cluster.close()

    def __enter__(self) -> "ArenoEngine":
        """Return self for `with ArenoEngine...` usage."""

        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        """Close workers when leaving a context manager."""

        self.close()


def _chunk_prompts_for_prefill_budget(
    prompts: list[list[int]],
    *,
    max_running_prompts: int,
    dp_size: int,
    max_prefill_tokens: int,
) -> list[list[list[int]]]:
    """Split prompts so each DP rank stays within prefill token budget.

    Greedy packer that walks the prompt list once and starts a new chunk when
    either the per-chunk count cap (``max_running_prompts * dp_size``) or the
    per-DP-rank token cap (``max_prefill_tokens``) would be exceeded. The
    per-rank index is computed via round-robin position inside the chunk.
    """

    # Hard cap on prompts per chunk. max_running_prompts is a global flat
    # rollout cap; DP splitting happens after chunking.
    max_chunk_size = max_running_prompts
    chunks: list[list[list[int]]] = []
    chunk: list[list[int]] = []
    token_sums = [0 for _ in range(dp_size)]
    for prompt in prompts:
        # Round-robin DP assignment matches split_list_by_dp's layout.
        dp_rank = len(chunk) % dp_size
        would_exceed_count = len(chunk) >= max_chunk_size
        # token_sums[dp_rank] > 0 guard: always allow the first prompt for a
        # rank even if it alone exceeds the prefill budget.
        would_exceed_tokens = token_sums[dp_rank] > 0 and token_sums[dp_rank] + len(prompt) > max_prefill_tokens
        if chunk and (would_exceed_count or would_exceed_tokens):
            # Flush current chunk and start fresh; reset per-rank token tally.
            chunks.append(chunk)
            chunk = []
            token_sums = [0 for _ in range(dp_size)]
            dp_rank = 0
        chunk.append(prompt)
        token_sums[dp_rank] += len(prompt)
    if chunk:
        chunks.append(chunk)
    return chunks


def _first_rank0_result(results: list[Any], tp_size: int, dp_size: int):
    """Return the first non-empty rank-0 DP result, or ``[]`` if none."""

    # Strip TP duplicates; only the DP rank-0 of each TP group carries the
    # actual scoring output, the rest return None.
    rank0_results = [result for result in dp_rank0_results(results, tp_size, dp_size) if result is not None]
    if not rank0_results:
        return []
    return rank0_results[0]


def _merge_dp_rank0_strided_results(results: list[Any], tp_size: int, dp_size: int) -> list[Any]:
    """Merge per-DP local score results returned by each TP rank-0 worker."""

    parts = [result if result is not None else [] for result in dp_rank0_results(results, tp_size, dp_size)]
    merged = []
    max_len = max((len(part) for part in parts), default=0)
    for item_idx in range(max_len):
        for part in parts:
            if item_idx < len(part):
                merged.append(part[item_idx])
    return merged
