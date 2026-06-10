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
  ``broadcast`` semantics.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from areno.engine.checkpoints.io import resolve_model_path
from areno.engine.config import EngineConfig, OptimizerConfig, RuntimeConfig
from areno.engine.data import RolloutOutput, SamplingParams, TrainStats, to_cpu
from areno.engine.protocol import (
    EnsureRolesPayload,
    Op,
    RoleSpecPayload,
    RolloutAddPayload,
    RolloutPartialPayload,
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
from areno.engine.runtime.rollout import _merge_dp_rollouts_in_input_order, _merge_rollouts


class ArenoEngine:
    """User-facing coordinator for one local training/inference engine.

    The engine itself does not hold model weights. It splits user batches by DP,
    sends commands to rank workers via the :class:`TPCluster` protocol, and
    merges rank-0 DP results back into the simple API shape returned to the
    caller.

    Two dispatch patterns are used against the cluster:

    * ``cluster.call(op, payload)`` -- blocking RPC-style; waits for every rank
      to return a result. Used for ops where the caller needs outputs
      (rollout, train step, scoring, checkpoint).
    * ``cluster.broadcast(op, payload)`` -- fire-and-forget object broadcast;
      returns immediately without waiting for workers. Used for the
      continuous-batching attach path (``add_rollout_batch``).
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
        partial_callback: Callable[[int, list[int], str], None] | None = None,
        cancel_flags: torch.Tensor | None = None,
        session_id: int | None = None,
    ) -> RolloutOutput:
        """Generate rollout tokens for pre-tokenized prompts.

        Dispatches ``Op.INFER_ROLLOUT`` via the blocking ``cluster.call``
        path. Inputs are token-id rows (one row per prompt); the returned
        :class:`RolloutOutput` carries response token ids and finish reasons
        merged across DP ranks in the original input order.

        Prompts are chunked by DP size and prefill-token budget so each worker
        can run bounded prefill batches while decode continues to use a fixed
        paged-KV allocation. When ``session_id`` is provided the call follows
        the continuous-batching contract: each chunk inherits a slightly larger
        ``local_max_running`` than the chunk's own size so newly-arriving
        prompts (via :meth:`add_rollout_batch`) can attach without forcing the
        worker to rebuild its KV pool.
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
        # Worst-case prefill = every running slot prefilling its full prompt.
        max_prefill_tokens = max_running_prompts * rollout_max_prompt_len
        dp_size = int(self.config.dp_size)
        for chunk in _chunk_prompts_for_prefill_budget(
            prompts,
            max_running_prompts=max_running_prompts,
            dp_size=dp_size,
            max_prefill_tokens=max_prefill_tokens,
        ):
            # Global offset of this chunk inside the user's input list, used to
            # restore original indices when merging DP results / partial events.
            chunk_start = sum(len(output.prompt_ids) for output in outputs)
            # Round-robin chunk rows across DP ranks; per-rank token rows.
            prompts_by_dp = split_list_by_dp(chunk, int(self.config.dp_size))
            # Parallel split of the global prompt indices for downstream mapping.
            prompt_indices_by_dp = split_list_by_dp(list(range(chunk_start, chunk_start + len(chunk))), int(self.config.dp_size))
            # Largest per-rank queue depth for the current chunk; floor of 1
            # avoids zero-sized KV pools for trailing partial chunks.
            current_local_running = max(max((len(rows) for rows in prompts_by_dp), default=0), 1)
            # In session mode reserve +8 headroom so add_rollout_batch can
            # attach late arrivals without re-allocating the KV pool; outside
            # session mode the pool is sized exactly to the chunk.
            local_max_running = (
                min(int(max_running_prompts), current_local_running + 8)
                if session_id is not None
                else current_local_running
            )
            # Honour per-prompt cache length if any prompt+max_new exceeds the
            # global ceiling computed from rollout_max_prompt_len.
            max_cache_len = max(max(len(prompt) + max_new_tokens for prompt in chunk), rollout_max_cache_len)
            max_blocks_per_seq = ceil_div(max_cache_len, self.config.runtime.kv_block_size)
            partial = None
            if partial_callback is not None:
                # Wrap the user callback so worker-side payloads (which carry
                # chunk-local rows) are translated back into global indices.
                partial = _partial_callback_for_chunk(
                    partial_callback,
                    chunk_start=chunk_start,
                    chunk_size=len(chunk),
                    dp_size=dp_size,
                )
            # Blocking RPC: returns one result per rank after every prompt
            # finishes (or the cancel flag fires).
            results = self.cluster.call(
                Op.INFER_ROLLOUT,
                RolloutPayload(
                    prompts_by_dp=prompts_by_dp,
                    prompt_indices_by_dp=prompt_indices_by_dp,
                    session_id=session_id,
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
                partial_callback=partial,
            )
            # Drop TP duplicates (only DP rank 0 carries real results) and
            # re-order DP-shuffled rows back into chunk input order.
            outputs.append(
                _merge_dp_rollouts_in_input_order(
                    dp_rank0_results(results, self.config.tp_size, int(self.config.dp_size)),
                    total_count=len(chunk),
                )
            )
        # Fast path for single-chunk inputs; otherwise concat per-chunk outputs.
        return outputs[0] if len(outputs) == 1 else _merge_rollouts(outputs)

    def add_rollout_batch(
        self,
        prompts: list[list[int]],
        *,
        session_id: int,
        prompt_offset: int,
        cancel_flags: torch.Tensor | None = None,
    ) -> None:
        """Attach more prompts to an in-flight rollout session.

        Dispatches ``Op.INFER_ROLLOUT_ADD`` via the non-blocking
        ``cluster.broadcast`` path. The matching :meth:`generate_rollout` call
        must already be running with the same ``session_id``; the worker side
        slots these new prompts into the existing continuous-batching loop
        using ``prompt_offset`` to keep global indices stable.

        ``cancel_flags`` shares the rollout caller's shared-memory tensor so
        cancellations apply uniformly to original and attached prompts.

        Non-blocking: returns once the command is enqueued; does not wait for
        workers to acknowledge.
        """

        dp_size = int(self.config.dp_size)
        # Round-robin attach across DP ranks (matches generate_rollout policy).
        prompts_by_dp = split_list_by_dp(prompts, dp_size)
        # prompt_offset is the absolute index of the first new prompt inside
        # the session's input space; preserves global ordering across attaches.
        prompt_indices_by_dp = split_list_by_dp(list(range(prompt_offset, prompt_offset + len(prompts))), dp_size)
        # broadcast = object send only, no result collection. The worker's
        # ongoing rollout loop will pick this command up between decode steps.
        self.cluster.broadcast(
            Op.INFER_ROLLOUT_ADD,
            RolloutAddPayload(
                session_id=int(session_id),
                prompts_by_dp=prompts_by_dp,
                prompt_indices_by_dp=prompt_indices_by_dp,
                cancel_flags=cancel_flags,
                cancel_indices_by_dp=split_list_by_dp(list(range(len(prompts))), dp_size) if cancel_flags is not None else None,
            ),
        )

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

    # Hard cap on prompts per chunk: every rank holds up to max_running_prompts
    # active sequences at once.
    max_chunk_size = max_running_prompts * dp_size
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


def _partial_callback_for_chunk(
    callback: Callable[[int, list[int], str], None],
    *,
    chunk_start: int,
    chunk_size: int,
    dp_size: int,
) -> Callable[[int, Any], None]:
    """Wrap a user callback to translate per-DP partial events to globals.

    The worker emits ``infer_rollout_partial`` payloads with chunk-local row
    indices (and optionally explicit ``prompt_indices``). This wrapper rewrites
    them back into the caller's global prompt index space so the public
    callback signature stays simple.
    """

    def _handle(_rank: int, payload: Any) -> None:
        # Filter out unrelated broadcast events.
        if not isinstance(payload, RolloutPartialPayload):
            return
        dp_rank = int(payload.dp_rank)
        rows = payload.rows
        response_ids = payload.response_ids
        finish_reason = payload.finish_reason
        prompt_indices = payload.prompt_indices
        if prompt_indices is not None:
            # Worker already provided absolute indices (continuous-batching
            # path): pass them through verbatim.
            for prompt_index, ids, reason in zip(prompt_indices, response_ids, finish_reason, strict=True):
                callback(int(prompt_index), ids, str(reason))
            return
        # Otherwise reconstruct the global index from the chunk-local row and
        # the DP rank using the same round-robin layout as split_list_by_dp.
        for row, ids, reason in zip(rows, response_ids, finish_reason, strict=True):
            index = chunk_start + int(row) * dp_size + dp_rank
            # Guard against padding rows that exceed the actual chunk size.
            if index < chunk_start + chunk_size:
                callback(index, ids, str(reason))

    return _handle


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
