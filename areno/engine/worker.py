"""Per-rank worker process for the areno TP/DP engine.

Each `ArenoWorker` owns one tensor-parallel shard of the model on a single
device and drives the four lifecycle phases of an RL step:

* rollout (prefill + paged-KV decode);
* reference / critic / reward scoring via swap-in `WorkerRole`s;
* training step (FP32 master weights, packed or padded);
* KV-cache lifecycle (allocate, reset, scratch block, CUDA-graph capture)
  and weight onload/offload between train and infer states.

The engine driver dispatches `Command` objects to `handle()`, which fans out
to the matching public method.
"""

from __future__ import annotations

import queue
import time
from dataclasses import replace

import torch
import torch.distributed as dist

from areno.engine.config import EngineConfig
from areno.engine.data import RolloutOutput
from areno.engine.inference import InferenceManager
from areno.engine.modeling import build_model_on_device, build_optimizer, param_grad
from areno.engine.data.sampling import _truncate_generated
from areno.engine.protocol import Command, Op, RolloutPayload, SaveCheckpointPayload, WorkerResult
from areno.engine.roles import RoleManager, WorkerRole
from areno.engine.runtime.common import pad_rollout_rows
from areno.engine.runtime.rollout import _empty_rollout, partial_tail_threshold
from areno.engine.training import TrainingManager
from areno.engine.parallel.context import get_tp_context
from areno.models.registry import load_model_weights, save_model_weights
from areno.engine.runtime.decode_graph import DecodeGraph

class ArenoWorker:
    """Single-rank executor for model work.

    The worker owns exactly one model shard and one optimizer shard. Before
    rollout it prepares inference weights and may offload training-only weights;
    before training it reloads the authoritative train weights for backward and
    optimizer updates.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        ctx = get_tp_context()
        self.device = ctx.device
        # Build the actor model directly on the shard's device, then wrap in
        # torch.compile so subsequent forward calls use the compiled graph.
        self.model = build_model_on_device(config, self.device)
        if config.model_path is not None and not config.dummy_load:
            load_model_weights(self.model, config.model, config.model_path)
        self.model = torch.compile(self.model)
        opt = config.optimizer
        self.optimizer = build_optimizer(self.model.parameters(), opt, ctx)
        self.grad_clip_norm = opt.grad_clip_norm
        self.base_lr = opt.lr
        self.min_lr = opt.min_lr
        self.lr_decay_steps = opt.lr_decay_steps
        self.lr_warmup_steps = opt.lr_warmup_steps
        self.lr_decay_style = opt.lr_decay_style
        self._global_step = 0
        # Paged-KV state: refreshed when the rollout spec changes.
        self._infer_batch_size = 0  # max concurrent sequences supported
        self._infer_cache_blocks = 0  # num_blocks + 1 (extra is scratch)
        self._scratch_block = 0  # index of the scratch block (last slot)
        self._max_cache_len = 0
        self._max_blocks_per_seq = 0
        # Per-bucket captured decode CUDA graphs; buckets that OOM during
        # capture get tracked in `_skipped` and fall back to eager forward.
        self._decode_graphs: dict[int, DecodeGraph] = {}
        self._decode_graph_skipped_buckets: set[int] = set()
        self._decode_graph_init_attempted = False
        # 5-tuple summarising the active cache config; used to decide whether
        # a new `_init_infer_cache` call can reuse the existing allocation.
        self._infer_cache_spec: tuple[int, int, int, int, int] | None = None
        self._train_state_ready = False
        self._actor_on_device = True
        self.inference = InferenceManager(self)
        self.roles = RoleManager(self)
        self.training = TrainingManager(self)
        if config.train_loss_fn is None:
            raise ValueError("ArenoEngine requires train_loss_fn")
        self.loss_fn = config.train_loss_fn

    def handle(self, cmd: Command):
        """Dispatch a `Command` to the matching method on this worker."""
        if cmd.op is Op.ENSURE_ROLES:
            return self.ensure_roles(cmd.payload)
        if cmd.op is Op.INFER_ROLLOUT:
            return self.infer_rollout(cmd.payload)
        if cmd.op is Op.ROLLOUT_SESSION_BEGIN:
            return self.rollout_session_begin(cmd.payload)
        if cmd.op is Op.ROLLOUT_SESSION_END:
            return self.rollout_session_end(cmd.payload)
        if cmd.op is Op.TRAIN:
            return self.train(cmd.payload)
        if cmd.op is Op.SCORE_LOGPROBS:
            return self.score_logprobs(cmd.payload)
        if cmd.op is Op.SCORE_VALUES:
            return self.score_values(cmd.payload)
        if cmd.op is Op.SCORE_REWARDS:
            return self.score_rewards(cmd.payload)
        if cmd.op is Op.TRAIN_VALUES:
            return self.train_values(cmd.payload)
        if cmd.op is Op.SAVE_CHECKPOINT:
            return self.save_checkpoint(cmd.payload)
        raise ValueError(f"unsupported areno op: {cmd.op}")

    def ensure_roles(self, payload: dict) -> None:
        """Delegate non-actor role lifecycle to `RoleManager`."""
        return self.roles.ensure_roles(payload)

    def infer_rollout(self, payload: dict, finished_callback=None, partial_tail_ready_callback=None) -> RolloutOutput | None:
        """Delegate rollout generation to `InferenceManager`."""
        return self.inference.infer_rollout(
            payload,
            finished_callback=finished_callback,
            partial_tail_ready_callback=partial_tail_ready_callback,
        )

    def collect_rollout_commands(self, first_cmd: Command) -> list[Command]:
        """Collect compatible commands until the worker batch fills or reaches DDL."""

        first_payload = first_cmd.payload
        if not _rollout_coalesce_enabled(first_payload):
            return [first_cmd]
        commands = [first_cmd]
        queued_count = _rollout_global_count(first_payload)
        target = _rollout_coalesce_target(first_payload) * len(first_payload.prompts_by_dp)
        deadline = time.monotonic() + float(first_payload.coalesce_timeout_s)
        while queued_count < target:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                cmd = self._cmd_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if cmd.op is not Op.INFER_ROLLOUT or not _rollout_payloads_compatible(first_payload, cmd.payload):
                self._deferred_commands.append(cmd)
                break
            commands.append(cmd)
            queued_count += _rollout_global_count(cmd.payload)
        return commands

    def run_rollout_commands(self, commands: list[Command]) -> list[tuple[int | None, RolloutOutput | None]]:
        """Run one coalesced rollout and split outputs back by request id."""

        ctx = get_tp_context()
        if len(commands) == 1:
            payload = commands[0].payload
            return [
                (
                    commands[0].request_id,
                    self.infer_rollout(
                        payload,
                        partial_tail_ready_callback=lambda active_count: self._has_enough_waiting_rollouts(payload, active_count),
                    ),
                )
            ]
        merged, counts = _merge_rollout_payloads([cmd.payload for cmd in commands], ctx.dp_rank)
        request_ids = [cmd.request_id for cmd in commands]
        return self.run_coalesced_rollout_payload(merged, request_ids, counts)

    def run_coalesced_rollout_payload(
        self,
        payload: RolloutPayload,
        request_ids: list[int | None],
        counts: list[int],
    ) -> list[tuple[int | None, RolloutOutput | None]]:
        """Run one coordinator-coalesced rollout and split outputs by request id."""

        ctx = get_tp_context()
        ranges = _rollout_ranges(counts)
        finished = [False] * sum(counts)
        finish_reasons = [""] * sum(counts)
        sent: set[int] = set()

        def send_finished(
            rows: torch.Tensor,
            generated: torch.Tensor,
            logprobs: torch.Tensor,
            response_lens: torch.Tensor,
            finish_reason: str,
            truncate_stop_token_ids: tuple[int, ...],
        ) -> None:
            for row in rows.detach().cpu().tolist():
                row_idx = int(row)
                if 0 <= row_idx < len(finished):
                    finished[row_idx] = True
                    finish_reasons[row_idx] = finish_reason
            for request_idx, (start, end) in enumerate(ranges):
                if request_idx in sent or not all(finished[start:end]):
                    continue
                result_payload = None
                if ctx.is_rank0:
                    result_payload = _build_rollout_from_tensor_rows(
                        payload.prompts_by_dp[ctx.dp_rank],
                        generated,
                        logprobs,
                        response_lens,
                        finish_reasons,
                        start,
                        end,
                        truncate_stop_token_ids,
                    )
                request_id = request_ids[request_idx]
                self._result_queue.put((self._rank, WorkerResult(ok=True, payload=result_payload, request_id=request_id)))
                current_request_ids = getattr(self, "_current_request_ids", None)
                if current_request_ids is not None:
                    self._current_request_ids = [pending_id for pending_id in current_request_ids if pending_id != request_id]
                sent.add(request_idx)

        output = self.infer_rollout(
            payload,
            finished_callback=send_finished,
            partial_tail_ready_callback=lambda active_count: self._has_enough_waiting_rollouts(payload, active_count),
        )
        parts = _split_rollout_output(output, counts)
        return [(request_id, part) for idx, (request_id, part) in enumerate(zip(request_ids, parts, strict=True)) if idx not in sent]

    def _has_enough_waiting_rollouts(self, payload: RolloutPayload, active_count: int) -> bool:
        """Return true when queued compatible requests can usefully refill the tail."""

        ctx = get_tp_context()
        ready = False
        if ctx.is_rank0:
            ready = self._waiting_rollout_count(payload, active_count) >= _rollout_refill_target(payload, active_count)
        if ctx.group is not None:
            flag = [ready]
            dist.broadcast_object_list(flag, src=ctx.dp_rank * ctx.world_size, group=ctx.group)
            ready = bool(flag[0])
        return ready

    def _waiting_rollout_count(self, payload: RolloutPayload, active_count: int) -> int:
        """Count compatible waiting rollout rows, draining queue entries to deferred."""

        count = 0
        for cmd in self._deferred_commands:
            if cmd.op is Op.INFER_ROLLOUT and _rollout_payloads_compatible(payload, cmd.payload):
                count += _rollout_global_count(cmd.payload)
                continue
            return count
        target = _rollout_refill_target(payload, active_count)
        while count < target:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            self._deferred_commands.append(cmd)
            if cmd.op is not Op.INFER_ROLLOUT or not _rollout_payloads_compatible(payload, cmd.payload):
                break
            count += _rollout_global_count(cmd.payload)
        return count

    def rollout_session_begin(self, payload: None) -> None:
        """Prepare actor state for one or more rollout calls."""

        del payload
        self._prepare_actor_onloaded()

    def rollout_session_end(self, payload: None) -> None:
        """Finalize rollout state before scoring or training starts."""

        del payload
        if not self.config.runtime.keep_rollout_state:
            self._drop_rollout_hbm()
        self._prepare_for_train()

    def _prepare_for_train(self) -> None:
        """Ensure the actor is on-device and train weights are loaded."""
        self._prepare_actor_onloaded()
        self.model.onload_train_weights(self.device)
        self._train_state_ready = True

    def _prepare_actor_onloaded(self) -> None:
        """Move the actor model + optimizer state back to `device` if offloaded."""
        if self._actor_on_device:
            return
        self.model.to(self.device)
        self.model.onload_train_weights(self.device)
        self.optimizer.onload_state(self.device)
        self._actor_on_device = True

    def _prepare_actor_offloaded(self) -> None:
        """Push the actor to CPU and drop all HBM state, including decode graphs.

        Decode graphs and the KV cache are tied to specific HBM allocations,
        so offloading invalidates them and a future rollout must re-init.
        """
        if not self._actor_on_device:
            return
        self._release_decode_graphs()
        self._infer_cache_spec = None
        self.model.clear_infer_weights()
        self.model.clear_kv_caches()
        self.model.offload_train_weights()
        self.model.to("cpu")
        self.optimizer.offload_state()
        self._train_state_ready = False
        self._actor_on_device = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _release_decode_graphs(self) -> None:
        """Drop captured decode CUDA graphs and release their cached memory."""

        self._decode_graphs.clear()
        self._decode_graph_skipped_buckets.clear()
        self._decode_graph_init_attempted = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _drop_rollout_hbm(self) -> None:
        """Release rollout-only GPU state while keeping CPU-reloadable handles."""

        self._release_decode_graphs()
        self.model.clear_infer_weights()
        offload_kv = getattr(self.model, "offload_kv_caches", None)
        if offload_kv is not None:
            offload_kv()
        self._train_state_ready = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def score_logprobs(self, payload: dict) -> list[list[float]] | None:
        """Delegate logprob scoring to `RoleManager`."""
        return self.roles.score_logprobs(payload)

    @torch.inference_mode()
    def score_values(self, payload: dict) -> list[list[float]] | None:
        """Delegate value scoring to `RoleManager`."""
        return self.roles.score_values(payload)

    @torch.inference_mode()
    def score_rewards(self, payload: dict) -> list[float] | None:
        """Delegate reward scoring to `RoleManager`."""
        return self.roles.score_rewards(payload)

    def train_values(self, payload: dict) -> dict | None:
        """Delegate critic value training to `RoleManager`."""
        return self.roles.train_values(payload)

    def train(self, payload: dict) -> list[dict | None]:
        """Delegate actor training to `TrainingManager`."""
        return self.training.train(payload)

    def _sync_role_grads(self, role: WorkerRole) -> None:
        """Sync a role's gradients across DP and TP groups.

        DP: average. TP-replicated value-head params (`role_tp_average=True`)
        get averaged across the whole world; TP-sharded params marked with
        `tp_grad_allreduce` get summed.
        """
        ctx = get_tp_context()
        if ctx.dp_size > 1:
            for param in role.parameters():
                grad = param_grad(param)
                if grad is not None:
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.dp_group)
                    grad.div_(ctx.dp_size)
        if ctx.world_size > 1:
            for param in role.parameters():
                grad = param_grad(param)
                if grad is None:
                    continue
                if bool(getattr(param, "role_tp_average", False)):
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
                    grad.div_(ctx.world_size)
                elif bool(getattr(param, "tp_grad_allreduce", False)):
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)

    def save_checkpoint(self, payload: SaveCheckpointPayload) -> dict | None:
        """Persist the actor's weights to disk (rank 0 returns the resolved path)."""
        self._prepare_actor_onloaded()
        path = save_model_weights(self.model, self.config.model, payload.path, self.config.model_path)
        return {"path": path} if path is not None else None


def _rollout_coalesce_enabled(payload: RolloutPayload) -> bool:
    """Return whether this rollout payload can wait for worker-local batching."""

    return bool(float(payload.coalesce_timeout_s) > 0.0 and payload.cancel_flags is None)


def _rollout_local_count(payload: RolloutPayload, dp_rank: int) -> int:
    """Number of prompt rows this payload asks the current DP rank to run."""

    return len(payload.prompts_by_dp[dp_rank])


def _rollout_global_count(payload: RolloutPayload) -> int:
    """Number of prompt rows in the request, independent of current DP rank."""

    return sum(len(rows) for rows in payload.prompts_by_dp)


def _rollout_coalesce_target(payload: RolloutPayload) -> int:
    """Worker-local batch target for coalescing compatible rollout commands."""

    return max(int(payload.coalesce_max_running_seqs or payload.max_running_seqs), 1)


def _rollout_refill_target(payload: RolloutPayload, active_count: int) -> int:
    """Global queued-row count needed before cutting the current active tail."""

    del active_count
    local_target = _rollout_coalesce_target(payload)
    lower = max(1, (local_target + 7) // 8)
    local_needed = max(lower, local_target // 2)
    return local_needed * len(payload.prompts_by_dp)


def _rollout_payloads_compatible(first: RolloutPayload, other: RolloutPayload) -> bool:
    """Return whether two rollout payloads can share one InferenceBatchState."""

    if other.cancel_flags is not None:
        return False
    return (
        first.max_new_tokens == other.max_new_tokens
        and first.eos_token_id == other.eos_token_id
        and first.sampling_params == other.sampling_params
        and first.block_size == other.block_size
        and first.decode_progress_interval_s == other.decode_progress_interval_s
        and first.cancel_flags is None
        and other.cancel_flags is None
        and first.cancel_indices_by_dp is None
        and other.cancel_indices_by_dp is None
    )


def _merge_rollout_payloads(payloads: list[RolloutPayload], current_dp_rank: int) -> tuple[RolloutPayload, list[int]]:
    """Merge compatible rollout payloads into one worker batch."""

    first = payloads[0]
    dp_size = len(first.prompts_by_dp)
    prompts_by_dp = [[] for _ in range(dp_size)]
    prompt_indices_by_dp = [[] for _ in range(dp_size)]
    cooldown_until_by_dp = [[] for _ in range(dp_size)]
    counts_by_dp = [[0 for _ in payloads] for _ in range(dp_size)]
    row_idx = 0
    for request_idx, payload in enumerate(payloads):
        for prompt, prompt_index, cooldown_until_s in _iter_rollout_payload_rows(payload):
            dp_rank = row_idx % dp_size
            prompts_by_dp[dp_rank].append(prompt)
            prompt_indices_by_dp[dp_rank].append(prompt_index)
            cooldown_until_by_dp[dp_rank].append(cooldown_until_s)
            counts_by_dp[dp_rank][request_idx] += 1
            row_idx += 1
    local_capacity = max(int(first.coalesce_max_running_seqs or first.max_running_seqs), 1)
    max_running_seqs = max(max((len(rows) for rows in prompts_by_dp), default=0), local_capacity)
    max_cache_len = max(payload.max_cache_len for payload in payloads)
    max_blocks_per_seq = max(payload.max_blocks_per_seq for payload in payloads)
    max_prefill_tokens = max(payload.max_prefill_tokens for payload in payloads)
    return replace(
        first,
        prompts_by_dp=prompts_by_dp,
        prompt_indices_by_dp=prompt_indices_by_dp,
        max_running_seqs=max_running_seqs,
        max_cache_len=max_cache_len,
        max_blocks_per_seq=max_blocks_per_seq,
        max_prefill_tokens=max_prefill_tokens,
        num_blocks=max_running_seqs * int(max_blocks_per_seq),
        partial_tail_threshold=partial_tail_threshold(local_capacity, float(first.coalesce_timeout_s)),
        partial_tail_min_tokens=max(int(payload.partial_tail_min_tokens) for payload in payloads),
        partial_tail_cooldown_until_s_by_dp=cooldown_until_by_dp,
    ), counts_by_dp[current_dp_rank]


def _iter_rollout_payload_rows(payload: RolloutPayload):
    """Yield prompt rows in the payload's original pre-DP-split order."""

    dp_size = len(payload.prompts_by_dp)
    total = _rollout_global_count(payload)
    for original_idx in range(total):
        dp_rank = original_idx % dp_size
        local_idx = original_idx // dp_size
        cooldown_until_s = 0.0
        if payload.partial_tail_cooldown_until_s_by_dp is not None:
            cooldown_until_s = float(payload.partial_tail_cooldown_until_s_by_dp[dp_rank][local_idx])
        yield payload.prompts_by_dp[dp_rank][local_idx], payload.prompt_indices_by_dp[dp_rank][local_idx], cooldown_until_s


def _split_rollout_output(output: RolloutOutput | None, counts: list[int]) -> list[RolloutOutput | None]:
    """Split a merged worker rollout back into per-request outputs."""

    if output is None:
        return [None for _ in counts]
    parts = []
    offset = 0
    for count in counts:
        end = offset + count
        parts.append(_slice_rollout_output(output, offset, end))
        offset = end
    return parts


def _rollout_ranges(counts: list[int]) -> list[tuple[int, int]]:
    """Return half-open row ranges for each coalesced request."""

    ranges = []
    offset = 0
    for count in counts:
        end = offset + count
        ranges.append((offset, end))
        offset = end
    return ranges


def _build_rollout_from_tensor_rows(
    prompts: list[list[int]],
    generated: torch.Tensor,
    logprobs: torch.Tensor,
    response_lens: torch.Tensor,
    finish_reasons: list[str],
    start: int,
    end: int,
    truncate_stop_token_ids: tuple[int, ...],
) -> RolloutOutput:
    """Build a RolloutOutput for completed tensor rows before the full batch ends."""

    if start == end:
        return _empty_rollout()
    prompt_ids = prompts[start:end]
    lengths = response_lens[start:end].detach().cpu().tolist()
    generated_rows = [row[: int(length)] for row, length in zip(generated[start:end].detach().cpu().tolist(), lengths, strict=True)]
    response_ids, truncated_finish_reasons = _truncate_generated(generated_rows, truncate_stop_token_ids)
    finish_reason = [finish_reasons[idx] or truncated_finish_reasons[idx - start] for idx in range(start, end)]
    logprob_rows_cpu = logprobs[start:end].detach().cpu().tolist()
    logprob_rows = [torch.tensor(row[: len(response)], dtype=torch.float32) for row, response in zip(logprob_rows_cpu, response_ids, strict=True)]
    input_ids, attention_mask, response_mask, padded_logprobs = pad_rollout_rows(prompt_ids, response_ids, logprob_rows)
    return RolloutOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        logprobs=padded_logprobs,
        finish_reason=finish_reason,
        metrics=None,
    )


def _slice_rollout_output(output: RolloutOutput, start: int, end: int) -> RolloutOutput:
    """Build a RolloutOutput view for rows [start, end)."""

    if start == end:
        return _empty_rollout()
    prompt_ids = output.prompt_ids[start:end]
    response_ids = output.response_ids[start:end]
    finish_reason = output.finish_reason[start:end]
    logprob_rows = [output.logprobs[idx, : len(output.response_ids[idx])].detach().cpu() for idx in range(start, end)]
    input_ids, attention_mask, response_mask, logprobs = pad_rollout_rows(prompt_ids, response_ids, logprob_rows)
    return RolloutOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        logprobs=logprobs,
        finish_reason=finish_reason,
        metrics=output.metrics,
    )
