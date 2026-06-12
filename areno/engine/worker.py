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
from areno.engine.runtime.rollout import _empty_rollout
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
        self._current_request_ids: list[int | None] = []
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
        if cmd.op is Op.ROLLOUT_SESSION_SYNC:
            return self.rollout_session_sync(cmd.payload)
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

    def infer_rollout(self, payload: dict, finished_callback=None, refill_callback=None) -> RolloutOutput | None:
        """Delegate rollout generation to `InferenceManager`."""
        return self.inference.infer_rollout(
            payload,
            finished_callback=finished_callback,
            refill_callback=refill_callback,
        )

    def run_rollout_command(self, command: Command) -> list[tuple[int | None, RolloutOutput | None]]:
        """Run one rollout command and continuously refill from queued requests."""

        ctx = get_tp_context()
        payload = command.payload
        counts = [len(payload.prompts_by_dp[ctx.dp_rank])]
        request_ids = [command.request_id]
        return self.run_continuous_rollout_payload(payload, request_ids, counts)

    def run_continuous_rollout_payload(
        self,
        payload: RolloutPayload,
        request_ids: list[int | None],
        counts: list[int],
    ) -> list[tuple[int | None, RolloutOutput | None]]:
        """Run one rollout and append compatible queued requests while decoding."""

        ctx = get_tp_context()
        request_rows = _rollout_request_rows(counts)
        finished = [False] * sum(counts)
        finish_reasons = [""] * sum(counts)
        sent: set[int] = set()

        def send_empty_requests() -> None:
            for request_idx, count in enumerate(counts):
                if request_idx in sent or count != 0:
                    continue
                request_id = request_ids[request_idx]
                self._result_queue.put((self._rank, WorkerResult(ok=True, payload=_empty_rollout() if ctx.is_rank0 else None, request_id=request_id)))
                self._current_request_ids = [pending_id for pending_id in self._current_request_ids if pending_id != request_id]
                sent.add(request_idx)

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
            for request_idx, row_ids in enumerate(request_rows):
                if request_idx in sent or not all(finished[row] for row in row_ids):
                    continue
                result_payload = None
                if ctx.is_rank0:
                    result_payload = _build_rollout_from_tensor_row_ids(
                        payload.prompts_by_dp[ctx.dp_rank],
                        generated,
                        logprobs,
                        response_lens,
                        finish_reasons,
                        row_ids,
                        truncate_stop_token_ids,
                    )
                request_id = request_ids[request_idx]
                self._result_queue.put((self._rank, WorkerResult(ok=True, payload=result_payload, request_id=request_id)))
                self._current_request_ids = [pending_id for pending_id in self._current_request_ids if pending_id != request_id]
                sent.add(request_idx)

        send_empty_requests()

        def refill_waiting(state) -> list[int]:
            new_prompt_indices: list[int] = []
            while True:
                cmd = self._next_refill_command()
                if cmd is None:
                    break
                if cmd.op is not Op.INFER_ROLLOUT or not _rollout_payloads_compatible(payload, cmd.payload):
                    self._deferred_commands.append(cmd)
                    break
                new_payload = cmd.payload
                new_request_ids = [cmd.request_id]
                new_counts = [len(new_payload.prompts_by_dp[ctx.dp_rank])]
                prompts = [list(prompt) for prompt in new_payload.prompts_by_dp[ctx.dp_rank]]
                prompt_indices = list(new_payload.prompt_indices_by_dp[ctx.dp_rank])
                request_ids.extend(new_request_ids)
                counts.extend(new_counts)
                request_rows.append([])
                self._current_request_ids = [*self._current_request_ids, *new_request_ids]
                if not prompts:
                    send_empty_requests()
                    continue
                appended_rows = state.append_prompts(prompts)
                if payload.prompts_by_dp[ctx.dp_rank] is not state.prompts:
                    payload.prompts_by_dp[ctx.dp_rank].extend(prompts)
                payload.prompt_indices_by_dp[ctx.dp_rank].extend(prompt_indices)
                request_rows[-1] = appended_rows
                new_prompt_indices.extend(prompt_indices)
                finished.extend(False for _ in prompts)
                finish_reasons.extend("" for _ in prompts)
            return new_prompt_indices

        output = self.infer_rollout(payload, finished_callback=send_finished, refill_callback=refill_waiting)
        parts = _split_rollout_output_by_rows(output, request_rows)
        return [(request_id, part) for idx, (request_id, part) in enumerate(zip(request_ids, parts, strict=True)) if idx not in sent]

    def _next_refill_command(self) -> Command | None:
        """Fetch the next queued command consistently across TP ranks."""

        ctx = get_tp_context()
        cmd = None
        if ctx.is_rank0:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                cmd = None
        command_header = self._broadcast_tp_command_header(cmd)
        if command_header is None:
            return None
        if not ctx.is_rank0:
            # Rank 0 picked the command that the TP group will execute next.
            # Sibling ranks may have stale deferred commands after prior async
            # returns, so they must consume until the same request id appears
            # instead of blindly taking the next local queue item.
            cmd = self._pop_matching_refill_command(*command_header)
        return cmd

    def _broadcast_tp_command_header(self, cmd: Command | None) -> tuple[Op, int | None] | None:
        """Broadcast the TP-rank0 refill command identity."""

        ctx = get_tp_context()
        request_id = -1
        op_value = -1
        has_command = 0
        if cmd is not None:
            has_command = 1
            op_value = int(cmd.op.value)
            request_id = -1 if cmd.request_id is None else int(cmd.request_id)
        header = torch.tensor([has_command, op_value, request_id], device=ctx.device, dtype=torch.long)
        if ctx.world_size > 1:
            src = ctx.dp_rank * ctx.world_size
            dist.broadcast(header, src=src, group=ctx.group)
        if int(header[0].item()) == 0:
            return None
        return Op(int(header[1].item())), None if int(header[2].item()) < 0 else int(header[2].item())

    def _pop_matching_refill_command(self, op: Op, request_id: int | None) -> Command:
        """Consume local commands until the TP-rank0 command is found."""

        while True:
            cmd = self._cmd_queue.get(timeout=5.0)
            if cmd.op is op and cmd.request_id == request_id:
                return cmd
            self._deferred_commands.append(cmd)

    def rollout_session_begin(self, payload: None) -> None:
        """Prepare actor state for one or more rollout calls."""

        del payload
        self._prepare_actor_onloaded()

    def rollout_session_sync(self, payload: None) -> None:
        """Synchronize TP ranks before agentic request-driven rollout starts."""

        del payload
        ctx = get_tp_context()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if ctx.group is not None:
            if ctx.device.type == "cuda":
                dist.barrier(group=ctx.group, device_ids=[ctx.device.index if ctx.device.index is not None else torch.cuda.current_device()])
            else:
                dist.barrier(group=ctx.group)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

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


def _rollout_payloads_compatible(first: RolloutPayload, other: RolloutPayload) -> bool:
    """Return whether two rollout payloads can share one InferenceBatchState."""

    if other.cancel_flags is not None:
        return False
    return (
        first.max_new_tokens == other.max_new_tokens
        and first.max_cache_len >= other.max_cache_len
        and first.max_blocks_per_seq >= other.max_blocks_per_seq
        and first.eos_token_id == other.eos_token_id
        and first.sampling_params == other.sampling_params
        and first.block_size == other.block_size
        and first.decode_progress_interval_s == other.decode_progress_interval_s
        and first.cancel_flags is None
        and other.cancel_flags is None
        and first.cancel_indices_by_dp is None
        and other.cancel_indices_by_dp is None
    )


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


def _split_rollout_output_by_rows(output: RolloutOutput | None, request_rows: list[list[int]]) -> list[RolloutOutput | None]:
    """Split a merged worker rollout by explicit state row ids."""

    if output is None:
        return [None for _ in request_rows]
    return [_slice_rollout_output_rows(output, rows) for rows in request_rows]


def _rollout_request_rows(counts: list[int]) -> list[list[int]]:
    """Return explicit state row ids for each request."""

    rows = []
    offset = 0
    for count in counts:
        end = offset + count
        rows.append(list(range(offset, end)))
        offset = end
    return rows


def _rollout_ranges(counts: list[int]) -> list[tuple[int, int]]:
    """Return half-open row ranges for each request in the active rollout."""

    ranges = []
    offset = 0
    for count in counts:
        end = offset + count
        ranges.append((offset, end))
        offset = end
    return ranges


def _build_rollout_from_tensor_row_ids(
    prompts: list[list[int]],
    generated: torch.Tensor,
    logprobs: torch.Tensor,
    response_lens: torch.Tensor,
    finish_reasons: list[str],
    row_ids: list[int],
    truncate_stop_token_ids: tuple[int, ...],
) -> RolloutOutput:
    """Build a RolloutOutput for non-contiguous completed tensor rows."""

    if not row_ids:
        return _empty_rollout()
    prompt_ids = [prompts[row] for row in row_ids]
    lengths = response_lens[row_ids].detach().cpu().tolist()
    generated_rows = [row[: int(length)] for row, length in zip(generated[row_ids].detach().cpu().tolist(), lengths, strict=True)]
    response_ids, truncated_finish_reasons = _truncate_generated(generated_rows, truncate_stop_token_ids)
    finish_reason = [finish_reasons[row] or truncated_finish_reasons[idx] for idx, row in enumerate(row_ids)]
    logprob_rows_cpu = logprobs[row_ids].detach().cpu().tolist()
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


def _slice_rollout_output_rows(output: RolloutOutput, rows: list[int]) -> RolloutOutput:
    """Build a RolloutOutput view for explicit row ids."""

    if not rows:
        return _empty_rollout()
    prompt_ids = [output.prompt_ids[row] for row in rows]
    response_ids = [output.response_ids[row] for row in rows]
    finish_reason = [output.finish_reason[row] for row in rows]
    logprob_rows = [output.logprobs[row, : len(output.response_ids[row])].detach().cpu() for row in rows]
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
