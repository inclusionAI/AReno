"""Per-rank worker process for the areno TP/DP engine.

Each `ArenoWorker` owns one tensor-parallel shard of the model on a single
device and drives the four lifecycle phases of an RL step:

* rollout (prefill + paged-KV decode) with continuous batching;
* reference / critic / reward scoring via swap-in `WorkerRole`s;
* training step (FP32 master weights, packed or padded);
* KV-cache lifecycle (allocate, reset, scratch block, CUDA-graph capture)
  and weight onload/offload between train and infer states.

The engine driver dispatches `Command` objects to `handle()`, which fans out
to the matching public method.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from areno.engine.config import EngineConfig
from areno.engine.data import RolloutOutput
from areno.engine.inference import InferenceManager
from areno.engine.modeling import build_model_on_device, build_optimizer, param_grad
from areno.engine.protocol import Command, Op, SaveCheckpointPayload
from areno.engine.roles import RoleManager, WorkerRole
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

    def infer_rollout(self, payload: dict) -> RolloutOutput | None:
        """Delegate rollout generation to `InferenceManager`."""
        return self.inference.infer_rollout(payload)

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
