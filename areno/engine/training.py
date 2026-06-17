"""Actor training manager for `ArenoWorker`."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from areno.engine.data import to_device
from areno.engine.modeling import param_grad
from areno.engine.parallel.context import get_tp_context
from areno.engine.protocol import TrainPayload
from areno.engine.runtime.logprobs import next_token_logprobs, packed_next_token_logprobs
from areno.engine.runtime.train_step import (
    _clip_grad_norm,
    _grad_norm,
    _grad_zero_metrics,
    _merge_metrics,
    _pack_train_data,
    _train_meta,
)


class TrainingManager:
    """Own actor forward/backward, gradient sync, and optimizer stepping."""

    def __init__(self, worker):
        self.worker = worker

    def train(self, payload: TrainPayload) -> list[dict | None]:
        """Run all microbatches for one actor optimizer step."""

        worker = self.worker
        packs = payload.data_packs_by_dp
        if not isinstance(packs, list):
            raise TypeError("TRAIN payload must contain a list data_packs_by_dp")
        accumulation_steps = payload.gradient_accumulation_steps
        accumulation_steps = len(packs) if accumulation_steps is None else max(int(accumulation_steps), 1)
        worker.optimizer.zero_grad(set_to_none=True)
        results = []
        try:
            for index, data_pack_shards in enumerate(packs):
                group_start = (index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(packs) - group_start)
                allow_step = (index + 1) % accumulation_steps == 0 or index == len(packs) - 1
                results.append(
                    self._train_step(
                        data_pack_shards,
                        allow_step=allow_step,
                        grad_scale=group_size,
                    )
                )
            return results
        finally:
            if not worker.config.runtime.keep_rollout_state:
                worker.optimizer.offload_state()
                if worker.device.type == "cuda":
                    torch.cuda.empty_cache()

    def _train_step(self, data_pack_shards: list[dict], *, allow_step: bool, grad_scale: int) -> dict | None:
        """Run a single actor forward + backward microbatch."""

        worker = self.worker
        ctx = get_tp_context()
        if not worker._train_state_ready:
            worker._prepare_for_train()
        worker.model.train()
        data_pack_obj = data_pack_shards[ctx.dp_rank]
        data_pack = _pack_train_data(data_pack_obj)
        data_pack = to_device(data_pack, worker.device)
        data_pack["_sequence_parallel_enabled"] = worker.config.model.sequence_parallel
        data_pack["_activation_checkpointing_enabled"] = worker.config.runtime.activation_checkpointing
        tokens = data_pack["input_ids"].long()
        position_ids = data_pack.get("position_ids")
        out = worker.model(input_ids=tokens, position_ids=position_ids, train_meta=_train_meta(data_pack, tokens))
        if "train_cu_seqlens" in data_pack:
            logprobs = packed_next_token_logprobs(out.logits_shard, tokens, data_pack["train_cu_seqlens"])
        else:
            logprobs = next_token_logprobs(out.logits_shard, tokens)
        loss_out = worker.loss_fn(data_pack, logprobs)
        metrics = None
        if isinstance(loss_out, tuple):
            loss, metrics = loss_out
        else:
            loss = loss_out
        if not isinstance(loss, torch.Tensor):
            raise TypeError("train_loss_fn must return a torch.Tensor")
        (loss / max(grad_scale, 1)).backward()
        self._accumulate_main_gradients()
        stepped = allow_step
        grad_norm = None
        grad_zero_metrics = None
        if stepped:
            self._sync_data_parallel_gradients()
            self._sync_tensor_parallel_replicated_gradients()
            self._finalize_router_expert_bias()
            grad_norm = _grad_norm(worker.model.parameters())
            grad_zero_metrics = _grad_zero_metrics(worker.model.parameters())
            if worker.grad_clip_norm is not None:
                _clip_grad_norm(worker.model.parameters(), grad_norm, worker.grad_clip_norm)
            current_lr = self._lr_for_step(worker._global_step + 1)
            worker.optimizer.lr = current_lr
            worker.optimizer.step()
            worker.optimizer.zero_grad(set_to_none=True)
            worker._global_step += 1
            if worker.device.type == "cuda" and worker._global_step % 10 == 0:
                torch.cuda.empty_cache()
        else:
            current_lr = worker.optimizer.lr
        if ctx.is_rank0:
            return {
                "loss": float(loss.detach().cpu()),
                "stepped": stepped,
                "global_step": worker._global_step,
                "metrics": _merge_metrics(
                    metrics,
                    None,
                    {"lr": current_lr},
                    {"grad_norm": grad_norm} if grad_norm is not None else None,
                    grad_zero_metrics,
                ),
            }
        return None

    def _lr_for_step(self, step: int) -> float:
        """Compute the actor learning rate for a given optimizer step."""

        worker = self.worker
        if worker.lr_warmup_steps > 0 and step <= worker.lr_warmup_steps:
            return worker.base_lr * step / worker.lr_warmup_steps
        if worker.lr_decay_style == "constant" or worker.lr_decay_steps <= 0:
            return worker.base_lr
        decay_step = step - worker.lr_warmup_steps
        decay_steps = max(worker.lr_decay_steps - worker.lr_warmup_steps, 1)
        progress = min(max(decay_step / decay_steps, 0.0), 1.0)
        if worker.lr_decay_style == "linear":
            coeff = 1.0 - progress
        elif worker.lr_decay_style == "cosine":
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"unsupported lr_decay_style: {worker.lr_decay_style}")
        return worker.min_lr + coeff * (worker.base_lr - worker.min_lr)

    def _sync_data_parallel_gradients(self) -> None:
        """Average actor gradients across data-parallel replicas."""

        worker = self.worker
        ctx = get_tp_context()
        if ctx.dp_size == 1:
            return
        for param in worker.model.parameters():
            grad = param_grad(param)
            if grad is None:
                continue
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.dp_group)
            grad.div_(ctx.dp_size)

    def _sync_tensor_parallel_replicated_gradients(self) -> None:
        """Sum TP-replicated actor gradients with shard-local contributions."""

        worker = self.worker
        ctx = get_tp_context()
        if ctx.world_size == 1:
            return
        for param in worker.model.parameters():
            grad = param_grad(param)
            if grad is None or not bool(getattr(param, "tp_grad_allreduce", False)):
                continue
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)

    def _accumulate_main_gradients(self) -> None:
        """Move actor autograd `.grad` into the FP32 master accumulator."""

        for param in self.worker.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            main_grad = getattr(param, "main_grad", None)
            if not isinstance(main_grad, torch.Tensor):
                param.main_grad = grad.to(dtype=torch.float32)
            else:
                main_grad.add_(grad.to(dtype=main_grad.dtype))
            param.grad = None

    def _finalize_router_expert_bias(self) -> None:
        """Apply MoE router/expert bias corrections after gradient sync."""

        ctx = get_tp_context()
        self.worker.model.finalize_router_expert_bias(ctx.group, ctx.dp_group)
