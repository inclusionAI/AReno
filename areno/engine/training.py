"""Actor training manager for `ArenoWorker`."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from areno.engine.data import to_device
from areno.engine.modeling import param_grad, unwrap_model
from areno.engine.parallel.context import get_tp_context
from areno.engine.protocol import TrainPayload
from areno.engine.runtime.logprobs import (
    packed_next_token_logprobs,
    packed_next_token_logprobs_from_hidden,
)
from areno.engine.runtime.routing_replay import routing_replay_context
from areno.engine.runtime.train_step import (
    _clip_grad_norm,
    _grad_norms,
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
        offload_mode = getattr(worker.config.runtime, "optimizer_state_offload", "none")
        if isinstance(offload_mode, bool):
            offload_mode = "cpu" if offload_mode else "none"
        if offload_mode == "none" and not worker.config.runtime.keep_rollout_state:
            # Preserve --drop-rollout-state's historical CPU optimizer offload.
            offload_mode = "cpu"
        offload_directory = getattr(worker.config.runtime, "optimizer_state_offload_dir", None)
        offload_batch_size = getattr(worker.config.runtime, "optimizer_state_offload_batch_size", 1)
        if not worker._train_state_ready:
            # onload_state() clears residency policy, so prepare the actor
            # before configuring first-step disk streaming.
            worker._prepare_for_train()
        if offload_mode != "none":
            # Disk streaming must be active before the first optimizer step;
            # otherwise step zero materializes every bucket on the GPU first.
            worker.optimizer.configure_state_offload(
                mode=offload_mode,
                directory=offload_directory,
                batch_size=offload_batch_size,
            )
            if offload_mode == "disk":
                prefetch_state = getattr(worker.optimizer, "prefetch_state", None)
                if callable(prefetch_state):
                    # Prime a two-bucket window while forward/backward runs;
                    # step() keeps that bounded lookahead moving.
                    prefetch_state()
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
            if offload_mode != "none":
                worker.optimizer.offload_state(
                    mode=offload_mode,
                    directory=offload_directory,
                    batch_size=offload_batch_size,
                )
                if worker.device.type == "cuda":
                    torch.cuda.empty_cache()

    def _train_step(
        self,
        data_pack_shards: list[dict],
        *,
        allow_step: bool,
        grad_scale: int,
    ) -> dict | None:
        """Run a single actor forward + backward microbatch."""

        worker = self.worker
        ctx = get_tp_context()
        if not worker._train_state_ready:
            worker._prepare_for_train()
        train_model = _actor_train_model(worker)
        train_model.train()
        data_pack = dict(data_pack_shards[ctx.dp_rank])
        data_pack = _pack_train_data(data_pack)
        pack_loss_fn = data_pack.get("_loss_fn")
        auto_tune_probe = callable(pack_loss_fn) and getattr(pack_loss_fn, "__name__", "") == "_dummy_policy_loss"
        if auto_tune_probe and worker.device.type == "cuda":
            torch.cuda.synchronize(worker.device)
            torch.cuda.reset_peak_memory_stats(worker.device)
        data_pack = to_device(data_pack, worker.device)
        data_pack["_activation_checkpointing_enabled"] = worker.config.runtime.activation_checkpointing
        tokens = data_pack["input_ids"].long()
        position_ids = data_pack.get("position_ids")
        train_meta = _train_meta(
            data_pack,
            tokens,
            sequence_parallel=worker.config.effective_sequence_parallel,
        )
        model_kwargs = {
            "input_ids": tokens,
            "position_ids": position_ids,
            "train_meta": train_meta,
        }
        if data_pack.get("features") is not None:
            model_kwargs["features"] = data_pack["features"]
        defer_lm_head = (
            worker.config.model.model_type == "gemma4" and ctx.world_size == 1 and "train_cu_seqlens" in data_pack
        )
        if defer_lm_head:
            model_kwargs["defer_lm_head"] = True
        with routing_replay_context(train_meta):
            out = train_model(**model_kwargs)
        if defer_lm_head:
            logprobs = packed_next_token_logprobs_from_hidden(
                out.hidden_states,
                tokens,
                data_pack["train_cu_seqlens"],
                train_model.lm_head,
                logit_softcap=getattr(train_model, "final_logit_softcapping", None),
            )
        else:
            logprobs = packed_next_token_logprobs(out.logits_shard, tokens, data_pack["train_cu_seqlens"])
        loss_out = worker.loss_fn(data_pack, logprobs)
        metrics = None
        if isinstance(loss_out, tuple):
            loss, metrics = loss_out
        else:
            loss = loss_out
        if not isinstance(loss, torch.Tensor):
            raise TypeError("train_loss_fn must return a torch.Tensor")
        (loss / max(grad_scale, 1)).backward()
        # Keep the original full-gradient path for every optimizer residency
        # mode, including disk. Disk offload still streams optimizer state,
        # but it does not alter gradient accumulation or synchronization.
        self._accumulate_main_gradients()
        stepped = allow_step
        grad_norm = None
        multimodal_grad_metrics = None
        clipped_grad_norm = None
        if stepped:
            self._sync_data_parallel_gradients()
            self._sync_tensor_parallel_replicated_gradients()
            self._finalize_router_expert_bias()
            multimodal_groups = tuple(
                group
                for group in worker.multimodal_lr_schedules
                if any(getattr(param, "_areno_lr_group", None) == group for param in worker.model.parameters())
            )
            grad_norms = _grad_norms(worker.model.parameters(), multimodal_groups)
            grad_norm = grad_norms.pop("global")
            multimodal_grad_metrics = (
                {f"{group}_grad_norm": grad_norms[group] for group in multimodal_groups} if multimodal_groups else None
            )
            clipped_grad_norm = grad_norm
            if worker.grad_clip_norm is not None:
                _clip_grad_norm(worker.model.parameters(), grad_norm, worker.grad_clip_norm)
                clipped_grad_norm = min(grad_norm, float(worker.grad_clip_norm))
            current_lr = self._lr_for_step(worker._global_step + 1)
            worker.optimizer.lr = current_lr
            multimodal_lrs = self._set_multimodal_lrs(worker._global_step + 1)
            worker.optimizer.step()
            worker.optimizer.zero_grad(set_to_none=True)
            worker._global_step += 1
            if worker.adapter_registry is not None:
                worker.adapter_registry.increment_version()
            if worker.device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            current_lr = worker.optimizer.lr
            multimodal_lrs = self._current_multimodal_lrs()
        if ctx.is_rank0:
            if auto_tune_probe and worker.device.type == "cuda":
                torch.cuda.synchronize(worker.device)
                total = torch.cuda.get_device_properties(worker.device).total_memory
                peak = torch.cuda.max_memory_allocated(worker.device)
                if metrics is None:
                    metrics = {}
                metrics["auto_tune_worker_peak_mem_frac"] = float(peak) / float(total)
            return {
                "loss": float(loss.detach().cpu()),
                "stepped": stepped,
                "global_step": worker._global_step,
                "adapter_version": (worker.adapter_registry.version if worker.adapter_registry is not None else None),
                "metrics": _merge_metrics(
                    metrics,
                    None,
                    {"lr": current_lr},
                    multimodal_lrs,
                    {"sequence_parallel": float(model_kwargs["train_meta"].sequence_parallel)},
                    {"grad_norm": grad_norm} if grad_norm is not None else None,
                    multimodal_grad_metrics,
                    {"clipped_grad_norm": clipped_grad_norm} if clipped_grad_norm is not None else None,
                ),
            }
        return None

    def _lr_for_step(self, step: int) -> float:
        """Compute the actor learning rate for a given optimizer step."""

        worker = self.worker
        return self._scheduled_lr(
            step,
            base_lr=worker.base_lr,
            min_lr=worker.min_lr,
            decay_steps=worker.lr_decay_steps,
            decay_style=worker.lr_decay_style,
            warmup_steps=worker.lr_warmup_steps,
        )

    @staticmethod
    def _scheduled_lr(
        step: int, *, base_lr: float, min_lr: float, decay_steps: int, decay_style: str, warmup_steps: int = 0
    ) -> float:
        if warmup_steps > 0 and step <= warmup_steps:
            return base_lr * step / warmup_steps
        if decay_style == "constant" or decay_steps <= 0:
            return base_lr
        decay_step = step - warmup_steps
        decay_steps = max(decay_steps - warmup_steps, 1)
        progress = min(max(decay_step / decay_steps, 0.0), 1.0)
        if decay_style == "linear":
            coeff = 1.0 - progress
        elif decay_style == "cosine":
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"unsupported lr_decay_style: {decay_style}")
        return min_lr + coeff * (base_lr - min_lr)

    def _set_multimodal_lrs(self, step: int) -> dict[str, float]:
        values = {}
        for group, schedule in self.worker.multimodal_lr_schedules.items():
            lr = self._scheduled_lr(
                step,
                base_lr=schedule["lr"],
                min_lr=schedule["min_lr"],
                decay_steps=schedule["decay_steps"],
                decay_style=schedule["decay_style"],
            )
            found = False
            for param in self.worker.optimizer.model_params:
                if getattr(param, "_areno_lr_group", None) == group:
                    param._areno_lr = lr
                    found = True
            if found:
                values[f"{group}_lr"] = lr
        return values

    def _current_multimodal_lrs(self) -> dict[str, float]:
        values = {}
        for param in self.worker.optimizer.model_params:
            group = getattr(param, "_areno_lr_group", None)
            if group is not None:
                values[f"{group}_lr"] = float(getattr(param, "_areno_lr", self.worker.optimizer.lr))
        return values

    def _sync_tensor_parallel_replicated_gradients(self) -> None:
        """Sum TP-replicated actor gradients with shard-local contributions."""

        worker = self.worker
        ctx = get_tp_context()
        if ctx.world_size == 1:
            return
        ranged = []
        for param in worker.model.parameters():
            grad = param_grad(param)
            if grad is None:
                continue
            output_range = getattr(param, "tp_replicated_output_range", None)
            if output_range is not None:
                start, end, global_size = output_range
                canonical_numel = global_size * grad[0].numel()
                ranged.append((grad, start, end, global_size, canonical_numel))
            elif bool(getattr(param, "tp_grad_allreduce", False)):
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        if not ranged:
            return
        packed = ranged[0][0].new_zeros(sum(item[-1] for item in ranged))
        offset = 0
        for grad, start, end, global_size, canonical_numel in ranged:
            canonical = packed.narrow(0, offset, canonical_numel).view(global_size, *grad.shape[1:])
            canonical[start:end].copy_(grad)
            offset += canonical_numel
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=ctx.group)
        offset = 0
        for grad, start, end, global_size, canonical_numel in ranged:
            canonical = packed.narrow(0, offset, canonical_numel).view(global_size, *grad.shape[1:])
            grad.copy_(canonical[start:end])
            offset += canonical_numel

    def _sync_data_parallel_gradients(self) -> None:
        """Average resident full gradients across data-parallel replicas."""

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

    def _accumulate_main_gradients(self) -> None:
        """Fold one microbatch into resident FP32 parameter gradients."""

        for param in self.worker.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            main_grad = getattr(param, "main_grad", None)
            if isinstance(main_grad, torch.Tensor):
                main_grad.add_(grad.to(dtype=main_grad.dtype))
            else:
                param.main_grad = grad.to(dtype=torch.float32)
            param.grad = None

    def _finalize_router_expert_bias(self) -> None:
        """Apply MoE router/expert bias corrections after gradient sync."""

        ctx = get_tp_context()
        self.worker.model.finalize_router_expert_bias(ctx.group, ctx.dp_group)


def _actor_train_model(worker):
    """Use eager Gemma4 backward while preserving compiled rollout forward."""

    if worker.config.model.model_type == "gemma4":
        return unwrap_model(worker.model)
    return worker.model
