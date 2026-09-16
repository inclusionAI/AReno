"""Gaudi device/HCCL selection; all model workflows use the CUDA engine."""

import os
from copy import copy
from dataclasses import replace

import torch

from areno.api.backend.base import BackendCapabilities, register_backend
from areno.api.backend.cuda.backend import CudaBackend
from areno.api.config import HpuConfig
from areno.api.context import Context
from areno.api.models import BackendType
from areno.engine.parallel.context import destroy_process_group, init_process_group
from areno.engine.worker import ArenoWorker


class HpuProcess:
    core = None

    @classmethod
    def initialize_process(cls, rank, world_size, device_id, world_spec, partition, config):
        global_rank = partition.global_rank_offset + rank
        os.environ.update(
            RANK=str(global_rank), LOCAL_RANK=str(device_id), WORLD_SIZE=str(world_spec.global_world_size)
        )
        from areno.accel._extension import configure_hpu_kernel_library

        configure_hpu_kernel_library()
        import habana_frameworks.torch.core as core
        import habana_frameworks.torch.distributed.hccl  # noqa: F401

        if not torch.hpu.is_available():
            raise RuntimeError("no HPU device is available")
        torch.hpu.set_device(device_id)
        cls.core = core
        init_process_group(
            rank=rank,
            world_size=world_size,
            master_addr=world_spec.master_addr,
            master_port=world_spec.master_port,
            device_id=device_id,
            tp_size=config.tp_size,
            global_rank=global_rank,
            global_world_size=world_spec.global_world_size,
            train_world_size=world_spec.train.local_world_size,
            train_tp_size=world_spec.train.tp_size,
            rollout_world_size=world_spec.rollout.local_world_size if world_spec.rollout else None,
            rollout_tp_size=world_spec.rollout.tp_size if world_spec.rollout else None,
            train_devices=world_spec.train.devices,
            rollout_devices=world_spec.rollout.devices if world_spec.rollout else None,
            role=partition.role,
            device=torch.device("hpu"),
            backend="hccl",
        )

    @classmethod
    def close_process(cls):
        destroy_process_group()
        cls.core = None


class HpuWorker(ArenoWorker):
    process_lifecycle = HpuProcess

    def __init__(self, config):
        if config.model.model_type not in {"llama", "qwen3", "qwen3_moe", "gemma4", "phi4mm"}:
            raise ValueError(
                "this model still calls FLA directly; its operators must be routed through shared accel for HPU"
            )
        super().__init__(config)
        self.model.register_forward_hook(lambda *args: self.process_lifecycle.core.mark_step())

    def train(self, payload):
        probe = any(
            getattr(shard.get("_loss_fn"), "__name__", "") == "_dummy_policy_loss"
            for pack in payload.data_packs_by_dp
            for shard in pack
        )
        if probe:
            self.process_lifecycle.core.mark_step()
            torch.hpu.synchronize()
            torch.hpu.reset_peak_memory_stats()
        result = super().train(payload)
        self.process_lifecycle.core.mark_step()
        if probe and result is not None:
            torch.hpu.synchronize()
            stats = torch.hpu.memory_stats()
            for microbatch in result:
                if microbatch is not None:
                    microbatch.setdefault("metrics", {})["auto_tune_worker_peak_mem_frac"] = (
                        stats["MaxInUse"] / stats["Limit"]
                    )
        return result

    def probe_rollout_cache(self, payload):
        self.process_lifecycle.core.mark_step()
        torch.hpu.synchronize()
        torch.hpu.reset_peak_memory_stats()
        super().probe_rollout_cache(payload)
        self.process_lifecycle.core.mark_step()
        torch.hpu.synchronize()
        stats = torch.hpu.memory_stats()
        return stats["MaxInUse"] / stats["Limit"]

    def rollout_session_sync(self, payload):
        self.process_lifecycle.core.mark_step()
        torch.hpu.synchronize()
        super().rollout_session_sync(payload)
        torch.hpu.synchronize()


@register_backend(BackendType.HPU)
class HpuBackend(CudaBackend):
    worker_cls = HpuWorker

    def initialize(self, ctx: Context):
        cfg = ctx.custom_config if ctx.custom_config is not None else HpuConfig()
        if not isinstance(cfg, HpuConfig):
            raise TypeError(f"HpuBackend requires HpuConfig, got {type(cfg)!r}")
        hpu_ctx = copy(ctx)
        hpu_ctx.custom_config = replace(cfg, runtime={**cfg.runtime, "device_type": "hpu"})
        return super().initialize(hpu_ctx)

    @classmethod
    def capabilities(cls):
        return BackendCapabilities(
            algorithms=CudaBackend.capabilities().algorithms,
            model_roles=CudaBackend.capabilities().model_roles,
            distributed=True,
            custom_loss=True,
        )
