"""Ascend device/HCCL selection; all model workflows use the shared Torch engine."""

import os
from copy import copy
from dataclasses import replace

import torch

from areno.api.backend.base import BackendCapabilities, register_backend
from areno.api.backend.cuda.backend import CudaBackend
from areno.api.config import NpuConfig
from areno.api.context import Context
from areno.api.models import BackendType
from areno.engine.parallel.context import destroy_process_group, init_process_group
from areno.engine.worker import ArenoWorker


class NpuProcess:
    @classmethod
    def initialize_process(cls, rank, world_size, device_id, world_spec, partition, config):
        global_rank = partition.global_rank_offset + rank
        os.environ.update(
            RANK=str(global_rank), LOCAL_RANK=str(device_id), WORLD_SIZE=str(world_spec.global_world_size)
        )
        import torch_npu  # noqa: F401

        from areno.accel._extension import extension

        native = extension("npu")
        if not getattr(native, "supports_training_and_serving", False):
            raise RuntimeError(
                "The Ascend native extension currently provides kernel validation only; "
                "training/serving kernels are not complete."
            )

        if not torch.npu.is_available():
            raise RuntimeError("no NPU device is available")
        torch.npu.set_device(device_id)
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
            device=torch.device("npu"),
            backend="hccl",
        )

    @classmethod
    def close_process(cls):
        destroy_process_group()


class NpuWorker(ArenoWorker):
    process_lifecycle = NpuProcess

    def __init__(self, config):
        if config.model.model_type not in {"llama", "qwen3", "qwen3_moe", "gemma4", "phi4mm"}:
            raise ValueError(
                "Ascend hybrid-model integration has not yet passed the required operator and end-to-end validation"
            )
        super().__init__(config)


@register_backend(BackendType.NPU)
class NpuBackend(CudaBackend):
    worker_cls = NpuWorker

    def initialize(self, ctx: Context):
        cfg = ctx.custom_config if ctx.custom_config is not None else NpuConfig()
        if not isinstance(cfg, NpuConfig):
            raise TypeError(f"NpuBackend requires NpuConfig, got {type(cfg)!r}")
        npu_ctx = copy(ctx)
        npu_ctx.custom_config = replace(cfg, runtime={**cfg.runtime, "device_type": "npu"})
        return super().initialize(npu_ctx)

    @classmethod
    def capabilities(cls):
        return BackendCapabilities(
            algorithms=CudaBackend.capabilities().algorithms,
            model_roles=CudaBackend.capabilities().model_roles,
            distributed=True,
            custom_loss=True,
        )
