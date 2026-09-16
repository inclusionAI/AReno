"""NPU selects device hooks while inheriting the existing Torch engine."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areno.api import NPU, CudaConfig, NpuConfig
from areno.api.backend.base import get_backend_cls
from areno.api.backend.cuda.backend import CudaBackend
from areno.api.backend.npu.backend import NpuBackend, NpuProcess, NpuWorker
from areno.api.context import Context
from areno.engine.api import ArenoEngine
from areno.engine.config import EngineConfig, ModelConfig, RuntimeConfig
from areno.engine.worker import ArenoWorker


@pytest.fixture(scope="module", autouse=True)
def register_npu_device_name():
    if torch._C._get_privateuse1_backend_name() == "privateuseone":
        torch.utils.rename_privateuse1_backend("npu")


def test_npu_reuses_cuda_workflows():
    assert get_backend_cls(NPU) is NpuBackend
    assert NpuConfig is CudaConfig
    for method in ("train", "rollout_batch", "rollout_batch_async", "save_checkpoint", "close"):
        assert getattr(NpuBackend, method) is getattr(CudaBackend, method)
    assert NpuWorker.handle is ArenoWorker.handle
    assert NpuWorker.save_checkpoint is ArenoWorker.save_checkpoint
    assert NpuWorker.run_rollout_command is ArenoWorker.run_rollout_command
    assert {p.name for p in Path(__import__("areno.api.backend.npu", fromlist=["x"]).__file__).parent.glob("*.py")} == {
        "__init__.py",
        "backend.py",
    }


def test_incomplete_extension_rejects_jobs_before_device_or_collective_initialization(monkeypatch):
    def unexpected_initialization(*args, **kwargs):
        pytest.fail("incomplete NPU kernels must be rejected before initializing devices or collectives")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(
        "areno.accel._extension.extension", lambda device: SimpleNamespace(supports_training_and_serving=False)
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=unexpected_initialization), raising=False)
    monkeypatch.setattr("areno.api.backend.npu.backend.init_process_group", unexpected_initialization)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.setenv(key, "0")
    with pytest.raises(RuntimeError, match="kernel validation only"):
        NpuProcess.initialize_process(
            0, 1, 0, SimpleNamespace(global_world_size=1), SimpleNamespace(global_rank_offset=0), SimpleNamespace()
        )


@pytest.mark.parametrize("optimizer", [{}, {"adam_4bit": True}, {"adam_8bit": True}])
def test_npu_initializes_shared_engine_with_tp_dp_and_optimizer(monkeypatch, optimizer):
    captured = {}

    class Engine:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured.update(path=path, **kwargs)
            return SimpleNamespace(close=lambda: None)

    backend = NpuBackend()
    monkeypatch.setattr("areno.ArenoEngine", Engine)
    config = CudaConfig(tp_size=2, optimizer=optimizer, runtime={"activation_checkpointing": True})
    backend.initialize(Context(4, "local-model", None, config))
    assert captured["tp_size"] == 2
    assert captured["dp_size"] == 2
    assert captured["devices"] == [0, 1, 2, 3]
    assert captured["optimizer_config"].adam_4bit == optimizer.get("adam_4bit", False)
    assert captured["optimizer_config"].adam_8bit == optimizer.get("adam_8bit", False)
    assert captured["runtime_config"].device_type == "npu"
    assert captured["runtime_config"].activation_checkpointing
    assert captured["runtime_config"].eager_decode
    assert not captured["runtime_config"].compile_model
    assert captured["runtime_config"].attn_backend == "native"
    assert config.runtime == {"activation_checkpointing": True}
    backend.close()


@pytest.mark.parametrize("device_type,worker_cls", [("cuda", ArenoWorker), ("npu", NpuWorker)])
def test_shared_engine_selects_device_worker_without_starting_processes(device_type, worker_cls):
    config = EngineConfig(
        model=ModelConfig(),
        runtime=RuntimeConfig(device_type=device_type),
        devices=[0, 1, 2, 3],
        tp_size=2,
        role="rollout",
    )
    engine = ArenoEngine(config, start=False)
    assert engine.cluster.worker_cls is worker_cls
    assert engine.config.tp_size == 2
    assert engine.config.dp_size == 2


def test_npu_config_does_not_probe_cuda(monkeypatch):
    def unexpected_cuda_probe():
        pytest.fail("NPU configuration must not probe CUDA hardware")

    monkeypatch.setattr("torch.cuda.is_available", unexpected_cuda_probe)
    config = EngineConfig(model=ModelConfig(), runtime=RuntimeConfig(device_type="npu"), tp_size=2, dp_size=2)
    assert config.devices == [0, 1, 2, 3]
    assert config.model.attn_backend == "native"


def test_npu_serve_reuses_shared_engine_with_tp_dp(monkeypatch):
    import importlib

    serve = importlib.import_module("areno.cli.serve")
    captured = {}

    class Engine:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured.update(path=path, **kwargs)
            return SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(max_position_embeddings=4096)))

    monkeypatch.setattr(serve, "ArenoEngine", Engine)
    runtime = serve._create_serve_runtime(
        model_path="local-model",
        backend_type=NPU,
        tp_size=2,
        world_size=4,
        max_running_prompts=8,
        decode_progress_interval_s=0,
        eager_decode=False,
        attn_backend="flash",
        lora=None,
        base_model_name_or_path=None,
    )
    assert runtime.max_model_len == 4096
    assert captured["tp_size"] == 2
    assert captured["dp_size"] == 2
    assert captured["devices"] == [0, 1, 2, 3]
    assert captured["runtime_config"].device_type == "npu"
    assert captured["runtime_config"].eager_decode
    assert captured["runtime_config"].attn_backend == "native"


def test_npu_partition_groups_reuse_shared_rank_layout(monkeypatch):
    import torch

    from areno.engine.parallel import context
    from tests.test_parallel_partition_cpu import _mock_distributed

    monkeypatch.setattr(context, "_TP_CONTEXT", context.get_tp_context())
    layouts = []
    for role, rank, global_rank, world, tp in (("train", 3, 3, 4, 2), ("rollout", 1, 5, 2, 1)):
        calls = _mock_distributed(monkeypatch)

        def unexpected_cuda():
            pytest.fail("explicit NPU process initialization must not probe CUDA")

        monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda)
        ctx = context.init_process_group(
            rank=rank,
            world_size=world,
            master_addr="127.0.0.1",
            master_port=12345,
            device_id=global_rank,
            tp_size=tp,
            global_rank=global_rank,
            global_world_size=6,
            train_world_size=4,
            train_tp_size=2,
            rollout_world_size=2,
            rollout_tp_size=1,
            train_devices=(0, 1, 2, 3),
            rollout_devices=(4, 5),
            role=role,
            device=torch.device("npu"),
            backend="hccl",
        )
        assert ctx.device.type == "npu"
        assert ctx.global_rank == global_rank and ctx.global_world_size == 6
        assert ctx.role == role
        assert calls[0][1]["backend"] == "hccl"
        assert calls[0][1]["rank"] == global_rank
        assert ctx.train_tp_size == 2 and ctx.rollout_tp_size == 1
        layouts.append([value for kind, value in calls if kind == "group"])
    assert layouts[0] == layouts[1] == [(0, 1), (2, 3), (0, 2), (1, 3), (4,), (5,), (4, 5), (0, 4), (2, 4)]


def test_npu_worker_training_probe_uses_npu_memory_and_shared_train(monkeypatch):
    import torch

    from areno.engine.protocol import TrainPayload

    events = []

    def _dummy_policy_loss():
        pass

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: events.append("sync"),
            reset_peak_memory_stats=lambda: events.append("reset"),
            max_memory_allocated=lambda: 40,
            get_device_properties=lambda: SimpleNamespace(total_memory=100),
        ),
        raising=False,
    )
    monkeypatch.setattr(ArenoWorker, "train", lambda self, payload: [None, {"loss": 1.0, "metrics": {"existing": 2.0}}])
    worker = object.__new__(NpuWorker)
    result = worker.train(TrainPayload(data_packs_by_dp=[[{"_loss_fn": _dummy_policy_loss}]]))
    assert result == [None, {"loss": 1.0, "metrics": {"existing": 2.0, "auto_tune_worker_peak_mem_frac": 0.4}}]
    assert events == ["sync", "reset", "sync"]


def test_npu_registered_training_algorithms_and_roles_are_shared():
    capabilities = NpuBackend.capabilities()
    assert capabilities.algorithms == CudaBackend.capabilities().algorithms
    assert capabilities.model_roles == CudaBackend.capabilities().model_roles
    assert capabilities.distributed and capabilities.custom_loss
