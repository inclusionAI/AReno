"""Run separately with torchrun: native NPU/HCCL against CPU/Gloo DP shards."""

import importlib.util
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from areno.accel._extension import extension
from areno.engine.optim import AdamW4bit, AdamW8bit, AdamWFP32Master


@pytest.fixture(scope="module")
def groups():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("Run this file separately with torchrun --nproc_per_node=2 (or more)")
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend torch_npu and hardware are required")
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    assert torch.npu.device_count() > local_rank, "Each local rank needs a distinct NPU"
    torch.npu.set_device(local_rank)
    assert extension("npu").optimizer_implementation == "ascendc"
    dist.init_process_group("hccl", timeout=timedelta(seconds=90))
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
    try:
        yield torch.device("npu", local_rank), dist.group.WORLD, cpu_group
    finally:
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


def exercise_optimizer(optimizer_cls, device, candidate_group, cpu_group, checkpoint_dir, *, tiny=False):
    """The same DP layout/collectives run on each device with independent state."""
    rank, size = dist.get_rank(candidate_group), dist.get_world_size(candidate_group)
    initial = (
        [torch.ones(1, 1, dtype=torch.bfloat16)]
        if tiny
        else [
            torch.linspace(-1, 1, 3).to(torch.bfloat16),
            torch.linspace(-1.5, 1.5, 65 * 67).reshape(65, 67).to(torch.bfloat16),
            torch.linspace(-0.5, 0.5, 17).to(torch.bfloat16),
        ]
    )
    params = [torch.nn.Parameter(x.to(device).clone()) for x in initial]
    references = [torch.nn.Parameter(x.clone()) for x in initial]
    kwargs = dict(
        lr=0.003, betas=(0.8, 0.95), weight_decay=0.02, bucket_numel=1 if tiny else 1024, dp_rank=rank, dp_size=size
    )
    if optimizer_cls is not AdamWFP32Master:
        kwargs["quant_block_size"] = 128
    candidate = optimizer_cls(params, dp_group=candidate_group, **kwargs)
    reference = optimizer_cls(references, dp_group=cpu_group, **kwargs)
    for step in range(4):
        for index, (param, ref) in enumerate(zip(params, references, strict=True)):
            gradient = (torch.sin(torch.arange(param.numel()) * 0.02 + step + index) + rank * 0.25).to(torch.bfloat16)
            param.grad, ref.grad = gradient.reshape(param.shape).to(device), gradient.reshape(ref.shape).clone()
        candidate.reduce_scatter_gradients()
        reference.reduce_scatter_gradients()
        assert all(bucket.grad_shard.dtype == candidate.gradient_shard_dtype for bucket in candidate.buckets)
        if tiny and rank > 0:
            assert all(bucket.shard_numel == 0 for bucket in candidate.buckets)
        candidate.step()
        reference.step()
        for param, ref in zip(params, references, strict=True):
            torch.testing.assert_close(param.cpu(), ref, atol=2e-3, rtol=0)
            if optimizer_cls is AdamW4bit and param.ndim >= 2:
                torch.testing.assert_close(
                    candidate._factored_second_moments[id(param)].cpu(),
                    reference._factored_second_moments[id(ref)],
                    atol=4e-6,
                    rtol=4e-5,
                )
        if step == 1:
            path = checkpoint_dir / f"rank-{rank}.pt"
            torch.save(candidate.state_dict(), path)
            candidate = optimizer_cls(params, dp_group=candidate_group, **kwargs)
            candidate.load_state_dict(torch.load(path, weights_only=True))
            saved = reference.state_dict()
            reference = optimizer_cls(references, dp_group=cpu_group, **kwargs)
            reference.load_state_dict(saved)


@pytest.mark.parametrize("optimizer_cls", [AdamWFP32Master, AdamW8bit, AdamW4bit])
@pytest.mark.parametrize("tiny", [False, True])
def test_hccl_sharded_updates_and_checkpoint_resume(groups, tmp_path, optimizer_cls, tiny):
    device, native, cpu = groups
    exercise_optimizer(optimizer_cls, device, native, cpu, tmp_path, tiny=tiny)
