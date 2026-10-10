"""Binding execution gates on native CUDA kernels; no checkpoint qualification."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from areno.adapters import LoraConfig
from areno.adapters.lora import initialize_lora
from areno.engine.config import ModelConfig
from areno.engine.optim import AdamW4bit, AdamW8bit, AdamWFP32Master
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.models.bailing.model import BailingSparseMoeBlock as LegacyBailingMoe
from areno.models.bailing_v3.model import BailingSparseMoeBlock as BailingV3Moe
from areno.models.gemma4.model import Gemma4MoeMLP
from areno.models.qwen3.model import Qwen3MoeMLP
from areno.models.qwen3_5.model import Qwen35MoeMLP

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA execution is required")


@pytest.mark.parametrize(
    "model_type,factory",
    (
        ("qwen3_moe", Qwen3MoeMLP),
        ("qwen3_5_moe", Qwen35MoeMLP),
        ("gemma4", Gemma4MoeMLP),
        ("bailing_moe_linear_v2", LegacyBailingMoe),
        ("bailing_moe_v3", BailingV3Moe),
    ),
)
def test_expert_binding_two_updates_fused_graph_and_base_isolation(model_type, factory):
    previous = get_tp_context()
    device = torch.device("cuda", 0)
    set_tp_context(TPContext(rank=0, world_size=1, device=device, group=None))
    try:
        torch.manual_seed(19)
        config = ModelConfig(
            model_type=model_type,
            hidden_size=128,
            intermediate_size=256,
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            num_shared_experts=None,
            no_kda_lora=True,
            dtype=torch.bfloat16,
            sequence_parallel=False,
        )
        policy = nn.Module()
        policy.config = config
        policy.mlp = factory(config, routing_layer_slot=0).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.normal_(std=0.05)
        flat = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
        indices = (torch.arange(16, device=device) % 2).view(-1, 1)
        weights = torch.ones(16, 1, device=device)
        base_weights = {name: value.detach().clone() for name, value in policy.named_parameters()}
        fft = policy.mlp.experts(flat, indices, weights).detach()
        registry = initialize_lora(
            policy,
            LoraConfig(rank=4, alpha=4, target_modules=("mlp.experts.gate_proj", "mlp.experts.down_proj")),
            seed=7,
        )
        zero_delta = policy.mlp.experts(flat, indices, weights).detach()
        torch.testing.assert_close(zero_delta, fft, rtol=0, atol=0)
        optimizer = AdamWFP32Master(registry.parameters(), lr=0.02, betas=(0.9, 0.999), weight_decay=0)
        for _ in range(2):
            optimizer.zero_grad()
            output = policy.mlp.experts(flat, indices, weights)
            output.float().square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in registry.parameters())
            assert all(slot.lora_B.grad.abs().max() > 0 for slot in registry.slots.values())
            optimizer.step()
            registry.increment_version()
        assert registry.version == 2
        grouped = policy.mlp.experts(flat, indices, weights).detach()
        assert not torch.equal(grouped, fft)
        policy.eval()
        policy.mlp.prepare_infer_weights()
        with torch.no_grad():
            fused = policy.mlp._forward_fused_moe(flat, indices, weights)
        # BF16 rounds separate low-rank GEMMs and merged tiles at different
        # boundaries. Bound that error at a few BF16 ULPs, not exact equality.
        torch.testing.assert_close(fused, grouped, rtol=0.02, atol=0.005)
        for name, value in policy.named_parameters():
            if name in base_weights:
                torch.testing.assert_close(value, base_weights[name], rtol=0, atol=0)
        pointer = policy.mlp._infer_w1_weight.data_ptr()
        policy.mlp.prepare_infer_weights()
        assert policy.mlp._infer_w1_weight.data_ptr() == pointer
        with torch.no_grad():
            for _ in range(3):
                policy.mlp._forward_fused_moe(flat, indices, weights)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = policy.mlp._forward_fused_moe(flat, indices, weights)
            graph.replay()
            torch.testing.assert_close(captured, fused, rtol=0, atol=0)
            for slot in registry.slots.values():
                slot.lora_B.add_(0.1)
            policy.mlp.prepare_infer_weights()
            graph.replay()
            refreshed = policy.mlp._forward_fused_moe(flat, indices, weights)
            torch.testing.assert_close(captured, refreshed, rtol=0, atol=0)
            assert not torch.equal(captured, fused)
            with registry.base_only():
                policy.mlp.prepare_infer_weights()
                restored = policy.mlp._forward_fused_moe(flat, indices, weights)
                torch.testing.assert_close(restored, fft, rtol=0.02, atol=0.005)
    finally:
        set_tp_context(previous)


@pytest.mark.parametrize("optimizer_cls", (AdamWFP32Master, AdamW8bit, AdamW4bit))
def test_native_adapter_shapes_with_existing_fft_optimizers(optimizer_cls):
    from areno.engine.layers.linear import ColumnParallelLinear

    previous = get_tp_context()
    device = torch.device("cuda", 0)
    set_tp_context(TPContext(rank=0, world_size=1, device=device, group=None))
    try:
        policy = nn.Module()
        policy.config = ModelConfig(model_type="llama")
        policy.projection = ColumnParallelLinear(128, 128).to(device=device, dtype=torch.bfloat16)
        registry = initialize_lora(policy, LoraConfig(rank=4, alpha=4, target_modules=("projection",)), seed=7)
        optimizer = optimizer_cls(registry.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0)
        inputs = torch.ones(1, 4, 128, device=device, dtype=torch.bfloat16)
        compiled = torch.compile(policy.projection)
        initial_b = registry.slots["projection"].lora_B.detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            compiled(inputs).float().square().mean().backward()
            optimizer.step()
        assert all(torch.isfinite(p).all() for p in registry.parameters())
        assert not torch.equal(registry.slots["projection"].lora_B, initial_b)
        torch.testing.assert_close(compiled(inputs), policy.projection(inputs), rtol=0, atol=0)
    finally:
        set_tp_context(previous)


def _empty_route_tp_worker(rank, rendezvous, output_queue):
    from datetime import timedelta

    import torch.distributed as dist

    from areno.engine.parallel.collectives import gather_from_sequence_parallel_region

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    set_tp_context(
        TPContext(
            rank=rank,
            world_size=2,
            device=torch.device("cuda", rank),
            group=dist.group.WORLD,
            global_rank=rank,
            global_world_size=2,
        )
    )
    try:
        for model_type, factory in (
            ("qwen3_moe", Qwen3MoeMLP),
            ("qwen3_5_moe", Qwen35MoeMLP),
            ("gemma4", Gemma4MoeMLP),
            ("bailing_moe_linear_v2", LegacyBailingMoe),
            ("bailing_moe_v3", BailingV3Moe),
        ):
            config = ModelConfig(
                model_type=model_type,
                hidden_size=128,
                intermediate_size=256,
                num_experts=2,
                num_experts_per_tok=1,
                moe_intermediate_size=128,
                num_shared_experts=None,
                no_kda_lora=False,
                dtype=torch.bfloat16,
            )
            policy = nn.Module()
            policy.config = config
            policy.mlp = factory(config, routing_layer_slot=0).to(device=rank, dtype=torch.bfloat16)
            with torch.no_grad():
                for parameter in policy.parameters():
                    parameter.fill_(0.01)
            registry = initialize_lora(
                policy,
                LoraConfig(rank=4, alpha=4, target_modules=("mlp.experts.gate_proj", "mlp.experts.down_proj")),
                seed=7,
            )
            local = torch.full((1, 8, 128), 0.1, device=rank, dtype=torch.bfloat16, requires_grad=True)
            flat = gather_from_sequence_parallel_region(local).reshape(16, 128)
            indices = torch.zeros(16, 1, device=rank, dtype=torch.long)
            weights = torch.ones(16, 1, device=rank)
            policy.mlp.experts(flat, indices, weights).float().mean().backward()
            assert local.grad is not None and torch.isfinite(local.grad).all() and local.grad.abs().max() > 0
            assert all(parameter.grad is not None for parameter in registry.parameters())
            if rank == 1:
                assert all(not parameter.grad.any() for parameter in registry.parameters())
            output_queue.put((rank, model_type))
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_empty_expert_tp_rank_completes_sp_backward(tmp_path):
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_empty_route_tp_worker, args=(rank, str(tmp_path / "rendezvous"), queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=90)
        assert [process.exitcode for process in processes] == [0, 0]
        results = [queue.get(timeout=5) for _ in range(10)]
        assert len(set(results)) == 10
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
