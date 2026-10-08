"""CUDA QLoRA: quantify approximation separately from kernel/update correctness."""

import copy

import pytest
import torch
import torch.nn.functional as F

from areno.accel.nf4 import NF4Weight, nf4_grouped_linear, nf4_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="QLoRA requires CUDA")


def test_qlora_model_loss_and_all_adapter_gradients():
    from areno.adapters import LoraConfig
    from areno.adapters.lora import initialize_lora
    from areno.adapters.qlora import initialize_qlora
    from areno.engine.config import ModelConfig
    from areno.engine.runtime.metadata import TrainMeta
    from areno.models.qwen3.model import Qwen3ForCausalLM

    torch.manual_seed(82)
    model = Qwen3ForCausalLM(
        ModelConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            dtype=torch.bfloat16,
            attn_backend="native",
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    registry = initialize_lora(model, LoraConfig(qlora=True), seed=1)
    with torch.no_grad():
        for slot in registry.slots.values():
            slot.lora_B.normal_(0, 0.01)
    reference = copy.deepcopy(model)
    initialize_qlora(model)
    for name, module in model.named_modules():
        if getattr(module, "quantized_weight", None) is not None:
            reference.get_submodule(name).weight.data.copy_(module.quantized_weight.dequantize())
    tokens = torch.randint(256, (1, 16), device="cuda")
    positions = torch.arange(16, device="cuda")[None]
    meta = TrainMeta(
        cu_seqlens=torch.tensor([0, 16], device="cuda", dtype=torch.int32),
        max_seqlen=16,
        packed=True,
        activation_checkpointing=True,
    )
    losses, gradients = [], []
    for candidate in (reference, model):
        logits = candidate(tokens, position_ids=positions, train_meta=meta).logits_shard
        loss = F.cross_entropy(logits[0, :-1].float(), tokens[0, 1:])
        losses.append(loss.detach())
        gradients.append(torch.autograd.grad(loss, [p for p in candidate.parameters() if p.requires_grad]))
    torch.testing.assert_close(losses[0], losses[1], atol=2e-4, rtol=2e-4)
    for a, b in zip(*gradients, strict=True):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(a, b, atol=2e-4, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_nf4_cuda_dequant_and_linear_gradients(dtype):
    torch.manual_seed(24)
    weight = torch.randn(259, 129, dtype=dtype) * 0.03
    cpu = NF4Weight(weight)
    q = copy.deepcopy(cpu).cuda()
    expected_weight = cpu.dequantize().cuda()
    torch.testing.assert_close(q.dequantize(), expected_weight, atol=0, rtol=0)
    x = torch.randn(3, 7, 129, device="cuda", dtype=dtype, requires_grad=True)
    ref = F.linear(x, expected_weight)
    actual = nf4_linear(x, q)
    torch.testing.assert_close(actual, ref, atol=0, rtol=0)
    upstream = torch.randn_like(ref)
    (dx_ref,) = torch.autograd.grad(ref, x, upstream)
    (dx,) = torch.autograd.grad(actual, x, upstream)
    torch.testing.assert_close(
        dx, dx_ref, atol=1e-5 if dtype == torch.float32 else 2e-3, rtol=1e-4 if dtype == torch.float32 else 1e-2
    )
    assert torch.isfinite(dx).all()


def test_nf4_cuda_graph_and_grouped_gradients():
    from areno.accel import areno_grouped_linear

    torch.manual_seed(45)
    q = NF4Weight(torch.randn(3, 128, 64, device="cuda", dtype=torch.bfloat16) * 0.02)
    counts = torch.tensor([3, 0, 5], device="cuda", dtype=torch.int64)
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected = areno_grouped_linear(x, q.dequantize(), counts)
    actual = nf4_grouped_linear(x, q, counts)
    upstream = torch.randn_like(expected)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, upstream)[0], torch.autograd.grad(expected, x, upstream)[0], atol=0, rtol=0
    )
    dense = NF4Weight(torch.randn(128, 64, device="cuda", dtype=torch.bfloat16))
    x = x.detach()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            nf4_linear(x, dense)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = nf4_linear(x, dense)
    graph.replay()
    torch.testing.assert_close(out, F.linear(x, dense.dequantize()), atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["fp32", "8bit", "4bit"])
def test_paged_optimizer_matches_nonpaged_and_roundtrips(mode):
    from areno.accel._extension import extension
    from areno.engine.optim import AdamW4bit, AdamW8bit, AdamWFP32Master
    from areno.engine.optim.paged import PagedAdamW4bit, PagedAdamW8bit, PagedAdamWFP32Master

    torch.manual_seed(63)
    normal_cls, paged_cls = {
        "fp32": (AdamWFP32Master, PagedAdamWFP32Master),
        "8bit": (AdamW8bit, PagedAdamW8bit),
        "4bit": (AdamW4bit, PagedAdamW4bit),
    }[mode]
    a = torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
    b = torch.nn.Parameter(a.detach().clone())
    kwargs = dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)
    normal, paged = normal_cls([a], **kwargs), paged_cls([b], **kwargs)
    for step in range(5):
        grad = torch.randn_like(a)
        a.grad = grad.clone()
        b.grad = grad.clone()
        normal.step()
        paged.step()
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert paged.managed_memory_bytes() > 0
        if step == 2:
            saved = paged.state_dict()
            paged.clear_state()
            paged.load_state_dict(saved)
    states = getattr(paged, "_states", paged.buckets)
    for state in states:
        tensor = getattr(state, "exp_avg_q", None)
        if tensor is None:
            tensor = state.exp_avg
        assert extension(a.device).areno_is_managed(tensor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("adapters", [False, True])
def test_nf4_moe_graph_replays_changed_routes_and_adapters(dtype, adapters):
    from types import SimpleNamespace

    from areno.accel.kernels.nf4_moe import nf4_experts
    from areno.accel.ops import FusedMoeConfig

    torch.manual_seed(77)
    experts, hidden, width, tokens, top_k = 5, 65, 33, 7, 2
    q1 = NF4Weight(torch.randn(experts, 2 * width, hidden, device="cuda", dtype=dtype) * 0.03)
    q2 = NF4Weight(torch.randn(experts, hidden, width, device="cuda", dtype=dtype) * 0.03)
    w1, w2 = q1.dequantize(), q2.dequantize()
    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    ids = torch.randint(experts, (tokens, top_k), device="cuda", dtype=torch.int32)
    weights = torch.rand(tokens, top_k, device="cuda")
    config = FusedMoeConfig(experts, hidden, width, top_k, routed_scaling_factor=1.25)
    slots = {}
    if adapters:
        for name, size_in, size_out in (
            ("gate_proj", hidden, width),
            ("up_proj", hidden, width),
            ("down_proj", width, hidden),
        ):
            slots[name] = SimpleNamespace(
                rank=8,
                out_features=size_out,
                lora_A=torch.randn(experts, 8, size_in, device="cuda", dtype=dtype) * 0.1,
                lora_B=torch.randn(experts, size_out, 8, device="cuda", dtype=dtype) * 0.1,
                scale=torch.tensor(2.0, device="cuda"),
            )

    def reference():
        out = torch.zeros_like(x, dtype=torch.float32)
        for t in range(tokens):
            for k in range(top_k):
                expert = int(ids[t, k])
                gu = F.linear(x[t], w1[expert])
                gate, up = gu.chunk(2)
                for name, target in (("gate_proj", gate), ("up_proj", up)):
                    if name in slots:
                        slot = slots[name]
                        target.add_(F.linear(F.linear(x[t], slot.lora_A[expert]), slot.lora_B[expert]) * slot.scale)
                # Native SiLU kernel computes the activation and product in FP32.
                activated = (F.silu(gate.float()) * up.float()).to(dtype) * weights[t, k].to(dtype)
                y = F.linear(activated, w2[expert])
                if "down_proj" in slots:
                    slot = slots["down_proj"]
                    y += F.linear(F.linear(activated, slot.lora_A[expert]), slot.lora_B[expert]) * slot.scale
                out[t] += y.float()
        return out.to(dtype) * config.routed_scaling_factor

    def candidate():
        return nf4_experts(x, q1, q2, ids, weights, slots, config)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            candidate()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = candidate()
    for iteration in range(3):
        # Populate formerly unused experts, all-zero masked routes, and live
        # adapter updates without recapturing or caching dense merged weights.
        ids.copy_((torch.arange(tokens * top_k, device="cuda").view(tokens, top_k) + iteration) % experts)
        weights[0].zero_()
        x.mul_(0.9)
        for slot in slots.values():
            slot.lora_B.add_(0.002)
        graph.replay()
        torch.testing.assert_close(output, candidate(), atol=0, rtol=0)
        torch.testing.assert_close(output, reference(), atol=2e-3, rtol=3e-2)
        assert torch.isfinite(output).all()
