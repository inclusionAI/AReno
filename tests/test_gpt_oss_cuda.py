"""Native GPT-OSS components and tiny-model differential CUDA gates."""

import pytest
import torch

from areno.accel.sink_attention import sink_attention
from areno.engine.runtime.metadata import TrainMeta
from areno.models.gpt_oss import GptOssAdapter
from tests.test_gpt_oss_cpu import dense_reference, tiny_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def ieee_reference_matmuls():
    # NGC enables TF32 by default; an FP32 correctness reference must not
    # silently round its inputs to ten mantissa bits.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("window", [None, 3])
def test_sink_cuda(window):
    torch.manual_seed(5)
    q = torch.randn(1, 8, 4, 64, device="cuda", requires_grad=True)
    k = torch.randn(1, 8, 2, 64, device="cuda", requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    sinks = torch.randn(4, device="cuda", requires_grad=True)
    seq = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device="cuda")
    actual = sink_attention(q, k, v, sinks, sequence_ids=seq, window=window)
    expected = dense_reference(q, k, v, sinks, sequence_ids=seq, window=window)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    grad = torch.randn_like(actual)
    a = torch.autograd.grad(actual, (q, k, v, sinks), grad, retain_graph=True)
    b = torch.autograd.grad(expected, (q, k, v, sinks), grad)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=5e-6, rtol=3e-5)


@pytest.mark.parametrize(
    "rope",
    [
        None,
        {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 4096,
        },
    ],
)
def test_tiny_hf_cuda(tmp_path, rope):
    from transformers import GptOssConfig, GptOssForCausalLM

    hf = tiny_config()
    hf["rope_scaling"] = rope
    hf["head_dim"] = 64
    hf["max_position_embeddings"] = 256
    ref_config = GptOssConfig(**hf)
    ref_config._attn_implementation = "eager"
    torch.manual_seed(33)
    ref = GptOssForCausalLM(ref_config).cuda().train()
    ref.save_pretrained(tmp_path)
    adapter = GptOssAdapter()
    model = adapter.build(adapter.config_from_hf(hf)).cuda().train()
    adapter.load_weights(model, tmp_path)
    tokens = torch.tensor([[1, 3, 5, 7, 9, 11, 13]], device="cuda")
    expected = ref(tokens).logits
    actual = model(tokens, train_meta=TrainMeta()).logits_shard
    print("TINY_LOGIT_DIFF", (actual - expected).abs().max().item())
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    # Packed sequences must not leak attention across question boundaries.
    packed = model(
        tokens,
        position_ids=torch.tensor([[0, 1, 2, 0, 1, 2, 3]], device="cuda"),
        train_meta=TrainMeta(
            cu_seqlens=torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32), max_seqlen=4, packed=True
        ),
    ).logits_shard
    separated = torch.cat([model(tokens[:, :3]).logits_shard, model(tokens[:, 3:]).logits_shard], 1)
    torch.testing.assert_close(packed, separated, atol=3e-5, rtol=3e-4)
    loss = actual.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    saved = tmp_path / "roundtrip"
    adapter.save_weights(model, saved, tmp_path)
    restored = adapter.build(adapter.config_from_hf(hf)).cuda().train()
    adapter.load_weights(restored, saved)
    torch.testing.assert_close(restored(tokens).logits_shard, actual.detach(), atol=0, rtol=0)


@pytest.mark.parametrize("window", [None, 3])
def test_paged_sink_cuda(window):
    from areno.accel.sink_attention import paged_sink_attention

    torch.manual_seed(31)
    q = torch.randn(1, 6, 4, 64, device="cuda")
    k = torch.randn(1, 6, 2, 64, device="cuda")
    v = torch.randn_like(k)
    s = torch.randn(4, device="cuda")
    kc = torch.zeros(2, 4, 2, 64, device="cuda")
    vc = torch.zeros_like(kc)
    kc.reshape(-1, 2, 64)[:6] = k[0]
    vc.reshape(-1, 2, 64)[:6] = v[0]
    table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    sequences = torch.zeros(6, device="cuda", dtype=torch.long)
    positions = torch.arange(6, device="cuda")

    def run():
        return paged_sink_attention(q[0], kc, vc, s, table, sequences, positions, window)

    actual = run()
    expected = dense_reference(q, k, v, s, window=window)[0]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=3e-6, rtol=2e-5)


def test_gpt_oss_routing_capture_and_checkpoint_replay_cuda():
    from areno.engine.runtime.metadata import InferMeta
    from areno.engine.runtime.recompute import checkpoint_layer
    from areno.engine.runtime.routing_replay import captured_routing, routing_replay_context
    from areno.models.gpt_oss.model import GptOssMLP

    torch.manual_seed(17)
    config = GptOssAdapter().config_from_hf({**tiny_config(), "torch_dtype": "bfloat16"})
    mlp = GptOssMLP(config, 0).cuda()
    for parameter in mlp.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    states = torch.randn(1, 3, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    infer_meta = InferMeta(mode="prefill", capture_routing=True)
    with torch.no_grad(), routing_replay_context(infer_meta):
        mlp(states)
    routes = captured_routing(infer_meta)
    assert routes.shape == (3, 1, config.num_experts_per_tok)
    replay = (routes + 1) % config.num_experts
    train_meta = TrainMeta(routing_replay=replay, activation_checkpointing=True)
    observed = []
    hook = mlp.experts.register_forward_pre_hook(lambda module, args: observed.append(args[1].detach().clone()))
    try:
        output = checkpoint_layer(lambda x: mlp(x, train_meta=train_meta), states, train_meta=train_meta)
        output.float().square().sum().backward()
    finally:
        hook.remove()
    assert len(observed) >= 2
    for ids in observed:
        torch.testing.assert_close(ids.long(), replay[:, 0].long(), atol=0, rtol=0)
    assert torch.isfinite(mlp.router.weight.grad).all()
    assert mlp.router.weight.grad.abs().sum() > 0
