"""Native CUDA hybrid model, backward, optimizer update and prompt-only serving."""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_qwen35_bi_attn_train_and_serve_cuda(tmp_path):
    from areno.api.jev.labels.cuda import score_prompt_labels
    from areno.engine.config import ModelConfig
    from areno.engine.runtime.metadata import TrainMeta
    from areno.models.qwen3_5.model import Qwen35ForCausalLM

    torch.manual_seed(17)
    config = ModelConfig(
        model_type="qwen3_5",
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=("linear_attention", "full_attention"),
        attn_output_gate=True,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        dtype=torch.bfloat16,
        attn_backend="native",
    )
    model = Qwen35ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7]], device="cuda")
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]], device="cuda")
    meta = TrainMeta(
        cu_seqlens=torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32),
        max_seqlen=3,
        packed=True,
        bi_attn=True,
        activation_checkpointing=True,
    )
    output = model(tokens, position_ids=positions, train_meta=meta).logits_shard
    changed = tokens.clone()
    changed[0, 3:] = 12
    other = model(changed, position_ids=positions, train_meta=meta).logits_shard
    torch.testing.assert_close(output[0, :3], other[0, :3], atol=2e-3, rtol=2e-3)
    loss = -output[0, [2, 5]][:, [8, 9]].float().log_softmax(-1)[:, 0].mean()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        actual = score_prompt_labels(model, [[2, 3, 4]], [[8, 9]], device="cuda", bi_attn=True)
        one = TrainMeta(
            cu_seqlens=torch.tensor([0, 3], device="cuda", dtype=torch.int32), max_seqlen=3, packed=True, bi_attn=True
        )
        logits = model(tokens[:, :3], position_ids=positions[:, :3], train_meta=one).logits_shard[0, -1]
        expected = logits.float().log_softmax(-1)[[8, 9]]
        torch.testing.assert_close(torch.tensor(actual[0], device="cuda"), expected, atol=2e-3, rtol=2e-3)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = copy.deepcopy(model)
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.inference_mode():
        repeat = score_prompt_labels(restored, [[2, 3, 4]], [[8, 9]], device="cuda", bi_attn=True)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(repeat))


def test_ling_full_attention_bi_attn_cuda():
    from areno.engine.config import ModelConfig
    from areno.engine.runtime.metadata import TrainMeta
    from areno.models.bailing.model import BailingSoftmaxAttention

    config = ModelConfig(
        model_type="bailing_moe_linear",
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        dtype=torch.bfloat16,
        attn_backend="native",
    )
    model = BailingSoftmaxAttention(config, 0).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)
    x = torch.randn(1, 6, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]], device="cuda")
    meta = TrainMeta(
        cu_seqlens=torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32), max_seqlen=3, packed=True, bi_attn=True
    )
    out = model(x, positions, meta, None)
    out.float().square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    changed = x.detach().clone()
    changed[:, 3:] += 10
    other = model(changed, positions, meta, None)
    torch.testing.assert_close(out[:, :3], other[:, :3])
