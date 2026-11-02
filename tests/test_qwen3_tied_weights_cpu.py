"""Dense Qwen3 parameter identity, gradient ownership and checkpoint contract."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import areno.engine.layers.norm as norm_module
import areno.engine.layers.vocab as vocab_module
import areno.models.qwen3.model as qwen_module
from areno.engine.checkpoints.io import SafetensorsIndex
from areno.engine.config import ModelConfig, OptimizerConfig
from areno.engine.modeling import build_optimizer
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.models.qwen3.model import Qwen3Adapter, Qwen3ForCausalLM


@pytest.fixture(autouse=True)
def cpu_context():
    previous = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    try:
        yield
    finally:
        set_tp_context(previous)


def tiny_config(tied=True):
    return ModelConfig(
        vocab_size=13,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        tie_word_embeddings=tied,
        dtype=torch.float32,
    )


@pytest.fixture
def cpu_vocabulary_math(monkeypatch):
    # CUDA/NPU kernels are unavailable on CPU. Substitute only their PyTorch
    # mathematical equivalents, keeping native modules/parameters/checkpoint IO.
    monkeypatch.setattr(
        vocab_module, "areno_vocab_embedding", lambda ids, weight, start, end: F.embedding(ids - start, weight)
    )
    monkeypatch.setattr(
        norm_module,
        "_areno_rmsnorm_no_compile",
        lambda x, weight, eps: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight,
    )


@pytest.mark.parametrize("tied", [True, False])
def test_parameter_identity_and_unique_count(tied):
    model = Qwen3ForCausalLM(tiny_config(tied))
    assert (model.embed_tokens.weight is model.lm_head.weight) == tied
    parameters = list(model.parameters())
    assert sum(p is model.embed_tokens.weight for p in parameters) == 1
    expected = 13 * 8 * (1 if tied else 2) + 8
    assert sum(p.numel() for p in parameters) == expected
    model = model.to(dtype=torch.float64)
    assert (model.embed_tokens.weight is model.lm_head.weight) == tied


@pytest.mark.parametrize("tied,expected", [(True, 596049920), (False, 751632384)])
def test_pinned_qwen06b_unique_parameter_count_on_meta(tied, expected):
    # Architecture fields from Qwen3-0.6B revision09b42cad, config SHA256
    # 660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd.
    hf_config = dict(
        model_type="qwen3",
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        max_position_embeddings=40960,
        tie_word_embeddings=tied,
        torch_dtype="bfloat16",
        attention_bias=False,
    )
    with torch.device("meta"):
        model = Qwen3Adapter().build(Qwen3Adapter().config_from_hf(hf_config))
    assert sum(p.numel() for p in model.parameters()) == expected
    assert (model.embed_tokens.weight is model.lm_head.weight) == tied


@pytest.mark.parametrize("galore", [False, True])
def test_summed_lookup_output_gradient_and_single_adam_update(cpu_vocabulary_math, galore):
    torch.manual_seed(593)
    model = Qwen3ForCausalLM(tiny_config())
    tokens = torch.tensor([[1, 3, 1], [4, 2, 8]])
    labels = torch.tensor([[3, 1, 5], [2, 8, 9]])
    initial = model.embed_tokens.weight.detach().clone()
    lookup = torch.nn.Parameter(initial.clone())
    head = torch.nn.Parameter(initial.clone())
    hidden = F.embedding(tokens, lookup)
    hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + model.norm.eps)
    reference_loss = F.cross_entropy(F.linear(hidden, head).reshape(-1, 13), labels.flatten())
    reference_loss.backward()
    expected_grad = lookup.grad + head.grad
    actual_loss = F.cross_entropy(model(tokens).logits_shard.reshape(-1, 13), labels.flatten())
    actual_loss.backward()
    torch.testing.assert_close(model.embed_tokens.weight.grad, expected_grad, atol=1e-6, rtol=1e-4)
    assert model.lm_head.weight.grad is model.embed_tokens.weight.grad
    config = OptimizerConfig(lr=0.003, betas=(0.8, 0.9), weight_decay=0.1, galore_rank=128 if galore else None)
    context = SimpleNamespace(world_size=1, dp_size=1, dp_rank=0, dp_group=None)
    optimizer = build_optimizer(model.parameters(), config, context)
    assert sum(p is model.embed_tokens.weight for p in optimizer.model_params) == 1
    expected = torch.nn.Parameter(initial.clone())
    reference = torch.optim.AdamW([expected], lr=0.003, betas=(0.8, 0.9), weight_decay=0.1, foreach=False)
    expected.grad = expected_grad.clone()
    reference.step()
    optimizer.step()
    torch.testing.assert_close(model.embed_tokens.weight, expected, atol=1e-6, rtol=1e-4)
    assert model.embed_tokens.weight is model.lm_head.weight


@pytest.mark.parametrize("galore", [False, True])
def test_zero_gradient_full_decay_happens_once(galore):
    model = Qwen3ForCausalLM(tiny_config())
    before = model.embed_tokens.weight.detach().clone()
    model.embed_tokens.weight.grad = torch.zeros_like(before)
    config = OptimizerConfig(lr=0.01, weight_decay=0.2, galore_rank=128 if galore else None)
    optimizer = build_optimizer(
        model.parameters(), config, SimpleNamespace(world_size=1, dp_size=1, dp_rank=0, dp_group=None)
    )
    optimizer.step()
    torch.testing.assert_close(model.embed_tokens.weight, before * 0.998, atol=1e-6, rtol=1e-4)
    assert sum(p is model.embed_tokens.weight for p in optimizer.model_params) == 1


@pytest.mark.parametrize("tied", [True, False])
def test_trained_model_only_checkpoint_preserves_logits(tmp_path, monkeypatch, cpu_vocabulary_math, tied):
    monkeypatch.setenv("ARENO_CKPT_PROGRESS", "0")
    torch.manual_seed(7)
    config = tiny_config(tied)
    model = Qwen3ForCausalLM(config)
    tokens = torch.tensor([[1, 3, 1], [4, 2, 8]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        F.cross_entropy(model(tokens).logits_shard.reshape(-1, 13), tokens.flatten()).backward()
        optimizer.step()
    expected = model(tokens).logits_shard.detach().clone()
    if not tied:
        assert not torch.equal(model.embed_tokens.weight, model.lm_head.weight)
    path = tmp_path / "checkpoint"
    Qwen3Adapter().save_weights(model, path, None)
    index = SafetensorsIndex(path, progress=False)
    try:
        assert ("lm_head.weight" in index.weight_map) == (not tied)
    finally:
        index.close()
    assert not (path / "training_state.pt").exists()
    restored = Qwen3ForCausalLM(config)
    Qwen3Adapter().load_weights(restored, path)
    assert (restored.embed_tokens.weight is restored.lm_head.weight) == tied
    torch.testing.assert_close(restored(tokens).logits_shard, expected, atol=0, rtol=0)
    assert json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]


def test_tie_rejects_mismatched_local_layout(monkeypatch):
    original = qwen_module.VocabParallelLMHead

    def mismatched(*args, **kwargs):
        module = original(*args, **kwargs)
        module.vocab_start += 1
        return module

    monkeypatch.setattr(qwen_module, "VocabParallelLMHead", mismatched)
    with pytest.raises(ValueError, match="identical local vocabulary layouts"):
        Qwen3ForCausalLM(tiny_config())


@pytest.mark.parametrize("tied", [True, False])
def test_source_passthrough_does_not_restore_runtime_owned_head(tmp_path, monkeypatch, cpu_vocabulary_math, tied):
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    monkeypatch.setenv("ARENO_CKPT_PROGRESS", "0")
    torch.manual_seed(21)
    config = tiny_config(tied)
    source = tmp_path / "source"
    source.mkdir()
    model = Qwen3ForCausalLM(config)
    initial_embedding = model.embed_tokens.weight.detach().clone()
    source_tensors = {
        "model.embed_tokens.weight": initial_embedding,
        "lm_head.weight": initial_embedding.clone(),
        "model.norm.weight": model.norm.weight.detach().clone(),
        "vision_tower.passthrough.weight": torch.tensor([3.0, 7.0]),
    }
    save_file(source_tensors, source / "model.safetensors", metadata={"format": "pt"})
    hf_config = dict(
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        vocab_size=13,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        tie_word_embeddings=tied,
        torch_dtype="float32",
        rms_norm_eps=config.rms_norm_eps,
    )
    (source / "config.json").write_text(json.dumps(hf_config))
    Qwen3Adapter().load_weights(model, source)
    ids = torch.tensor([[1, 3, 1], [4, 2, 8]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        F.cross_entropy(model(ids).logits_shard.reshape(-1, 13), ids.flatten()).backward()
        optimizer.step()
    assert not torch.equal(model.embed_tokens.weight, initial_embedding)
    expected = model(ids).logits_shard.detach().clone()
    output = tmp_path / "trained"
    Qwen3Adapter().save_weights(model, output, source)
    index = SafetensorsIndex(output, progress=False)
    try:
        assert ("lm_head.weight" in index.weight_map) == (not tied)
        torch.testing.assert_close(
            index.get_tensor("model.embed_tokens.weight"), model.embed_tokens.weight, atol=0, rtol=0
        )
        torch.testing.assert_close(index.get_tensor("vision_tower.passthrough.weight"), torch.tensor([3.0, 7.0]))
        if not tied:
            torch.testing.assert_close(index.get_tensor("lm_head.weight"), model.lm_head.weight, atol=0, rtol=0)
    finally:
        index.close()
    native = Qwen3ForCausalLM(config)
    Qwen3Adapter().load_weights(native, output)
    torch.testing.assert_close(native(ids).logits_shard, expected, atol=0, rtol=0)
    hf = AutoModelForCausalLM.from_pretrained(output, local_files_only=True)
    assert (hf.model.embed_tokens.weight is hf.lm_head.weight) == tied
    torch.testing.assert_close(hf.model.embed_tokens.weight, model.embed_tokens.weight, atol=0, rtol=0)
    torch.testing.assert_close(hf.lm_head.weight, model.lm_head.weight, atol=0, rtol=0)
    torch.testing.assert_close(hf(ids).logits, expected, atol=1e-6, rtol=1e-4)
