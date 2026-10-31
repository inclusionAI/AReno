"""Native CUDA checks for dense Qwen3 tied-weight gradients and checkpoints."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from areno.engine.checkpoints.training_state import load_training_state, save_training_state
from areno.engine.config import ModelConfig, OptimizerConfig
from areno.engine.modeling import build_optimizer
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.models.qwen3.model import Qwen3Adapter, Qwen3ForCausalLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def native_context():
    previous = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cuda", 0), group=None))
    try:
        yield
    finally:
        set_tp_context(previous)


def make_model(tied=True, dtype=torch.float32):
    config = ModelConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        tie_word_embeddings=tied,
        dtype=dtype,
    )
    with torch.device("cuda"):
        return Qwen3ForCausalLM(config)


def tokens():
    return torch.tensor([[1, 3, 1, 7], [4, 2, 8, 2]], device="cuda", dtype=torch.long)


def train_loss(model):
    ids = tokens()
    return F.cross_entropy(model(ids).logits_shard.float().reshape(-1, 128), ids.flatten())


def logits(model):
    with torch.no_grad():
        return model(tokens()).logits_shard.detach().clone()


def test_native_gradient_sum_and_one_engine_adamw_update():
    torch.manual_seed(593)
    tied = make_model()
    untied = make_model(False)
    with torch.no_grad():
        untied.embed_tokens.weight.copy_(tied.embed_tokens.weight)
        untied.lm_head.weight.copy_(tied.embed_tokens.weight)
        untied.norm.weight.copy_(tied.norm.weight)
    train_loss(tied).backward()
    train_loss(untied).backward()
    expected_gradient = untied.embed_tokens.weight.grad + untied.lm_head.weight.grad
    assert tied.embed_tokens.weight is tied.lm_head.weight
    torch.testing.assert_close(tied.embed_tokens.weight.grad, expected_gradient, atol=1e-6, rtol=1e-4)
    expected = torch.nn.Parameter(tied.embed_tokens.weight.detach().clone())
    reference = torch.optim.AdamW([expected], lr=0.003, betas=(0.8, 0.9), weight_decay=0.1, foreach=False)
    expected.grad = expected_gradient.clone()
    reference.step()
    optimizer = build_optimizer(
        tied.parameters(), OptimizerConfig(lr=0.003, betas=(0.8, 0.9), weight_decay=0.1), get_tp_context()
    )
    assert sum(p is tied.embed_tokens.weight for p in optimizer.model_params) == 1
    optimizer.step()
    torch.testing.assert_close(tied.embed_tokens.weight, expected, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tied", [True, False])
def test_native_model_only_checkpoint_keeps_trained_logits(tmp_path, monkeypatch, dtype, tied):
    monkeypatch.setenv("ARENO_CKPT_PROGRESS", "0")
    torch.manual_seed(7)
    model = make_model(tied, dtype)
    before = model.embed_tokens.weight.detach().clone()
    optimizer = build_optimizer(model.parameters(), OptimizerConfig(lr=0.003, weight_decay=0.1), get_tp_context())
    for _ in range(3):
        optimizer.zero_grad()
        train_loss(model).backward()
        optimizer.step()
    assert not torch.equal(before, model.embed_tokens.weight)
    expected = logits(model)
    directory = tmp_path / "model-only"
    Qwen3Adapter().save_weights(model, directory, None)
    assert not (directory / "training_state.pt").exists()
    restored = make_model(tied, dtype)
    Qwen3Adapter().load_weights(restored, directory)
    assert (restored.embed_tokens.weight is restored.lm_head.weight) == tied
    torch.testing.assert_close(logits(restored), expected, atol=0, rtol=0)


def test_native_bfloat16_galore_sidecar_preserves_shared_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENO_CKPT_PROGRESS", "0")
    model = make_model(dtype=torch.bfloat16)
    config = OptimizerConfig(lr=0.003, weight_decay=0.1, galore_rank=128)
    optimizer = build_optimizer(model.parameters(), config, get_tp_context())
    train_loss(model).backward()
    optimizer.step()
    optimizer.zero_grad()
    directory = tmp_path / "with-state"
    Qwen3Adapter().save_weights(model, directory, None)

    def worker(m, opt):
        return SimpleNamespace(
            model=m,
            optimizer=opt,
            device=torch.device("cuda", 0),
            _global_step=1,
            base_lr=0.003,
            min_lr=0.003,
            lr_decay_steps=0,
            lr_warmup_steps=0,
            lr_decay_style="constant",
        )

    save_training_state(worker(model, optimizer), str(directory))
    payload = torch.load(directory / "training_state.pt", weights_only=True, map_location="cpu")
    assert payload["parameter_names"] == ["embed_tokens.weight", "norm.weight"]
    expected = logits(model)
    restored = make_model(dtype=torch.bfloat16)
    Qwen3Adapter().load_weights(restored, directory)
    new_optimizer = build_optimizer(restored.parameters(), config, get_tp_context())
    load_training_state(worker(restored, new_optimizer), str(directory))
    assert restored.embed_tokens.weight is restored.lm_head.weight
    assert sum(p is restored.embed_tokens.weight for p in new_optimizer.model_params) == 1
    torch.testing.assert_close(logits(restored), expected, atol=0, rtol=0)
    for old, new in zip(optimizer.state_dict()["state"], new_optimizer.state_dict()["state"], strict=True):
        for key, value in old.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, new[key])
            else:
                assert value == new[key]
