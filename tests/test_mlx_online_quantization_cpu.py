"""Capability-based online quantization: no model family or checkpoint rewriting."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from areno.api.backend.mlx.quantization import METADATA_FILE, adapter_quantization, validate_quantization


def test_quantization_metadata_validates_and_missing_metadata_preserves_dense_mode(tmp_path):
    assert adapter_quantization(tmp_path) is None
    (tmp_path / METADATA_FILE).write_text(json.dumps({"format": "mlx-affine-v1", "bits": 4, "group_size": 64}))
    assert adapter_quantization(tmp_path)["bits"] == 4
    for bits, group in [(7, 64), (4, 48), (0, 64)]:
        with pytest.raises(ValueError):
            validate_quantization(bits, group)


@pytest.mark.parametrize("wrapped", [False, True])
def test_generic_online_quantization_lora_training_and_reload(tmp_path, wrapped):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from areno import LoraConfig
    from areno.api.backend.mlx.lora import export_peft_adapter, initialize_lora, load_peft_adapter
    from areno.api.backend.mlx.quantization import quantize_online

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)

        def __call__(self, x):
            return nn.gelu(self.q_proj(x))

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 64)
            self.layers = [Block(), Block()]

        def __call__(self, tokens):
            h = mx.cumsum(self.embed(tokens), axis=1)
            for layer in self.layers:
                h = h + layer(h)
            return h

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            if wrapped:
                self.encoder = nn.Module()
                self.encoder.model = Body()
            else:
                self.model = Body()

        @property
        def layers(self):
            return self.body.layers

        @property
        def body(self):
            return self.encoder.model if wrapped else self.model

    with mx.stream(mx.cpu):
        mx.random.seed(12)
        model = Model()
        original = dict(tree_flatten(model.parameters()))
        original_arrays = {name: np.array(value) for name, value in original.items()}
        config = quantize_online(model, {"model_type": "arbitrary", "hidden_size": 64}, bits=4, group_size=32)
        assert config["areno_online_quantization"]["modules"] == 3
        assert all(np.array_equal(np.array(original[name]), value) for name, value in original_arrays.items())
        # A second arbitrary layout uses the same capability-based adapter path.
        lora = initialize_lora(model, LoraConfig(rank=4, alpha=8, target_modules=("q_proj",)), model_type="arbitrary")
        assert len(lora.slots) == 2
        assert all(isinstance(slot.linear, nn.QuantizedLinear) for slot in lora.slots.values())
        optimizer = optim.AdamW(learning_rate=1e-3)
        tokens = mx.array([[1, 2, 3], [4, 5, 6]])
        target = mx.zeros((2, 3, 64))

        def objective(policy):
            return mx.mean((policy.body(tokens) - target) ** 2)

        before = {name: np.array(slot.lora_b) for name, slot in lora.slots.items()}
        frozen = {
            name: np.array(value)
            for name, value in tree_flatten(model.parameters())
            if name not in dict(tree_flatten(model.trainable_parameters()))
        }
        for _ in range(2):
            loss, gradients = nn.value_and_grad(model, objective)(model)
            optimizer.update(model, gradients)
            mx.eval(loss, model.parameters(), optimizer.state)
            assert np.isfinite(loss.item())
        assert any(not np.array_equal(value, np.array(lora.slots[name].lora_b)) for name, value in before.items())
        after = dict(tree_flatten(model.parameters()))
        assert all(np.array_equal(value, np.array(after[name])) for name, value in frozen.items())
        expected = np.array(model.body(tokens))
        expected_adapters = {name: (np.array(slot.lora_a), np.array(slot.lora_b)) for name, slot in lora.slots.items()}
        export_peft_adapter(lora, tmp_path, base_model_name_or_path="original")
        for slot in lora.slots.values():
            slot.lora_b = mx.zeros_like(slot.lora_b)
        assert not np.array_equal(np.array(model.body(tokens)), expected)
        load_peft_adapter(lora, tmp_path)
        for name, (lora_a, lora_b) in expected_adapters.items():
            np.testing.assert_array_equal(np.array(lora.slots[name].lora_a), lora_a)
            np.testing.assert_array_equal(np.array(lora.slots[name].lora_b), lora_b)
        # PEFT transposes can change strides and FP32 matmul accumulation order.
        np.testing.assert_allclose(np.array(model.body(tokens)), expected, atol=1e-7, rtol=1e-6)


def test_online_provider_loads_tokenizer_independently_and_model_lazily(monkeypatch, tmp_path):
    pytest.importorskip("mlx.core")
    from areno.api.backend.mlx.provider import MlxLmProvider

    events = []
    model, tokenizer = object(), SimpleNamespace(eos_token_id=2)
    monkeypatch.setattr("areno.api.tokenizer.load_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr("areno.api.tokenizer.eos_token_ids", lambda path, tok: (2,))
    monkeypatch.setattr("mlx_lm.tokenizer_utils.TokenizerWrapper", lambda tok, **kwargs: tok)

    def load_model(path, *, lazy):
        assert path == tmp_path and lazy
        events.append("lazy_model")
        return model, {"hidden_size": 64}

    monkeypatch.setattr("mlx_lm.utils.load_model", load_model)

    def quantize(policy, config, **kwargs):
        assert policy is model and kwargs == {"bits": 4, "group_size": 64}
        events.append("quantize")
        return {**config, "quantization": kwargs}

    monkeypatch.setattr("areno.api.backend.mlx.quantization.quantize_online", quantize)
    provider = MlxLmProvider.load(
        str(tmp_path), config={}, adapter_path=None, quantization={"bits": 4, "group_size": 64}
    )
    assert provider.model is model and provider.tokenizer is tokenizer
    assert events == ["lazy_model", "quantize"]
