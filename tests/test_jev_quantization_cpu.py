"""JEV integration with generic online Q4 and quantized backbone LoRA."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("wrapped", [False, True])
def test_generic_online_quantization_joint_head_lora_training_and_reload(tmp_path, wrapped):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from areno import LoraConfig
    from areno.api.backend.mlx.decision import initialize_head, train_questions
    from areno.api.backend.mlx.lora import export_peft_adapter, initialize_lora, load_peft_adapter
    from areno.api.backend.mlx.provider import MlxModelProvider
    from areno.api.backend.mlx.quantization import quantize_online
    from areno.api.jev.data import encode_question

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

    class Provider(MlxModelProvider):
        def forward_hidden_states(self, batch):
            return self.model.body(batch["input_ids"])

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(c) % 64 for c in text]

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
        provider = Provider(model, Tokenizer(), None, config)
        optimizer = optim.AdamW(learning_rate=1e-3)
        backend = SimpleNamespace(
            model=model,
            provider=provider,
            model_config=config,
            optimizer=optimizer,
            config=SimpleNamespace(
                head_lr=1e-3,
                decision_checkpoint=None,
                decision_head=True,
                model_path="original",
                base_model_name_or_path="original",
                optimizer={"grad_clip_norm": 1.0},
            ),
            _optimizer_groups=[("model", optimizer, {"lr": 1e-3})],
            _policy_version=0,
        )
        initialize_head(backend)
        rows = encode_question(
            {"prompt": "state", "candidates": ["a", "bc", "def"], "question_id": "q", "target": [0.1, 0.2, 0.7]},
            Tokenizer(),
            20,
        )
        frozen = {
            name: np.array(value)
            for name, value in tree_flatten(model.parameters())
            if name not in dict(tree_flatten(model.trainable_parameters()))
        }
        for step in range(2):
            result = train_questions(backend, SimpleNamespace(global_step=step), rows)
            assert np.isfinite(result["loss"])
            assert result["head_update_norm"] > 0 and result["lora_update_norm"] > 0
        after = dict(tree_flatten(model.parameters()))
        assert all(np.array_equal(value, np.array(after[name])) for name, value in frozen.items())
        export_peft_adapter(lora, tmp_path, base_model_name_or_path="original")
        before = {name: np.array(slot.lora_b) for name, slot in lora.slots.items()}
        load_peft_adapter(lora, tmp_path)
        assert all(np.array_equal(value, np.array(lora.slots[name].lora_b)) for name, value in before.items())
