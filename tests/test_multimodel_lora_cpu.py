"""Focused native-LoRA target-contract checks across model adapters."""

from __future__ import annotations

import pytest
import torch

from areno.adapters import LoraConfig
from areno.adapters.lora import initialize_lora
from areno.engine.config import ModelConfig
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.models.olmo2 import Olmo2ForCausalLM
from areno.models.phi4mm import Phi4MMForCausalLM


@pytest.fixture(autouse=True)
def _cpu_tp_context():
    previous = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    try:
        yield
    finally:
        set_tp_context(previous)


def _dense_config(model_type: str) -> ModelConfig:
    return ModelConfig(
        model_type=model_type,
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        dtype=torch.float32,
        attn_backend="native",
        sequence_parallel=False,
        tie_word_embeddings=model_type == "phi4mm",
        hf_text_config=(
            {
                "original_max_position_embeddings": 32,
                "rope_scaling": {
                    "type": "longrope",
                    "short_factor": (1.0, 1.0, 1.0, 1.0),
                    "long_factor": (1.0, 2.0, 3.0, 4.0),
                },
            }
            if model_type == "phi4mm"
            else None
        ),
    )


@pytest.mark.parametrize(
    ("model", "targets"),
    (
        (
            lambda: Olmo2ForCausalLM(_dense_config("olmo2")),
            ("layers.0.self_attn.q_proj", "layers.0.mlp.down_proj"),
        ),
        (
            lambda: Phi4MMForCausalLM(_dense_config("phi4mm")),
            ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"),
        ),
    ),
)
def test_dense_model_exact_lora_targets_resolve(model, targets: tuple[str, ...]) -> None:
    policy = model()
    registry = initialize_lora(
        policy,
        LoraConfig(rank=2, alpha=2.0, target_modules=targets),
        seed=7,
    )

    assert tuple(registry.slots) == targets
    assert all(parameter.requires_grad for parameter in registry.parameters())
    trainable_ids = {id(parameter) for parameter in registry.parameters()}
    assert all(parameter.requires_grad == (id(parameter) in trainable_ids) for parameter in policy.parameters())
