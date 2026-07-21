from __future__ import annotations

import torch
from torch import nn

from areno.models.olmo2.checkpoint import CHECKPOINT_SPEC
from areno.models.olmo2.config import config_from_hf
from areno.models.olmo2.semantics import post_norm_residual, projected_rms_norm


def test_olmo2_config_translation_matches_checkpoint():
    config = config_from_hf(
        {
            "model_type": "olmo2",
            "vocab_size": 100352,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 500000,
            "max_position_embeddings": 4096,
            "torch_dtype": "bfloat16",
            "hidden_act": "silu",
            "tie_word_embeddings": False,
            "attention_bias": False,
        }
    )

    assert config.model_type == "olmo2"
    assert config.head_dim == 128
    assert config.rope_theta == 500000
    assert config.dtype == torch.bfloat16
    assert config.qk_norm is False


def test_olmo2_checkpoint_maps_projected_qk_norms():
    mappings = {(spec.key, spec.attr, spec.dim) for spec in CHECKPOINT_SPEC.layer.load_ops if hasattr(spec, "dim")}

    assert ("{prefix}.self_attn.q_norm.weight", "self_attn.q_norm.weight", 0) in mappings
    assert ("{prefix}.self_attn.k_norm.weight", "self_attn.k_norm.weight", 0) in mappings


def test_olmo2_projected_rmsnorm_matches_full_vector_reference():
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    weight = torch.ones(4)
    squared_sum = hidden.float().square().sum(dim=-1, keepdim=True)

    output = projected_rms_norm(hidden, squared_sum, weight, global_size=4, eps=1e-6)
    expected = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + 1e-6)

    torch.testing.assert_close(output, expected)


def test_olmo2_decoder_applies_post_norm_before_residual_add():
    hidden = torch.ones(1, 2, 32)

    output = post_norm_residual(hidden, hidden, nn.Identity())
    output = post_norm_residual(output, output, nn.Identity())

    torch.testing.assert_close(output, hidden * 4)
