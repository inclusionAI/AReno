from __future__ import annotations

import torch
from torch import nn

from areno.engine.config import ModelConfig
from areno.models.olmo2.model import Olmo2Adapter, Olmo2DecoderLayer, Olmo2ForCausalLM, Olmo2ProjectedRMSNorm


class _IdentityAttention(nn.Module):
    def forward(self, hidden_states, position_ids, train_meta=None, infer_meta=None):
        del position_ids, train_meta, infer_meta
        return hidden_states


def _config() -> ModelConfig:
    return ModelConfig(
        model_type="olmo2",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        dtype=torch.float32,
        attn_backend="native",
        sequence_parallel=False,
    )


def test_olmo2_config_translation_matches_checkpoint():
    config = Olmo2Adapter().config_from_hf(
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


def test_olmo2_build_uses_full_projected_qk_norms():
    model = Olmo2Adapter().build(_config())

    assert isinstance(model, Olmo2ForCausalLM)
    assert tuple(model.layers[0].self_attn.q_norm.weight.shape) == (32,)
    assert tuple(model.layers[0].self_attn.k_norm.weight.shape) == (32,)


def test_olmo2_projected_rmsnorm_matches_full_vector_reference():
    norm = Olmo2ProjectedRMSNorm(local_size=4, global_size=4, eps=1e-6)
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    output = norm(hidden)
    expected = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + 1e-6)

    torch.testing.assert_close(output, expected)


def test_olmo2_decoder_applies_post_norm_before_residual_add():
    layer = Olmo2DecoderLayer(_config(), 0)
    layer.self_attn = _IdentityAttention()
    layer.mlp = nn.Identity()
    layer.post_attention_layernorm = nn.Identity()
    layer.post_feedforward_layernorm = nn.Identity()
    hidden = torch.ones(1, 2, 32)

    output = layer(hidden, torch.arange(2).unsqueeze(0))

    torch.testing.assert_close(output, hidden * 4)
