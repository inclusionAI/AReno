"""Lightweight OLMo 2 configuration translation."""

from __future__ import annotations

from typing import Any

from areno.engine.config import ModelConfig, _parse_dtype


def config_from_hf(hf_config: dict[str, Any]) -> ModelConfig:
    """Translate a Hugging Face OLMo 2 config into AReno's model config."""

    hidden_size = int(hf_config["hidden_size"])
    num_attention_heads = int(hf_config["num_attention_heads"])
    return ModelConfig(
        model_type="olmo2",
        vocab_size=int(hf_config["vocab_size"]),
        hidden_size=hidden_size,
        intermediate_size=int(hf_config["intermediate_size"]),
        num_hidden_layers=int(hf_config["num_hidden_layers"]),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(hf_config.get("num_key_value_heads", num_attention_heads)),
        head_dim=int(hf_config.get("head_dim", hidden_size // num_attention_heads)),
        rms_norm_eps=float(hf_config.get("rms_norm_eps", 1e-5)),
        rope_theta=float(hf_config.get("rope_theta", 10_000.0)),
        max_position_embeddings=int(hf_config.get("max_position_embeddings", 2048)),
        tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
        qkv_bias=bool(hf_config.get("attention_bias", False)),
        qk_norm=False,
        dtype=_parse_dtype(hf_config.get("torch_dtype") or hf_config.get("dtype")),
        hidden_act=str(hf_config.get("hidden_act", "silu")),
        sequence_parallel=bool(hf_config.get("sequence_parallel", True)),
    )
