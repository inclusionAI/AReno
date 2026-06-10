"""Generic Llama-family causal-LM adapter.

This covers HF checkpoints that expose ``model_type == "llama"`` and the
standard Llama parameter layout. MiniCPM5-1B falls into this path: it uses
SwiGLU MLPs, GQA attention, RMSNorm, full RoPE, and untied output embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from areno.engine.checkpoints.common import load_checkpoint_weights, save_checkpoint_weights
from areno.engine.config import ModelConfig, _parse_dtype
from areno.models.base import ModelAdapter
from areno.models.llama.checkpoint import CHECKPOINT_SPEC
from areno.models.qwen3.model import Qwen3ForCausalLM


class LlamaAdapter(ModelAdapter):
    """Adapter glue for standard Llama-style dense decoder checkpoints."""

    name = "llama"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        return str(hf_config.get("model_type", "")).lower() == "llama"

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        dtype = _parse_dtype(hf_config.get("torch_dtype") or hf_config.get("dtype"))
        hidden_size = int(hf_config["hidden_size"])
        num_attention_heads = int(hf_config["num_attention_heads"])
        return ModelConfig(
            model_type=self.name,
            vocab_size=int(hf_config["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(hf_config["intermediate_size"]),
            num_hidden_layers=int(hf_config["num_hidden_layers"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(hf_config.get("num_key_value_heads", num_attention_heads)),
            head_dim=int(hf_config.get("head_dim", hidden_size // num_attention_heads)),
            rms_norm_eps=float(hf_config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(hf_config.get("rope_theta", 10_000.0)),
            max_position_embeddings=int(hf_config.get("max_position_embeddings", 8192)),
            tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
            qkv_bias=bool(hf_config.get("attention_bias", hf_config.get("qkv_bias", hf_config.get("bias", False)))),
            qk_norm=False,
            dtype=dtype,
            hidden_act=str(hf_config.get("hidden_act", "silu")),
            sequence_parallel=bool(hf_config.get("sequence_parallel", True)),
        )

    def build(self, config: ModelConfig) -> nn.Module:
        if config.hidden_act != "silu":
            raise ValueError(f"LlamaAdapter only supports hidden_act='silu', got {config.hidden_act!r}")
        return Qwen3ForCausalLM(config)

    @torch.no_grad()
    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        if not isinstance(model, Qwen3ForCausalLM):
            raise TypeError(f"LlamaAdapter cannot load weights into {type(model)!r}")
        load_checkpoint_weights(model, model_path, CHECKPOINT_SPEC)

    @torch.no_grad()
    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        if not isinstance(model, Qwen3ForCausalLM):
            raise TypeError(f"LlamaAdapter cannot save weights from {type(model)!r}")
        return save_checkpoint_weights(model, output_path, source_path, CHECKPOINT_SPEC)
