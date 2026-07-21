"""AReno-native OLMo 2 causal language model.

OLMo 2 differs from Llama-style decoders in two material ways: attention and
MLP outputs are normalized before their residual additions, and Q/K RMSNorm is
applied across each complete projected vector before it is split into heads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from areno.engine.checkpoints.common import load_checkpoint_weights, save_checkpoint_weights
from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.attention import CausalSelfAttention
from areno.engine.layers.norm import RMSNorm
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.models.base import ModelAdapter
from areno.models.olmo2.checkpoint import CHECKPOINT_SPEC
from areno.models.qwen3.model import Qwen3ForCausalLM, QwenDecoderLayer


class Olmo2SelfAttention(CausalSelfAttention):
    """OLMo 2 attention with RMSNorm over the full local Q/K projections."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = RMSNorm(self.local_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.local_kv_heads * self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> torch.Tensor:
        batch, seqlen, _ = hidden_states.shape
        q_size = self.local_heads * self.head_dim
        kv_size = self.local_kv_heads * self.head_dim
        q, k, v = self.qkv_proj(hidden_states).split((q_size, kv_size, kv_size), dim=-1)
        q = self.q_norm(q).view(batch, seqlen, self.local_heads, self.head_dim)
        k = self.k_norm(k).view(batch, seqlen, self.local_kv_heads, self.head_dim)
        v = v.view(batch, seqlen, self.local_kv_heads, self.head_dim)
        q, k = self.rope(q, k, position_ids)
        if infer_meta is not None:
            return self.forward_infer(q, k, v, infer_meta)
        return self.forward_train(q, k, v, train_meta)


class Olmo2DecoderLayer(QwenDecoderLayer):
    """OLMo 2 post-norm attention and MLP residual block."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.input_layernorm
        self.self_attn = Olmo2SelfAttention(config, layer_idx)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, position_ids, train_meta, infer_meta)
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        return residual + self.post_feedforward_layernorm(hidden_states)


class Olmo2ForCausalLM(Qwen3ForCausalLM):
    """OLMo 2 causal LM using AReno's shared dense runtime lifecycle."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Olmo2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])


class Olmo2Adapter(ModelAdapter):
    """Adapter for Hugging Face checkpoints with ``model_type == 'olmo2'``."""

    name = "olmo2"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        return str(hf_config.get("model_type", "")).lower() == "olmo2"

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
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

    def build(self, config: ModelConfig) -> nn.Module:
        if config.hidden_act != "silu":
            raise ValueError(f"Olmo2Adapter only supports hidden_act='silu', got {config.hidden_act!r}")
        return Olmo2ForCausalLM(config)

    @torch.no_grad()
    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        if not isinstance(model, Olmo2ForCausalLM):
            raise TypeError(f"Olmo2Adapter cannot load weights into {type(model)!r}")
        load_checkpoint_weights(model, model_path, CHECKPOINT_SPEC)

    @torch.no_grad()
    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        if not isinstance(model, Olmo2ForCausalLM):
            raise TypeError(f"Olmo2Adapter cannot save weights from {type(model)!r}")
        return save_checkpoint_weights(model, output_path, source_path, CHECKPOINT_SPEC)
