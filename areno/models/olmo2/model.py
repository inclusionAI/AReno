"""AReno-native OLMo 2 causal language model.

OLMo 2 differs from Llama-style decoders in two material ways: attention and
MLP outputs are normalized before their residual additions, and Q/K RMSNorm is
applied across each complete projected vector before it is split into heads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed.nn.functional as dist_nn
from torch import nn

from areno.engine.checkpoints.common import load_checkpoint_weights, save_checkpoint_weights
from areno.engine.config import ModelConfig
from areno.engine.layers.attention import CausalSelfAttention
from areno.engine.layers.linear import mark_tensor_parallel_parameter
from areno.engine.layers.norm import RMSNorm
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.models.base import ModelAdapter
from areno.models.olmo2.checkpoint import CHECKPOINT_SPEC
from areno.models.olmo2.config import config_from_hf
from areno.models.olmo2.semantics import post_norm_residual, projected_rms_norm
from areno.models.qwen3.model import Qwen3ForCausalLM, QwenDecoderLayer


class Olmo2ProjectedRMSNorm(nn.Module):
    """RMSNorm a TP-sharded projection using a global channel statistic."""

    def __init__(self, local_size: int, global_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(local_size, dtype=torch.float32))
        mark_tensor_parallel_parameter(self.weight, True, sequence_parallel=False)
        self.global_size = global_size
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        squared_sum = hidden_states.float().square().sum(dim=-1, keepdim=True)
        ctx = get_tp_context()
        if ctx.world_size > 1:
            squared_sum = dist_nn.all_reduce(squared_sum, group=ctx.group)
        return projected_rms_norm(hidden_states, squared_sum, self.weight, self.global_size, self.eps)


class Olmo2SelfAttention(CausalSelfAttention):
    """OLMo 2 attention with RMSNorm over the full global Q/K projections."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = Olmo2ProjectedRMSNorm(
            self.local_heads * self.head_dim,
            config.num_attention_heads * config.head_dim,
            config.rms_norm_eps,
        )
        self.k_norm = Olmo2ProjectedRMSNorm(
            self.local_kv_heads * self.head_dim,
            config.num_key_value_heads * config.head_dim,
            config.rms_norm_eps,
        )

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
        hidden_states = post_norm_residual(residual, hidden_states, self.post_attention_layernorm)
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        return post_norm_residual(residual, hidden_states, self.post_feedforward_layernorm)


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
        return config_from_hf(hf_config)

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
