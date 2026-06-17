"""Causal self-attention module shared by all dense transformer models.

Wires the tensor-parallel QKV/output projections, RoPE, optional QK
normalization and the two FlashAttention backends (train vs. infer) into a
single nn.Module. The forward dispatches by inspecting which metadata
container is present.
"""

from __future__ import annotations

import torch
from torch import nn

from areno.engine.config import ModelConfig
from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend, build_infer_attention_backend
from areno.engine.layers.attention_backend.train import build_train_attention_backend
from areno.engine.layers.linear import QKVParallelLinear, RowParallelLinear
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.rotary import RotaryEmbedding
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.metadata import InferMeta, TrainMeta


class CausalSelfAttention(nn.Module):
    """Grouped-query causal self-attention with TP-sharded projections.

    Heads are split across the tensor-parallel group: each rank owns
    ``num_heads // world_size`` query heads and the matching fraction of
    KV heads. The fused QKV column-parallel projection produces a single
    contiguous tensor that is split into Q/K/V slices using the local sizes
    reported by ``QKVParallelLinear``. The row-parallel output projection
    reduces across ranks to reassemble the full hidden state.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        ctx = get_tp_context()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.attn_backend = config.attn_backend
        # Local query head count after column-parallel sharding.
        self.local_heads = self.num_heads // ctx.world_size
        # Fused QKV column-parallel projection; ranks each own a contiguous
        # slice of the Q, K and V output channels.
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.qkv_bias,
        )
        # Local KV head count is derived from the shard layout so that GQA
        # configurations stay consistent with the actual weight partition.
        self.local_kv_heads = self.qkv_proj.local_out_features[1] // self.head_dim
        # Row-parallel output projection: input is already sharded along
        # head dimension, output is all-reduced across ranks.
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        # Optional per-head QK normalization (used by some recent models).
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.train_backend = build_train_attention_backend(self.attn_backend)
        # Inference backend and KV cache buffers are lazily attached: training
        # uses neither, and inference plumbs the cache in via set_kv_cache.
        self.infer_backend: FlashAttnInferBackend | None = None
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> torch.Tensor:
        """Project, normalize, apply RoPE, then dispatch to train or infer."""

        batch, seqlen, _ = hidden_states.shape
        q_size = self.local_heads * self.head_dim
        kv_size = self.local_kv_heads * self.head_dim
        qkv = self.qkv_proj(hidden_states)
        # Split the fused projection into Q, K, V using local sizes (after TP).
        q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(batch, seqlen, self.local_heads, self.head_dim)
        k = k.view(batch, seqlen, self.local_kv_heads, self.head_dim)
        v = v.view(batch, seqlen, self.local_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        # Rotary embedding is applied on the head dim using position-indexed
        # cos/sin tables; positions are broadcast across heads.
        q, k = self.rope(q, k, position_ids)

        # Presence of infer_meta selects the paged KV-cache backend; otherwise
        # we run the training-mode FlashAttention (padded or varlen packed).
        if infer_meta is not None:
            return self.forward_infer(q, k, v, infer_meta)
        return self.forward_train(q, k, v, train_meta)

    def forward_train(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, train_meta: TrainMeta | None
    ) -> torch.Tensor:
        """Run training FlashAttention then collapse heads back to hidden dim."""

        out = self.train_backend(q, k, v, train_meta)
        out = out.contiguous().view(q.shape[0], q.shape[1], self.local_heads * self.head_dim)
        return self.o_proj(out)

    def forward_infer(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, infer_meta: InferMeta) -> torch.Tensor:
        """Run inference attention against the per-layer paged KV cache."""

        if self.k_cache.numel() == 0 or self.v_cache.numel() == 0:
            raise RuntimeError("inference requires per-layer KV cache tensors")
        # Lazy backend init so the FlashAttention functions are only bound when
        # the layer actually serves an inference request.
        if self.infer_backend is None:
            self.infer_backend = build_infer_attention_backend(self.attn_backend)
        out = self.infer_backend(q, k, v, self.k_cache, self.v_cache, infer_meta)
        out = out.contiguous().view(q.shape[0], q.shape[1], self.local_heads * self.head_dim)
        return self.o_proj(out)

    def set_kv_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        """Attach paged KV-cache tensors owned by the runtime."""

        self.k_cache = k_cache
        self.v_cache = v_cache

    def clear_kv_cache(self) -> None:
        """Detach KV-cache buffers and the cached inference backend."""

        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        self.infer_backend = None
