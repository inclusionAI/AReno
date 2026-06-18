"""Qwen3.5 text-backbone causal-LM adapter.

Qwen3.5 uses a hybrid decoder stack: most layers are Gated Delta-Net linear
attention and every ``full_attention_interval`` layer is standard softmax
attention. The multimodal wrapper is ignored here; callers feed token ids into
the language backbone directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from areno.accel import (
    areno_linear,
    areno_topk_softmax,
)
from areno.accel.ops import FusedMoeConfig, areno_fused_experts, areno_silu_and_mul, log_once
from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend, build_infer_attention_backend
from areno.engine.layers.attention_backend.train import build_train_attention_backend
from areno.engine.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    _shard_range,
    mark_tensor_parallel_parameter,
)
from areno.engine.layers.mlp import GatedMLP
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.rotary import PartialRotaryEmbedding
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.parallel.collectives import (
    all_reduce,
    copy_to_tensor_parallel_region,
    gather_from_sequence_parallel_region,
    is_sequence_parallel_active,
    scatter_to_sequence_parallel_region,
    sequence_parallel_region,
)
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.recompute import checkpoint_layer
from areno.models._shared.dynamo_wrappers import (
    _areno_depthwise_causal_conv1d_silu_decode_no_compile,
    _areno_depthwise_causal_conv1d_silu_no_compile,
    _areno_packed_depthwise_causal_conv1d_silu_no_compile,
    _areno_rmsnorm_silu_gate_no_compile,
    _areno_sigmoid_no_compile,
    _areno_softplus_no_compile,
    _fla_causal_conv1d_no_compile,
    _fla_chunk_gated_delta_rule_no_compile,
    _fla_fused_recurrent_gated_delta_rule_no_compile,
    _require_fla_gdn,
)
from areno.models.base import CausalLMOutput, ModelAdapter
from areno.models.qwen3.model import Qwen3MoeExperts

_REPLICATED_KV_GROUPS: dict[tuple[int, int, int, int, int], tuple[dist.ProcessGroup | None, int]] = {}


def _replicated_kv_grad_group(num_kv_heads: int) -> tuple[torch.distributed.ProcessGroup | None, int]:
    ctx = get_tp_context()
    replication = ctx.world_size // num_kv_heads
    if replication <= 1 or not dist.is_available() or not dist.is_initialized():
        return None, replication
    key = (ctx.global_world_size, ctx.dp_size, ctx.dp_rank, ctx.world_size, num_kv_heads)
    cached = _REPLICATED_KV_GROUPS.get(key)
    if cached is not None:
        return cached
    current_group = None
    for group_dp_rank in range(ctx.dp_size):
        base = group_dp_rank * ctx.world_size
        for kv_rank in range(num_kv_heads):
            start = base + kv_rank * replication
            ranks = [start + offset for offset in range(replication)]
            group = dist.new_group(ranks=ranks)
            if group_dp_rank == ctx.dp_rank and ctx.rank in [rank - base for rank in ranks]:
                current_group = group
    cached = (current_group, replication)
    _REPLICATED_KV_GROUPS[key] = cached
    return cached


class Qwen35FullAttention(nn.Module):
    """Softmax attention layer with optional Q output gate."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        ctx = get_tp_context()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.local_heads = self.num_heads // ctx.world_size
        self.attn_output_gate = config.attn_output_gate
        q_size = self.num_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        kv_size = self.num_kv_heads * self.head_dim
        if self.num_kv_heads % ctx.world_size == 0:
            self.qkv_proj = MergedColumnParallelLinear(config.hidden_size, (q_size, kv_size, kv_size), bias=False)
        elif ctx.world_size % self.num_kv_heads == 0:
            self.qkv_proj = Qwen35ReplicatedKVQKVLinear(
                config.hidden_size,
                self.head_dim,
                self.num_heads,
                self.num_kv_heads,
                attn_output_gate=self.attn_output_gate,
            )
        else:
            raise ValueError(
                "Qwen3.5 full attention requires num_key_value_heads to divide TP or TP to divide into replicated KV groups"
            )
        self.local_kv_heads = self.qkv_proj.local_out_features[1] // self.head_dim
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.rope = PartialRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.partial_rotary_factor,
            is_neox_style=True,
        )
        self.attn_backend = config.attn_backend
        self.train_backend = build_train_attention_backend(self.attn_backend)
        self.infer_backend: FlashAttnInferBackend | None = None
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> torch.Tensor:
        batch, seqlen, _ = hidden_states.shape
        q_size = self.local_heads * self.head_dim
        q_gate_size = q_size * (2 if self.attn_output_gate else 1)
        kv_size = self.local_kv_heads * self.head_dim
        q_gate, k, v = self.qkv_proj(hidden_states).split((q_gate_size, kv_size, kv_size), dim=-1)
        if self.attn_output_gate:
            q, gate = q_gate.view(batch, seqlen, self.local_heads, 2, self.head_dim).unbind(dim=3)
            gate = gate.reshape(batch, seqlen, q_size)
        else:
            q = q_gate.view(batch, seqlen, self.local_heads, self.head_dim)
            gate = None
        k = k.view(batch, seqlen, self.local_kv_heads, self.head_dim)
        v = v.view(batch, seqlen, self.local_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, position_ids)
        out = (
            self._forward_infer(q, k, v, infer_meta)
            if infer_meta is not None
            else self._forward_train(q, k, v, train_meta)
        )
        out = out.contiguous().view(batch, seqlen, q_size)
        if gate is not None:
            out = out * _areno_sigmoid_no_compile(gate)
        return self.o_proj(out)

    def _forward_train(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, train_meta: TrainMeta | None
    ) -> torch.Tensor:
        return self.train_backend(q, k, v, train_meta)

    def _forward_infer(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, infer_meta: InferMeta) -> torch.Tensor:
        if self.k_cache.numel() == 0 or self.v_cache.numel() == 0:
            raise RuntimeError("Qwen3.5 full attention inference requires KV cache")
        if self.infer_backend is None:
            self.infer_backend = build_infer_attention_backend(self.attn_backend)
        return self.infer_backend(q, k, v, self.k_cache, self.v_cache, infer_meta)

    def set_kv_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        self.k_cache = k_cache
        self.v_cache = v_cache

    def clear_kv_cache(self) -> None:
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        self.infer_backend = None

    @torch.no_grad()
    def reset_kv_cache(self) -> None:
        return None


class Qwen35ReplicatedKVQKVLinear(nn.Module):
    """Q-sharded QKV projection with K/V heads replicated across TP ranks."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        attn_output_gate: bool,
    ):
        super().__init__()
        ctx = get_tp_context()
        q_size = num_heads * head_dim * (2 if attn_output_gate else 1)
        kv_size = num_kv_heads * head_dim
        q_range = _shard_range(q_size, ctx.rank, ctx.world_size)
        replication = ctx.world_size // num_kv_heads
        kv_rank = ctx.rank // replication
        kv_range = (kv_rank * head_dim, (kv_rank + 1) * head_dim)
        self.in_features = hidden_size
        self.out_features = (q_size, kv_size, kv_size)
        self.local_out_features = [q_range[1] - q_range[0], head_dim, head_dim]
        self.shard_ranges = (q_range, kv_range, kv_range)
        self.weight = nn.Parameter(torch.empty(sum(self.local_out_features), hidden_size))
        mark_tensor_parallel_parameter(self.weight, True, sequence_parallel=True)
        self._kv_grad_group, self._kv_grad_replication = _replicated_kv_grad_group(num_kv_heads)
        self.weight.register_hook(self._sync_replicated_kv_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            gather_from_sequence_parallel_region(x)
            if is_sequence_parallel_active()
            else copy_to_tensor_parallel_region(x)
        )
        return _areno_linear_no_compile(x, self.weight)

    def _sync_replicated_kv_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if self._kv_grad_group is None or self._kv_grad_replication <= 1:
            return grad
        q_rows = self.local_out_features[0]
        kv_grad = grad[q_rows:]
        dist.all_reduce(kv_grad, group=self._kv_grad_group)
        kv_grad.div_(self._kv_grad_replication)
        return grad


class Qwen35GatedDeltaNet(nn.Module):
    """Qwen3.5 Gated Delta-Net layer backed by FLA kernels."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        ctx = get_tp_context()
        self.layer_idx = layer_idx
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.local_key_heads = self.num_key_heads // ctx.world_size
        self.local_value_heads = self.num_value_heads // ctx.world_size
        self.key_dim = self.num_key_heads * self.head_k_dim
        self.value_dim = self.num_value_heads * self.head_v_dim
        self.local_key_dim = self.local_key_heads * self.head_k_dim
        self.local_value_dim = self.local_value_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.in_proj_qkvz = MergedColumnParallelLinear(
            config.hidden_size,
            (self.key_dim, self.key_dim, self.value_dim, self.value_dim),
            bias=False,
        )
        self.in_proj_ba = MergedColumnParallelLinear(
            config.hidden_size, (self.num_value_heads, self.num_value_heads), bias=False
        )
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.local_key_dim * 2 + self.local_value_dim, 1, self.conv_kernel_size)
        )
        mark_tensor_parallel_parameter(self.conv1d_weight, True, sequence_parallel=True)
        self.dt_bias = nn.Parameter(torch.empty(self.local_value_heads))
        self.A_log = nn.Parameter(torch.empty(self.local_value_heads, dtype=torch.float32))
        mark_tensor_parallel_parameter(self.dt_bias, True, sequence_parallel=True)
        mark_tensor_parallel_parameter(self.A_log, True, sequence_parallel=True)
        self.norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        self.out_proj = RowParallelLinear(self.value_dim, config.hidden_size, bias=False)
        self.eps = config.rms_norm_eps
        self.scale = self.head_k_dim**-0.5
        self.state_cache = torch.tensor([])
        self.conv_cache = torch.tensor([])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> torch.Tensor:
        del position_ids
        batch, seqlen, _ = hidden_states.shape
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        mixed_qkv = self._causal_conv(
            qkvz[..., : self.local_key_dim * 2 + self.local_value_dim], train_meta, infer_meta
        )
        query_key, value = mixed_qkv.split((self.local_key_dim * 2, self.local_value_dim), dim=-1)
        z = qkvz[..., self.local_key_dim * 2 + self.local_value_dim :]
        b_gate, a_gate = ba.split((self.local_value_heads, self.local_value_heads), dim=-1)
        query_key = query_key.view(batch, seqlen, self.local_key_heads * 2, self.head_k_dim)
        query, key = query_key.split((self.local_key_heads, self.local_key_heads), dim=2)
        value = value.view(batch, seqlen, self.local_value_heads, self.head_v_dim)
        z = z.view(batch, seqlen, self.local_value_heads, self.head_v_dim)
        if infer_meta is not None:
            out = self._forward_infer(query, key, value, a_gate, b_gate, infer_meta)
        else:
            g = -self.A_log.float().exp().view(1, 1, -1) * F.softplus(
                a_gate.float() + self.dt_bias.float().view(1, 1, -1)
            )
            beta = torch.sigmoid(b_gate)
            out = self._forward_train(query, key, value, g, beta, train_meta)
        out = self._rmsnorm_gate(out, z).reshape(batch, seqlen, self.local_value_dim)
        return self.out_proj(out.to(dtype=hidden_states.dtype))

    def _causal_conv(self, x: torch.Tensor, train_meta: TrainMeta | None, infer_meta: InferMeta | None) -> torch.Tensor:
        if infer_meta is not None:
            return self._causal_conv_infer(x, infer_meta)
        if train_meta is not None and train_meta.packed and train_meta.cu_seqlens is not None:
            return self._causal_conv_train_packed(x, train_meta.cu_seqlens)
        _require_fla_gdn()
        log_once("qwen35_gdn_fla_conv", "using FLA causal-conv training kernel")
        out = _fla_causal_conv1d_no_compile(x, weight=self.conv1d_weight.squeeze(1), activation="silu")
        return out

    @torch._dynamo.disable
    def _causal_conv_train_packed(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != 1:
            raise ValueError("packed ARENO causal-conv expects flattened packed input with batch size 1")
        log_once("qwen35_gdn_areno_packed_conv", "using ARENO packed causal-conv training kernel")
        return _areno_packed_depthwise_causal_conv1d_silu_no_compile(x, self.conv1d_weight, cu_seqlens)

    def _causal_conv_dense(self, x: torch.Tensor) -> torch.Tensor:
        return _areno_depthwise_causal_conv1d_silu_no_compile(x, self.conv1d_weight)

    def _causal_conv_infer(self, x: torch.Tensor, infer_meta: InferMeta) -> torch.Tensor:
        if infer_meta.block_table is None:
            raise RuntimeError("Qwen3.5 GDN inference requires block_table")
        if self.conv_cache.numel() == 0:
            raise RuntimeError("Qwen3.5 GDN inference requires conv state cache")
        slots = infer_meta.block_table[:, 0].long()
        if infer_meta.mode == "decode":
            current = x.reshape(-1, x.shape[-1])
            history = self.conv_cache.index_select(0, slots).to(dtype=current.dtype)
            out = _areno_depthwise_causal_conv1d_silu_decode_no_compile(current, history, self.conv1d_weight)
            window = torch.cat((history, current.unsqueeze(-1)), dim=-1)
            self.conv_cache[slots] = window[:, :, 1:].detach().to(dtype=self.conv_cache.dtype)
            return out.view(x.shape)
        if infer_meta.mode == "prefill":
            if infer_meta.cu_seqlens is None:
                raise RuntimeError("Qwen3.5 GDN prefill requires cu_seqlens")
            return self._causal_conv_infer_prefill(x, infer_meta.cu_seqlens, slots)
        raise ValueError(f"unsupported inference mode: {infer_meta.mode}")

    @torch._dynamo.disable
    def _causal_conv_infer_prefill(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, slots: torch.Tensor
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        cu = cu_seqlens.to(device=x.device, dtype=torch.long)
        for idx, slot in enumerate(slots):
            start, end = int(cu[idx].item()), int(cu[idx + 1].item())
            segment = x[:, start:end]
            out[:, start:end] = self._causal_conv_dense(segment)
            tail = segment.reshape(-1, x.shape[-1])[-(self.conv_kernel_size - 1) :]
            cache = torch.zeros(x.shape[-1], self.conv_kernel_size - 1, device=x.device, dtype=self.conv_cache.dtype)
            cache[:, -tail.shape[0] :] = tail.transpose(0, 1).to(dtype=self.conv_cache.dtype)
            self.conv_cache[slot] = cache
        return out

    def _forward_train(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        train_meta: TrainMeta | None,
    ) -> torch.Tensor:
        cu = (
            train_meta.cu_seqlens.to(device=q.device, dtype=torch.long)
            if train_meta is not None and train_meta.packed and train_meta.cu_seqlens is not None
            else None
        )
        _require_fla_gdn()
        if cu is not None and q.shape[0] != 1:
            raise ValueError("FLA chunk gated-delta expects flattened packed input with batch size 1")
        log_once("qwen35_gdn_fla_chunk", "using FLA chunk gated-delta training kernel")
        out, _ = _fla_chunk_gated_delta_rule_no_compile(
            q, k, v, g=g, beta=beta, scale=self.scale, cu_seqlens=cu, use_qk_l2norm_in_kernel=True
        )
        return out

    def _forward_infer(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, infer_meta: InferMeta
    ) -> torch.Tensor:
        if infer_meta.block_table is None:
            raise RuntimeError("Qwen3.5 GDN inference requires block_table")
        if self.state_cache.numel() == 0:
            raise RuntimeError("Qwen3.5 GDN inference requires recurrent state cache")
        slots = infer_meta.block_table[:, 0].long()
        cu = infer_meta.cu_seqlens
        _require_fla_gdn()
        if infer_meta.mode == "decode":
            decode_shape = q.shape
            q = q.reshape(1, -1, self.local_key_heads, self.head_k_dim)
            k = k.reshape(1, -1, self.local_key_heads, self.head_k_dim)
            v = v.reshape(1, -1, self.local_value_heads, self.head_v_dim)
            a = a.reshape(1, -1, self.local_value_heads)
            b = b.reshape(1, -1, self.local_value_heads)
            g = -self.A_log.float().exp().view(1, 1, -1) * _areno_softplus_no_compile(
                a.float() + self.dt_bias.float().view(1, 1, -1)
            )
            beta = _areno_sigmoid_no_compile(b)
            initial_state = self.state_cache.index_select(0, slots).to(device=q.device)
            out, state = _fla_fused_recurrent_gated_delta_rule_no_compile(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                scale=self.scale,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=torch.arange(slots.numel() + 1, device=q.device, dtype=torch.long),
            )
            self.state_cache[slots] = state.detach().to(dtype=self.state_cache.dtype)
            return out.reshape(decode_shape[:-2] + (self.local_value_heads, self.head_v_dim))
        if cu is None:
            raise RuntimeError("Qwen3.5 GDN inference requires cu_seqlens")
        log_once("qwen35_gdn_fla_infer", "using FLA recurrent gated-delta inference kernel")
        g = -self.A_log.float().exp().view(1, 1, -1) * _areno_softplus_no_compile(
            a.float() + self.dt_bias.float().view(1, 1, -1)
        )
        beta = _areno_sigmoid_no_compile(b)
        initial_state = self.state_cache.index_select(0, slots).to(device=q.device)
        out, state = _fla_fused_recurrent_gated_delta_rule_no_compile(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=self.scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu.to(device=q.device, dtype=torch.long),
            use_qk_l2norm_in_kernel=True,
        )
        self.state_cache[slots] = state.detach().to(dtype=self.state_cache.dtype)
        return out

    def _rmsnorm_gate(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return _areno_rmsnorm_silu_gate_no_compile(x, gate, self.norm_weight, self.eps)

    def set_state_cache(self, state_cache: torch.Tensor, conv_cache: torch.Tensor) -> None:
        self.state_cache = state_cache
        self.conv_cache = conv_cache

    def clear_kv_cache(self) -> None:
        self.state_cache = torch.tensor([])
        self.conv_cache = torch.tensor([])

    @torch.no_grad()
    def reset_kv_cache(self) -> None:
        if self.state_cache.numel() > 0:
            self.state_cache.zero_()
        if self.conv_cache.numel() > 0:
            self.conv_cache.zero_()


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        layer_type = (config.layer_types or ())[layer_idx]
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = (
            Qwen35FullAttention(config, layer_idx)
            if layer_type == "full_attention"
            else Qwen35GatedDeltaNet(config, layer_idx)
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = residual + self.attention(
            self.input_layernorm(hidden_states), position_ids, train_meta, infer_meta
        )
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Qwen35MoeMLP(nn.Module):
    """Qwen3.5-MoE router, routed experts, and dense shared expert."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.num_experts is None:
            raise ValueError("qwen3_5_moe requires num_experts")
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, dtype=torch.float32))
        mark_tensor_parallel_parameter(self.gate, False, sequence_parallel=False, tp_grad_allreduce=True)
        self.experts = Qwen3MoeExperts(config)
        shared_size = int(config.shared_expert_intermediate_size or 0)
        self.shared_expert = Qwen35SharedExpert(config.hidden_size, shared_size) if shared_size > 0 else None
        self.shared_expert_gate = (
            nn.Parameter(torch.empty(config.hidden_size, dtype=config.dtype)) if shared_size > 0 else None
        )
        if self.shared_expert_gate is not None:
            mark_tensor_parallel_parameter(
                self.shared_expert_gate, False, sequence_parallel=False, tp_grad_allreduce=True
            )
        self.register_buffer("_infer_w1_weight", torch.empty(0), persistent=False)
        self.register_buffer("_infer_w2_weight", torch.empty(0), persistent=False)
        self._infer_weights_ready = False
        self._fused_moe_config = FusedMoeConfig(
            num_experts=self.experts.local_num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            top_k=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seqlen, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        logits = _areno_linear_no_compile(flat.to(dtype=self.gate.dtype), self.gate)
        log_once("qwen35_moe_topk_softmax", "using ARENO fused topk_softmax router for Qwen3.5-MoE")
        topk_idx, topk_weight = _areno_topk_softmax_no_compile(logits, self.top_k, self.norm_topk_prob)
        if self.training:
            out = self.experts(flat, topk_idx.to(torch.long), topk_weight)
        else:
            out = self._forward_fused_moe(flat, topk_idx, topk_weight)
        if self.shared_expert is not None:
            shared = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared = shared * torch.sigmoid(F.linear(hidden_states, self.shared_expert_gate.unsqueeze(0)))
            out = out + shared.reshape(-1, hidden)
        return out.view(batch, seqlen, hidden)

    @torch.no_grad()
    def prepare_infer_weights(self) -> None:
        self._infer_w1_weight = self._updated_infer_weight(
            self._infer_w1_weight,
            self.experts.gate_up_weight.detach().to(dtype=self.experts.gate_up_weight.dtype).contiguous(),
        )
        self._infer_w2_weight = self._updated_infer_weight(
            self._infer_w2_weight,
            self.experts.down_weight.detach().contiguous(),
        )
        self._infer_weights_ready = True

    @torch.no_grad()
    def _updated_infer_weight(self, current: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if current.shape == value.shape and current.device == value.device and current.dtype == value.dtype:
            current.copy_(value)
            return current
        return value

    @torch.no_grad()
    def clear_infer_weights(self) -> None:
        device = self._infer_w1_weight.device
        dtype = self._infer_w1_weight.dtype
        self._infer_w1_weight = torch.empty(0, device=device, dtype=dtype)
        self._infer_w2_weight = torch.empty(0, device=device, dtype=dtype)
        self._infer_weights_ready = False

    def _forward_fused_moe(self, flat: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        if self._infer_w1_weight.numel() == 0 or self._infer_w2_weight.numel() == 0:
            raise RuntimeError("Qwen3.5-MoE fused inference weights are not prepared")
        log_once("qwen35_moe_fused_experts", "using areno fused MoE expert kernel for Qwen3.5-MoE inference")
        local_idx, local_weight = self.experts.local_routes(topk_idx, topk_weight)
        out = _areno_fused_experts_no_compile(
            flat.contiguous(),
            self._infer_w1_weight,
            self._infer_w2_weight,
            local_weight.float(),
            local_idx.int(),
            self._fused_moe_config,
        )
        return all_reduce(out)


class Qwen35SharedExpert(nn.Module):
    """Dense shared expert with HF-compatible gate/up/down projection names."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden = _areno_silu_and_mul_no_compile(torch.cat((gate, up), dim=-1))
        return self.down_proj(hidden)


class Qwen35MoeDecoderLayer(Qwen35DecoderLayer):
    handles_activation_checkpointing = True

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = Qwen35MoeMLP(config)

    def _attention_block(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> torch.Tensor:
        return self.attention(self.input_layernorm(hidden_states), position_ids, train_meta, infer_meta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = residual + checkpoint_layer(
            self._attention_block,
            hidden_states,
            position_ids,
            train_meta,
            infer_meta,
            train_meta=train_meta,
            infer_meta=infer_meta,
        )
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Qwen35ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> CausalLMOutput:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden_states = self.embed_tokens(input_ids)
        use_sequence_parallel = bool(train_meta is not None and train_meta.sequence_parallel)
        if use_sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        with sequence_parallel_region(use_sequence_parallel):
            for layer in self.layers:
                if getattr(layer, "handles_activation_checkpointing", False):
                    hidden_states = layer(hidden_states, position_ids, train_meta, infer_meta)
                else:
                    hidden_states = checkpoint_layer(
                        layer,
                        hidden_states,
                        position_ids,
                        train_meta,
                        infer_meta,
                        train_meta=train_meta,
                        infer_meta=infer_meta,
                    )
            hidden_states = self.norm(hidden_states)
            logits_shard = self.lm_head(hidden_states)
        return CausalLMOutput(logits_shard=logits_shard, hidden_states=hidden_states)

    def set_kv_caches(self, kv_caches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        idx = 0
        device = next(self.parameters()).device
        num_slots = kv_caches[0][0].shape[0] if kv_caches else 1
        for layer in self.layers:
            attn = layer.attention
            if isinstance(attn, Qwen35FullAttention):
                attn.set_kv_cache(*kv_caches[idx])
                idx += 1
            else:
                state = torch.zeros(
                    num_slots,
                    attn.local_value_heads,
                    attn.head_k_dim,
                    attn.head_v_dim,
                    device=device,
                    dtype=torch.float32,
                )
                conv = torch.zeros(
                    num_slots,
                    attn.local_key_dim * 2 + attn.local_value_dim,
                    attn.conv_kernel_size - 1,
                    device=device,
                    dtype=self.config.dtype,
                )
                attn.set_state_cache(state, conv)

    @torch.no_grad()
    def prepare_infer_weights(self) -> None:
        return None

    @torch.no_grad()
    def clear_infer_weights(self) -> None:
        return None

    @torch.no_grad()
    def offload_train_weights(self) -> None:
        return None

    @torch.no_grad()
    def onload_train_weights(self, device: torch.device) -> None:
        del device
        return None

    @torch.no_grad()
    def finalize_router_expert_bias(self, tp_group, dp_group) -> None:
        del tp_group, dp_group
        return None

    def allocate_kv_caches(
        self, num_blocks: int, block_size: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        caches = []
        for layer in self.layers:
            attention = layer.attention
            if isinstance(attention, Qwen35FullAttention):
                k_cache = torch.empty(
                    num_blocks,
                    block_size,
                    attention.local_kv_heads,
                    attention.head_dim,
                    device=device,
                    dtype=self.config.dtype,
                )
                v_cache = torch.empty(
                    num_blocks,
                    block_size,
                    attention.local_kv_heads,
                    attention.head_dim,
                    device=device,
                    dtype=self.config.dtype,
                )
                caches.append((k_cache, v_cache))
        return caches

    def clear_kv_caches(self) -> None:
        for layer in self.layers:
            layer.attention.clear_kv_cache()

    @torch.no_grad()
    def reset_kv_caches(self) -> None:
        for layer in self.layers:
            layer.attention.reset_kv_cache()

    @torch.no_grad()
    def offload_kv_caches(self) -> None:
        for layer in self.layers:
            attn = layer.attention
            for name in ("k_cache", "v_cache", "state_cache", "conv_cache"):
                cache = getattr(attn, name, None)
                if cache is not None and cache.numel() > 0:
                    setattr(attn, name, cache.to(device="cpu"))
            if isinstance(attn, Qwen35FullAttention):
                attn.infer_backend = None

    @torch.no_grad()
    def onload_kv_caches(self, device: torch.device) -> bool:
        found = False
        for layer in self.layers:
            attn = layer.attention
            for name in ("k_cache", "v_cache", "state_cache", "conv_cache"):
                cache = getattr(attn, name, None)
                if cache is not None and cache.numel() > 0:
                    found = True
                    if cache.device != device:
                        setattr(attn, name, cache.to(device=device))
        return found


class Qwen35MoeForCausalLM(Qwen35ForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Qwen35MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])

    @torch.no_grad()
    def prepare_infer_weights(self) -> None:
        for layer in self.layers:
            layer.mlp.prepare_infer_weights()

    @torch.no_grad()
    def clear_infer_weights(self) -> None:
        for layer in self.layers:
            layer.mlp.clear_infer_weights()


class Qwen35Adapter(ModelAdapter):
    name = "qwen3_5"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        architectures = set(hf_config.get("architectures") or [])
        return (
            str(hf_config.get("model_type", "")).lower() == "qwen3_5"
            or "Qwen3_5ForConditionalGeneration" in architectures
        )

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        text = hf_config.get("text_config") or hf_config
        dtype = _parse_dtype(
            text.get("torch_dtype") or text.get("dtype") or hf_config.get("torch_dtype") or hf_config.get("dtype")
        )
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        layer_types = tuple(
            text.get("layer_types")
            or _layer_types_from_interval(int(text["num_hidden_layers"]), int(text.get("full_attention_interval", 1)))
        )
        return ModelConfig(
            model_type=self.name,
            checkpoint_prefix="model",
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text.get("num_key_value_heads", text["num_attention_heads"])),
            head_dim=int(text.get("head_dim", text["hidden_size"] // text["num_attention_heads"])),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10_000.0))),
            max_position_embeddings=int(text.get("max_position_embeddings", 262144)),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", hf_config.get("tie_word_embeddings", True))),
            qkv_bias=bool(text.get("attention_bias", False)),
            qk_norm=True,
            dtype=dtype,
            hidden_act=str(text.get("hidden_act", "silu")),
            layer_types=layer_types,
            partial_rotary_factor=float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))),
            sequence_parallel=bool(text.get("sequence_parallel", True)),
            attn_output_gate=bool(text.get("attn_output_gate", True)),
            linear_conv_kernel_dim=int(text.get("linear_conv_kernel_dim", 4)),
            linear_key_head_dim=int(text.get("linear_key_head_dim", 128)),
            linear_value_head_dim=int(text.get("linear_value_head_dim", 128)),
            linear_num_key_heads=int(text.get("linear_num_key_heads", 16)),
            linear_num_value_heads=int(text.get("linear_num_value_heads", 32)),
        )

    def build(self, config: ModelConfig) -> nn.Module:
        return Qwen35ForCausalLM(config)

    @torch.no_grad()
    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        if not isinstance(model, Qwen35ForCausalLM):
            raise TypeError(f"Qwen35Adapter cannot load weights into {type(model)!r}")
        from areno.models.qwen3_5.checkpoint import load_qwen35_weights

        load_qwen35_weights(model, model_path)

    @torch.no_grad()
    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        if not isinstance(model, Qwen35ForCausalLM):
            raise TypeError(f"Qwen35Adapter cannot save weights from {type(model)!r}")
        from areno.models.qwen3_5.checkpoint import save_qwen35_weights

        return save_qwen35_weights(model, output_path, source_path)


class Qwen35MoeAdapter(ModelAdapter):
    name = "qwen3_5_moe"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        architectures = set(hf_config.get("architectures") or [])
        return (
            str(hf_config.get("model_type", "")).lower() == "qwen3_5_moe"
            or "Qwen3_5MoeForConditionalGeneration" in architectures
        )

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        text = hf_config.get("text_config") or hf_config
        dtype = _parse_dtype(
            text.get("torch_dtype") or text.get("dtype") or hf_config.get("torch_dtype") or hf_config.get("dtype")
        )
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        layer_types = tuple(
            text.get("layer_types")
            or _layer_types_from_interval(int(text["num_hidden_layers"]), int(text.get("full_attention_interval", 1)))
        )
        return ModelConfig(
            model_type=self.name,
            checkpoint_prefix="model",
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(
                text.get(
                    "intermediate_size", text.get("shared_expert_intermediate_size", text["moe_intermediate_size"])
                )
            ),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text.get("num_key_value_heads", text["num_attention_heads"])),
            head_dim=int(text.get("head_dim", text["hidden_size"] // text["num_attention_heads"])),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10_000.0))),
            max_position_embeddings=int(text.get("max_position_embeddings", 262144)),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", hf_config.get("tie_word_embeddings", False))),
            qkv_bias=bool(text.get("attention_bias", False)),
            qk_norm=True,
            dtype=dtype,
            hidden_act=str(text.get("hidden_act", "silu")),
            layer_types=layer_types,
            partial_rotary_factor=float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))),
            sequence_parallel=False,
            attn_output_gate=bool(text.get("attn_output_gate", True)),
            linear_conv_kernel_dim=int(text.get("linear_conv_kernel_dim", 4)),
            linear_key_head_dim=int(text.get("linear_key_head_dim", 128)),
            linear_value_head_dim=int(text.get("linear_value_head_dim", 128)),
            linear_num_key_heads=int(text.get("linear_num_key_heads", 16)),
            linear_num_value_heads=int(text.get("linear_num_value_heads", 32)),
            enable_moe_block=True,
            num_experts=int(text["num_experts"]),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            shared_expert_intermediate_size=int(text.get("shared_expert_intermediate_size", 0)),
            norm_topk_prob=bool(text.get("norm_topk_prob", True)),
            score_function="softmax",
        )

    def build(self, config: ModelConfig) -> nn.Module:
        return Qwen35MoeForCausalLM(config)

    @torch.no_grad()
    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        if not isinstance(model, Qwen35MoeForCausalLM):
            raise TypeError(f"Qwen35MoeAdapter cannot load weights into {type(model)!r}")
        from areno.models.qwen3_5.checkpoint import load_qwen35_weights

        load_qwen35_weights(model, model_path)

    @torch.no_grad()
    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        if not isinstance(model, Qwen35MoeForCausalLM):
            raise TypeError(f"Qwen35MoeAdapter cannot save weights from {type(model)!r}")
        from areno.models.qwen3_5.checkpoint import save_qwen35_weights

        return save_qwen35_weights(model, output_path, source_path)


def _layer_types_from_interval(num_layers: int, full_attention_interval: int) -> list[str]:
    if full_attention_interval <= 0:
        raise ValueError("full_attention_interval must be positive")
    return [
        "full_attention" if (idx + 1) % full_attention_interval == 0 else "linear_attention"
        for idx in range(num_layers)
    ]


@torch._dynamo.disable
def _areno_linear_no_compile(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return areno_linear(x, weight, None)


@torch._dynamo.disable
def _areno_topk_softmax_no_compile(
    logits: torch.Tensor, top_k: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return areno_topk_softmax(logits, top_k, renormalize)


@torch._dynamo.disable
def _areno_silu_and_mul_no_compile(x: torch.Tensor) -> torch.Tensor:
    return areno_silu_and_mul(x)


@torch._dynamo.disable
def _areno_fused_experts_no_compile(
    flat: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_idx: torch.Tensor,
    config: FusedMoeConfig,
) -> torch.Tensor:
    return areno_fused_experts(flat, w1, w2, topk_weight, topk_idx, config)
