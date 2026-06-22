"""Gemma4 causal-LM adapter.

Targets Google Gemma4 checkpoints (``model_type == "gemma4"`` or
``"gemma4_unified"``, architectures ``Gemma4ForCausalLM`` /
``Gemma4ForConditionalGeneration`` / ``Gemma4UnifiedForConditionalGeneration``).
Notable
peculiarities the implementation has to handle:
    * Layered attention types — each layer is either ``full_attention`` or
      ``sliding_attention``; head_dim, kv_head_count, and RoPE settings can
      differ per type. See ``layer_types`` in the HF config.
    * KV-shared tail layers — the last ``num_kv_shared_layers`` decoders reuse
      K/V from a matching earlier layer of the same attention type, but still
      run their own Q projection + rope.
    * Sandwich RMSNorm — four norms per layer (pre/post around both attn and
      MLP) and Gemma4's RMSNorm stores ``(scale - 1)`` on disk, so loader adds
      1 to the weight (handled implicitly via Gemma4RMSNorm/the optional-scale
      kernel below).
    * Per-layer input embeddings (PLE) — a separate vocab embedding plus a
      projection from hidden states form a per-layer side-channel that is
      added back into each decoder via a gated GELU-tanh pathway.
    * GeLU-tanh activation, partial rotary fraction, optional final logit
      softcapping, ``embed_scale = sqrt(hidden_size)``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from areno.accel import (
    areno_grouped_linear,
    areno_moe_topk_permute,
    areno_moe_unpermute,
    areno_optional_scale_rmsnorm,
    areno_topk_softmax,
)
from areno.accel.ops import FusedMoeConfig, areno_fused_experts, areno_gelu_tanh_and_mul, log_once
from areno.engine.checkpoints.common import load_checkpoint_weights, save_checkpoint_weights
from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend, build_infer_attention_backend
from areno.engine.layers.attention_backend.train import build_train_attention_backend
from areno.engine.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    _areno_linear_forward,
    mark_tensor_parallel_parameter,
)
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.rotary import Gemma4RotaryEmbedding
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.parallel.collectives import all_reduce, scatter_to_sequence_parallel_region, sequence_parallel_region
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.recompute import checkpoint_layer
from areno.models.base import CausalLMOutput, ModelAdapter
from areno.models.gemma4.checkpoint import checkpoint_spec


class Gemma4RMSNorm(nn.Module):
    """RMSNorm with an optional learnable scale stored as ``(scale - 1)``.

    Gemma4 stores norm weights in the HF checkpoint as offsets from 1 (so
    "no-op" is encoded as 0). The loader adds 1 before copying into ``weight``,
    and ``with_scale=False`` keeps the layer purely normalizing (used for V
    pre-norm where no learnable scale exists).
    """

    def __init__(self, hidden_size: int, eps: float, *, with_scale: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32)) if with_scale else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _areno_optional_scale_rmsnorm_no_compile(x, self.weight, self.eps)


class Gemma4ReplicatedLinear(nn.Module):
    """Plain linear whose weight is replicated across TP ranks.

    Used for the per-layer-input projection paths where the input/output dims
    are small (PLE width) and sharding would add latency without saving memory.
    Sequence parallelism is configurable so SP-active forward passes still
    avoid all-gathers.
    """

    def __init__(
        self, in_features: int, out_features: int, *, sequence_parallel: bool, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        mark_tensor_parallel_parameter(self.weight, False, sequence_parallel=sequence_parallel, tp_grad_allreduce=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _areno_linear_forward(x, self.weight, None)


class Gemma4MLP(nn.Module):
    """SwiGLU-style MLP with GELU-tanh activation.

    For KV-shared tail layers Gemma4 doubles the intermediate width
    (``use_double_wide_mlp=True``); the constructor inflates the size in that
    case so the wider FF compensates for the missing attention K/V slots.
    """

    def __init__(self, config: ModelConfig, use_double_wide_mlp: bool):
        super().__init__()
        intermediate_size = int(config.intermediate_size) * (2 if use_double_wide_mlp else 1)
        # gate + up share one column-parallel slab and are then split along the
        # last dim before activation/mul.
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            (intermediate_size, intermediate_size),
            bias=False,
        )
        self.down_proj = RowParallelLinear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        # areno_gelu_tanh_and_mul implements gelu_tanh(gate) * up in a fused kernel.
        hidden = _areno_gelu_tanh_and_mul_no_compile(gate_up)
        return self.down_proj(hidden)


class Gemma4Router(nn.Module):
    """Gemma4 MoE router: norm + root-size scale + learned scale + projection."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps, with_scale=False)
        self.scale = nn.Parameter(torch.ones(config.hidden_size, dtype=torch.float32))
        mark_tensor_parallel_parameter(self.scale, False, sequence_parallel=False, tp_grad_allreduce=True)
        self.proj = Gemma4ReplicatedLinear(
            config.hidden_size, int(config.num_experts or 0), sequence_parallel=False, dtype=config.dtype
        )
        self.register_buffer("root_size", torch.tensor(config.hidden_size**-0.5), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        x = x * (self.scale.to(dtype=x.dtype) * self.root_size.to(device=x.device, dtype=x.dtype))
        return self.proj(x)


class Gemma4MoeExperts(nn.Module):
    """Expert-parallel Gemma4 MoE experts with GELU-tanh gated activation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        ctx = get_tp_context()
        if config.num_experts is None:
            raise ValueError("Gemma4 MoE requires num_experts")
        if config.num_experts % ctx.world_size != 0:
            raise ValueError("Gemma4 num_experts must be divisible by tensor parallel world size")
        self.local_num_experts = config.num_experts // ctx.world_size
        self.local_expert_start = ctx.rank * self.local_num_experts
        self.local_expert_end = self.local_expert_start + self.local_num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_weight = nn.Parameter(
            torch.empty(self.local_num_experts, 2 * self.intermediate_size, self.hidden_size, dtype=config.dtype)
        )
        self.down_weight = nn.Parameter(
            torch.empty(self.local_num_experts, self.hidden_size, self.intermediate_size, dtype=config.dtype)
        )
        mark_tensor_parallel_parameter(self.gate_up_weight, True, sequence_parallel=False, tp_grad_allreduce=False)
        mark_tensor_parallel_parameter(self.down_weight, True, sequence_parallel=False, tp_grad_allreduce=False)

    def forward(self, flat: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        x, route_weight, token_idx, tokens_per_expert = _areno_moe_topk_permute_no_compile(
            flat,
            topk_idx,
            topk_weight.float(),
            self.local_expert_start,
            self.local_num_experts,
        )
        if x.shape[0] == 0:
            return all_reduce(flat.new_zeros(flat.shape))
        hidden = _areno_grouped_linear_no_compile(x.contiguous(), self.gate_up_weight, tokens_per_expert)
        hidden = (
            _areno_gelu_tanh_and_mul_no_compile(hidden) * route_weight.unsqueeze(-1).to(dtype=hidden.dtype)
        ).contiguous()
        out = _areno_grouped_linear_no_compile(hidden, self.down_weight, tokens_per_expert)
        out = _areno_moe_unpermute_no_compile(out, token_idx, flat.shape)
        return all_reduce(out)

    def local_routes(self, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_mask = (topk_idx >= self.local_expert_start) & (topk_idx < self.local_expert_end)
        local_idx = (topk_idx - self.local_expert_start).clamp(0, self.local_num_experts - 1)
        return local_idx, topk_weight * local_mask.to(dtype=topk_weight.dtype)

    @torch.no_grad()
    def copy_expert(
        self, expert_id: int, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, rank: int, world_size: int
    ) -> None:
        del rank, world_size
        if expert_id < self.local_expert_start or expert_id >= self.local_expert_end:
            return
        local_expert_id = expert_id - self.local_expert_start
        self.gate_up_weight[local_expert_id].copy_(torch.cat((gate, up), dim=0).to(dtype=self.gate_up_weight.dtype))
        self.down_weight[local_expert_id].copy_(down.to(dtype=self.down_weight.dtype))

    @torch.no_grad()
    def expert_weights(self) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        gate_weights = []
        up_weights = []
        down_weights = []
        for expert_id in range(self.local_num_experts):
            gate, up = self.gate_up_weight[expert_id].detach().chunk(2, dim=0)
            gate_weights.append(gate)
            up_weights.append(up)
            down_weights.append(self.down_weight[expert_id].detach())
        return gate_weights, up_weights, down_weights


class Gemma4MoeMLP(nn.Module):
    """Gemma4 sparse MoE expert branch.

    Gemma4 MoE keeps the dense MLP branch and adds this routed branch in
    parallel; router logits are produced outside this module from the residual.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.num_experts is None:
            raise ValueError("Gemma4 MoE requires num_experts")
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))
        mark_tensor_parallel_parameter(self.per_expert_scale, False, sequence_parallel=False, tp_grad_allreduce=True)
        self.experts = Gemma4MoeExperts(config)
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

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        batch, seqlen, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        topk_idx, topk_weight = _areno_topk_softmax_no_compile(
            router_logits.reshape(-1, self.num_experts), self.top_k, self.norm_topk_prob
        )
        topk_weight = topk_weight * self.per_expert_scale[topk_idx].to(dtype=topk_weight.dtype)
        if self.training:
            out = self.experts(flat, topk_idx.to(torch.long), topk_weight)
        else:
            out = self._forward_fused_moe(flat, topk_idx, topk_weight)
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
            raise RuntimeError("Gemma4 MoE fused inference weights are not prepared")
        log_once("gemma4_moe_fused_experts", "using areno fused MoE expert kernel for Gemma4-MoE inference")
        local_idx, local_weight = self.experts.local_routes(topk_idx, topk_weight)
        out = _areno_fused_experts_gelu_no_compile(
            flat.contiguous(),
            self._infer_w1_weight,
            self._infer_w2_weight,
            local_weight.float(),
            local_idx.int(),
            self._fused_moe_config,
        )
        return all_reduce(out)


class Gemma4Attention(nn.Module):
    """Per-layer attention block with optional KV sharing and sliding window.

    The per-layer ``layer_type`` selects head_dim/kv_heads/RoPE settings and
    decides whether a sliding-window constraint is applied during attention.
    KV-shared layers skip the K/V projections entirely and read tensors
    produced by an earlier matching layer.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        ctx = get_tp_context()
        self.layer_idx = layer_idx
        layer_type = _layer_type(config, layer_idx)
        self.layer_type = layer_type
        self.attention_k_eq_v = config.attention_k_eq_v
        # Sliding-attention layers can use a smaller head_dim/kv-head budget;
        # _attention_head_dim / _attention_kv_heads pick the right value.
        self.head_dim = _attention_head_dim(config, layer_type)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = _attention_kv_heads(config, layer_type)
        self.local_heads = self.num_heads // ctx.world_size
        # The tail ``num_kv_shared_layers`` layers reuse a previous layer's K/V.
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = config.num_kv_shared_layers > 0 and layer_idx >= first_shared
        self.kv_shared_layer_index = _kv_shared_layer_index(config, layer_idx) if self.is_kv_shared_layer else None
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.qkv_bias,
        )
        # Read the actual local KV-head count from the linear (rounded to TP shape).
        self.local_kv_heads = self.qkv_proj.local_out_features[1] // self.head_dim
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, config.hidden_size, bias=config.qkv_bias)
        # Per-head Q/K/V RMSNorm applied before RoPE (Gemma4 convention).
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        # V norm has no learnable scale -- only the raw RMS division.
        self.v_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False) if config.v_norm else None
        # Gemma4/HF treats sliding_window as inclusive with the current token;
        # FlashAttention window_size is exclusive on the left side.
        self.window_size = (
            (max(int(config.sliding_window) - 1, 0), 0)
            if layer_type == "sliding_attention" and config.sliding_window
            else None
        )
        self.softmax_scale = config.attention_softmax_scale
        self.attn_backend = config.attn_backend
        # Rotary frequency base and partial-rotary fraction may differ per
        # layer_type (full vs sliding may use different long-context settings).
        theta = _rope_theta(config, layer_type)
        partial = _rope_partial_rotary_factor(config, layer_type)
        self.rope = Gemma4RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            theta,
            partial,
        )
        self.train_backend = build_train_attention_backend(self.attn_backend)
        self.infer_backend: FlashAttnInferBackend | None = None
        # KV cache slots are lazily bound; numel==0 means "not yet allocated".
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seqlen, _ = hidden_states.shape
        q_size = self.local_heads * self.head_dim
        kv_size = self.local_kv_heads * self.head_dim
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
        # Q/K/V norms operate in [B, S, H, D] layout.
        q = self.q_norm(q.view(batch, seqlen, self.local_heads, self.head_dim))
        kv_for_share = None
        if self.is_kv_shared_layer:
            # Skip K/V computation entirely; only rotate Q using a zero K of the
            # right shape so the rope op still produces a valid Q tensor.
            if shared_kv is None:
                raise RuntimeError(
                    f"Gemma4 layer {self.layer_idx} requires shared KV from layer {self.kv_shared_layer_index}"
                )
            q, _ = self.rope(
                q,
                torch.zeros(batch, seqlen, self.local_kv_heads, self.head_dim, device=q.device, dtype=q.dtype),
                position_ids,
            )
            k, v = shared_kv
        else:
            k = self.k_norm(k.view(batch, seqlen, self.local_kv_heads, self.head_dim))
            v = v.view(batch, seqlen, self.local_kv_heads, self.head_dim)
            if self.v_norm is not None:
                v = self.v_norm(v)
            q, k = self.rope(q, k, position_ids)
            # Expose the post-norm/post-rope K/V so later KV-shared layers can
            # consume them without re-projecting.
            kv_for_share = (k, v)
        if infer_meta is not None:
            return self.forward_infer(q, k, v, infer_meta), kv_for_share
        return self.forward_train(q, k, v, train_meta), kv_for_share

    def forward_train(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, train_meta: TrainMeta | None
    ) -> torch.Tensor:
        # Window size and softmax_scale are passed through so the backend can
        # honour sliding attention and the Gemma-specific scale of 1.0.
        out = self.train_backend(q, k, v, train_meta, self.window_size, self.softmax_scale)
        out = out.contiguous().view(q.shape[0], q.shape[1], self.local_heads * self.head_dim)
        return self.o_proj(out)

    def forward_infer(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, infer_meta: InferMeta) -> torch.Tensor:
        if self.k_cache.numel() == 0 or self.v_cache.numel() == 0:
            raise RuntimeError("inference requires per-layer KV cache tensors")
        if self.infer_backend is None:
            # Lazily instantiate once we know what we are running against.
            self.infer_backend = build_infer_attention_backend(self.attn_backend)
        out = self.infer_backend(
            q,
            k,
            v,
            self.k_cache,
            self.v_cache,
            infer_meta,
            self.window_size,
            self.softmax_scale,
        )
        out = out.contiguous().view(q.shape[0], q.shape[1], self.local_heads * self.head_dim)
        return self.o_proj(out)

    def set_kv_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        self.k_cache = k_cache
        self.v_cache = v_cache

    def clear_kv_cache(self) -> None:
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        self.infer_backend = None


class Gemma4DecoderLayer(nn.Module):
    """Gemma4 transformer block with sandwich norms and optional PLE pathway.

    Each block has four RMSNorms (pre/post for attention and FF), a learnable
    scalar applied to the final residual contribution, and -- when the model
    ships per-layer-input embeddings -- a gated GELU-tanh injection of the
    per-layer signal after the FF block.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Gemma4Attention(config, layer_idx)
        # Double-wide MLP only kicks in for KV-shared tail layers (compensates
        # for the missing K/V capacity by widening the FF block).
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        use_double_wide_mlp = (
            config.use_double_wide_mlp and config.num_kv_shared_layers > 0 and layer_idx >= first_shared
        )
        self.mlp = Gemma4MLP(config, use_double_wide_mlp)
        self.router = Gemma4Router(config) if config.enable_moe_block else None
        self.moe = Gemma4MoeMLP(config) if config.enable_moe_block else None
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = (
            RMSNorm(config.hidden_size, config.rms_norm_eps) if config.enable_moe_block else None
        )
        self.post_feedforward_layernorm_1 = (
            RMSNorm(config.hidden_size, config.rms_norm_eps) if config.enable_moe_block else None
        )
        self.post_feedforward_layernorm_2 = (
            RMSNorm(config.hidden_size, config.rms_norm_eps) if config.enable_moe_block else None
        )
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input > 0:
            # PLE modules: gate projects hidden -> PLE width; projection brings
            # the gated GELU(gate, per_layer_input) back to hidden width.
            self.per_layer_input_gate = Gemma4ReplicatedLinear(
                config.hidden_size,
                self.hidden_size_per_layer_input,
                sequence_parallel=config.sequence_parallel,
            )
            self.per_layer_projection = Gemma4ReplicatedLinear(
                self.hidden_size_per_layer_input,
                config.hidden_size,
                sequence_parallel=config.sequence_parallel,
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None
        # Learnable per-layer scalar multiplier on the block output; persistent
        # so it round-trips through the checkpoint as ``{prefix}.layer_scalar``.
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        per_layer_input: torch.Tensor | None = None,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Pre-norm attention path (note: norm comes both before AND after attn).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, kv_for_share = self.self_attn(hidden_states, position_ids, shared_kv, train_meta, infer_meta)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        # Pre-norm FF path with matching sandwich norm on output.
        residual = hidden_states
        dense_input = self.pre_feedforward_layernorm(hidden_states)
        if self.moe is not None:
            if (
                self.router is None
                or self.pre_feedforward_layernorm_2 is None
                or self.post_feedforward_layernorm_1 is None
                or self.post_feedforward_layernorm_2 is None
            ):
                raise RuntimeError("Gemma4 MoE layer is missing router or MoE norms")
            dense_hidden = self.mlp(dense_input)
            router_logits = self.router(residual)
            moe_hidden = self.moe(self.pre_feedforward_layernorm_2(residual), router_logits)
            hidden_states = self.post_feedforward_layernorm_1(dense_hidden) + self.post_feedforward_layernorm_2(
                moe_hidden
            )
        else:
            hidden_states = self.mlp(dense_input)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        if per_layer_input is not None:
            # PLE injection: concatenate the gated hidden with the per-layer
            # signal and run them through gelu_tanh + mul before projecting
            # back to hidden_size and normalizing.
            if (
                self.per_layer_input_gate is None
                or self.per_layer_projection is None
                or self.post_per_layer_input_norm is None
            ):
                raise RuntimeError("Gemma4 PLE tensors were provided but layer PLE modules are missing")
            gate_input = self.per_layer_input_gate(hidden_states)
            per_layer_contribution = self.per_layer_projection(
                _areno_gelu_tanh_and_mul_no_compile(torch.cat((gate_input, per_layer_input), dim=-1))
            )
            hidden_states = hidden_states + self.post_per_layer_input_norm(per_layer_contribution)
        # Scale the final residual by the learnable per-layer scalar.
        return hidden_states * self.layer_scalar, kv_for_share


class Gemma4ForCausalLM(nn.Module):
    """Top-level Gemma4 model orchestrating PLE, KV sharing, and the LM head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        # PLE may use a smaller vocab subset; fall back to the full vocab when
        # the HF config does not override it.
        self.vocab_size_per_layer_input = config.vocab_size_per_layer_input or config.vocab_size
        if self.hidden_size_per_layer_input > 0:
            # Per-layer-input embeddings: one vector per (token, layer).
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                self.vocab_size_per_layer_input,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                dtype=config.dtype,
            )
            # Projects hidden -> num_layers * PLE_dim so each layer gets its
            # own contribution from the model state.
            self.per_layer_model_projection = Gemma4ReplicatedLinear(
                config.hidden_size,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                sequence_parallel=config.sequence_parallel,
            )
            self.per_layer_projection_norm = RMSNorm(self.hidden_size_per_layer_input, config.rms_norm_eps)
            # Static scales used when combining the embedding signal with the
            # projected hidden state (1/sqrt(2) preserves variance of the sum).
            self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)
            self.register_buffer("per_layer_projection_scale", torch.tensor(config.hidden_size**-0.5), persistent=False)
            self.embed_per_layer_scale = math.sqrt(self.hidden_size_per_layer_input)
        else:
            # PLE disabled -- keep attribute slots populated with no-ops so the
            # forward path can branch without missing attributes.
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.register_buffer("per_layer_input_scale", torch.tensor(1.0), persistent=False)
            self.register_buffer("per_layer_projection_scale", torch.tensor(1.0), persistent=False)
            self.embed_per_layer_scale = 1.0
        self.layers = nn.ModuleList([Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)
        # Gemma scales embeddings by sqrt(hidden_size) before the first layer.
        self.embed_scale = math.sqrt(config.hidden_size)
        self.final_logit_softcapping = config.final_logit_softcapping

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> CausalLMOutput:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        # Build per-layer PLE inputs (or None when PLE disabled). Wrapped in a
        # dynamo-disabled helper because the gather/reshape is data-dependent.
        per_layer_inputs = _gemma4_per_layer_inputs_no_compile(self, hidden_states, input_ids)
        use_sequence_parallel = bool(train_meta is not None and train_meta.sequence_parallel)
        if use_sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
            position_ids = scatter_to_sequence_parallel_region(position_ids)
            if per_layer_inputs is not None:
                per_layer_inputs = scatter_to_sequence_parallel_region(per_layer_inputs)
        with sequence_parallel_region(use_sequence_parallel):
            # Cache K/V tensors that the tail KV-shared layers will consume.
            shared_kv_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for layer_idx, layer in enumerate(self.layers):
                # Slice out the current layer's PLE slot if PLE is active.
                per_layer_input = per_layer_inputs[:, :, layer_idx, :] if per_layer_inputs is not None else None
                shared_kv = None
                shared_idx = layer.self_attn.kv_shared_layer_index
                if shared_idx is not None:
                    shared_kv = shared_kv_by_layer.get(shared_idx)
                if shared_kv is None:
                    hidden_states, kv_for_share = checkpoint_layer(
                        layer,
                        hidden_states,
                        position_ids,
                        per_layer_input,
                        shared_kv,
                        train_meta,
                        infer_meta,
                        train_meta=train_meta,
                        infer_meta=infer_meta,
                    )
                else:
                    hidden_states, kv_for_share = layer(
                        hidden_states, position_ids, per_layer_input, shared_kv, train_meta, infer_meta
                    )
                if kv_for_share is not None:
                    shared_kv_by_layer[layer_idx] = kv_for_share
            hidden_states = self.norm(hidden_states)
            logits_shard = self.lm_head(hidden_states)
            if self.final_logit_softcapping:
                # Gemma uses tanh-based logit softcap to limit extreme values.
                cap = float(self.final_logit_softcapping)
                logits_shard = cap * torch.tanh(logits_shard / cap)
        return CausalLMOutput(logits_shard=logits_shard, hidden_states=hidden_states)

    @torch._dynamo.disable
    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Look up PLE embeddings for each token, returning [B, S, L, PLE_dim]."""
        if self.embed_tokens_per_layer is None:
            return None
        # Tokens outside the PLE vocab are clamped to 0 (a dedicated pad slot);
        # the mask keeps the original index in scope without out-of-bound lookup.
        mask = torch.logical_and(input_ids >= 0, input_ids < self.vocab_size_per_layer_input)
        tokens = torch.where(mask, input_ids, torch.zeros_like(input_ids))
        embeds = self.embed_tokens_per_layer(tokens) * self.embed_per_layer_scale
        return embeds.reshape(*input_ids.shape, self.config.num_hidden_layers, self.hidden_size_per_layer_input)

    @torch._dynamo.disable
    def project_per_layer_inputs(
        self, hidden_states: torch.Tensor, per_layer_inputs: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Combine projected hidden states with PLE embeddings (variance-preserving)."""
        if self.per_layer_model_projection is None:
            return None
        projected = self.per_layer_model_projection(hidden_states) * self.per_layer_projection_scale.to(
            hidden_states.device
        )
        projected = projected.reshape(
            *hidden_states.shape[:-1], self.config.num_hidden_layers, self.hidden_size_per_layer_input
        )
        if self.per_layer_projection_norm is None:
            raise RuntimeError("Gemma4 PLE projection norm is missing")
        projected = self.per_layer_projection_norm(projected)
        if per_layer_inputs is None:
            return projected
        # Average the projection and the PLE embedding (scaled by 1/sqrt(2)) so
        # the combined signal keeps unit variance.
        return (projected + per_layer_inputs) * self.per_layer_input_scale.to(hidden_states.device)

    def set_kv_caches(self, kv_caches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Bind KV cache slots, redirecting KV-shared layers to their source."""
        if len(kv_caches) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} layer caches, got {len(kv_caches)}")
        for layer_idx, layer in enumerate(self.layers):
            shared_idx = layer.self_attn.kv_shared_layer_index
            # KV-shared layers point at the cache of the layer they share with.
            k_cache, v_cache = kv_caches[shared_idx if shared_idx is not None else layer_idx]
            layer.self_attn.set_kv_cache(k_cache, v_cache)

    @torch.no_grad()
    def prepare_infer_weights(self) -> None:
        for layer in self.layers:
            for module in (layer.mlp, getattr(layer, "moe", None)):
                prepare = getattr(module, "prepare_infer_weights", None)
                if prepare is not None:
                    prepare()

    @torch.no_grad()
    def clear_infer_weights(self) -> None:
        for layer in self.layers:
            for module in (layer.mlp, getattr(layer, "moe", None)):
                clear = getattr(module, "clear_infer_weights", None)
                if clear is not None:
                    clear()

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
        """Allocate per-layer KV caches, deduplicating slots for KV-shared layers."""
        caches = []
        for layer in self.layers:
            attention = layer.self_attn
            if attention.kv_shared_layer_index is not None:
                # KV-shared layer reuses the buffer of the matching source layer.
                caches.append(caches[attention.kv_shared_layer_index])
                continue
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
            layer.self_attn.clear_kv_cache()

    @torch.no_grad()
    def reset_kv_caches(self) -> None:
        return None

    @torch.no_grad()
    def offload_kv_caches(self) -> None:
        # Move KV cache tensors back to host memory to release GPU memory
        # between sessions; drops the cached backend so it gets rebuilt.
        for layer in self.layers:
            attention = layer.self_attn
            if attention.kv_shared_layer_index is not None:
                attention.infer_backend = None
                continue
            if attention.k_cache.numel() > 0:
                attention.k_cache = attention.k_cache.to(device="cpu")
            if attention.v_cache.numel() > 0:
                attention.v_cache = attention.v_cache.to(device="cpu")
            attention.infer_backend = None
        self._rebind_shared_kv_caches()

    @torch.no_grad()
    def onload_kv_caches(self, device: torch.device) -> bool:
        found = False
        for layer in self.layers:
            attention = layer.self_attn
            if attention.kv_shared_layer_index is not None:
                continue
            if attention.k_cache.numel() > 0:
                found = True
                if attention.k_cache.device != device:
                    attention.k_cache = attention.k_cache.to(device=device)
            if attention.v_cache.numel() > 0 and attention.v_cache.device != device:
                attention.v_cache = attention.v_cache.to(device=device)
        self._rebind_shared_kv_caches()
        return found

    def _rebind_shared_kv_caches(self) -> None:
        for layer in self.layers:
            attention = layer.self_attn
            shared_idx = attention.kv_shared_layer_index
            if shared_idx is None:
                continue
            source = self.layers[shared_idx].self_attn
            attention.set_kv_cache(source.k_cache, source.v_cache)
            attention.infer_backend = None


class Gemma4Adapter(ModelAdapter):
    """Adapter glue for both Gemma4 LM and Gemma4 multimodal trunks."""

    name = "gemma4"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        architectures = set(hf_config.get("architectures") or [])
        model_type = str(hf_config.get("model_type", "")).lower()
        return model_type in {"gemma4", "gemma4_unified"} or bool(
            architectures
            & {
                "Gemma4ForCausalLM",
                "Gemma4ForConditionalGeneration",
                "Gemma4UnifiedForConditionalGeneration",
            }
        )

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        # Multimodal and unified checkpoints wrap text settings under
        # text_config and live at model.language_model.* in the safetensors;
        # non-MM stays under the standard model.* prefix.
        text_config = hf_config.get("text_config")
        checkpoint_prefix = "model.language_model" if isinstance(text_config, dict) else "model"
        cfg = text_config if isinstance(text_config, dict) else hf_config
        _reject_unsupported_gemma4(cfg)
        # layer_types tells each layer whether it runs full or sliding attention.
        layer_types = tuple(cfg.get("layer_types") or ("full_attention",) * int(cfg["num_hidden_layers"]))
        base_head_dim = int(cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]))
        # global_head_dim overrides head_dim for full-attention layers when set;
        # swa_head_dim does the same for sliding-attention layers (handled in
        # _attention_head_dim).
        head_dim = int(cfg.get("global_head_dim") or base_head_dim)
        kv_heads = int(
            cfg.get("num_global_key_value_heads") or cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        )
        enable_moe_block = bool(cfg.get("enable_moe_block", False))
        top_k_experts = cfg.get("top_k_experts")
        if top_k_experts is None:
            top_k_experts = cfg.get("num_experts_per_tok")
        return ModelConfig(
            model_type=self.name,
            checkpoint_prefix=checkpoint_prefix,
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 131072)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
            qkv_bias=bool(cfg.get("attention_bias", False)),
            qk_norm=True,
            v_norm=True,
            dtype=_parse_dtype(
                hf_config.get("torch_dtype") or cfg.get("torch_dtype") or hf_config.get("dtype") or cfg.get("dtype")
            ),
            hidden_act=str(cfg.get("hidden_activation", cfg.get("hidden_act", "gelu_pytorch_tanh"))),
            layer_types=layer_types,
            sliding_window=cfg.get("sliding_window"),
            # Sliding-attention layers may carry a different head_dim/kv-head
            # count from the full-attention layers.
            swa_head_dim=int(cfg.get("swa_head_dim") or base_head_dim),
            swa_num_key_value_heads=int(
                cfg.get("swa_num_key_value_heads") or cfg.get("num_key_value_heads", cfg["num_attention_heads"])
            ),
            rope_parameters=cfg.get("rope_parameters"),
            attention_k_eq_v=bool(cfg.get("attention_k_eq_v", False)),
            num_kv_shared_layers=int(cfg.get("num_kv_shared_layers") or 0),
            partial_rotary_factor=float(cfg.get("partial_rotary_factor", 1.0)),
            hidden_size_per_layer_input=int(cfg.get("hidden_size_per_layer_input") or 0),
            vocab_size_per_layer_input=cfg.get("vocab_size_per_layer_input"),
            use_double_wide_mlp=bool(cfg.get("use_double_wide_mlp", False)),
            enable_moe_block=enable_moe_block,
            num_experts=int(cfg["num_experts"]) if enable_moe_block and cfg.get("num_experts") is not None else None,
            num_experts_per_tok=int(top_k_experts) if enable_moe_block and top_k_experts is not None else 1,
            moe_intermediate_size=int(cfg.get("moe_intermediate_size") or 0) if enable_moe_block else 0,
            norm_topk_prob=bool(cfg.get("norm_topk_prob", True)),
            sequence_parallel=bool(cfg.get("sequence_parallel", True)),
            # Gemma4 uses softmax scale of 1.0 (Q is pre-divided by sqrt(d) via
            # head-norm); the actual scale is enforced by the attention backend.
            attention_softmax_scale=1.0,
            final_logit_softcapping=cfg.get("final_logit_softcapping"),
        )

    def build(self, config: ModelConfig) -> nn.Module:
        _validate_gemma4_config(config)
        return Gemma4ForCausalLM(config)

    @torch.no_grad()
    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        if not isinstance(model, Gemma4ForCausalLM):
            raise TypeError(f"Gemma4Adapter cannot load weights into {type(model)!r}")
        # The runtime prefix selects between LM-only and multimodal layouts.
        load_checkpoint_weights(model, model_path, checkpoint_spec(model.config.checkpoint_prefix))

    @torch.no_grad()
    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        if not isinstance(model, Gemma4ForCausalLM):
            raise TypeError(f"Gemma4Adapter cannot save weights from {type(model)!r}")
        return save_checkpoint_weights(model, output_path, source_path, checkpoint_spec(model.config.checkpoint_prefix))


def _layer_type(config: ModelConfig, layer_idx: int) -> str:
    """Return ``full_attention``/``sliding_attention`` for the given layer index."""
    if config.layer_types is None:
        return "full_attention"
    return config.layer_types[layer_idx]


def _attention_head_dim(config: ModelConfig, layer_type: str) -> int:
    """Select head_dim per layer_type (sliding layers may use swa_head_dim)."""
    if layer_type == "sliding_attention" and config.swa_head_dim is not None:
        return int(config.swa_head_dim)
    return int(config.head_dim)


def _attention_kv_heads(config: ModelConfig, layer_type: str) -> int:
    """Select num_key_value_heads per layer_type (sliding may differ from global)."""
    if layer_type == "sliding_attention" and config.swa_num_key_value_heads is not None:
        return int(config.swa_num_key_value_heads)
    return int(config.num_key_value_heads)


def _kv_shared_layer_index(config: ModelConfig, layer_idx: int) -> int:
    """Walk backwards from ``layer_idx`` to find the previous layer of the same type.

    KV-shared layers can only share K/V with a layer that ran the same
    attention type (full vs sliding) because the head shapes must match.
    """
    if config.layer_types is None:
        raise ValueError("Gemma4 KV-sharing requires layer_types")
    first_shared = config.num_hidden_layers - config.num_kv_shared_layers
    current_type = config.layer_types[layer_idx]
    for idx in range(first_shared - 1, -1, -1):
        if config.layer_types[idx] == current_type:
            return idx
    raise ValueError(f"Gemma4 shared KV layer {layer_idx} has no previous {current_type!r} layer")


def _rope_theta(config: ModelConfig, layer_type: str) -> float:
    """Per-layer-type RoPE theta override; falls back to global rope_theta."""
    params = config.rope_parameters or {}
    if layer_type in params:
        return float(params[layer_type].get("rope_theta", config.rope_theta))
    return config.rope_theta


def _rope_partial_rotary_factor(config: ModelConfig, layer_type: str) -> float:
    """Per-layer-type partial rotary fraction override (1.0 = full rotation)."""
    params = config.rope_parameters or {}
    if layer_type in params:
        return float(params[layer_type].get("partial_rotary_factor", config.partial_rotary_factor))
    return config.partial_rotary_factor


def _reject_unsupported_gemma4(cfg: dict[str, Any]) -> None:
    """Fail fast on Gemma4 variants the adapter does not implement."""
    if str(cfg.get("hidden_activation", cfg.get("hidden_act", "gelu_pytorch_tanh"))) != "gelu_pytorch_tanh":
        raise ValueError("Gemma4 support requires hidden_activation='gelu_pytorch_tanh'")
    # The HF config exposes two aliases per dim; reject configs that set both
    # to avoid ambiguity.
    if cfg.get("swa_head_dim") is not None and cfg.get("global_head_dim") is not None:
        raise ValueError("Gemma4 configs should use either global_head_dim or swa_head_dim, not both")
    if cfg.get("swa_num_key_value_heads") is not None and cfg.get("num_global_key_value_heads") is not None:
        raise ValueError(
            "Gemma4 configs should use either num_global_key_value_heads or swa_num_key_value_heads, not both"
        )


def _validate_gemma4_config(config: ModelConfig) -> None:
    """Cross-check ModelConfig against TP world size and Gemma4 invariants."""
    if config.enable_moe_block:
        if config.num_experts is None or config.num_experts <= 0:
            raise ValueError("Gemma4 MoE requires num_experts")
        if config.num_experts_per_tok <= 0:
            raise ValueError("Gemma4 MoE requires top_k_experts > 0")
        if config.moe_intermediate_size <= 0:
            raise ValueError("Gemma4 MoE requires moe_intermediate_size")
    if config.layer_types is not None and len(config.layer_types) != config.num_hidden_layers:
        raise ValueError("Gemma4 layer_types length must match num_hidden_layers")
    if config.layer_types is not None and "sliding_attention" in config.layer_types and not config.sliding_window:
        raise ValueError("Gemma4 sliding_attention layers require sliding_window")
    if config.num_kv_shared_layers < 0 or config.num_kv_shared_layers >= config.num_hidden_layers:
        raise ValueError("Gemma4 num_kv_shared_layers must be in [0, num_hidden_layers)")
    if config.num_kv_shared_layers and config.layer_types is None:
        raise ValueError("Gemma4 KV-sharing requires layer_types")
    ctx = get_tp_context()
    # Each unique layer_type must have head counts divisible by TP world size.
    for layer_type in set(config.layer_types or ("full_attention",)):
        if config.num_attention_heads % ctx.world_size != 0:
            raise ValueError("Gemma4 num_attention_heads must divide tensor parallel world size")
        kv_heads = _attention_kv_heads(config, layer_type)
        if kv_heads % ctx.world_size != 0 and ctx.world_size % kv_heads != 0:
            raise ValueError(
                f"Gemma4 {layer_type} num_key_value_heads must either divide or be replicated across tensor parallel world size"
            )


@torch._dynamo.disable
def _gemma4_per_layer_inputs_no_compile(
    model: Gemma4ForCausalLM,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor | None:
    # Dynamo-disabled wrapper around the PLE pipeline because get/project rely
    # on data-dependent indexing and reshapes.
    return model.project_per_layer_inputs(hidden_states, model.get_per_layer_inputs(input_ids))


@torch._dynamo.disable
def _areno_optional_scale_rmsnorm_no_compile(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    return areno_optional_scale_rmsnorm(x, weight, eps)


@torch._dynamo.disable
def _areno_gelu_tanh_and_mul_no_compile(x: torch.Tensor) -> torch.Tensor:
    return areno_gelu_tanh_and_mul(x)


@torch._dynamo.disable
def _areno_linear_no_compile(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _areno_linear_forward(x, weight, None)


@torch._dynamo.disable
def _areno_topk_softmax_no_compile(
    logits: torch.Tensor, top_k: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return areno_topk_softmax(logits, top_k, renormalize)


@torch._dynamo.disable
def _areno_moe_topk_permute_no_compile(
    flat: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    local_expert_start: int,
    local_num_experts: int,
):
    return areno_moe_topk_permute(flat, topk_idx, topk_weight, local_expert_start, local_num_experts)


@torch._dynamo.disable
def _areno_moe_unpermute_no_compile(
    expert_out: torch.Tensor, token_idx: torch.Tensor, restore_shape: tuple[int, int]
) -> torch.Tensor:
    return areno_moe_unpermute(expert_out, token_idx, restore_shape)


@torch._dynamo.disable
def _areno_grouped_linear_no_compile(
    x: torch.Tensor, weight: torch.Tensor, tokens_per_expert: torch.Tensor
) -> torch.Tensor:
    return areno_grouped_linear(x, weight, tokens_per_expert)


@torch._dynamo.disable
def _areno_fused_experts_gelu_no_compile(
    flat: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_idx: torch.Tensor,
    config: FusedMoeConfig,
) -> torch.Tensor:
    return areno_fused_experts(flat, w1, w2, topk_weight, topk_idx, config, activation="gelu_tanh")
