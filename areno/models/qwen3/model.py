"""Qwen3 causal-LM adapter.

Targets the Qwen/Qwen3-* family of HF checkpoints (``model_type == "qwen3"``).
The architecture is a vanilla GQA decoder:
    * Pre-norm RMSNorm + CausalSelfAttention + GatedMLP (SwiGLU).
    * GQA head ratio comes from ``num_attention_heads`` / ``num_key_value_heads``
      in the HF config (the adapter trusts the HF value as-is rather than
      deriving anything special).
    * Optional QK RMSNorm (``qk_layernorm``) is enabled by default to match the
      Qwen3 reference; the underlying CausalSelfAttention picks it up from
      ``config.qk_norm``.
    * Rotary uses full rotary fraction with the HF-provided ``rope_theta`` (1e6
      for the public Qwen3 checkpoints).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from areno.accel import areno_grouped_linear, areno_linear, areno_moe_topk_permute, areno_moe_unpermute, areno_topk_softmax
from torch import nn

from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.checkpoints.common import load_checkpoint_weights, save_checkpoint_weights
from areno.models.qwen3.checkpoint import CHECKPOINT_SPEC, QWEN3_MOE_CHECKPOINT_SPEC
from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.parallel.collectives import all_reduce, scatter_to_sequence_parallel_region, sequence_parallel_region
from areno.engine.layers.attention import CausalSelfAttention
from areno.engine.layers.mlp import GatedMLP
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.linear import mark_tensor_parallel_parameter
from areno.models.base import CausalLMOutput, ModelAdapter
from areno.accel.ops import FusedMoeConfig, areno_silu_and_mul, areno_fused_experts, log_once
from areno.engine.runtime.recompute import checkpoint_layer
from areno.engine.parallel.context import get_tp_context


class QwenDecoderLayer(nn.Module):
    """One Qwen3 transformer block: pre-norm attention + pre-norm SwiGLU MLP."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> torch.Tensor:
        # Standard pre-norm residual: norm -> sublayer -> add.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, position_ids, train_meta, infer_meta)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen3MoeExperts(nn.Module):
    """Expert-parallel fused SwiGLU experts for Qwen3-MoE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        ctx = get_tp_context()
        if config.num_experts is None:
            raise ValueError("qwen3_moe requires num_experts")
        if config.num_experts % ctx.world_size != 0:
            raise ValueError("num_experts must be divisible by tp_size for qwen3_moe")
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
        log_once("qwen3_moe_silu_and_mul", "using ARENO fused silu_and_mul kernel for Qwen3-MoE experts")
        hidden = (_areno_silu_and_mul_no_compile(hidden) * route_weight.unsqueeze(-1).to(dtype=hidden.dtype)).contiguous()
        out = _areno_grouped_linear_no_compile(hidden, self.down_weight, tokens_per_expert)
        out = _areno_moe_unpermute_no_compile(out, token_idx, flat.shape)
        return all_reduce(out)

    def local_routes(self, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_mask = (topk_idx >= self.local_expert_start) & (topk_idx < self.local_expert_end)
        local_idx = (topk_idx - self.local_expert_start).clamp(0, self.local_num_experts - 1)
        return local_idx, topk_weight * local_mask.to(dtype=topk_weight.dtype)

    @torch.no_grad()
    def copy_expert(self, expert_id: int, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, rank: int, world_size: int) -> None:
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


class Qwen3MoeMLP(nn.Module):
    """Qwen3-MoE router and expert block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.num_experts is None:
            raise ValueError("qwen3_moe requires num_experts")
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, dtype=torch.float32))
        mark_tensor_parallel_parameter(self.gate, False, sequence_parallel=False, tp_grad_allreduce=True)
        self.experts = Qwen3MoeExperts(config)
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
        log_once("qwen3_moe_topk_softmax", "using ARENO fused topk_softmax router for Qwen3-MoE")
        topk_idx, topk_weight = _areno_topk_softmax_no_compile(logits, self.top_k, self.norm_topk_prob)
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
            raise RuntimeError("Qwen3-MoE fused inference weights are not prepared")
        log_once("qwen3_moe_fused_experts", "using areno fused MoE expert kernel for Qwen3-MoE inference")
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


class Qwen3MoeDecoderLayer(QwenDecoderLayer):
    """Qwen3 block with routed MoE MLP."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = Qwen3MoeMLP(config)


class Qwen3ForCausalLM(nn.Module):
    """Top-level Qwen3 causal LM with vocab-parallel embedding/LM head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([QwenDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Optional weight tying with embed_tokens is wired up by the runtime
        # based on config.tie_word_embeddings; LM head allocates its own weight
        # otherwise.
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> CausalLMOutput:
        if position_ids is None:
            # Default to monotonic 0..S-1 positions broadcast across the batch.
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden_states = self.embed_tokens(input_ids)
        use_sequence_parallel = bool(train_meta is not None and train_meta.sequence_parallel)
        if use_sequence_parallel:
            # Split sequence dim across TP ranks before entering the SP region.
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        with sequence_parallel_region(use_sequence_parallel):
            for layer in self.layers:
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
        """Bind one (k_cache, v_cache) pair per decoder layer for inference."""
        if len(kv_caches) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} layer caches, got {len(kv_caches)}")
        for layer, (k_cache, v_cache) in zip(self.layers, kv_caches, strict=True):
            layer.self_attn.set_kv_cache(k_cache, v_cache)

    @torch.no_grad()
    def prepare_infer_weights(self) -> None:
        # Qwen3 has no MoE/fused-inference weights; nothing to do.
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
        # Dense model; no MoE router state to finalize.
        del tp_group, dp_group
        return None

    def allocate_kv_caches(self, num_blocks: int, block_size: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Allocate paged KV cache tensors shaped [blocks, block_size, local_kv_heads, head_dim]."""
        caches = []
        for layer in self.layers:
            attention = layer.self_attn
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
        # Qwen3 caches are write-on-step and don't need zeroing between requests.
        return None

    @torch.no_grad()
    def offload_kv_caches(self) -> None:
        # Move populated KV caches to CPU to free device memory; drop the
        # backend handle so it gets rebuilt against the new tensors on reload.
        for layer in self.layers:
            attention = layer.self_attn
            if attention.k_cache.numel() > 0:
                attention.k_cache = attention.k_cache.to(device="cpu")
            if attention.v_cache.numel() > 0:
                attention.v_cache = attention.v_cache.to(device="cpu")
            attention.infer_backend = None

    @torch.no_grad()
    def onload_kv_caches(self, device: torch.device) -> bool:
        # Returns True if at least one layer had a non-empty cache to move,
        # used by the runtime to short-circuit when nothing was offloaded.
        found = False
        for layer in self.layers:
            attention = layer.self_attn
            if attention.k_cache.numel() > 0:
                found = True
                if attention.k_cache.device != device:
                    attention.k_cache = attention.k_cache.to(device=device)
            if attention.v_cache.numel() > 0 and attention.v_cache.device != device:
                attention.v_cache = attention.v_cache.to(device=device)
        return found


class Qwen3MoeForCausalLM(Qwen3ForCausalLM):
    """Qwen3-MoE causal LM reusing Qwen3 attention and KV-cache layout."""

    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)

    @torch.no_grad()
    def prepare_infer_weights(self) -> None:
        for layer in self.layers:
            layer.mlp.prepare_infer_weights()

    @torch.no_grad()
    def clear_infer_weights(self) -> None:
        for layer in self.layers:
            layer.mlp.clear_infer_weights()


class Qwen3Adapter(ModelAdapter):
    """Adapter glue: HF config detection, build, and checkpoint I/O."""

    name = "qwen3"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        return str(hf_config.get("model_type", "")).lower() == "qwen3"

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        # Convert HF JSON into areno's ModelConfig. ``head_dim`` defaults to
        # hidden/num_heads when the HF config omits it (older Qwen3 variants).
        dtype = _parse_dtype(hf_config.get("torch_dtype") or hf_config.get("dtype"))
        return ModelConfig(
            model_type=self.name,
            vocab_size=int(hf_config["vocab_size"]),
            hidden_size=int(hf_config["hidden_size"]),
            intermediate_size=int(hf_config["intermediate_size"]),
            num_hidden_layers=int(hf_config["num_hidden_layers"]),
            num_attention_heads=int(hf_config["num_attention_heads"]),
            num_key_value_heads=int(hf_config.get("num_key_value_heads", hf_config["num_attention_heads"])),
            head_dim=int(hf_config.get("head_dim", hf_config["hidden_size"] // hf_config["num_attention_heads"])),
            rms_norm_eps=float(hf_config.get("rms_norm_eps", 1e-6)),
            # Qwen3 ships with a very long-context rope_theta (1e6).
            rope_theta=float(hf_config.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=int(hf_config.get("max_position_embeddings", 40960)),
            tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
            qkv_bias=bool(hf_config.get("attention_bias", hf_config.get("qkv_bias", False))),
            qk_norm=bool(hf_config.get("qk_layernorm", True)),
            dtype=dtype,
            sequence_parallel=bool(hf_config.get("sequence_parallel", True)),
        )

    def build(self, config: ModelConfig) -> nn.Module:
        return Qwen3ForCausalLM(config)

    @torch.no_grad()
    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        if not isinstance(model, Qwen3ForCausalLM):
            raise TypeError(f"Qwen3Adapter cannot load weights into {type(model)!r}")
        # Generic loader walks CHECKPOINT_SPEC, fetching/sharding from HF safetensors.
        load_checkpoint_weights(model, model_path, CHECKPOINT_SPEC)

    @torch.no_grad()
    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        if not isinstance(model, Qwen3ForCausalLM):
            raise TypeError(f"Qwen3Adapter cannot save weights from {type(model)!r}")
        return save_checkpoint_weights(model, output_path, source_path, CHECKPOINT_SPEC)


class Qwen3MoeAdapter(ModelAdapter):
    """Adapter for Qwen3 MoE checkpoints (for example Qwen3-30B-A3B)."""

    name = "qwen3_moe"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        return str(hf_config.get("model_type", "")).lower() == "qwen3_moe"

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        dtype = _parse_dtype(hf_config.get("torch_dtype") or hf_config.get("dtype"))
        return ModelConfig(
            model_type=self.name,
            vocab_size=int(hf_config["vocab_size"]),
            hidden_size=int(hf_config["hidden_size"]),
            intermediate_size=int(hf_config["intermediate_size"]),
            num_hidden_layers=int(hf_config["num_hidden_layers"]),
            num_attention_heads=int(hf_config["num_attention_heads"]),
            num_key_value_heads=int(hf_config.get("num_key_value_heads", hf_config["num_attention_heads"])),
            head_dim=int(hf_config.get("head_dim", hf_config["hidden_size"] // hf_config["num_attention_heads"])),
            rms_norm_eps=float(hf_config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(hf_config.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=int(hf_config.get("max_position_embeddings", 40960)),
            tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
            qkv_bias=bool(hf_config.get("attention_bias", hf_config.get("qkv_bias", False))),
            qk_norm=bool(hf_config.get("qk_layernorm", hf_config.get("qk_norm", True))),
            dtype=dtype,
            sequence_parallel=False,
            enable_moe_block=True,
            num_experts=int(hf_config["num_experts"]),
            num_experts_per_tok=int(hf_config["num_experts_per_tok"]),
            moe_intermediate_size=int(hf_config["moe_intermediate_size"]),
            norm_topk_prob=bool(hf_config.get("norm_topk_prob", True)),
            score_function="softmax",
        )

    def build(self, config: ModelConfig) -> nn.Module:
        return Qwen3MoeForCausalLM(config)

    @torch.no_grad()
    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        if not isinstance(model, Qwen3MoeForCausalLM):
            raise TypeError(f"Qwen3MoeAdapter cannot load weights into {type(model)!r}")
        load_checkpoint_weights(model, model_path, QWEN3_MOE_CHECKPOINT_SPEC)

    @torch.no_grad()
    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        if not isinstance(model, Qwen3MoeForCausalLM):
            raise TypeError(f"Qwen3MoeAdapter cannot save weights from {type(model)!r}")
        return save_checkpoint_weights(model, output_path, source_path, QWEN3_MOE_CHECKPOINT_SPEC)


@torch._dynamo.disable
def _areno_moe_topk_permute_no_compile(
    flat: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    local_expert_start: int,
    local_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return areno_moe_topk_permute(flat, topk_idx, topk_weight, local_expert_start, local_num_experts)


@torch._dynamo.disable
def _areno_moe_unpermute_no_compile(expert_out: torch.Tensor, token_idx: torch.Tensor, restore_shape: tuple[int, int]) -> torch.Tensor:
    return areno_moe_unpermute(expert_out, token_idx, restore_shape)


@torch._dynamo.disable
def _areno_grouped_linear_no_compile(x: torch.Tensor, weight: torch.Tensor, tokens_per_expert: torch.Tensor | Sequence[int]) -> torch.Tensor:
    return areno_grouped_linear(x, weight, tokens_per_expert)


@torch._dynamo.disable
def _areno_linear_no_compile(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return areno_linear(x, weight, None)


@torch._dynamo.disable
def _areno_silu_and_mul_no_compile(x: torch.Tensor) -> torch.Tensor:
    return areno_silu_and_mul(x)


@torch._dynamo.disable
def _areno_topk_softmax_no_compile(logits: torch.Tensor, top_k: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return areno_topk_softmax(logits, top_k, renormalize)


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
