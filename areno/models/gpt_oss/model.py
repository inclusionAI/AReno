"""AReno-native GPT-OSS: sinks, YaRN, alternating windows and biased MoE.

HF's MXFP4 experts are materialized to the configured training dtype when
loading. Training and inference never import Transformers model code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from areno.accel import (
    areno_grouped_linear,
    areno_linear,
    areno_moe_topk_permute,
    areno_moe_unpermute,
    areno_topk_softmax,
)
from areno.accel.grouped_bias import grouped_bias, limited_swiglu
from areno.accel.ops import FusedMoeConfig, areno_fused_experts
from areno.accel.sink_attention import paged_sink_attention, sink_attention, sink_sequence_starts
from areno.engine.checkpoints.common import load_checkpoint_weights, save_checkpoint_weights
from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.attention import CausalSelfAttention
from areno.engine.layers.attention_backend.infer import _store_decode_cache, _store_prefill_cache
from areno.engine.layers.linear import RowParallelLinear, mark_tensor_parallel_parameter
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.parallel.collectives import all_reduce, copy_to_tensor_parallel_region
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.recompute import checkpoint_layer
from areno.engine.runtime.routing_replay import resolve_softmax_routes, routing_replay_context
from areno.models.base import CausalLMOutput, ModelAdapter
from areno.models.gpt_oss.checkpoint import SPEC
from areno.models.qwen3.model import Qwen3ForCausalLM


class YarnRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        rope = (config.hf_text_config or {}).get("rope_scaling") or {}
        dim = config.head_dim
        factor = float(rope.get("factor", 1.0))
        freq = config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv = 1 / freq
        if rope.get("rope_type", "default") == "yarn":
            original = rope.get("original_max_position_embeddings", 4096)

            def correction(rotations):
                return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(config.rope_theta))

            lo = correction(rope.get("beta_fast", 32))
            hi = correction(rope.get("beta_slow", 1))
            if rope.get("truncate", True):
                lo, hi = math.floor(lo), math.ceil(hi)
            lo, hi = max(lo, 0), min(hi, dim - 1)
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - lo) / (hi - lo if hi != lo else 0.001)).clamp(0, 1)
            inv = inv * (1 - ramp) + inv / factor * ramp
        self.scale = float(rope.get("attention_factor", 1.0 + 0.1 * math.log(factor) if factor > 1 else 1.0))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, q, k, position_ids):
        angles = position_ids.float()[..., None] * self.inv_freq.float()
        cos = torch.cat((angles.cos(), angles.cos()), -1).mul(self.scale).to(q.dtype).unsqueeze(2)
        sin = torch.cat((angles.sin(), angles.sin()), -1).mul(self.scale).to(q.dtype).unsqueeze(2)

        def rotate(x):
            a, b = x.chunk(2, -1)
            return torch.cat((-b, a), -1)

        return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


class GptOssAttention(CausalSelfAttention):
    def __init__(self, config, index):
        super().__init__(config, index, rotary_embedding=YarnRotaryEmbedding(config))
        self.o_proj = RowParallelLinear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=True)
        self.qkv_proj.to(dtype=config.dtype)
        self.o_proj.to(dtype=config.dtype)
        self.sinks = nn.Parameter(torch.empty(self.local_heads, dtype=config.dtype))
        mark_tensor_parallel_parameter(self.sinks, True, sequence_parallel=False)
        self.window = config.sliding_window if config.layer_types[index] == "sliding_attention" else None
        self._sequence_starts = None

    def forward_train(self, q, k, v, meta):
        out = sink_attention(q, k, v, self.sinks, sequence_starts=self._sequence_starts, window=self.window)
        return self.o_proj(out.reshape(q.shape[0], q.shape[1], -1))

    def forward_infer(self, q, k, v, meta):
        qf, kf, vf = (x.reshape(-1, x.shape[-2], x.shape[-1]) for x in (q, k, v))
        if meta.mode == "prefill":
            _store_prefill_cache(kf, vf, self.k_cache, self.v_cache, meta)
            seq = torch.bucketize(torch.arange(qf.shape[0], device=q.device), meta.cu_seqlens[1:], right=True)
            positions = torch.arange(qf.shape[0], device=q.device) - meta.cu_seqlens[seq]
            if meta.cache_seqlens is not None:
                positions = positions + meta.cache_seqlens[seq]
        else:
            if meta.tokens_per_seq != 1:
                raise ValueError("GPT-OSS speculative multi-token decode is not supported")
            _store_decode_cache(kf, vf, self.k_cache, self.v_cache, meta)
            seq = torch.arange(qf.shape[0], device=q.device)
            positions = meta.cache_seqlens
        out = paged_sink_attention(
            qf, self.k_cache, self.v_cache, self.sinks, meta.block_table, seq, positions, self.window
        )
        return self.o_proj(out.reshape(q.shape[0], q.shape[1], -1))


class GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        ctx = get_tp_context()
        if config.num_experts % ctx.world_size:
            raise ValueError("expert count must be divisible by TP")
        self.count = config.num_experts // ctx.world_size
        self.start = ctx.rank * self.count
        h, i = config.hidden_size, config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.count, 2 * i, h, dtype=config.dtype))
        self.down_proj = nn.Parameter(torch.empty(self.count, h, i, dtype=config.dtype))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.count, 2 * i, dtype=config.dtype))
        self.down_proj_bias = nn.Parameter(torch.empty(self.count, h, dtype=config.dtype))
        for p in self.parameters():
            mark_tensor_parallel_parameter(p, True, sequence_parallel=False, tp_grad_allreduce=False)
        self.fused_config = FusedMoeConfig(self.count, h, i, config.num_experts_per_tok)
        hf = config.hf_text_config or {}
        self.alpha = float(hf.get("swiglu_alpha", 1.702))
        self.limit = float(hf.get("swiglu_limit", 7.0))

    def forward(self, x, ids, weights):
        if not torch.is_grad_enabled():
            local = (ids >= self.start) & (ids < self.start + self.count)
            local_ids = (ids - self.start).clamp(0, self.count - 1).int()
            out = areno_fused_experts(
                x,
                self.gate_up_proj,
                self.down_proj,
                weights.float() * local,
                local_ids,
                self.fused_config,
                activation="limited_swiglu",
                bias1=self.gate_up_proj_bias,
                bias2=self.down_proj_bias,
                activation_alpha=self.alpha,
                activation_limit=self.limit,
            )
            return all_reduce(out)
        routed, route_weights, token_ids, counts = areno_moe_topk_permute(
            x, ids.long(), weights.float(), self.start, self.count
        )
        hidden = areno_grouped_linear(routed, self.gate_up_proj, counts)
        hidden = limited_swiglu(grouped_bias(hidden, self.gate_up_proj_bias, counts), self.alpha, self.limit)
        out = grouped_bias(areno_grouped_linear(hidden, self.down_proj, counts), self.down_proj_bias, counts)
        out = (out * route_weights[:, None].to(out.dtype)).contiguous()
        return all_reduce(areno_moe_unpermute(out, token_ids, x.shape))


class GptOssMLP(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        self.routing_layer_slot = layer_index
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=True, dtype=config.dtype)
        for p in self.router.parameters():
            mark_tensor_parallel_parameter(p, False, sequence_parallel=False, tp_grad_allreduce=True)
        self.experts = GptOssExperts(config)

    def forward(self, x, train_meta=None):
        flat = copy_to_tensor_parallel_region(x.reshape(-1, x.shape[-1]))
        logits = areno_linear(flat, self.router.weight, self.router.bias)
        ids, weights = areno_topk_softmax(logits, self.top_k, True)
        # Layer checkpoint recompute runs after the outer routing context exits.
        # Restore replay metadata here while preserving inference capture.
        with routing_replay_context(train_meta):
            ids, weights = resolve_softmax_routes(self.routing_layer_slot, logits, ids, weights, renormalize=True)
        return self.experts(flat, ids, weights.to(flat.dtype)).reshape(x.shape)


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GptOssAttention(config, index)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GptOssMLP(config, index)

    def forward(self, x, positions, train_meta, infer_meta, mask):
        self.self_attn._sequence_starts = mask
        x = x + self.self_attn(self.input_layernorm(x), positions, train_meta, infer_meta)
        return x + self.mlp(self.post_attention_layernorm(x), train_meta=train_meta)


class GptOssForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([GptOssDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)

    def forward(self, input_ids, position_ids=None, train_meta=None, infer_meta=None, defer_lm_head=False):
        if train_meta is not None and train_meta.sequence_parallel:
            raise ValueError("GPT-OSS sequence parallelism is not yet supported; use sequence_parallel=False")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].expand_as(input_ids)
        hidden = self.embed_tokens(input_ids)
        sequence_starts = None
        if infer_meta is None and hidden.is_cuda:
            cu = getattr(train_meta, "cu_seqlens", None)
            seq = (
                None
                if cu is None
                else torch.bucketize(torch.arange(input_ids.shape[1], device=hidden.device), cu[1:], right=True)
            )
            sequence_starts = sink_sequence_starts(input_ids.shape[1], hidden.device, seq)
        for layer in self.layers:
            hidden = checkpoint_layer(
                layer,
                hidden,
                position_ids,
                train_meta,
                infer_meta,
                sequence_starts,
                train_meta=train_meta,
                infer_meta=infer_meta,
            )
        hidden = self.norm(hidden)
        return CausalLMOutput(None if defer_lm_head else self.lm_head(hidden), hidden)


class GptOssAdapter(ModelAdapter):
    name = "gpt_oss"

    def match_hf_config(self, hf_config):
        return hf_config.get("model_type") == "gpt_oss"

    def config_from_hf(self, hf):
        rope = hf.get("rope_scaling") or {}
        if rope.get("rope_type", "default") not in ("default", "yarn"):
            raise ValueError("GPT-OSS supports default/YaRN RoPE")
        layers = hf.get("layer_types") or [
            "sliding_attention" if i % 2 == 0 else "full_attention" for i in range(hf["num_hidden_layers"])
        ]
        if (
            any(x not in ("sliding_attention", "full_attention") for x in layers)
            or len(layers) != hf["num_hidden_layers"]
        ):
            raise ValueError("invalid GPT-OSS layer_types")
        return ModelConfig(
            model_type=self.name,
            vocab_size=int(hf["vocab_size"]),
            hidden_size=int(hf["hidden_size"]),
            intermediate_size=int(hf["intermediate_size"]),
            moe_intermediate_size=int(hf["intermediate_size"]),
            num_hidden_layers=int(hf["num_hidden_layers"]),
            num_attention_heads=int(hf["num_attention_heads"]),
            num_key_value_heads=int(hf["num_key_value_heads"]),
            head_dim=int(hf.get("head_dim", 64)),
            max_position_embeddings=int(hf.get("max_position_embeddings", 131072)),
            rope_theta=float(hf.get("rope_theta", 150000)),
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
            num_experts=int(hf["num_local_experts"]),
            num_experts_per_tok=int(hf["num_experts_per_tok"]),
            qkv_bias=True,
            qk_norm=False,
            dtype=_parse_dtype(hf.get("dtype", hf.get("torch_dtype", "bfloat16"))),
            layer_types=tuple(layers),
            sliding_window=int(hf.get("sliding_window", 128)),
            hf_text_config=dict(hf),
            sequence_parallel=False,
            pad_token_id=int(hf.get("pad_token_id", 0)),
            tie_word_embeddings=False,
        )

    def build(self, config):
        return GptOssForCausalLM(config)

    def load_weights(self, model, model_path):
        load_checkpoint_weights(model, str(model_path), SPEC)

    def save_weights(self, model, output_path, source_path):
        result = save_checkpoint_weights(
            model, str(output_path), str(source_path) if source_path else None, SPEC, copy_passthrough=False
        )
        if result is not None:
            cfg = dict(model.config.hf_text_config or {})
            cfg.pop("quantization_config", None)
            cfg["torch_dtype"] = str(model.config.dtype).split(".")[-1]
            (Path(output_path) / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        return result

    def build_policy_plan(self, model):
        raise NotImplementedError("GPT-OSS direct policy synchronization of transposed experts is not yet supported")
