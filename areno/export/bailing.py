"""Bailing V3 (Ling 3) KDA/MLA and MoE tensor contracts."""

from __future__ import annotations

import numpy as np

from areno.export.architectures import Architecture, register_architecture


@register_architecture("bailing_hybrid", "bailing_moe_v3")
class Bailing3Export(Architecture):
    arch = "bailingmoe3"

    def full_attention(self, layer: int) -> bool:
        c = self.config
        count, group = int(c["num_hidden_layers"]), int(c.get("layer_group_size", 1))
        return layer >= count or (layer + 1) % group == 0 or layer >= count // group * group

    def build(self) -> None:
        c = self.config
        if (
            not c.get("no_kda_lora")
            or not c.get("kda_safe_gate")
            or c.get("gated_attention_proj_granularity_type") != "head_wise"
        ):
            raise ValueError("Ollama Bailing V3 requires no_kda_lora, safe KDA gates and head-wise attention gates")
        if c.get("rope_scaling") or c.get("use_bias") or c.get("use_qkv_bias") or c.get("hidden_act", "silu") != "silu":
            raise ValueError("unsupported Bailing V3 scaled RoPE, biases or activation")
        if c.get("expert_swiglu_limit_list") or c.get("share_expert_swiglu_limit_list"):
            raise ValueError("AReno's live Bailing V3 layout does not expose SwiGLU clamp semantics")
        c.setdefault("num_experts", c.get("n_routed_experts"))
        c.setdefault("num_experts_per_tok", c.get("moe_router_topk", 1))
        c.setdefault("routed_scaling_factor", c.get("moe_router_topk_scaling_factor", 1.0))
        if c.get("score_function", c.get("scoring_func", "sigmoid")) != "sigmoid" or not c.get("norm_topk_prob", True):
            raise ValueError("Bailing V3 export requires sigmoid routing and top-k normalization")
        hidden, heads = int(c["hidden_size"]), int(c["num_attention_heads"])
        vocab = int(c["vocab_size"])
        self.add("model.word_embeddings.weight", (vocab, hidden), "token_embd.weight")
        self.add("model.norm.weight", (hidden,), "output_norm.weight")
        if not c.get("tie_word_embeddings", False):
            self.add("lm_head.weight", (vocab, hidden), "output.weight")
        count = int(c["num_hidden_layers"]) + int(c.get("num_nextn_predict_layers") or 0)
        for layer in range(count):
            prefix = f"model.layers.{layer}"
            for name, target in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")):
                self.add(f"{prefix}.{name}.weight", (hidden,), f"blk.{layer}.{target}.weight")
            if layer >= int(c["num_hidden_layers"]):
                for name, shape, target in (
                    ("enorm", (hidden,), "nextn.enorm"),
                    ("hnorm", (hidden,), "nextn.hnorm"),
                    ("eh_proj", (hidden, 2 * hidden), "nextn.eh_proj"),
                    ("final_layernorm", (hidden,), "nextn.shared_head_norm"),
                ):
                    self.add(f"{prefix}.{name}.weight", shape, f"blk.{layer}.{target}.weight")
            if self.full_attention(layer):
                self.mla(layer)
            else:
                dim, kernel = int(c["head_dim"]), int(c.get("short_conv_kernel_size", 4))
                width = heads * dim
                for name, target, rows in (
                    ("q", "attn_q", width),
                    ("k", "attn_k", width),
                    ("v", "attn_v", width),
                    ("f", "ssm_f_a", width),
                    ("g", "ssm_g_a", width),
                    ("b", "ssm_beta", heads),
                ):
                    self.add(f"{prefix}.attention.{name}_proj.weight", (rows, hidden), f"blk.{layer}.{target}.weight")
                for name in ("q", "k", "v"):
                    self.add(
                        f"{prefix}.attention.{name}_conv1d.weight",
                        (width, 1, kernel),
                        f"blk.{layer}.ssm_conv1d_{name}.weight",
                        output_shape=(1, width, 1, kernel),
                        transform=lambda value: value.reshape(1, value.shape[0], 1, value.shape[-1]),
                        force_f32=True,
                    )
                self.add(f"{prefix}.attention.dt_bias", (width,), f"blk.{layer}.ssm_dt.bias")
                self.add(
                    f"{prefix}.attention.A_log",
                    (heads,),
                    f"blk.{layer}.ssm_a",
                    output_shape=(heads, 1),
                    transform=lambda value: np.exp(value).reshape(-1, 1),
                    force_f32=True,
                )
                self.add(f"{prefix}.attention.o_norm.weight", (dim,), f"blk.{layer}.ssm_norm.weight")
                self.add(f"{prefix}.attention.o_proj.weight", (hidden, width), f"blk.{layer}.attn_output.weight")
            if layer < int(c.get("first_k_dense_replace", 0)):
                self.dense_mlp(layer)
            else:
                self.moe(layer)
                experts = int(c["num_experts"])
                self.add(f"{prefix}.mlp.gate.expert_bias", (experts,), f"blk.{layer}.exp_probs_b.bias")
                shared = int(
                    c.get("moe_shared_expert_intermediate_size")
                    or int(c.get("num_shared_experts") or 0) * int(c["moe_intermediate_size"])
                )
                for name, shape in (("gate", (shared, hidden)), ("up", (shared, hidden)), ("down", (hidden, shared))):
                    if shared:
                        self.add(
                            f"{prefix}.mlp.shared_experts.{name}_proj.weight",
                            shape,
                            f"blk.{layer}.ffn_{name}_shexp.weight",
                        )

    def mla(self, layer: int) -> None:
        c = self.config
        hidden, heads = int(c["hidden_size"]), int(c["num_attention_heads"])
        rope, nope, vd, rank = (
            int(c["qk_rope_head_dim"]),
            int(c["qk_nope_head_dim"]),
            int(c["v_head_dim"]),
            int(c["kv_lora_rank"]),
        )
        prefix = f"model.layers.{layer}.attention"
        q_rank = c.get("q_lora_rank")
        if q_rank is None:
            self.add(f"{prefix}.q_proj.weight", (heads * (nope + rope), hidden), f"blk.{layer}.attn_q.weight")
        else:
            self.add(f"{prefix}.q_a_proj.weight", (int(q_rank), hidden), f"blk.{layer}.attn_q_a.weight")
            self.add(f"{prefix}.q_a_layernorm.weight", (int(q_rank),), f"blk.{layer}.attn_q_a_norm.weight")
            self.add(f"{prefix}.q_b_proj.weight", (heads * (nope + rope), int(q_rank)), f"blk.{layer}.attn_q_b.weight")
        self.add(f"{prefix}.kv_a_proj_with_mqa.weight", (rank + rope, hidden), f"blk.{layer}.attn_kv_a_mqa.weight")
        self.add(f"{prefix}.kv_a_layernorm.weight", (rank,), f"blk.{layer}.attn_kv_a_norm.weight")
        key = f"{prefix}.kv_b_proj.weight"
        self.add(
            key,
            (heads * (nope + vd), rank),
            f"blk.{layer}.attn_k_b.weight",
            output_shape=(heads, rank, nope),
            transform=lambda value: value.reshape(heads, nope + vd, rank)[:, :nope, :].swapaxes(1, 2),
        )
        self.add(
            key,
            (heads * (nope + vd), rank),
            f"blk.{layer}.attn_v_b.weight",
            output_shape=(heads, vd, rank),
            transform=lambda value: value.reshape(heads, nope + vd, rank)[:, nope:, :],
        )
        self.add(f"{prefix}.g_proj.weight", (heads, hidden), f"blk.{layer}.attn_gate.weight")
        self.add(f"{prefix}.dense.weight", (hidden, heads * vd), f"blk.{layer}.attn_output.weight")

    def metadata(self, writer) -> None:
        import gguf

        super().metadata(writer)
        c = self.config
        count = int(c["num_hidden_layers"]) + int(c.get("num_nextn_predict_layers") or 0)
        writer.add_block_count(count)
        writer.add_head_count_kv([1 if self.full_attention(layer) else 0 for layer in range(count)])
        writer.add_vocab_size(int(c["vocab_size"]))
        writer.add_ssm_conv_kernel(int(c.get("short_conv_kernel_size", 4)))
        writer.add_kda_head_dim(int(c["head_dim"]))
        writer.add_bool(f"{self.arch}.kda.safe_gate", True)
        writer.add_float32(f"{self.arch}.kda.gate_lower_bound", float(c["kda_lower_bound"]))
        if c.get("q_lora_rank") is not None:
            writer.add_q_lora_rank(int(c["q_lora_rank"]))
        writer.add_kv_lora_rank(int(c["kv_lora_rank"]))
        writer.add_rope_dimension_count(int(c["qk_rope_head_dim"]))
        writer.add_key_length(int(c["kv_lora_rank"]) + int(c["qk_rope_head_dim"]))
        writer.add_key_length_mla(int(c["qk_nope_head_dim"]) + int(c["qk_rope_head_dim"]))
        writer.add_value_length_mla(int(c["v_head_dim"]))
        self.moe_metadata(writer)
        writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        writer.add_expert_shared_count(int(c.get("num_shared_experts") or 0))
        writer.add_expert_shared_feed_forward_length(
            int(
                c.get("moe_shared_expert_intermediate_size")
                or int(c.get("num_shared_experts") or 0) * int(c["moe_intermediate_size"])
            )
        )
        writer.add_expert_group_count(int(c.get("n_group", c.get("moe_router_num_groups", 1))))
        writer.add_expert_group_used_count(int(c.get("topk_group", c.get("moe_router_group_topk", 1))))
        writer.add_leading_dense_block_count(int(c.get("first_k_dense_replace", 0)))
        writer.add_expert_weights_scale(float(c["routed_scaling_factor"]))
        if c.get("num_nextn_predict_layers"):
            writer.add_nextn_predict_layers(int(c["num_nextn_predict_layers"]))
