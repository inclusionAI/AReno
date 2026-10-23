"""Architecture contracts for live AReno tensors, independent of model execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

_ARCHITECTURES: dict[str, type[Architecture]] = {}


def register_architecture(*names: str):
    def register(cls):
        for name in names:
            if name in _ARCHITECTURES:
                raise ValueError(f"duplicate GGUF architecture: {name}")
            _ARCHITECTURES[name] = cls
        return cls

    return register


@dataclass(frozen=True)
class TensorRule:
    name: str
    shape: tuple[int, ...]
    transform: Callable[[np.ndarray], np.ndarray] | None = None
    expert: int | None = None
    packed_experts: bool = False
    force_f32: bool = False


def collection_blocks(name: str, shape: tuple[int, ...]) -> Iterator[tuple[int | None, int, tuple[int, ...]]]:
    """Read packed experts separately, never materializing an entire MoE slab."""
    if ".experts." in name and len(shape) == 3:
        size = int(np.prod(shape[1:]))
        for expert in range(shape[0]):
            yield expert, expert * size, shape[1:]
    else:
        yield None, 0, shape


class Architecture:
    arch: str
    tokenizer_profile = "auto"
    thinking_off = False

    def __init__(self, raw: dict[str, Any], shapes: dict[str, tuple[int, ...]]):
        self.raw = raw
        self.config = {**raw, **raw.get("text_config", {})}
        self.prefix = "model.language_model" if "model.language_model.embed_tokens.weight" in shapes else "model"
        self.head_key = (
            "model.language_model.lm_head.weight"
            if "model.language_model.lm_head.weight" in shapes
            else "lm_head.weight"
        )
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.rules: dict[str, list[TensorRule]] = {}
        self.extra: dict[str, np.ndarray] = {}
        self.build()

    def add(self, source: str, shape: tuple[int, ...], target: str, **kwargs) -> None:
        self.shapes[source] = shape
        self.rules.setdefault(source, []).append(TensorRule(target, kwargs.pop("output_shape", shape), **kwargs))

    def build(self) -> None:
        raise NotImplementedError

    @property
    def context_length(self) -> int:
        return int(self.config.get("max_position_embeddings", 2048))

    def metadata(self, writer) -> None:
        c = self.config
        hidden, heads = int(c["hidden_size"]), int(c["num_attention_heads"])
        dim = int(c.get("head_dim", hidden // heads))
        writer.add_name(f"AReno trained {self.arch}")
        writer.add_context_length(self.context_length)
        writer.add_embedding_length(hidden)
        writer.add_block_count(int(c["num_hidden_layers"]))
        writer.add_feed_forward_length(int(c.get("intermediate_size", c.get("moe_intermediate_size", 0))))
        writer.add_head_count(heads)
        writer.add_head_count_kv(int(c.get("num_key_value_heads", heads)))
        writer.add_key_length(dim)
        writer.add_value_length(dim)
        writer.add_rope_dimension_count(dim)
        writer.add_rope_freq_base(float(c.get("rope_theta", 10000)))
        writer.add_layer_norm_rms_eps(float(c.get("rms_norm_eps", 1e-6)))

    def embeddings(self, *, tied_default: bool = False) -> None:
        c = self.config
        hidden, vocab = int(c["hidden_size"]), int(c["vocab_size"])
        self.add(f"{self.prefix}.embed_tokens.weight", (vocab, hidden), "token_embd.weight")
        self.add(f"{self.prefix}.norm.weight", (hidden,), "output_norm.weight")
        if not c.get("tie_word_embeddings", tied_default):
            self.add(self.head_key, (vocab, hidden), "output.weight")

    def dense_mlp(self, layer: int, intermediate: int | None = None) -> None:
        c = self.config
        hidden = int(c["hidden_size"])
        intermediate = int(c["intermediate_size"]) if intermediate is None else intermediate
        for name, shape in (
            ("gate", (intermediate, hidden)),
            ("up", (intermediate, hidden)),
            ("down", (hidden, intermediate)),
        ):
            self.add(f"{self.prefix}.layers.{layer}.mlp.{name}_proj.weight", shape, f"blk.{layer}.ffn_{name}.weight")

    def attention(self, layer: int, *, qk_norm: str | None = "head", permute: bool = False) -> None:
        c = self.config
        hidden, heads = int(c["hidden_size"]), int(c["num_attention_heads"])
        kv_heads = int(c.get("num_key_value_heads", heads))
        dim = int(c.get("head_dim", hidden // heads))
        prefix = f"{self.prefix}.layers.{layer}"
        for name, count in (("q", heads), ("k", kv_heads), ("v", kv_heads)):
            transform = None
            if permute and name != "v":
                if dim % 2:
                    raise ValueError("Llama GGUF requires an even head dimension")

                def transform(tensor, count=count):
                    return tensor.reshape(count, 2, dim // 2, *tensor.shape[1:]).swapaxes(1, 2).reshape(tensor.shape)

            for suffix, shape in (("weight", (count * dim, hidden)), ("bias", (count * dim,))):
                if suffix == "bias" and not c.get("attention_bias", c.get("qkv_bias", False)):
                    continue
                self.add(
                    f"{prefix}.self_attn.{name}_proj.{suffix}",
                    shape,
                    f"blk.{layer}.attn_{name}.{suffix}",
                    transform=transform,
                )
        self.add(f"{prefix}.self_attn.o_proj.weight", (hidden, heads * dim), f"blk.{layer}.attn_output.weight")
        if qk_norm:
            for name, count in (("q", heads), ("k", kv_heads)):
                self.add(
                    f"{prefix}.self_attn.{name}_norm.weight",
                    (dim * count if qk_norm == "projection" else dim,),
                    f"blk.{layer}.attn_{name}_norm.weight",
                )

    def moe(self, layer: int, *, packed: bool = False, shared: bool = False) -> None:
        c = self.config
        hidden, count, intermediate = int(c["hidden_size"]), int(c["num_experts"]), int(c["moe_intermediate_size"])
        prefix = f"{self.prefix}.layers.{layer}.mlp"
        self.add(f"{prefix}.gate.weight", (count, hidden), f"blk.{layer}.ffn_gate_inp.weight", force_f32=True)
        for projection in ("gate", "up", "down"):
            shape = (hidden, intermediate) if projection == "down" else (intermediate, hidden)
            target = f"blk.{layer}.ffn_{projection}_exps.weight"
            if packed:
                if projection == "down":
                    self.add(f"{prefix}.experts.down_proj", (count, *shape), target, packed_experts=True)
                else:
                    start = 0 if projection == "gate" else intermediate
                    self.add(
                        f"{prefix}.experts.gate_up_proj",
                        (count, 2 * intermediate, hidden),
                        target,
                        output_shape=(count, *shape),
                        transform=lambda value, start=start: value[start : start + intermediate],
                        packed_experts=True,
                    )
            else:
                for expert in range(count):
                    self.add(
                        f"{prefix}.experts.{expert}.{projection}_proj.weight",
                        shape,
                        target,
                        output_shape=(count, *shape),
                        expert=expert,
                    )
        if shared:
            size = int(c.get("shared_expert_intermediate_size", 0))
            if size:
                self.add(
                    f"{prefix}.shared_expert_gate.weight",
                    (1, hidden),
                    f"blk.{layer}.ffn_gate_inp_shexp.weight",
                    force_f32=True,
                )
                for name, shape in (("gate", (size, hidden)), ("up", (size, hidden)), ("down", (hidden, size))):
                    self.add(
                        f"{prefix}.shared_expert.{name}_proj.weight", shape, f"blk.{layer}.ffn_{name}_shexp.weight"
                    )

    def moe_metadata(self, writer) -> None:
        c = self.config
        writer.add_expert_count(int(c["num_experts"]))
        writer.add_expert_used_count(int(c["num_experts_per_tok"]))
        writer.add_expert_feed_forward_length(int(c["moe_intermediate_size"]))
        writer.add_expert_weights_norm(bool(c.get("norm_topk_prob", True)))
        if c.get("shared_expert_intermediate_size"):
            writer.add_expert_shared_feed_forward_length(int(c["shared_expert_intermediate_size"]))


@register_architecture("qwen3", "qwen3_moe")
class Qwen3Export(Architecture):
    tokenizer_profile = "qwen2"
    thinking_off = True

    def build(self) -> None:
        c = self.config
        self.arch = "qwen3moe" if c["model_type"] == "qwen3_moe" else "qwen3"
        if c.get("rope_scaling") or c.get("use_sliding_window"):
            raise ValueError("GGUF Qwen3 export does not support scaled RoPE or sliding-window attention")
        if c.get("hidden_act", "silu") != "silu" or not c.get("qk_layernorm", c.get("qk_norm", True)):
            raise ValueError("GGUF Qwen3 export requires silu and Q/K RMSNorm")
        c.setdefault("rope_theta", 1_000_000)
        self.embeddings()
        for layer in range(int(c["num_hidden_layers"])):
            for name, target in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")):
                self.add(
                    f"{self.prefix}.layers.{layer}.{name}.weight",
                    (int(c["hidden_size"]),),
                    f"blk.{layer}.{target}.weight",
                )
            self.attention(layer)
            if self.arch == "qwen3moe":
                self.moe(layer)
            else:
                self.dense_mlp(layer)

    def metadata(self, writer) -> None:
        super().metadata(writer)
        if self.arch == "qwen3moe":
            self.moe_metadata(writer)


@register_architecture("llama", "olmo2")
class DenseExport(Architecture):
    def build(self) -> None:
        c = self.config
        self.arch = c["model_type"]
        c.setdefault("rms_norm_eps", 1e-5 if self.arch == "olmo2" else 1e-6)
        c.setdefault("max_position_embeddings", 2048 if self.arch == "olmo2" else 8192)
        c.setdefault("attention_bias", c.get("qkv_bias", c.get("bias", False)))
        if self.arch == "olmo2":
            self.tokenizer_profile = "olmo"
        if c.get("rope_scaling") or c.get("use_sliding_window") or c.get("sliding_window"):
            raise ValueError(f"GGUF {self.arch} export does not support scaled RoPE or sliding-window attention")
        if c.get("hidden_act", "silu") != "silu":
            raise ValueError(f"GGUF {self.arch} export requires hidden_act='silu'")
        self.embeddings()
        for layer in range(int(c["num_hidden_layers"])):
            norms = (
                (("post_attention_layernorm", "post_attention_norm"), ("post_feedforward_layernorm", "post_ffw_norm"))
                if self.arch == "olmo2"
                else (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"))
            )
            for name, target in norms:
                self.add(
                    f"{self.prefix}.layers.{layer}.{name}.weight",
                    (int(c["hidden_size"]),),
                    f"blk.{layer}.{target}.weight",
                )
            self.attention(layer, qk_norm="projection" if self.arch == "olmo2" else None, permute=self.arch == "llama")
            self.dense_mlp(layer)


@register_architecture(
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
)
class Qwen35Export(Architecture):
    tokenizer_profile = "qwen35"
    thinking_off = True

    def build(self) -> None:
        c = self.config
        self.arch = "qwen35moe" if c.get("num_experts") else "qwen35"
        self.rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
        if self.rope.get("rope_type", self.rope.get("type", "default")) not in {"default", "mrope"}:
            raise ValueError("scaled Qwen3.5 RoPE is not implemented by the AReno exporter")
        if (
            c.get("hidden_act", "silu") != "silu"
            or not c.get("attn_output_gate", True)
            or c.get("attention_bias", False)
        ):
            raise ValueError("Qwen3.5 GGUF requires silu, attention output gates and bias-free attention")
        c["rope_theta"] = self.rope.get("rope_theta", c.get("rope_theta", 10000))
        c.setdefault("max_position_embeddings", 262144)
        interval = int(c.get("full_attention_interval", 1))
        self.layers = c.get("layer_types") or [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(int(c["num_hidden_layers"]))
        ]
        if len(self.layers) != int(c["num_hidden_layers"]) or set(self.layers) - {"full_attention", "linear_attention"}:
            raise ValueError("invalid Qwen3.5 layer_types")
        self.embeddings(tied_default=self.arch == "qwen35")
        hidden = int(c["hidden_size"])
        for layer, kind in enumerate(self.layers):
            prefix = f"{self.prefix}.layers.{layer}"
            for name, target in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "post_attention_norm")):
                # AReno's policy layout already contains actual scales (HF + 1).
                self.add(f"{prefix}.{name}.weight", (hidden,), f"blk.{layer}.{target}.weight")
            if kind == "full_attention":
                self.attention(layer)
                q = f"{prefix}.self_attn.q_proj.weight"
                rule = self.rules.pop(q)[0]
                self.shapes.pop(q)
                self.add(q, (rule.shape[0] * 2, hidden), rule.name)
            else:
                self.linear_attention(layer)
            if self.arch == "qwen35moe":
                self.moe(layer, packed=True, shared=True)
            else:
                self.dense_mlp(layer)

    def linear_attention(self, layer: int) -> None:
        c = self.config
        hidden = int(c["hidden_size"])
        kh, vh = int(c.get("linear_num_key_heads", 16)), int(c.get("linear_num_value_heads", 32))
        kd, vd = int(c.get("linear_key_head_dim", 128)), int(c.get("linear_value_head_dim", 128))
        if vh % kh:
            raise ValueError("linear value-head count must be divisible by key-head count")
        k, v, kernel = kh * kd, vh * vd, int(c.get("linear_conv_kernel_dim", 4))
        entries = {
            "in_proj_qkv.weight": ((2 * k + v, hidden), "attn_qkv.weight", False),
            "in_proj_z.weight": ((v, hidden), "attn_gate.weight", False),
            "in_proj_b.weight": ((vh, hidden), "ssm_beta.weight", True),
            "in_proj_a.weight": ((vh, hidden), "ssm_alpha.weight", True),
            "conv1d.weight": ((2 * k + v, 1, kernel), "ssm_conv1d.weight", True),
            "A_log": ((vh,), "ssm_a", True),
            "dt_bias": ((vh,), "ssm_dt.bias", True),
            "norm.weight": ((vd,), "ssm_norm.weight", True),
            "out_proj.weight": ((hidden, v), "ssm_out.weight", False),
        }
        for name, (shape, target, f32) in entries.items():
            output_shape = (shape[0], kernel) if name == "conv1d.weight" else shape
            self.add(
                f"{self.prefix}.layers.{layer}.linear_attn.{name}",
                shape,
                f"blk.{layer}.{target}",
                output_shape=output_shape,
                transform=lambda value, name=name: self.linear_transform(name, value),
                force_f32=f32,
            )

    def linear_transform(self, name: str, value: np.ndarray) -> np.ndarray:
        c = self.config
        kh, vh = int(c.get("linear_num_key_heads", 16)), int(c.get("linear_num_value_heads", 32))
        kd, vd = int(c.get("linear_key_head_dim", 128)), int(c.get("linear_value_head_dim", 128))

        def reorder(tensor, axis, dim):
            shape = tensor.shape
            expanded = (*shape[:axis], kh, vh // kh, dim, *shape[axis + 1 :])
            return tensor.reshape(expanded).swapaxes(axis, axis + 1).reshape(shape)

        if name == "conv1d.weight":
            value = value[:, 0, :]
        if kh != vh:
            if name in {"in_proj_qkv.weight", "conv1d.weight"}:
                split = 2 * kh * kd
                value = np.concatenate((value[:split], reorder(value[split:], 0, vd)), axis=0)
            elif name == "in_proj_z.weight":
                value = reorder(value, 0, vd)
            elif name in {"in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias"}:
                value = reorder(value, 0, 1)
            elif name == "out_proj.weight":
                value = reorder(value, 1, vd)
        return -np.exp(value) if name == "A_log" else value

    def metadata(self, writer) -> None:
        super().metadata(writer)
        c = self.config
        dim = int(c.get("head_dim", int(c["hidden_size"]) // int(c["num_attention_heads"])))
        rotary = int(dim * float(self.rope.get("partial_rotary_factor", c.get("partial_rotary_factor", 0.25))))
        writer.add_rope_dimension_count(rotary)
        sections = list(self.rope.get("mrope_section", [rotary // 2, 0, 0]))
        if len(sections) != 3 or sum(sections) != rotary // 2:
            raise ValueError("Qwen3.5 mrope_section must match the rotary dimension")
        writer.add_rope_dimension_sections([*sections, 0])
        writer.add_array(
            f"{self.arch}.attention.recurrent_layers", [kind == "linear_attention" for kind in self.layers]
        )
        writer.add_ssm_conv_kernel(int(c.get("linear_conv_kernel_dim", 4)))
        writer.add_ssm_state_size(int(c.get("linear_key_head_dim", 128)))
        writer.add_ssm_group_count(int(c.get("linear_num_key_heads", 16)))
        writer.add_ssm_time_step_rank(int(c.get("linear_num_value_heads", 32)))
        writer.add_ssm_inner_size(int(c.get("linear_value_head_dim", 128)) * int(c.get("linear_num_value_heads", 32)))
        if self.arch == "qwen35moe":
            self.moe_metadata(writer)


@register_architecture("gemma4")
class Gemma4Export(Architecture):
    arch = "gemma4"
    tokenizer_profile = "gemma4"

    def build(self) -> None:
        import gguf

        c = self.config
        if c.get("attention_bias", False):
            raise ValueError("Gemma4 GGUF does not implement attention projection biases")
        if c.get("hidden_activation", c.get("hidden_act", "gelu_pytorch_tanh")) != "gelu_pytorch_tanh":
            raise ValueError("Gemma4 GGUF requires gelu_pytorch_tanh")
        hidden, count = int(c["hidden_size"]), int(c["num_hidden_layers"])
        self.layers = c.get("layer_types") or ["full_attention"] * count
        if len(self.layers) != count or set(self.layers) - {"full_attention", "sliding_attention"}:
            raise ValueError("invalid Gemma4 layer_types")
        self.full_dim = int(c.get("global_head_dim") or c.get("head_dim", hidden // int(c["num_attention_heads"])))
        self.swa_dim = int(c.get("swa_head_dim") or c.get("head_dim", self.full_dim))
        self.full_kv = int(
            c.get("num_global_key_value_heads") or c.get("num_key_value_heads", c["num_attention_heads"])
        )
        self.swa_kv = int(c.get("swa_num_key_value_heads") or c.get("num_key_value_heads", c["num_attention_heads"]))
        self.embeddings(tied_default=True)
        ple = int(c.get("hidden_size_per_layer_input") or 0)
        if ple:
            if int(c.get("vocab_size_per_layer_input") or c["vocab_size"]) != int(c["vocab_size"]):
                raise ValueError("Gemma4 GGUF requires PLE and text vocabularies of equal size")
            for name, shape, target in (
                ("embed_tokens_per_layer", (int(c["vocab_size"]), count * ple), "per_layer_token_embd"),
                ("per_layer_model_projection", (count * ple, hidden), "per_layer_model_proj"),
                ("per_layer_projection_norm", (ple,), "per_layer_proj_norm"),
            ):
                self.add(f"{self.prefix}.{name}.weight", shape, target + ".weight")
        name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GEMMA4, count)
        first_shared = count - int(c.get("num_kv_shared_layers") or 0)
        for layer, kind in enumerate(self.layers):
            prefix = f"{self.prefix}.layers.{layer}"
            old_dim, old_kv = c.get("head_dim"), c.get("num_key_value_heads")
            c["head_dim"] = self.swa_dim if kind == "sliding_attention" else self.full_dim
            c["num_key_value_heads"] = self.swa_kv if kind == "sliding_attention" else self.full_kv
            self.attention(layer)
            if old_dim is None:
                c.pop("head_dim")
            else:
                c["head_dim"] = old_dim
            if old_kv is None:
                c.pop("num_key_value_heads")
            else:
                c["num_key_value_heads"] = old_kv
            names = {
                "input_layernorm.weight": (hidden,),
                "post_attention_layernorm.weight": (hidden,),
                "pre_feedforward_layernorm.weight": (hidden,),
                "post_feedforward_layernorm.weight": (hidden,),
                "layer_scalar": (1,),
            }
            if ple:
                names.update(
                    {
                        "per_layer_input_gate.weight": (ple, hidden),
                        "per_layer_projection.weight": (hidden, ple),
                        "post_per_layer_input_norm.weight": (hidden,),
                    }
                )
            if c.get("enable_moe_block"):
                for name in (
                    "pre_feedforward_layernorm_2",
                    "post_feedforward_layernorm_1",
                    "post_feedforward_layernorm_2",
                ):
                    names[name + ".weight"] = (hidden,)
                experts, intermediate = int(c["num_experts"]), int(c["moe_intermediate_size"])
                names.update(
                    {
                        "router.scale": (hidden,),
                        "router.proj.weight": (experts, hidden),
                        "router.per_expert_scale": (experts,),
                    }
                )
                self.add(
                    f"{prefix}.experts.gate_up_proj",
                    (experts, 2 * intermediate, hidden),
                    f"blk.{layer}.ffn_gate_up_exps.weight",
                    packed_experts=True,
                )
                self.add(
                    f"{prefix}.experts.down_proj",
                    (experts, hidden, intermediate),
                    f"blk.{layer}.ffn_down_exps.weight",
                    packed_experts=True,
                )
            for name, shape in names.items():
                source = f"{prefix}.{name}"
                mapped = "model.layers." + source.split(".layers.", 1)[1]
                mapped += ".weight" if name == "layer_scalar" else ""
                target = name_map.get_name(mapped, try_suffixes=(".weight", ".bias"))
                if name == "router.scale":
                    target = f"blk.{layer}.ffn_gate_inp.scale"
                elif name == "router.per_expert_scale":
                    target = f"blk.{layer}.ffn_down_exps.scale"
                if target is None:
                    raise ValueError(f"missing Gemma4 GGUF mapping: {source}")
                self.add(source, shape, target, force_f32="router" in name)
            width = int(c["intermediate_size"]) * (2 if c.get("use_double_wide_mlp") and layer >= first_shared else 1)
            self.dense_mlp(layer, width)
        rope = c.get("rope_parameters") or {}
        full = rope.get("full_attention", {})
        if full.get("rope_type", "proportional") != "proportional":
            raise ValueError("Gemma4 full-attention GGUF requires proportional RoPE")
        n_rot = int(self.full_dim * float(full.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0))) / 2)
        self.extra["rope_freqs.weight"] = np.array(
            [1.0] * n_rot + [1e30] * (self.full_dim // 2 - n_rot), dtype=np.float32
        )

    def metadata(self, writer) -> None:
        super().metadata(writer)
        c = self.config
        rope = c.get("rope_parameters") or {}
        writer.add_key_length(self.full_dim)
        writer.add_value_length(self.full_dim)
        writer.add_key_length_swa(self.swa_dim)
        writer.add_value_length_swa(self.swa_dim)
        writer.add_rope_dimension_count(self.full_dim)
        writer.add_rope_dimension_count_swa(
            int(
                self.swa_dim
                * float(
                    rope.get("sliding_attention", {}).get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0))
                )
            )
        )
        writer.add_rope_freq_base(float(rope.get("full_attention", {}).get("rope_theta", c.get("rope_theta", 10000))))
        writer.add_rope_freq_base_swa(
            float(rope.get("sliding_attention", {}).get("rope_theta", c.get("rope_theta", 10000)))
        )
        writer.add_head_count_kv([self.swa_kv if kind == "sliding_attention" else self.full_kv for kind in self.layers])
        writer.add_sliding_window_pattern([kind == "sliding_attention" for kind in self.layers])
        writer.add_sliding_window(int(c.get("sliding_window") or 0))
        writer.add_shared_kv_layers(int(c.get("num_kv_shared_layers") or 0))
        writer.add_embedding_length_per_layer_input(int(c.get("hidden_size_per_layer_input") or 0))
        first_shared = len(self.layers) - int(c.get("num_kv_shared_layers") or 0)
        writer.add_feed_forward_length(
            [
                int(c["intermediate_size"]) * (2 if c.get("use_double_wide_mlp") and layer >= first_shared else 1)
                for layer in range(len(self.layers))
            ]
        )
        if c.get("enable_moe_block"):
            writer.add_expert_count(int(c["num_experts"]))
            writer.add_expert_used_count(int(c.get("top_k_experts", c.get("num_experts_per_tok", 1))))
            writer.add_expert_feed_forward_length(int(c["moe_intermediate_size"]))
        if c.get("final_logit_softcapping"):
            writer.add_final_logit_softcapping(float(c["final_logit_softcapping"]))


def build_architecture(
    config: dict[str, Any], shapes: dict[str, tuple[int, ...]], *, model_type: str | None = None
) -> Architecture:
    # Import plugins only when exporting, without loading model execution code.
    from areno.export import bailing  # noqa: F401

    kind = model_type or config.get("model_type")
    if (
        config.get("vision_config")
        or config.get("audio_config")
        or kind
        in {"qwen3_5_vl", "qwen3_5_moe_vl", "phi4mm", "phi4_multimodal", "minicpmv4_6", "minicpmv46", "gemma4_unified"}
    ):
        raise ValueError("multimodal GGUF export is not implemented; refusing to omit vision/audio weights")
    if kind not in _ARCHITECTURES:
        raise ValueError(f"unsupported GGUF model architecture: {kind!r}; supported: {sorted(_ARCHITECTURES)}")
    return _ARCHITECTURES[kind](config, shapes)
