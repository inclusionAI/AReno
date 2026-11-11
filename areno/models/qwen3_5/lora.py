"""Canonical Qwen3.5 dense language targets over native fused projections."""

from areno.adapters.lora import _install_merged_slot, _install_row_slot
from areno.models.qwen3_5.model import Qwen35FullAttention


def initialize_qwen35_lora(model, requested, config, seed, runtime_state, slots):
    if getattr(model, "_train_multimodal_tower", False) or getattr(model, "_train_multimodal_projector", False):
        raise ValueError("Qwen3.5 native LoRA requires a frozen vision tower and projector")
    language = getattr(model, "language_model", model)
    prefix = "language_model." if language is not model else ""
    matched = set()
    for index, layer in enumerate(language.layers):
        layer_prefix = f"{prefix}layers.{index}"
        attention = layer.attention
        if isinstance(attention, Qwen35FullAttention):
            attention_prefix = f"{layer_prefix}.self_attn"
            projections = (
                ("q_proj", attention.qkv_proj, 0),
                ("k_proj", attention.qkv_proj, 1),
                ("v_proj", attention.qkv_proj, 2),
            )
            output_name, output = "o_proj", attention.o_proj
        else:
            attention_prefix = f"{layer_prefix}.linear_attn"
            projections = (
                ("in_proj_qkv", attention.in_proj_qkvz, (0, 1, 2)),
                ("in_proj_z", attention.in_proj_qkvz, 3),
                ("in_proj_b", attention.in_proj_ba, 0),
                ("in_proj_a", attention.in_proj_ba, 1),
            )
            output_name, output = "out_proj", attention.out_proj
        for name, owner, components in projections:
            if name in requested:
                matched.add(name)
                _install_merged_slot(
                    f"{attention_prefix}.{name}", owner, components, config, seed, runtime_state, slots
                )
        if output_name in requested:
            matched.add(output_name)
            _install_row_slot(f"{attention_prefix}.{output_name}", output, config, seed, runtime_state, slots)
        for component, name in enumerate(("gate_proj", "up_proj")):
            if name in requested:
                matched.add(name)
                _install_merged_slot(
                    f"{layer_prefix}.mlp.{name}", layer.mlp.gate_up_proj, component, config, seed, runtime_state, slots
                )
        if "down_proj" in requested:
            matched.add("down_proj")
            _install_row_slot(f"{layer_prefix}.mlp.down_proj", layer.mlp.down_proj, config, seed, runtime_state, slots)
    return matched
