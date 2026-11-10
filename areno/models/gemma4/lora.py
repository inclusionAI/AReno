"""Canonical dense Gemma4 LoRA targets, including shared K/V projections."""

from areno.adapters.lora import _install_merged_slot, _install_row_slot


def initialize_gemma4_lora(model, requested, config, seed, runtime_state, slots):
    if model.config.enable_moe_block:
        raise ValueError("Gemma4 native LoRA currently supports dense text trunks only")
    if getattr(model, "_train_multimodal_tower", False) or getattr(model, "_train_multimodal_projector", False):
        raise ValueError("Gemma4 native LoRA requires frozen multimodal towers and projectors")
    # PEFT prepends base_model.model.model; multimodal checkpoints place the
    # text decoder below model.language_model rather than model directly.
    root = "language_model." if model.config.checkpoint_prefix == "model.language_model" else ""
    matched = set()
    for index, layer in enumerate(model.layers):
        prefix = f"{root}layers.{index}"
        attention = layer.self_attn
        qkv = attention.qkv_proj
        shared = attention.attention_k_eq_v and attention.layer_type == "full_attention"
        for component, name in enumerate(("q_proj", "k_proj", "v_proj")):
            # KV-shared tail layers do not own operative K/V projections.
            if component and attention.is_kv_shared_layer:
                continue
            # A full-attention K=V layer has no v_proj in its HF checkpoint.
            # Its single k_proj adapter must contribute to both fused outputs.
            if name == "v_proj" and shared:
                continue
            if name in requested:
                matched.add(name)
                logical = f"{prefix}.self_attn.{name}"
                _install_merged_slot(logical, qkv, component, config, seed, runtime_state, slots)
                if name == "k_proj" and shared:
                    qkv.install_lora_component("shared_v_proj", 2, slots[logical])
        if "o_proj" in requested:
            matched.add("o_proj")
            _install_row_slot(f"{prefix}.self_attn.o_proj", attention.o_proj, config, seed, runtime_state, slots)
        for component, name in enumerate(("gate_proj", "up_proj")):
            if name in requested:
                matched.add(name)
                _install_merged_slot(
                    f"{prefix}.mlp.{name}", layer.mlp.gate_up_proj, component, config, seed, runtime_state, slots
                )
        if "down_proj" in requested:
            matched.add("down_proj")
            _install_row_slot(f"{prefix}.mlp.down_proj", layer.mlp.down_proj, config, seed, runtime_state, slots)
    return matched
