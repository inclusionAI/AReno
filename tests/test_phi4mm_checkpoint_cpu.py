from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from areno.engine.checkpoints.common import load_packed_section_column_spec, save_packed_section_column_spec
from areno.engine.checkpoints.io import PolicyTensorStore, SafetensorsIndex
from areno.engine.config import ModelConfig
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.models.phi4mm import Phi4MMAdapter
from areno.models.phi4mm.checkpoint import QKV_SPEC, audit_phi4mm_checkpoint


@pytest.fixture(autouse=True)
def _isolate_tp_context():
    previous_context = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    try:
        yield
    finally:
        set_tp_context(previous_context)


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        model_type="phi4mm",
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        qkv_bias=False,
        qk_norm=False,
        dtype=torch.float32,
        hidden_act="silu",
        partial_rotary_factor=0.75,
        sequence_parallel=False,
        attn_backend="native",
        hf_text_config={
            "original_max_position_embeddings": 32,
            "rope_scaling": {
                "type": "longrope",
                "short_factor": (1.0, 1.0, 1.0),
                "long_factor": (1.0, 2.0, 3.0),
            },
        },
    )


def _row_values(rows: int, columns: int, base: float) -> torch.Tensor:
    return (base + torch.arange(rows, dtype=torch.float32)).unsqueeze(1).expand(rows, columns).clone()


def _column_values(rows: int, columns: int, base: float) -> torch.Tensor:
    return (base + torch.arange(columns, dtype=torch.float32)).unsqueeze(0).expand(rows, columns).clone()


def _synthetic_weights(*, skipped: bool = False) -> dict[str, torch.Tensor]:
    config = _tiny_config()
    tensors = {
        "model.embed_tokens.weight": torch.arange(config.vocab_size * config.hidden_size, dtype=torch.float32).view(
            config.vocab_size, config.hidden_size
        ),
        "model.norm.weight": torch.arange(config.hidden_size, dtype=torch.float32) + 10,
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        offset = layer * 10_000
        q = _row_values(64, 64, 1_000 + offset)
        k = _row_values(32, 64, 2_000 + offset)
        v = _row_values(32, 64, 3_000 + offset)
        gate = _row_values(128, 64, 4_000 + offset)
        up = _row_values(128, 64, 5_000 + offset)
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": torch.arange(64, dtype=torch.float32) + 20 + offset,
                f"{prefix}.post_attention_layernorm.weight": torch.arange(64, dtype=torch.float32) + 30 + offset,
                f"{prefix}.self_attn.qkv_proj.base_layer.weight": torch.cat((q, k, v)),
                f"{prefix}.self_attn.o_proj.base_layer.weight": _column_values(64, 64, 6_000 + offset),
                f"{prefix}.mlp.gate_up_proj.base_layer.weight": torch.cat((gate, up)),
                f"{prefix}.mlp.down_proj.base_layer.weight": _column_values(64, 128, 7_000 + offset),
            }
        )
    if skipped:
        tensors.update(
            {
                "model.layers.0.self_attn.qkv_proj.lora_A.vision.weight": torch.ones(1),
                "model.layers.0.self_attn.qkv_proj.lora_B.speech.weight": torch.ones(1),
                "model.embed_tokens_extend.image_embed.img_projection.weight": torch.ones(1),
                "model.embed_tokens_extend.audio_embed.audio_projection.weight": torch.ones(1),
            }
        )
    return tensors


def _write_checkpoint(path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir()
    save_file(tensors, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps({"model_type": "phi4mm"}), encoding="utf-8")


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_phi4mm_checkpoint_loads_each_packed_section_independently(tmp_path, monkeypatch, tp_size):
    monkeypatch.setenv("ARENO_CKPT_PROGRESS", "0")
    tensors = _synthetic_weights()
    checkpoint = tmp_path / "source"
    _write_checkpoint(checkpoint, tensors)
    old_context = get_tp_context()
    try:
        for rank in range(tp_size):
            set_tp_context(TPContext(rank=rank, world_size=tp_size, device=torch.device("cpu"), group=None))
            model = Phi4MMAdapter().build(_tiny_config())
            Phi4MMAdapter().load_weights(model, checkpoint)

            layer = model.model.layers[0]
            q, k, v = tensors["model.layers.0.self_attn.qkv_proj.base_layer.weight"].split((64, 32, 32))
            gate, up = tensors["model.layers.0.mlp.gate_up_proj.base_layer.weight"].split((128, 128))
            expected_qkv = torch.cat((q.chunk(tp_size)[rank], k.chunk(tp_size)[rank], v.chunk(tp_size)[rank]))
            expected_gate_up = torch.cat((gate.chunk(tp_size)[rank], up.chunk(tp_size)[rank]))

            torch.testing.assert_close(layer.self_attn.qkv_proj.weight, expected_qkv)
            torch.testing.assert_close(layer.mlp.gate_up_proj.weight, expected_gate_up)
            torch.testing.assert_close(
                layer.self_attn.o_proj.weight,
                tensors["model.layers.0.self_attn.o_proj.base_layer.weight"].chunk(tp_size, dim=1)[rank],
            )
            torch.testing.assert_close(
                layer.mlp.down_proj.weight,
                tensors["model.layers.0.mlp.down_proj.base_layer.weight"].chunk(tp_size, dim=1)[rank],
            )
            torch.testing.assert_close(
                model.model.embed_tokens.weight,
                tensors["model.embed_tokens.weight"].chunk(tp_size)[rank],
            )
            torch.testing.assert_close(layer.input_layernorm.weight, tensors["model.layers.0.input_layernorm.weight"])
            torch.testing.assert_close(
                layer.post_attention_layernorm.weight,
                tensors["model.layers.0.post_attention_layernorm.weight"],
            )
            torch.testing.assert_close(model.model.norm.weight, tensors["model.norm.weight"])
            assert model.lm_head.weight is model.model.embed_tokens.weight
    finally:
        set_tp_context(old_context)


def test_phi4mm_checkpoint_audit_accepts_only_documented_skips(tmp_path):
    checkpoint = tmp_path / "source"
    _write_checkpoint(checkpoint, _synthetic_weights(skipped=True))

    audit = audit_phi4mm_checkpoint(checkpoint, num_hidden_layers=2)

    assert audit.total == 18
    assert audit.consumed == 14
    assert audit.vision_lora_skipped == 1
    assert audit.speech_lora_skipped == 1
    assert audit.vision_skipped == 1
    assert audit.audio_skipped == 1
    assert audit.unknown == 0


def test_phi4mm_checkpoint_audit_rejects_unknown_base_key(tmp_path):
    tensors = _synthetic_weights()
    tensors["model.layers.0.self_attn.foo.weight"] = torch.ones(1)
    checkpoint = tmp_path / "source"
    _write_checkpoint(checkpoint, tensors)

    with pytest.raises(ValueError, match="unknown tensors.*self_attn.foo.weight"):
        audit_phi4mm_checkpoint(checkpoint, num_hidden_layers=2)


def test_phi4mm_checkpoint_audit_rejects_missing_required_key(tmp_path):
    tensors = _synthetic_weights()
    del tensors["model.layers.0.self_attn.qkv_proj.base_layer.weight"]
    checkpoint = tmp_path / "source"
    _write_checkpoint(checkpoint, tensors)

    with pytest.raises(ValueError, match="missing 1 required.*qkv_proj.base_layer.weight"):
        audit_phi4mm_checkpoint(checkpoint, num_hidden_layers=2)


def test_phi4mm_checkpoint_rejects_wrong_packed_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENO_CKPT_PROGRESS", "0")
    tensors = _synthetic_weights()
    tensors["model.layers.0.self_attn.qkv_proj.base_layer.weight"] = torch.zeros(127, 64)
    checkpoint = tmp_path / "source"
    _write_checkpoint(checkpoint, tensors)

    with pytest.raises(ValueError, match=r"shape \(127, 64\), expected \(128, 64\)"):
        Phi4MMAdapter().load_weights(Phi4MMAdapter().build(_tiny_config()), checkpoint)


def test_packed_section_loader_rejects_non_divisible_sections(tmp_path):
    checkpoint = tmp_path / "source"
    _write_checkpoint(checkpoint, _synthetic_weights())
    model = Phi4MMAdapter().build(_tiny_config())
    index = SafetensorsIndex(checkpoint, progress=False)
    try:
        with pytest.raises(ValueError, match="cannot shard size 64 across 3 ranks"):
            load_packed_section_column_spec(model.model.layers[0], index, "model.layers.0", QKV_SPEC, 0, 3)
    finally:
        index.close()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_packed_section_save_layout_is_inverse_of_section_sharding(tp_size):
    tensors = _synthetic_weights()
    full = tensors["model.layers.0.self_attn.qkv_proj.base_layer.weight"]
    q, k, v = full.split((64, 32, 32))
    old_context = get_tp_context()
    contributions = []
    try:
        for rank in range(tp_size):
            set_tp_context(TPContext(rank=rank, world_size=tp_size, device=torch.device("cpu"), group=None))
            layer = Phi4MMAdapter().build(_tiny_config()).model.layers[0]
            layer.self_attn.qkv_proj.weight.data.copy_(
                torch.cat((q.chunk(tp_size)[rank], k.chunk(tp_size)[rank], v.chunk(tp_size)[rank]))
            )
            store = PolicyTensorStore()
            save_packed_section_column_spec(store, layer, "model.layers.0", QKV_SPEC)
            layout = store["model.layers.0.self_attn.qkv_proj.base_layer.weight"].policy_layout()
            contribution = torch.empty(layout.numel, dtype=layout.dtype)
            layout.read_chunk(0, contribution)
            contributions.append(contribution)
    finally:
        set_tp_context(old_context)

    reconstructed = torch.stack(contributions).sum(dim=0).reshape_as(full)
    torch.testing.assert_close(reconstructed, full, rtol=0, atol=0)


def test_phi4mm_text_only_checkpoint_load_save_reload_closes(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENO_CKPT_PROGRESS", "0")
    source = tmp_path / "source"
    output = tmp_path / "output"
    tensors = _synthetic_weights(skipped=True)
    _write_checkpoint(source, tensors)
    first = Phi4MMAdapter().build(_tiny_config())
    Phi4MMAdapter().load_weights(first, source)

    saved_path = Phi4MMAdapter().save_weights(first, output, source)
    second = Phi4MMAdapter().build(_tiny_config())
    Phi4MMAdapter().load_weights(second, output)

    assert saved_path == str(output)
    assert (output / "config.json").exists()
    assert second.lm_head.weight is second.model.embed_tokens.weight
    for (first_name, first_parameter), (second_name, second_parameter) in zip(
        first.named_parameters(), second.named_parameters(), strict=True
    ):
        assert first_name == second_name
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)
    audit = audit_phi4mm_checkpoint(output, num_hidden_layers=2)
    assert audit.total == audit.consumed == 14
    with open(output / "model.safetensors.index.json", encoding="utf-8") as handle:
        saved_keys = set(json.load(handle)["weight_map"])
    assert not any("embed_tokens_extend" in key or ".lora_" in key for key in saved_keys)
    with safe_open(output / "model-rank00000-00002-layer-00000.safetensors", framework="pt") as handle:
        torch.testing.assert_close(
            handle.get_tensor("model.layers.0.self_attn.qkv_proj.base_layer.weight"),
            tensors["model.layers.0.self_attn.qkv_proj.base_layer.weight"],
        )
        torch.testing.assert_close(
            handle.get_tensor("model.layers.0.mlp.gate_up_proj.base_layer.weight"),
            tensors["model.layers.0.mlp.gate_up_proj.base_layer.weight"],
        )
