from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from areno.api.multimodal import _image_token_id
from areno.engine.data.rollout_state import InferenceBatchState, _slice_prompt_image_features, payload_to_infer_meta
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context


@pytest.fixture(autouse=True)
def _isolate_tp_context():
    previous_context = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    try:
        yield
    finally:
        set_tp_context(previous_context)


def _config() -> dict:
    return {
        "model_type": "phi4mm",
        "vocab_size": 128,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "partial_rotary_factor": 0.5,
        "original_max_position_embeddings": 16,
        "max_position_embeddings": 32,
        "rope_scaling": {"type": "longrope", "short_factor": [1.0], "long_factor": [2.0]},
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "lm_head_bias": False,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "vision_lora": {"r": 4, "lora_alpha": 8, "dp": 0.0},
        "embd_layer": {
            "image_embd_layer": {
                "embedding_cls": "tune_image",
                "crop_size": 8,
                "image_token_compression_cls": "avg_pool_2d",
                "projection_cls": "mlp",
                "use_hd_transform": True,
                "with_learnable_separator": True,
                "hd_transform_order": "sub_glb",
            }
        },
        "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 2,
            "feature_layer": -2,
            "crop_size": 8,
        },
    }


def test_phi4mm_adapter_constructs_native_vision_path():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    config = Phi4MMAdapter().config_from_hf(_config())
    model = Phi4MMAdapter().build(config).float()

    assert config.image_token_id == 200010
    assert config.vision_config["hidden_size"] == 8
    assert model.model.embed_tokens_extend.image_embed.img_processor.encoder.layers.__len__() == 2


def test_phi4mm_hd_projection_matches_expanded_image_token_count():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
    features = {
        "input_image_embeds": torch.zeros(1, 2, 3, 8, 8),
        "image_sizes": torch.tensor([[8, 8]], dtype=torch.long),
        "image_attention_mask": torch.ones(1, 2, 4, 4, dtype=torch.bool),
        "image_token_id": 99,
    }

    image_embeds = model.model._project_image_feature(features, torch.device("cpu"))

    assert image_embeds.shape == (13, 16)


def test_phi4mm_replaces_only_expanded_image_slots():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
    input_ids = torch.tensor([[1, *([99] * 13), 2]], dtype=torch.long)
    hidden = torch.randn(1, input_ids.shape[1], 16)
    features = {
        "input_image_embeds": torch.zeros(1, 2, 3, 8, 8),
        "image_sizes": torch.tensor([[8, 8]], dtype=torch.long),
        "image_attention_mask": torch.ones(1, 2, 4, 4, dtype=torch.bool),
        "image_token_id": 99,
    }

    replaced = model.model._apply_multimodal_features(hidden, input_ids, features)

    assert torch.equal(replaced[:, :1], hidden[:, :1])
    assert torch.equal(replaced[:, -1:], hidden[:, -1:])
    assert not torch.equal(replaced[:, 1:-1], hidden[:, 1:-1])


def test_phi4mm_processor_token_fallback_uses_endoftext10():
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 200010 if token == "<|endoftext10|>" else -1)

    assert _image_token_id(tokenizer, object()) == 200010


def test_phi4mm_rollout_chunk_keeps_processor_vision_fields():
    features = {
        "input_image_embeds": torch.zeros(1, 2, 3, 8, 8),
        "image_sizes": torch.tensor([[8, 8]], dtype=torch.long),
        "image_attention_mask": torch.ones(1, 2, 4, 4, dtype=torch.bool),
        "image_token_id": 99,
    }

    mask, payload = _slice_prompt_image_features(features, [1, 99, 99, 2], 0, 4)

    assert mask == [False, True, True, False]
    assert payload is not None
    assert payload["input_image_embeds"] is features["input_image_embeds"]
    assert payload["image_sizes"] is features["image_sizes"]
    assert payload["image_attention_mask"] is features["image_attention_mask"]
    assert payload["image_token_count"] == 2


def test_phi4mm_chunked_prefill_keeps_vision_lora_active_after_image_chunk():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    features = {
        "input_image_embeds": torch.zeros(1, 2, 3, 8, 8),
        "image_sizes": torch.tensor([[8, 8]], dtype=torch.long),
        "image_attention_mask": torch.ones(1, 2, 4, 4, dtype=torch.bool),
        "image_token_id": 99,
    }
    state = InferenceBatchState(
        [[99, 99, 1, 2]],
        max_new_tokens=1,
        max_prefill_tokens=2,
        max_cache_len=8,
        kv_block_size=2,
        num_cache_blocks=4,
        prompt_features=[features],
    )
    first = state.build_prefill_payload()
    second = state.build_prefill_payload()

    assert first["features"]["image_sequence_mask"].tolist() == [True]
    assert second["input_ids"].tolist() == [1, 2]
    assert second["features"]["image_sequence_mask"].tolist() == [True]

    model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
    model.model.vision_lora_slots = torch.zeros(1, dtype=torch.bool)
    input_ids = second["input_ids"].unsqueeze(0)
    infer_meta = payload_to_infer_meta(second, torch.device("cpu"))
    mask = model.model._vision_lora_mask(input_ids, second["features"], None, infer_meta)

    assert mask.tolist() == [[True, True]]
    assert model.model.vision_lora_slots.tolist() == [True]


def test_phi4mm_projects_multiple_images_with_different_crop_counts():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
    features = {
        "input_image_embeds": torch.zeros(2, 3, 3, 8, 8),
        "image_sizes": torch.tensor([[8, 8], [16, 8]], dtype=torch.long),
        "image_attention_mask": torch.ones(2, 3, 4, 4, dtype=torch.bool),
    }

    projected = model.model._project_image_feature(features, torch.device("cpu"))

    assert projected.shape == (32, 16)


def test_phi4mm_multiple_image_features_follow_placeholder_order():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
    first = torch.full((2, 16), 1.0)
    second = torch.full((3, 16), 2.0)
    input_ids = torch.tensor([[7, 99, 99, 8, 99, 99, 99, 9]])
    hidden = torch.randn(1, input_ids.shape[1], 16)
    features = {
        "image_feature_rows": [
            {"image_embeds": first, "image_token_count": 2},
            {"image_embeds": second, "image_token_count": 3},
        ],
        "image_token_id": 99,
    }

    merged = model.model._apply_multimodal_features(hidden, input_ids, features)

    torch.testing.assert_close(merged[0, 1:3], first)
    torch.testing.assert_close(merged[0, 4:7], second)
    torch.testing.assert_close(merged[0, [0, 3, 7]], hidden[0, [0, 3, 7]])


def test_phi4mm_mixed_batch_keeps_image_features_and_lora_row_local():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
    image_token = model.config.image_token_id
    input_ids = torch.tensor([[image_token, 1, 2], [3, 4, 5]])
    hidden = torch.randn(2, 3, 16)
    image_embeds = torch.full((1, 16), 4.0)
    features = [{"image_embeds": image_embeds, "image_token_id": image_token}, None]

    merged = model.model._apply_multimodal_features(hidden, input_ids, features)
    lora_mask = model.model._vision_lora_mask(input_ids, features, None, None)

    torch.testing.assert_close(merged[0, 0], image_embeds[0])
    torch.testing.assert_close(merged[0, 1:], hidden[0, 1:])
    torch.testing.assert_close(merged[1], hidden[1])
    assert lora_mask.tolist() == [[True, True, True], [False, False, False]]


def test_phi4mm_packed_batch_maps_vision_modes_to_recurrent_slots():
    pytest.importorskip("triton")
    from areno.engine.runtime.metadata import InferMeta
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
    model.model.vision_lora_slots = torch.zeros(2, dtype=torch.bool)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    features = {"image_sequence_mask": torch.tensor([True, False])}
    infer_meta = InferMeta(
        mode="prefill",
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
        recurrent_slots=torch.tensor([1, 0]),
    )

    mask = model.model._vision_lora_mask(input_ids, features, None, infer_meta)

    assert mask.tolist() == [[True, True, False, False]]
    assert model.model.vision_lora_slots.tolist() == [False, True]


def _lora_weights() -> dict[str, torch.Tensor]:
    prefix = "model.layers.0"
    return {
        f"{prefix}.self_attn.qkv_proj.lora_A.vision.weight": torch.arange(4 * 16).view(4, 16).float(),
        f"{prefix}.self_attn.qkv_proj.lora_B.vision.weight": torch.arange(48 * 4).view(48, 4).float(),
        f"{prefix}.self_attn.o_proj.lora_A.vision.weight": torch.arange(4 * 16).view(4, 16).float() + 1_000,
        f"{prefix}.self_attn.o_proj.lora_B.vision.weight": torch.arange(16 * 4).view(16, 4).float() + 2_000,
        f"{prefix}.mlp.gate_up_proj.lora_A.vision.weight": torch.arange(4 * 16).view(4, 16).float() + 3_000,
        f"{prefix}.mlp.gate_up_proj.lora_B.vision.weight": torch.arange(64 * 4).view(64, 4).float() + 4_000,
        f"{prefix}.mlp.down_proj.lora_A.vision.weight": torch.arange(4 * 32).view(4, 32).float() + 5_000,
        f"{prefix}.mlp.down_proj.lora_B.vision.weight": torch.arange(16 * 4).view(16, 4).float() + 6_000,
    }


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_phi4mm_vision_lora_tp_mapping_shards_each_fused_section(tmp_path, tp_size):
    pytest.importorskip("triton")
    from areno.models.phi4mm.checkpoint import _load_vision_lora_weights
    from areno.models.phi4mm.model import Phi4MMAdapter

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tensors = _lora_weights()
    save_file(tensors, checkpoint / "model.safetensors")
    previous = get_tp_context()
    try:
        for rank in range(tp_size):
            set_tp_context(TPContext(rank=rank, world_size=tp_size, device=torch.device("cpu"), group=None))
            model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
            _load_vision_lora_weights(model, checkpoint)
            layer = model.layers[0]

            q, k, v = tensors["model.layers.0.self_attn.qkv_proj.lora_B.vision.weight"].split((16, 16, 16))
            expected_qkv_b = torch.cat((q.chunk(tp_size)[rank], k.chunk(tp_size)[rank], v.chunk(tp_size)[rank]))
            gate, up = tensors["model.layers.0.mlp.gate_up_proj.lora_B.vision.weight"].split((32, 32))
            expected_gate_b = torch.cat((gate.chunk(tp_size)[rank], up.chunk(tp_size)[rank]))

            torch.testing.assert_close(layer.self_attn.qkv_proj.lora_B["vision"].weight, expected_qkv_b)
            torch.testing.assert_close(layer.mlp.gate_up_proj.lora_B["vision"].weight, expected_gate_b)
            torch.testing.assert_close(
                layer.self_attn.o_proj.lora_A["vision"].weight,
                tensors["model.layers.0.self_attn.o_proj.lora_A.vision.weight"].chunk(tp_size, dim=1)[rank],
            )
            torch.testing.assert_close(
                layer.mlp.down_proj.lora_A["vision"].weight,
                tensors["model.layers.0.mlp.down_proj.lora_A.vision.weight"].chunk(tp_size, dim=1)[rank],
            )
            torch.testing.assert_close(
                layer.self_attn.qkv_proj.lora_A["vision"].weight,
                tensors["model.layers.0.self_attn.qkv_proj.lora_A.vision.weight"],
            )
            torch.testing.assert_close(
                layer.self_attn.o_proj.lora_B["vision"].weight,
                tensors["model.layers.0.self_attn.o_proj.lora_B.vision.weight"],
            )
    finally:
        set_tp_context(previous)


def test_phi4mm_vision_lora_tp1_forward_matches_peft_formula():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
    projection = model.layers[0].self_attn.qkv_proj
    projection.weight.data.zero_()
    projection.lora_A["vision"].weight.data.copy_(torch.arange(4 * 16).view(4, 16).float() / 100)
    projection.lora_B["vision"].weight.data.copy_(torch.arange(48 * 4).view(48, 4).float() / 100)
    projection.vision_lora_mask = torch.tensor([[True, False, True]])
    inputs = torch.arange(3 * 16).view(1, 3, 16).float() / 100

    actual = projection(inputs)
    expected = 2.0 * F.linear(F.linear(inputs, projection.lora_A["vision"].weight), projection.lora_B["vision"].weight)
    expected[:, 1].zero_()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_phi4mm_vision_lora_tp_shards_reconstruct_peft_formula(tmp_path, tp_size):
    pytest.importorskip("triton")
    from areno.models.phi4mm.checkpoint import _load_vision_lora_weights
    from areno.models.phi4mm.model import Phi4MMAdapter

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tensors = {name: tensor / 10_000 for name, tensor in _lora_weights().items()}
    save_file(tensors, checkpoint / "model.safetensors")
    inputs = torch.arange(3 * 16).view(3, 16).float() / 100
    down_inputs = torch.arange(3 * 32).view(3, 32).float() / 100
    qkv_parts: list[tuple[torch.Tensor, ...]] = []
    gate_parts: list[tuple[torch.Tensor, ...]] = []
    o_latents = []
    down_latents = []
    previous = get_tp_context()
    try:
        for rank in range(tp_size):
            set_tp_context(TPContext(rank=rank, world_size=tp_size, device=torch.device("cpu"), group=None))
            model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
            _load_vision_lora_weights(model, checkpoint)
            layer = model.layers[0]
            qkv_delta = F.linear(
                F.linear(inputs, layer.self_attn.qkv_proj.lora_A["vision"].weight),
                layer.self_attn.qkv_proj.lora_B["vision"].weight,
            )
            gate_delta = F.linear(
                F.linear(inputs, layer.mlp.gate_up_proj.lora_A["vision"].weight),
                layer.mlp.gate_up_proj.lora_B["vision"].weight,
            )
            qkv_parts.append(qkv_delta.split((16 // tp_size,) * 3, dim=-1))
            gate_parts.append(gate_delta.split((32 // tp_size,) * 2, dim=-1))
            o_latents.append(
                F.linear(inputs.chunk(tp_size, dim=-1)[rank], layer.self_attn.o_proj.lora_A["vision"].weight)
            )
            down_latents.append(
                F.linear(down_inputs.chunk(tp_size, dim=-1)[rank], layer.mlp.down_proj.lora_A["vision"].weight)
            )
        qkv_actual = torch.cat(
            [torch.cat([parts[section] for parts in qkv_parts], dim=-1) for section in range(3)], dim=-1
        )
        gate_actual = torch.cat(
            [torch.cat([parts[section] for parts in gate_parts], dim=-1) for section in range(2)], dim=-1
        )
        o_actual = F.linear(sum(o_latents), layer.self_attn.o_proj.lora_B["vision"].weight)
        down_actual = F.linear(sum(down_latents), layer.mlp.down_proj.lora_B["vision"].weight)
    finally:
        set_tp_context(previous)

    prefix = "model.layers.0"
    qkv_expected = F.linear(
        F.linear(inputs, tensors[f"{prefix}.self_attn.qkv_proj.lora_A.vision.weight"]),
        tensors[f"{prefix}.self_attn.qkv_proj.lora_B.vision.weight"],
    )
    gate_expected = F.linear(
        F.linear(inputs, tensors[f"{prefix}.mlp.gate_up_proj.lora_A.vision.weight"]),
        tensors[f"{prefix}.mlp.gate_up_proj.lora_B.vision.weight"],
    )
    o_expected = F.linear(
        F.linear(inputs, tensors[f"{prefix}.self_attn.o_proj.lora_A.vision.weight"]),
        tensors[f"{prefix}.self_attn.o_proj.lora_B.vision.weight"],
    )
    down_expected = F.linear(
        F.linear(down_inputs, tensors[f"{prefix}.mlp.down_proj.lora_A.vision.weight"]),
        tensors[f"{prefix}.mlp.down_proj.lora_B.vision.weight"],
    )
    torch.testing.assert_close(qkv_actual, qkv_expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gate_actual, gate_expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(o_actual, o_expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(down_actual, down_expected, rtol=1e-5, atol=1e-5)


def test_phi4mm_vision_checkpoint_save_reload_closes(tmp_path, monkeypatch):
    pytest.importorskip("triton")
    from areno.engine.checkpoints.io import SafetensorsIndex
    from areno.models.phi4mm.checkpoint import (
        _vision_checkpoint_keys,
        _vision_lora_checkpoint_keys,
        audit_phi4mm_checkpoint,
    )
    from areno.models.phi4mm.model import Phi4MMAdapter

    monkeypatch.setenv("ARENO_CKPT_PROGRESS", "0")
    torch.manual_seed(7)
    adapter = Phi4MMAdapter()
    config = adapter.config_from_hf(_config())
    first = adapter.build(config).float()
    output = tmp_path / "output"

    saved_path = adapter.save_weights(first, output, None)
    second = adapter.build(config).float()
    adapter.load_weights(second, output)

    assert saved_path == str(output)
    assert second.lm_head.weight is second.model.embed_tokens.weight
    for (first_name, first_parameter), (second_name, second_parameter) in zip(
        first.named_parameters(), second.named_parameters(), strict=True
    ):
        assert first_name == second_name
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)

    vision_keys = _vision_checkpoint_keys(first)
    vision_lora_keys = _vision_lora_checkpoint_keys(first)
    audit = audit_phi4mm_checkpoint(output, len(first.layers), vision_keys, vision_lora_keys)
    assert audit.consumed == audit.total
    assert audit.speech_lora_skipped == audit.audio_skipped == audit.unknown == 0
    assert "model.embed_tokens_extend.image_embed.sub_GN" in vision_keys
    assert "model.embed_tokens_extend.image_embed.glb_GN" in vision_keys
    assert len(vision_lora_keys) == 8

    index = SafetensorsIndex(output, progress=False)
    try:
        saved_keys = set(index.weight_map)
    finally:
        index.close()
    assert vision_keys | vision_lora_keys <= saved_keys
    assert not any(".speech." in key or ".audio_embed." in key for key in saved_keys)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_phi4mm_vision_lora_save_layout_inverts_tp_sharding(tmp_path, tp_size):
    pytest.importorskip("triton")
    from areno.engine.checkpoints.io import PolicyTensorStore, policy_plan_scope
    from areno.models.phi4mm.checkpoint import _load_vision_lora_weights, _save_vision_weights
    from areno.models.phi4mm.model import Phi4MMAdapter

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    expected = _lora_weights()
    save_file(expected, checkpoint / "model.safetensors")
    sharded_keys = (
        "model.layers.0.self_attn.qkv_proj.lora_B.vision.weight",
        "model.layers.0.mlp.gate_up_proj.lora_B.vision.weight",
        "model.layers.0.self_attn.o_proj.lora_A.vision.weight",
        "model.layers.0.mlp.down_proj.lora_A.vision.weight",
    )
    contributions = {key: [] for key in sharded_keys}
    previous = get_tp_context()
    try:
        for rank in range(tp_size):
            set_tp_context(TPContext(rank=rank, world_size=tp_size, device=torch.device("cpu"), group=None))
            model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
            _load_vision_lora_weights(model, checkpoint)
            store = PolicyTensorStore()
            with policy_plan_scope():
                _save_vision_weights(store, model)
            for key in sharded_keys:
                layout = store[key].policy_layout()
                contribution = torch.empty(layout.numel, dtype=layout.dtype)
                layout.read_chunk(0, contribution)
                contributions[key].append(contribution)
    finally:
        set_tp_context(previous)

    for key in sharded_keys:
        reconstructed = torch.stack(contributions[key]).sum(dim=0).reshape_as(expected[key])
        torch.testing.assert_close(reconstructed, expected[key], rtol=0, atol=0)
