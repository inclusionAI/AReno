from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from areno.api.multimodal import _image_token_id, _processor_chat_text
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


def test_phi4mm_vision_attention_uses_memory_efficient_sdpa(monkeypatch):
    from areno.models.phi4mm import vision

    calls = []
    original = F.scaled_dot_product_attention

    def tracked_sdpa(query, key, value, **kwargs):
        calls.append(kwargs)
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(vision.F, "scaled_dot_product_attention", tracked_sdpa)
    attention = vision.Phi4MMVisionAttention(
        vision.Phi4MMVisionConfig(hidden_size=8, num_attention_heads=2),
        torch.float32,
    )
    mask = torch.tensor([[True, True, False]])

    output = attention(torch.randn(1, 3, 8), mask)

    assert output.shape == (1, 3, 8)
    assert len(calls) == 1
    assert torch.equal(calls[0]["attn_mask"], mask[:, None, None, :])


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


def test_phi4mm_processor_token_fallback_rejects_unknown_token_id():
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: 200010 if token == "<|endoftext10|>" else 199999,
        get_vocab=lambda: {"<|endoftext|>": 199999, "<|endoftext10|>": 200010},
    )

    assert _image_token_id(tokenizer, object()) == 200010


def test_phi4mm_processor_renders_ordered_images_with_string_chat_template():
    class Phi4MMProcessor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "".join("<|" + item["role"] + "|>" + item["content"] + "<|end|>" for item in messages)

    messages = [
        {"role": "system", "content": "Describe images."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "First?"}]},
        {"role": "assistant", "content": "A red square."},
        {"role": "user", "content": [{"type": "text", "text": "Compare: "}, {"type": "image"}]},
    ]

    rendered = _processor_chat_text(Phi4MMProcessor(), messages)

    assert rendered == (
        "<|system|>Describe images.<|end|>"
        "<|user|><|image_1|>\nFirst?<|end|>"
        "<|assistant|>A red square.<|end|>"
        "<|user|>Compare: <|image_2|>\n<|end|>"
    )
    assert isinstance(messages[1]["content"], list)


def test_phi4mm_serve_encoder_preserves_expanded_slots_and_resolves_real_image_token(tmp_path):
    from PIL import Image

    from areno.cli.serve import ChatMessage, _encode_messages_with_features

    image_path = tmp_path / "red.png"
    Image.new("RGB", (8, 8), "red").save(image_path)

    class Phi4MMProcessor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"] == "<|image_1|>\nDescribe."
            return "<|user|>" + messages[0]["content"] + "<|end|><|assistant|>"

        def __call__(self, *, text, images, return_tensors):
            assert text == ["<|user|><|image_1|>\nDescribe.<|end|><|assistant|>"]
            assert len(images) == 1
            return {
                "input_ids": torch.tensor([[1, 200010, 200010, 2]]),
                "input_image_embeds": torch.ones(1, 2, 3, 8, 8),
                "image_sizes": torch.tensor([[8, 8]]),
                "image_attention_mask": torch.ones(1, 2, 4, 4),
            }

    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"<|endoftext|>": 199999, "<|endoftext10|>": 200010},
        convert_tokens_to_ids=lambda token: 200010 if token == "<|endoftext10|>" else 199999,
    )
    messages = [
        ChatMessage(
            role="user",
            content=[
                {"type": "image_url", "image_url": {"url": str(image_path)}},
                {"type": "text", "text": "Describe."},
            ],
        )
    ]

    tokens, features = _encode_messages_with_features(tokenizer, Phi4MMProcessor(), messages)

    assert tokens == [1, 200010, 200010, 2]
    assert features["image_token_id"] == 200010
    assert features["input_image_embeds"].shape == (1, 2, 3, 8, 8)


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

    assert payload_to_infer_meta(first, torch.device("cpu")).cache_seqlens is None
    assert payload_to_infer_meta(second, torch.device("cpu")).cache_seqlens.tolist() == [2]
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


def test_phi4mm_cache_reprefill_restores_vision_features_and_lora_mode():
    features = {
        "input_image_embeds": torch.zeros(1, 2, 3, 8, 8),
        "image_sizes": torch.tensor([[8, 8]], dtype=torch.long),
        "image_attention_mask": torch.ones(1, 2, 4, 4, dtype=torch.bool),
        "image_token_id": 99,
    }
    state = InferenceBatchState(
        [[99, 1]],
        max_new_tokens=2,
        max_prefill_tokens=4,
        max_cache_len=4,
        kv_block_size=2,
        num_cache_blocks=2,
        prompt_features=[features],
    )
    state.build_prefill_payload()
    state.ensure_decode_blocks([0], [2])

    payload = state.build_cache_reprefill_payload(
        [0],
        generated=torch.tensor([[2]], dtype=torch.long),
        response_lens=torch.tensor([1], dtype=torch.long),
    )

    assert payload["features"]["image_sequence_mask"].tolist() == [True]
    assert payload["features"]["image_token_mask"].tolist() == [True, False, False]
    assert payload["features"]["image_feature_rows"][0]["input_image_embeds"] is features["input_image_embeds"]


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


def test_phi4mm_image_text_backward_reaches_encoder_projector_and_lora(monkeypatch):
    from areno.engine.layers import mlp, norm, vocab
    from areno.engine.runtime.metadata import TrainMeta
    from areno.models.phi4mm.model import Phi4MMAdapter

    monkeypatch.setattr(vocab, "areno_vocab_embedding", lambda ids, weight, start, end: F.embedding(ids, weight))
    monkeypatch.setattr(
        norm, "_areno_rmsnorm_no_compile", lambda x, weight, eps: F.rms_norm(x, (x.shape[-1],), weight, eps)
    )
    monkeypatch.setattr(
        mlp, "_areno_silu_and_mul_no_compile", lambda x: F.silu(x.chunk(2, dim=-1)[0]) * x.chunk(2, dim=-1)[1]
    )
    torch.manual_seed(91)
    config = Phi4MMAdapter().config_from_hf(_config())
    config.attn_backend = "native"
    config.image_token_id = 99
    model = Phi4MMAdapter().build(config).float().train()
    tokens = torch.tensor([[1, *([99] * 13), 2]])
    features = {
        "input_image_embeds": torch.randn(1, 2, 3, 8, 8),
        "image_sizes": torch.tensor([[8, 8]]),
        "image_attention_mask": torch.ones(1, 2, 4, 4, dtype=torch.bool),
        "image_token_id": 99,
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    checked = [
        model.model.embed_tokens_extend.image_embed.img_processor.embeddings.patch_embedding.weight,
        model.model.embed_tokens_extend.image_embed.img_projection[0].weight,
        model.layers[0].self_attn.qkv_proj.lora_A["vision"].weight,
        model.layers[0].mlp.down_proj.lora_B["vision"].weight,
    ]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(tokens, features=features, train_meta=TrainMeta(activation_checkpointing=True))
        loss = F.cross_entropy(output.logits_shard[:, -1], torch.tensor([3]))
        loss.backward()
        assert torch.isfinite(loss)
        for parameter in checked:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        optimizer.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phi4MM cached decode graph requires CUDA")
@torch.inference_mode()
def test_phi4mm_decode_graph_replays_current_vision_slots():
    from areno.engine.runtime.metadata import InferMeta
    from areno.models.phi4mm.model import Phi4MMAdapter

    torch.manual_seed(12)
    device = torch.device("cuda")
    config = Phi4MMAdapter().config_from_hf(_config())
    config.attn_backend = "native"
    model = Phi4MMAdapter().build(config).to(device).eval()
    model.set_kv_caches(model.allocate_kv_caches(2, 4, device), num_slots=2)
    tokens = torch.tensor([[3], [4]], device=device)
    positions = torch.zeros(2, 1, dtype=torch.long, device=device)
    meta = InferMeta(
        mode="decode",
        cache_seqlens=torch.zeros(2, dtype=torch.int32, device=device),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32, device=device),
        recurrent_slots=torch.tensor([0, 1], device=device),
    )
    model.model.vision_lora_slots.copy_(torch.tensor([True, False], device=device))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model(tokens, positions, infer_meta=meta)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = model(tokens, positions, infer_meta=meta).logits_shard
    for modes in ([True, False], [False, True], [False, False]):
        model.model.vision_lora_slots.copy_(torch.tensor(modes, device=device))
        expected = model(tokens, positions, infer_meta=meta).logits_shard
        graph.replay()
        torch.testing.assert_close(captured, expected)
    model.reset_recurrent_cache_slots(meta.recurrent_slots)
    assert not model.model.vision_lora_slots.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phi4MM paged prefill requires CUDA")
@torch.inference_mode()
def test_phi4mm_chunked_image_prefill_matches_full_forward_across_longrope_boundary():
    from areno.models.phi4mm.model import Phi4MMAdapter

    torch.manual_seed(53)
    device = torch.device("cuda")
    config = Phi4MMAdapter().config_from_hf(_config())
    config.attn_backend = "native"
    config.image_token_id = 99
    model = Phi4MMAdapter().build(config).to(device).eval()
    prompt = [1] * 5 + [99] * 13 + [2]
    features = {
        "input_image_embeds": torch.randn(1, 2, 3, 8, 8, device=device),
        "image_sizes": torch.tensor([[8, 8]]),
        "image_attention_mask": torch.ones(1, 2, 4, 4, dtype=torch.bool),
        "image_token_id": 99,
    }
    expected = model(torch.tensor([prompt], device=device), features=features).logits_shard[:, -1]
    model.set_kv_caches(model.allocate_kv_caches(5, 4, device), num_slots=1)
    state = InferenceBatchState(
        [prompt],
        max_new_tokens=1,
        max_prefill_tokens=4,
        kv_block_size=4,
        num_cache_blocks=5,
        prompt_features=[features],
    )
    while state.has_pending_prompts:
        payload = state.build_prefill_payload()
        meta = payload_to_infer_meta(payload, device)
        assert meta.sequence_lengths.tolist() == [19]
        output = model(
            payload["input_ids"].to(device).unsqueeze(0),
            payload["position_ids"].to(device).unsqueeze(0),
            infer_meta=meta,
            features=payload.get("features"),
        )
    torch.testing.assert_close(output.logits_shard[:, -1], expected, atol=2e-5, rtol=2e-4)


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


def test_phi4mm_row_lora_scatter_matches_sequence_parallel_output(monkeypatch):
    pytest.importorskip("triton")
    import areno.engine.layers.linear as linear
    import areno.models.phi4mm.model as phi4mm
    from areno.engine.parallel.collectives import sequence_parallel_region

    previous = get_tp_context()
    try:
        set_tp_context(TPContext(rank=0, world_size=2, device=torch.device("cpu"), group=None))
        projection = phi4mm._Phi4MMRowLoRA(32, 16, phi4mm.Phi4MMAdapter().config_from_hf(_config())).float()
        projection.weight.data.zero_()
        projection.lora_A["vision"].weight.data.copy_(torch.arange(4 * 16).view(4, 16).float() / 100)
        projection.lora_B["vision"].weight.data.copy_(torch.arange(16 * 4).view(16, 4).float() / 100)
        projection.vision_lora_mask = torch.tensor([[True, False, True, True]])
        inputs = torch.arange(4 * 16).view(1, 4, 16).float() / 100
        scatter_calls = []

        monkeypatch.setattr(linear, "reduce_scatter_to_sequence_parallel_region", lambda tensor: tensor[:, :2])
        monkeypatch.setattr(phi4mm, "all_reduce", lambda tensor: tensor)

        def fake_scatter(tensor):
            scatter_calls.append(tuple(tensor.shape))
            return tensor[:, :2]

        monkeypatch.setattr(phi4mm, "scatter_to_sequence_parallel_region", fake_scatter)
        with sequence_parallel_region(True):
            actual = projection(inputs)
        expected = 2.0 * F.linear(
            F.linear(inputs, projection.lora_A["vision"].weight),
            projection.lora_B["vision"].weight,
        )
        expected = (expected * projection.vision_lora_mask.unsqueeze(-1))[:, :2]

        assert scatter_calls == [(1, 4, 16)]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    finally:
        set_tp_context(previous)


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
    from areno.engine.checkpoints.common import save_checkpoint_weights
    from areno.engine.checkpoints.io import SafetensorsIndex
    from areno.models.phi4mm.checkpoint import (
        CHECKPOINT_SPEC,
        _save_vision_weights,
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
    seed = tmp_path / "seed"
    save_checkpoint_weights(
        first,
        str(seed),
        None,
        CHECKPOINT_SPEC,
        extra_tensors_fn=lambda tensors: _save_vision_weights(tensors, first),
    )
    seed_index = SafetensorsIndex(seed, progress=False)
    try:
        source_tensors = {key: seed_index.get_tensor(key) for key in seed_index.weight_map}
    finally:
        seed_index.close()
    source_tensors["model.layers.0.self_attn.qkv_proj.lora_A.speech.weight"] = torch.ones(1)
    source_tensors["model.embed_tokens_extend.audio_embed.dummy.weight"] = torch.ones(1)
    source = tmp_path / "source"
    source.mkdir()
    save_file(source_tensors, source / "model.safetensors")
    output = tmp_path / "output"

    saved_path = adapter.save_weights(first, output, source)
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
    assert audit.consumed + audit.speech_lora_skipped + audit.audio_skipped == audit.total
    assert audit.speech_lora_skipped == audit.audio_skipped == 1
    assert audit.unknown == 0
    assert "model.embed_tokens_extend.image_embed.sub_GN" in vision_keys
    assert "model.embed_tokens_extend.image_embed.glb_GN" in vision_keys
    assert len(vision_lora_keys) == 8

    index = SafetensorsIndex(output, progress=False)
    try:
        saved_keys = set(index.weight_map)
    finally:
        index.close()
    assert vision_keys | vision_lora_keys <= saved_keys
    assert "model.layers.0.self_attn.qkv_proj.lora_A.speech.weight" in saved_keys
    assert "model.embed_tokens_extend.audio_embed.dummy.weight" in saved_keys


def test_phi4mm_policy_plan_includes_vision_and_lora_weights():
    pytest.importorskip("triton")
    from areno.models.phi4mm.checkpoint import build_phi4mm_policy_plan
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(Phi4MMAdapter().config_from_hf(_config())).float()
    plan = build_phi4mm_policy_plan(model)

    assert "model.embed_tokens_extend.image_embed.sub_GN" in plan
    assert "model.layers.0.self_attn.qkv_proj.lora_B.vision.weight" in plan


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
