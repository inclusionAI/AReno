from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from areno.api.multimodal import encode_processor_messages, modality_token_ids
from areno.engine.config import ModelConfig
from areno.engine.data.rollout_state import InferenceBatchState, payload_to_infer_meta
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.models.phi4mm.audio import Phi4MMAudioConfig, Phi4MMAudioEmbedding


@pytest.fixture(autouse=True)
def _isolate_tp_context():
    previous_context = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    try:
        yield
    finally:
        set_tp_context(previous_context)


def _tiny_audio_config() -> Phi4MMAudioConfig:
    return Phi4MMAudioConfig(
        input_size=80,
        attention_dim=8,
        attention_heads=2,
        linear_units=12,
        num_blocks=2,
        kernel_size=3,
        time_reduction=8,
        relative_attention_max_distance=8,
    )


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        model_type="phi4mm",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        qkv_bias=False,
        qk_norm=False,
        dtype=torch.float32,
        hidden_act="silu",
        partial_rotary_factor=0.5,
        sequence_parallel=False,
        attn_backend="native",
        audio_config={
            "input_size": 80,
            "attention_dim": 8,
            "attention_heads": 2,
            "linear_units": 12,
            "num_blocks": 1,
            "kernel_size": 3,
            "time_reduction": 8,
            "relative_attention_bias_args": {"t5_bias_max_distance": 8},
        },
        audio_token_id=200011,
        hf_text_config={
            "original_max_position_embeddings": 16,
            "rope_scaling": {"type": "longrope", "short_factor": (1.0,), "long_factor": (2.0,)},
            "speech_lora": {"r": 4, "lora_alpha": 8, "dp": 0.0},
        },
    )


def _official_audio_sections() -> dict:
    return {
        "embd_layer": {
            "audio_embd_layer": {
                "embedding_cls": "audio",
                "projection_cls": "mlp",
                "compression_rate": 8,
                "downsample_rate": 1,
                "use_qformer": False,
                "use_conv_downsample": False,
            }
        },
        "audio_processor": {
            "name": "cascades",
            "config": {
                "input_layer": "nemo_conv",
                "input_size": 80,
                "attention_dim": 1024,
                "attention_heads": 16,
                "num_blocks": 24,
                "time_reduction": 8,
                "causal": True,
                "activation": "swish",
                "conv_activation": "swish",
                "conv_glu_type": "swish",
                "batch_norm": False,
            },
        },
    }


def test_phi4mm_audio_config_accepts_only_official_semantics():
    from areno.models.phi4mm.model import _phi4mm_audio_config

    config = _official_audio_sections()
    assert _phi4mm_audio_config(config)["num_blocks"] == 24

    config["embd_layer"]["audio_embd_layer"]["compression_rate"] = 4
    with pytest.raises(ValueError, match="compression_rate=8"):
        _phi4mm_audio_config(config)


def test_phi4mm_audio_encoder_reduces_time_and_respects_padding():
    module = Phi4MMAudioEmbedding(_tiny_audio_config(), language_hidden_size=16, dtype=torch.float32).eval()
    inputs = torch.randn(2, 25, 80)
    mask = torch.tensor([[True] * 25, [True] * 17 + [False] * 8])

    output = module(inputs, mask)

    assert output.shape == (2, 4, 16)
    assert torch.isfinite(output).all()


def test_phi4mm_audio_checkpoint_names_match_official_layout():
    module = Phi4MMAudioEmbedding(_tiny_audio_config(), language_hidden_size=16, dtype=torch.float32)
    keys = set(module.state_dict())

    assert "encoder.embed.conv.0.weight" in keys
    assert "encoder.encoder_embedding.global_mean" in keys
    assert "encoder.encoders.0.self_attn.linear_q.weight" in keys
    assert "encoder.encoders.0.conv.glu.b1" in keys
    assert "audio_projection.speech.2.weight" in keys
    assert "audio_projection.vision.2.weight" in keys


def test_phi4mm_audio_relative_bias_clamps_distant_positions():
    from areno.models.phi4mm.audio import _T5RelativeAttentionLogitBias

    bias = _T5RelativeAttentionLogitBias(num_heads=2, max_distance=3, dtype=torch.float32)
    output = bias(torch.zeros(1, 6, 8))

    assert output.shape == (1, 2, 6, 6)
    torch.testing.assert_close(output[:, :, 0, 2], output[:, :, 0, 5])
    torch.testing.assert_close(output[:, :, 3, 0], output[:, :, 5, 0])


def test_phi4mm_audio_merge_replaces_only_audio_placeholders():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(_tiny_model_config()).float()
    input_ids = torch.tensor([[1, 200011, 200011, 2]])
    hidden = torch.randn(1, 4, 16)
    audio_embeds = torch.full((2, 16), 3.0)

    merged = model.model._apply_multimodal_features(
        hidden,
        input_ids,
        {"audio_embeds": audio_embeds, "audio_token_id": 200011},
    )

    torch.testing.assert_close(merged[0, 1:3], audio_embeds)
    torch.testing.assert_close(merged[0, [0, 3]], hidden[0, [0, 3]])


def test_phi4mm_audio_merge_rejects_placeholder_count_mismatch():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(_tiny_model_config()).float()
    input_ids = torch.tensor([[1, 200011, 2]])

    with pytest.raises(ValueError, match="audio token count does not match"):
        model.model._apply_multimodal_features(
            torch.randn(1, 3, 16),
            input_ids,
            {"audio_embeds": torch.ones(2, 16), "audio_token_id": 200011},
        )


def test_phi4mm_speech_lora_state_survives_chunked_prefill_and_decode():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    features = {
        "audio_embeds": torch.ones(2, 16),
        "audio_token_id": 200011,
        "modality_token_ids": {"audio": 200011},
    }
    state = InferenceBatchState(
        [[200011, 200011, 1, 2]],
        max_new_tokens=1,
        max_prefill_tokens=2,
        max_cache_len=8,
        kv_block_size=2,
        num_cache_blocks=4,
        prompt_features=[features],
    )
    first = state.build_prefill_payload()
    second = state.build_prefill_payload()
    assert first["features"]["audio_sequence_mask"].tolist() == [True]
    assert second["features"]["audio_sequence_mask"].tolist() == [True]

    model = Phi4MMAdapter().build(_tiny_model_config()).float()
    model.model.speech_lora_slots = torch.zeros(1, dtype=torch.bool)
    _, speech = model.model._lora_masks(
        second["input_ids"].unsqueeze(0),
        second["features"],
        None,
        payload_to_infer_meta(second, torch.device("cpu")),
    )
    assert speech.tolist() == [[True, True]]
    assert model.model.speech_lora_slots.tolist() == [True]


def test_phi4mm_multi_audio_merge_preserves_segment_order():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(_tiny_model_config()).float()
    first = torch.full((2, 16), 1.0)
    second = torch.full((3, 16), 2.0)
    input_ids = torch.tensor([[1, 200011, 200011, 2, 200011, 200011, 200011, 3]])
    hidden = torch.randn(1, input_ids.shape[1], 16)
    features = {
        "audio_feature_rows": [
            {"audio_embeds": first, "audio_token_count": 2},
            {"audio_embeds": second, "audio_token_count": 3},
        ],
        "audio_token_id": 200011,
    }

    merged = model.model._apply_multimodal_features(hidden, input_ids, features)

    torch.testing.assert_close(merged[0, 1:3], first)
    torch.testing.assert_close(merged[0, 4:7], second)
    torch.testing.assert_close(merged[0, [0, 3, 7]], hidden[0, [0, 3, 7]])


def test_phi4mm_mixed_text_audio_batch_uses_speech_lora_per_row():
    pytest.importorskip("triton")
    from areno.models.phi4mm.model import Phi4MMAdapter

    model = Phi4MMAdapter().build(_tiny_model_config()).float()
    input_ids = torch.tensor([[200011, 1, 2], [3, 4, 5]])
    _, speech = model.model._lora_masks(
        input_ids,
        [{"audio_embeds": torch.ones(1, 16), "audio_token_id": 200011}, None],
        None,
        None,
    )

    assert speech.tolist() == [[True, True, True], [False, False, False]]


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_phi4mm_speech_lora_tp_mapping_reconstructs_full_weights(tmp_path, tp_size):
    pytest.importorskip("triton")
    from areno.models.phi4mm.checkpoint import _load_vision_lora_weights
    from areno.models.phi4mm.model import Phi4MMAdapter

    prefix = "model.layers.0"
    tensors = {
        f"{prefix}.self_attn.qkv_proj.lora_A.speech.weight": torch.arange(4 * 16).view(4, 16).float(),
        f"{prefix}.self_attn.qkv_proj.lora_B.speech.weight": torch.arange(48 * 4).view(48, 4).float(),
        f"{prefix}.self_attn.o_proj.lora_A.speech.weight": torch.arange(4 * 16).view(4, 16).float(),
        f"{prefix}.self_attn.o_proj.lora_B.speech.weight": torch.arange(16 * 4).view(16, 4).float(),
        f"{prefix}.mlp.gate_up_proj.lora_A.speech.weight": torch.arange(4 * 16).view(4, 16).float(),
        f"{prefix}.mlp.gate_up_proj.lora_B.speech.weight": torch.arange(64 * 4).view(64, 4).float(),
        f"{prefix}.mlp.down_proj.lora_A.speech.weight": torch.arange(4 * 32).view(4, 32).float(),
        f"{prefix}.mlp.down_proj.lora_B.speech.weight": torch.arange(16 * 4).view(16, 4).float(),
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(tensors, checkpoint / "model.safetensors")
    previous = get_tp_context()
    try:
        for rank in range(tp_size):
            set_tp_context(TPContext(rank=rank, world_size=tp_size, device=torch.device("cpu"), group=None))
            model = Phi4MMAdapter().build(_tiny_model_config()).float()
            _load_vision_lora_weights(model, checkpoint)
            qkv = model.layers[0].self_attn.qkv_proj
            expected = torch.cat(
                [
                    part.chunk(tp_size)[rank]
                    for part in tensors[f"{prefix}.self_attn.qkv_proj.lora_B.speech.weight"].split((16, 16, 16))
                ]
            )
            torch.testing.assert_close(qkv.lora_B["speech"].weight, expected)
            torch.testing.assert_close(
                model.layers[0].self_attn.o_proj.lora_A["speech"].weight,
                tensors[f"{prefix}.self_attn.o_proj.lora_A.speech.weight"].chunk(tp_size, dim=1)[rank],
            )
    finally:
        set_tp_context(previous)


def test_phi4mm_processor_token_fallback_finds_audio_token():
    class Tokenizer:
        def convert_tokens_to_ids(self, token):
            return {"<|endoftext10|>": 200010, "<|endoftext11|>": 200011}[token]

    class Phi4MMProcessor:
        tokenizer = Tokenizer()

    Phi4MMProcessor.__module__ = "transformers_modules.phi4mm"
    processor = Phi4MMProcessor()

    assert modality_token_ids(processor) == {"image": 200010, "audio": 200011}


def test_phi4mm_processor_bridge_numbers_audio_placeholders(monkeypatch):
    class Tokenizer:
        def convert_tokens_to_ids(self, token):
            return {"<|endoftext10|>": 200010, "<|endoftext11|>": 200011}[token]

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages[0]["content"] == "<|audio_1|><|audio_2|>Compare"
            return "rendered<|endoftext|>"

    class Phi4MMProcessor:
        tokenizer = Tokenizer()

        def __call__(self, *, text, images, audios, return_tensors):
            assert text == "rendered"
            assert images is None
            assert audios == ["first", "second"]
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[1, 200011, 200011, 2]]),
                "attention_mask": torch.ones(1, 4),
                "input_audio_embeds": torch.zeros(2, 8, 80),
                "audio_embed_sizes": torch.tensor([1, 1]),
            }

    Phi4MMProcessor.__module__ = "transformers_modules.phi4mm"
    monkeypatch.setattr(
        "areno.api.multimodal._load_phi4mm_audio",
        lambda reference: {"one.wav": "first", "two.wav": "second"}[reference],
    )
    tokens, features = encode_processor_messages(
        Phi4MMProcessor(),
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": "one.wav"},
                    {"type": "audio", "url": "two.wav"},
                    {"type": "text", "text": "Compare"},
                ],
            }
        ],
    )

    assert tokens == [1, 200011, 200011, 2]
    assert features["audio_token_id"] == 200011
    assert features["input_audio_embeds"].shape == (2, 8, 80)
