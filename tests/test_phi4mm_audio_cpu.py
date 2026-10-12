from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from areno.api.backend.cuda.roles import _pack_score_rows
from areno.api.multimodal import encode_processor_messages, modality_token_ids
from areno.engine.checkpoints.common import save_checkpoint_weights
from areno.engine.config import ModelConfig, OptimizerConfig
from areno.engine.data.rollout_state import InferenceBatchState, payload_to_infer_meta
from areno.engine.modeling import configure_multimodal_training
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.engine.runtime.train_step import _pack_multimodal_features
from areno.models.phi4mm import Phi4MMAdapter
from areno.models.phi4mm.audio import Phi4MMAudioConfig, Phi4MMAudioEmbedding
from areno.models.phi4mm.checkpoint import (
    CHECKPOINT_SPEC,
    _lora_checkpoint_keys,
    _multimodal_checkpoint_keys,
    _save_multimodal_weights,
    audit_phi4mm_checkpoint,
)


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


def test_phi4mm_audio_sdpa_matches_reference_attention_and_gradients():
    from areno.models.phi4mm.audio import _MultiHeadedAttention

    torch.manual_seed(9)
    module = _MultiHeadedAttention(hidden_size=8, num_heads=2, dtype=torch.float32).eval()
    value = torch.randn(2, 5, 8, requires_grad=True)
    relative_bias = torch.randn(1, 2, 5, 5)
    valid_keys = torch.tensor([[True] * 5, [True, True, True, False, False]])
    mask = valid_keys[:, None, :].expand(-1, 5, -1)

    actual = module(value, mask, relative_bias)
    actual.square().mean().backward()
    actual_input_grad = value.grad.detach().clone()
    actual_parameter_grads = {name: parameter.grad.detach().clone() for name, parameter in module.named_parameters()}

    module.zero_grad(set_to_none=True)
    reference_value = value.detach().clone().requires_grad_(True)

    query = module.linear_q(reference_value).view(2, 5, 2, 4).transpose(1, 2) * module.inv_sqrt_d_k
    key = module.linear_k(reference_value).view(2, 5, 2, 4).transpose(1, 2)
    projected = module.linear_v(reference_value).view(2, 5, 2, 4).transpose(1, 2)
    invalid = ~mask.unsqueeze(1)
    scores = (query @ key.transpose(-2, -1) + relative_bias).masked_fill(invalid, -torch.inf)
    attention = torch.softmax(scores, dim=-1).masked_fill(invalid, 0.0)
    expected = (attention @ projected).transpose(1, 2).contiguous().view(2, 5, 8)
    expected = module.linear_out(expected)
    expected.square().mean().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_input_grad, reference_value.grad, rtol=2e-5, atol=1e-6)
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(actual_parameter_grads[name], parameter.grad, rtol=2e-5, atol=1e-6)


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


def test_phi4mm_cache_reprefill_restores_audio_features_and_mode():
    features = {
        "audio_embeds": torch.ones(1, 16),
        "audio_token_id": 200011,
        "modality_token_ids": {"audio": 200011},
    }
    state = InferenceBatchState(
        [[200011, 1]],
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

    assert payload["features"]["audio_sequence_mask"].tolist() == [True]
    assert payload["features"]["audio_token_mask"].tolist() == [True, False, False]
    assert payload["features"]["image_token_mask"].tolist() == [False, False, False]
    assert payload["features"]["audio_feature_rows"][0]["audio_embeds"] is features["audio_embeds"]


def test_phi4mm_audio_only_prefill_slices_features_and_masks_audio_tokens():
    audio_embeds = torch.arange(2 * 16, dtype=torch.float32).view(2, 16)
    features = {
        "audio_embeds": audio_embeds,
        "audio_token_id": 200011,
        "modality_token_ids": {"audio": 200011},
    }
    state = InferenceBatchState(
        [[200011, 1, 200011, 2]],
        max_new_tokens=1,
        max_prefill_tokens=4,
        max_cache_len=8,
        kv_block_size=4,
        num_cache_blocks=2,
        prompt_features=[features],
    )

    payload = state.build_prefill_payload()

    assert payload["features"]["image_token_mask"].tolist() == [False, False, False, False]
    assert payload["features"]["audio_token_mask"].tolist() == [True, False, True, False]
    torch.testing.assert_close(payload["features"]["audio_feature_rows"][0]["audio_embeds"], audio_embeds)


def test_phi4mm_audio_training_groups_and_modes():
    model = Phi4MMAdapter().build(_tiny_model_config())
    audio = model.model.embed_tokens_extend.audio_embed
    configure_multimodal_training(model, OptimizerConfig(lr=1e-3), trainable=True)
    assert all(not parameter.requires_grad for parameter in audio.parameters())
    assert not audio.encoder.training
    assert not audio.audio_projection.training

    configure_multimodal_training(
        model,
        OptimizerConfig(
            lr=1e-3,
            unfreeze_multimodal_tower=True,
            unfreeze_multimodal_projector=True,
            multimodal_tower_lr=2e-4,
            multimodal_projector_lr=3e-4,
        ),
        trainable=True,
    )

    assert all(parameter.requires_grad for parameter in audio.encoder.parameters())
    assert all(parameter.requires_grad for parameter in audio.audio_projection.parameters())
    assert {parameter._areno_lr_group for parameter in audio.encoder.parameters()} == {"tower"}
    assert {parameter._areno_lr_group for parameter in audio.audio_projection.parameters()} == {"projector"}
    assert audio.encoder.training
    assert audio.audio_projection.training


def test_phi4mm_audio_train_save_reload_inference_closes(tmp_path, monkeypatch):
    from areno.engine.layers import linear, mlp, norm, vocab

    monkeypatch.setenv("ARENO_CKPT_PROGRESS", "0")
    monkeypatch.setattr(vocab, "areno_vocab_embedding", lambda ids, weight, start, end: F.embedding(ids, weight))
    monkeypatch.setattr(linear, "areno_linear", F.linear)
    monkeypatch.setattr(
        norm, "_areno_rmsnorm_no_compile", lambda value, weight, eps: F.rms_norm(value, (value.shape[-1],), weight, eps)
    )
    monkeypatch.setattr(
        mlp,
        "_areno_silu_and_mul_no_compile",
        lambda value: F.silu(value.chunk(2, dim=-1)[0]) * value.chunk(2, dim=-1)[1],
    )
    torch.manual_seed(17)
    adapter = Phi4MMAdapter()
    config = _tiny_model_config()
    config.attn_backend = "native"
    config.audio_token_id = 20
    first = adapter.build(config).float()
    seed = tmp_path / "seed"
    save_checkpoint_weights(
        first,
        str(seed),
        None,
        CHECKPOINT_SPEC,
        extra_tensors_fn=lambda tensors: _save_multimodal_weights(tensors, first),
    )
    configure_multimodal_training(
        first,
        OptimizerConfig(
            lr=1e-3,
            unfreeze_multimodal_tower=True,
            unfreeze_multimodal_projector=True,
        ),
        trainable=True,
    )
    audio = first.model.embed_tokens_extend.audio_embed
    trainable = [parameter for parameter in audio.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=1e-3)
    features = {
        "input_audio_embeds": torch.randn(1, 8, 80),
        "audio_attention_mask": torch.ones(1, 8, dtype=torch.bool),
        "audio_embed_sizes": torch.tensor([1]),
        "audio_token_id": 20,
        "input_mode": 2,
    }
    tokens = torch.tensor([[1, 20, 2]])
    projector_before = audio.audio_projection["speech"][0].weight.detach().clone()
    first.train()
    loss = F.cross_entropy(first(tokens, features=features).logits_shard[:, -1], torch.tensor([3]))
    loss.backward()
    assert any(parameter.grad is not None for parameter in audio.encoder.parameters())
    assert any(parameter.grad is not None for parameter in audio.audio_projection.parameters())
    optimizer.step()
    assert not torch.equal(projector_before, audio.audio_projection["speech"][0].weight)

    output = tmp_path / "output"
    adapter.save_weights(first, output, seed)

    second = adapter.build(config).float()
    adapter.load_weights(second, output)
    first.eval()
    second.eval()
    with torch.no_grad():
        trained_logits = first(tokens, features=features).logits_shard
        reloaded_logits = second(tokens, features=features).logits_shard
    torch.testing.assert_close(reloaded_logits, trained_logits, rtol=0, atol=0)
    audit = audit_phi4mm_checkpoint(
        output,
        len(second.layers),
        _multimodal_checkpoint_keys(second),
        _lora_checkpoint_keys(second),
    )
    assert audit.audio_skipped == 0


def test_phi4mm_chunked_audio_prefill_keeps_audio_embedding_offsets():
    audio_embeds = torch.arange(2 * 16, dtype=torch.float32).view(2, 16)
    features = {
        "audio_embeds": audio_embeds,
        "audio_token_id": 200011,
        "modality_token_ids": {"audio": 200011},
    }
    state = InferenceBatchState(
        [[200011, 1, 200011, 2]],
        max_new_tokens=1,
        max_prefill_tokens=2,
        max_cache_len=8,
        kv_block_size=2,
        num_cache_blocks=4,
        prompt_features=[features],
    )

    first = state.build_prefill_payload()
    second = state.build_prefill_payload()

    torch.testing.assert_close(first["features"]["audio_feature_rows"][0]["audio_embeds"], audio_embeds[:1])
    torch.testing.assert_close(second["features"]["audio_feature_rows"][0]["audio_embeds"], audio_embeds[1:])
    assert first["features"]["audio_token_mask"].tolist() == [True, False]
    assert second["features"]["audio_token_mask"].tolist() == [True, False]


def test_phi4mm_packed_training_keeps_audio_rows_and_masks():
    audio = torch.ones(2, 16)
    packed = _pack_multimodal_features(
        [
            {
                "audio_embeds": audio,
                "audio_token_id": 200011,
                "modality_token_ids": {"audio": 200011},
            },
            None,
        ],
        torch.tensor([[1, 200011, 200011, 2], [3, 4, 0, 0]]),
        torch.tensor([4, 2]),
        torch.device("cpu"),
    )

    assert packed["audio_token_id"] == 200011
    assert packed["audio_token_mask"].tolist() == [False, True, True, False, False, False]
    assert packed["image_token_mask"].tolist() == [False] * 6
    assert packed["audio_feature_rows"][0]["audio_embeds"] is audio


def test_phi4mm_logprob_packing_preserves_audio_features():
    audio = torch.ones(2, 16)
    packed, lengths = _pack_score_rows(
        [[1, 200011, 200011, 2]],
        torch.device("cpu"),
        0,
        features=[
            {
                "audio_embeds": audio,
                "audio_token_id": 200011,
                "modality_token_ids": {"audio": 200011},
            }
        ],
    )

    assert lengths == [4]
    assert packed["features"]["audio_token_mask"].tolist() == [False, True, True, False]
    assert packed["features"]["image_token_mask"].tolist() == [False] * 4
    assert packed["features"]["audio_feature_rows"][0]["audio_embeds"] is audio


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
    assert features["input_mode"] == 2


def test_phi4mm_processor_bridge_marks_combined_vision_speech_mode(monkeypatch):
    class Tokenizer:
        def convert_tokens_to_ids(self, token):
            return {"<|endoftext10|>": 200010, "<|endoftext11|>": 200011}[token]

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages[0]["content"] == "<|image_1|><|audio_1|>Describe"
            return "rendered"

    class Phi4MMProcessor:
        tokenizer = Tokenizer()

        def __call__(self, *, text, images, audios, return_tensors):
            assert text == "rendered"
            assert images == ["image"]
            assert audios == ["audio"]
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[1, 200010, 200011, 2]]),
                "input_image_embeds": torch.zeros(1, 1, 3, 8, 8),
                "input_audio_embeds": torch.zeros(1, 8, 80),
                "audio_embed_sizes": torch.tensor([1]),
            }

    Phi4MMProcessor.__module__ = "transformers_modules.phi4mm"
    monkeypatch.setattr("areno.api.multimodal._load_phi4mm_image", lambda _reference: "image")
    monkeypatch.setattr("areno.api.multimodal._load_phi4mm_audio", lambda _reference: "audio")

    _, features = encode_processor_messages(
        Phi4MMProcessor(),
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "one.png"},
                    {"type": "audio", "url": "one.wav"},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ],
    )

    assert features["input_mode"] == 3
