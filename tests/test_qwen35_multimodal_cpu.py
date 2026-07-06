from __future__ import annotations

import pytest
import torch

from areno.api import TrainSequence
from areno.api.backend.areno.backend import _make_train_pack
from areno.api.multimodal import (
    expand_image_tokens,
    image_token_counts_from_features,
    mrope_position_ids_from_image_grid,
)
from areno.api.trainers.sft import _record_to_train_sequence
from areno.engine.data.rollout_state import _prefill_multimodal_features
from areno.engine.layers.rotary import PartialRotaryEmbedding
from areno.engine.runtime.train_step import _pack_train_data


def _qwen35_text_config() -> dict:
    return {
        "model_type": "qwen3_5",
        "vocab_size": 128,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 4,
        "rms_norm_eps": 1e-6,
        "full_attention_interval": 1,
        "tie_word_embeddings": False,
    }


def _qwen35_vision_config() -> dict:
    return {
        "depth": 1,
        "hidden_size": 8,
        "hidden_act": "gelu_pytorch_tanh",
        "in_channels": 3,
        "intermediate_size": 16,
        "num_heads": 2,
        "num_position_embeddings": 16,
        "out_hidden_size": 16,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
    }


def test_train_pack_preserves_multimodal_features():
    image_embeds = torch.ones(1, 16)
    seq = TrainSequence(
        tokens=[1, 99, 2],
        prompt_mask=[True, True, False],
        logprobs=[0.0, 0.0, -0.1],
        advantages=[0.0, 0.0, 1.0],
        features={"image_token_id": 99, "image_embeds": image_embeds},
    )

    pack = _make_train_pack([seq])

    assert pack["features"][0]["image_token_id"] == 99
    assert torch.equal(pack["features"][0]["image_embeds"], image_embeds)


def test_packed_train_leaves_multimodal_features_dense():
    pack = {
        "input_ids": torch.tensor([[1, 99, 2]]),
        "lengths": torch.tensor([3], dtype=torch.int32),
        "prompt_mask": torch.tensor([[True, True, False]]),
        "logprobs": torch.zeros(1, 3),
        "advantages": torch.zeros(1, 3),
        "features": [{"image_token_id": 99, "image_embeds": torch.ones(1, 16)}],
    }

    packed = _pack_train_data(pack)

    assert packed is pack
    assert "train_cu_seqlens" not in packed


def test_qwen35_vl_adapter_matches_vision_text_config():
    pytest.importorskip("triton")
    from areno.models.qwen3_5.model import Qwen35VLAdapter

    adapter = Qwen35VLAdapter()
    hf_config = {
        "model_type": "qwen3_5_vl",
        "architectures": ["Qwen3_5_VLForConditionalGeneration"],
        "text_config": _qwen35_text_config(),
        "vision_config": _qwen35_vision_config(),
        "image_token_id": 99,
    }

    assert adapter.match_hf_config(hf_config)
    config = adapter.config_from_hf(hf_config)
    assert config.model_type == "qwen3_5_vl"
    assert config.hidden_size == 16
    assert config.vision_config["out_hidden_size"] == 16
    assert config.image_token_id == 99


def test_qwen35_multimodal_features_split_and_mask():
    pytest.importorskip("triton")
    from areno.models.qwen3_5.model import _features_by_row, _image_token_mask_for_row

    features = {
        "image_embeds": torch.arange(2 * 1 * 4, dtype=torch.float32).view(2, 1, 4),
        "image_token_id": 99,
    }

    rows = _features_by_row(features, batch=2)
    mask = _image_token_mask_for_row(rows[1], torch.tensor([7, 99, 8]), row_idx=1)

    assert rows[1]["image_embeds"].shape == (1, 4)
    assert mask.tolist() == [False, True, False]


def test_qwen35_image_grid_expands_placeholder_tokens():
    features = {
        "image_token_id": 99,
        "image_grid_thw": torch.tensor([[1, 16, 16]]),
        "spatial_merge_size": 2,
    }

    counts = image_token_counts_from_features(features)
    tokens, aligned = expand_image_tokens(
        [1, 99, 2],
        image_token_id=99,
        image_token_counts=counts,
        aligned_sequences={"prompt_mask": [True, True, False]},
    )

    assert counts == [64]
    assert len(tokens) == 66
    assert tokens.count(99) == 64
    assert aligned["prompt_mask"] == [True] * 65 + [False]


def test_qwen35_image_grid_keeps_processor_expanded_image_tokens():
    features = {
        "image_token_id": 99,
        "image_grid_thw": torch.tensor([[1, 16, 16]]),
        "spatial_merge_size": 2,
    }

    counts = image_token_counts_from_features(features)
    tokens, aligned = expand_image_tokens(
        [1] + [99] * 64 + [2],
        image_token_id=99,
        image_token_counts=counts,
        aligned_sequences={"prompt_mask": [True] * 66},
    )

    assert counts == [64]
    assert len(tokens) == 66
    assert tokens.count(99) == 64
    assert aligned["prompt_mask"] == [True] * 66


def test_qwen35_image_grid_builds_sglang_style_mrope_positions():
    features = {
        "image_token_id": 99,
        "image_grid_thw": torch.tensor([[1, 16, 16]]),
        "spatial_merge_size": 2,
    }
    counts = image_token_counts_from_features(features)
    tokens, _ = expand_image_tokens([1, 99, 2, 3], image_token_id=99, image_token_counts=counts)

    position_ids = mrope_position_ids_from_image_grid(tokens, image_token_id=99, features=features)

    assert position_ids.shape == (3, len(tokens))
    assert position_ids[:, 0].tolist() == [0, 0, 0]
    assert position_ids[:, 1].tolist() == [1, 1, 1]
    assert position_ids[:, 8].tolist() == [1, 1, 8]
    assert position_ids[:, 64].tolist() == [1, 8, 8]
    assert position_ids[:, -2:].tolist() == [[9, 10], [9, 10], [9, 10]]


def test_qwen35_mrope_selects_axes_before_repeating_half_dim_cache():
    rope = PartialRotaryEmbedding(
        head_dim=8,
        max_position=8,
        theta=10000.0,
        partial_rotary_factor=1.0,
        is_neox_style=True,
        mrope_section=(2, 1, 1),
        mrope_interleaved=True,
    )
    position_ids = torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=torch.long)

    cos, sin = rope._cos_sin(position_ids, torch.float32)

    half_dim = rope.rope_dim // 2
    expected_half_cos = rope.cos_cached[:, :half_dim][position_ids]
    expected_half_sin = rope.sin_cached[:, :half_dim][position_ids]
    expected_cos = expected_half_cos[0].clone()
    expected_sin = expected_half_sin[0].clone()
    expected_cos[..., 1:3:3] = expected_half_cos[1, ..., 1:3:3]
    expected_sin[..., 1:3:3] = expected_half_sin[1, ..., 1:3:3]
    expected_cos[..., 2:3:3] = expected_half_cos[2, ..., 2:3:3]
    expected_sin[..., 2:3:3] = expected_half_sin[2, ..., 2:3:3]
    expected_cos = torch.cat((expected_cos, expected_cos), dim=-1).unsqueeze(2)
    expected_sin = torch.cat((expected_sin, expected_sin), dim=-1).unsqueeze(2)

    assert torch.equal(cos, expected_cos)
    assert torch.equal(sin, expected_sin)


def test_sft_encoded_multimodal_row_expands_image_tokens():
    class Tokenizer:
        eos_token_id = 0

    seq = _record_to_train_sequence(
        {
            "tokens": [1, 99, 2],
            "prompt_mask": [True, True, False],
            "loss_mask": [False, False, True],
            "features": {
                "image_token_id": 99,
                "image_grid_thw": [[1, 16, 16]],
                "spatial_merge_size": 2,
            },
        },
        Tokenizer(),
        max_prompt_tokens=128,
        max_new_tokens=8,
    )

    assert seq is not None
    assert len(seq.tokens) == 66
    assert seq.tokens.count(99) == 64
    assert seq.prompt_mask == [True] * 65 + [False]
    assert seq.loss_mask == [False] * 65 + [True]


def test_qwen35_vl_projects_pixel_values_to_image_embeds():
    pytest.importorskip("triton")
    from areno.models.qwen3_5.model import Qwen35VLAdapter

    adapter = Qwen35VLAdapter()
    config = adapter.config_from_hf(
        {
            "model_type": "qwen3_5_vl",
            "text_config": _qwen35_text_config(),
            "vision_config": _qwen35_vision_config(),
            "image_token_id": 99,
        }
    )
    model = adapter.build(config)
    features = {
        "pixel_values": torch.zeros(4, 3 * 1 * 2 * 2),
        "image_token_id": 99,
    }

    projected = model._project_pixel_values(features, torch.device("cpu"), batch=1)

    assert projected["image_embeds"].shape == (1, 16)


def test_prefill_multimodal_features_support_multiple_rows():
    features = _prefill_multimodal_features(
        [False, True, False, True],
        [
            {"pixel_values": torch.zeros(4, 12), "image_grid_thw": torch.tensor([[1, 2, 2]])},
            {"pixel_values": torch.zeros(4, 12), "image_grid_thw": torch.tensor([[1, 2, 2]])},
        ],
    )

    assert features["image_token_mask"].tolist() == [False, True, False, True]
    assert len(features["image_feature_rows"]) == 2


def test_prefill_multimodal_features_carries_mrope_positions():
    mrope_positions = torch.arange(12, dtype=torch.long).view(3, 4)

    features = _prefill_multimodal_features(
        [False, False, False, False],
        [],
        [mrope_positions[:, :2], mrope_positions[:, 2:]],
    )

    assert "image_token_mask" not in features
    assert torch.equal(features["mrope_position_ids"], mrope_positions)
