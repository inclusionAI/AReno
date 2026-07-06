from __future__ import annotations

import pytest
import torch

from areno.api import TrainSequence
from areno.api.backend.areno.backend import _make_train_pack
from areno.engine.data.rollout_state import _prefill_multimodal_features
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
    input_ids = torch.tensor([[1, 99, 2]])
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
