from __future__ import annotations

from types import MethodType

import pytest
import torch
from torch import nn

from areno.api.multimodal import encode_multimodal_prompt, image_token_counts_from_features
from areno.engine.data.rollout_state import InferenceBatchState, _slice_prompt_image_features, payload_to_infer_meta


def _config(*, insert_layer_id: int = 6) -> dict:
    return {
        "model_type": "minicpmv4_6",
        "image_token_id": 99,
        "insert_layer_id": insert_layer_id,
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "layer_types": ["full_attention"],
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 32,
        },
        "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_channels": 3,
            "image_size": 8,
            "patch_size": 2,
            "layer_norm_eps": 1e-6,
            "window_kernel_size": [2, 2],
        },
    }


def test_minicpmv46_adapter_maps_nested_text_and_vision_config():
    pytest.importorskip("triton")
    from areno.models.minicpmv46.model import MiniCPMV46Adapter

    config = MiniCPMV46Adapter().config_from_hf(_config())

    assert config.linear_num_key_heads == 2
    assert config.linear_num_value_heads == 4
    assert config.vision_config["hidden_size"] == 8
    assert config.vision_config["insert_layer_id"] == 6
    assert config.image_token_id == 99


def test_minicpmv46_config_is_not_claimed_by_qwen35_vl_adapter():
    pytest.importorskip("triton")
    from areno.models.minicpmv46.model import MiniCPMV46Adapter
    from areno.models.qwen3_5.model import Qwen35VLAdapter

    hf_config = _config()

    assert MiniCPMV46Adapter().match_hf_config(hf_config)
    assert not Qwen35VLAdapter().match_hf_config(hf_config)


def test_minicpmv46_projects_target_sizes_to_visual_embeddings():
    pytest.importorskip("triton")
    from areno.models.minicpmv46.model import MiniCPMV46Adapter

    adapter = MiniCPMV46Adapter()
    model = adapter.build(adapter.config_from_hf(_config())).float()
    features = {
        "pixel_values": torch.zeros(1, 3, 2, 32),
        "target_sizes": torch.tensor([[4, 4]], dtype=torch.int32),
        "image_token_id": 99,
    }

    projected = model._project_pixel_values(features, torch.device("cpu"), batch=1)

    assert projected["image_embeds"].shape == (4, 32)


def test_minicpmv46_decode_skips_multimodal_helpers_without_features(monkeypatch):
    pytest.importorskip("triton")
    from areno.models.minicpmv46.model import MiniCPMV46ForCausalLM

    model = MiniCPMV46ForCausalLM.__new__(MiniCPMV46ForCausalLM)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Embedding(16, 4)
    model.layers = nn.ModuleList()
    model.norm = nn.Identity()
    model.lm_head = nn.Identity()

    def unexpected_helper(self, *args):
        del self, args
        raise AssertionError("multimodal helpers must be skipped for features=None")

    monkeypatch.setattr(model, "_project_pixel_values", MethodType(unexpected_helper, model))
    monkeypatch.setattr(model, "_apply_multimodal_features", MethodType(unexpected_helper, model))

    output = model(torch.tensor([[1, 2]], dtype=torch.long), features=None)

    assert output.logits_shard.shape == (1, 2, 4)


def test_minicpmv46_vision_checkpoint_keys_cover_tower_and_merger():
    pytest.importorskip("triton")
    from areno.models.minicpmv46.checkpoint import _load_vision_weights
    from areno.models.minicpmv46.model import MiniCPMV46Adapter

    adapter = MiniCPMV46Adapter()
    model = adapter.build(adapter.config_from_hf(_config())).float()
    source = {
        f"model.{module_name}.{name}": torch.full_like(parameter, 0.25)
        for module_name, module in (("vision_tower", model.vision_tower), ("merger", model.merger))
        for name, parameter in module.named_parameters()
    }

    class Index:
        weight_map = source

        def get_tensor(self, key):
            return self.weight_map[key]

    _load_vision_weights(model, Index())

    assert all(
        torch.equal(parameter, torch.full_like(parameter, 0.25)) for parameter in model.vision_tower.parameters()
    )
    assert all(torch.equal(parameter, torch.full_like(parameter, 0.25)) for parameter in model.merger.parameters())


def test_minicpmv46_gdn_uses_configured_key_and_value_heads():
    pytest.importorskip("triton")
    from areno.models.minicpmv46.model import MiniCPMV46Adapter

    config_data = _config()
    config_data["text_config"]["layer_types"] = ["linear_attention"]
    model = MiniCPMV46Adapter().build(MiniCPMV46Adapter().config_from_hf(config_data))
    model.set_kv_caches([], num_slots=3)
    attention = model.layers[0].attention

    assert attention.num_key_heads == 2
    assert attention.num_value_heads == 4
    assert tuple(attention.state_cache.shape) == (3, 4, 4, 4)
    assert tuple(attention.conv_cache.shape) == (3, 32, 3)


def test_minicpmv46_window_merger_downsamples_visual_tokens():
    pytest.importorskip("triton")
    from areno.models.minicpmv46.model import MiniCPMV46Adapter

    adapter = MiniCPMV46Adapter()
    config = adapter.config_from_hf(_config(insert_layer_id=0))
    config.vision_config["num_hidden_layers"] = 2
    model = adapter.build(config).float()
    features = {
        "pixel_values": torch.zeros(1, 3, 2, 32),
        "target_sizes": torch.tensor([[4, 4]], dtype=torch.int32),
        "image_token_id": 99,
    }

    projected = model._project_pixel_values(features, torch.device("cpu"), batch=1)

    assert projected["image_embeds"].shape == (1, 32)


def test_minicpmv46_rollout_chunk_keeps_processor_fields():
    mask, payload = _slice_prompt_image_features(
        {
            "pixel_values": torch.zeros(1, 3, 2, 32),
            "target_sizes": torch.tensor([[4, 4]], dtype=torch.int32),
            "num_patches_per_image": [1],
            "image_token_id": 99,
        },
        [1, 99, 2],
        0,
        3,
    )

    assert mask == [False, True, False]
    assert payload is not None
    assert payload["target_sizes"].tolist() == [[4, 4]]
    assert payload["num_patches_per_image"] == [1]


def test_minicpmv46_prefill_uses_dense_recurrent_slots():
    state = InferenceBatchState(
        [[1, 2], [3, 4]],
        max_new_tokens=1,
        max_running_seqs=2,
        max_cache_len=8,
        kv_block_size=1,
        num_cache_blocks=8,
    )

    payload = state.build_prefill_payload()
    meta = payload_to_infer_meta(payload, torch.device("cpu"))

    assert payload["block_table"][:, 0].tolist() == [0, 2]
    assert meta.recurrent_slots.tolist() == [0, 1]


def test_minicpmv46_token_count_handles_processor_expansion():
    features = {
        "target_sizes": torch.tensor([[4, 4], [4, 4]], dtype=torch.int32),
        "num_patches_per_image": [2],
    }

    assert image_token_counts_from_features(features) == [2]
    features["processor_expanded_image_tokens"] = True
    assert image_token_counts_from_features(features) == []


def test_minicpmv46_processor_expanded_tokens_are_not_repeated():
    image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP8z8BQDwAFgwJ/lwJw6QAAAABJRU5ErkJggg=="

    class Processor:
        image_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            del messages, kwargs
            return "<image> question"

        def __call__(self, *, text, images, return_tensors):
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([[1, 99, 2]]),
                "pixel_values": torch.zeros(1, 3, 2, 32),
                "target_sizes": torch.tensor([[4, 4]], dtype=torch.int32),
                "num_patches_per_image": [1],
            }

    tokens, features = encode_multimodal_prompt(
        tokenizer=object(),
        processor=Processor(),
        record={"prompt": "question", "image_base64": image},
    )

    assert tokens == [1, 99, 2]
    assert features["processor_expanded_image_tokens"] is True


@pytest.mark.parametrize("downsample_mode, expected", [("16x", 1), ("4x", 4)])
def test_minicpmv46_downsample_mode_controls_token_count(downsample_mode: str, expected: int):
    pytest.importorskip("triton")
    from areno.models.minicpmv46.model import MiniCPMV46Adapter

    adapter = MiniCPMV46Adapter()
    config = adapter.config_from_hf(_config(insert_layer_id=0))
    config.vision_config["num_hidden_layers"] = 2
    model = adapter.build(config).float()
    features = {
        "pixel_values": torch.zeros(1, 3, 2, 32),
        "target_sizes": torch.tensor([[4, 4]], dtype=torch.int32),
        "downsample_mode": downsample_mode,
        "image_token_id": 99,
    }

    projected = model._project_pixel_values(features, torch.device("cpu"), batch=1)

    assert projected["image_embeds"].shape == (expected, 32)
