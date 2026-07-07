from __future__ import annotations

import pytest
import torch

from areno.api import TrainSequence
from areno.api.backend.areno.backend import _make_train_pack
from areno.api.multimodal import (
    encode_multimodal_prompt,
    expand_image_tokens,
    image_token_counts_from_features,
    mrope_position_ids_from_image_grid,
)
from areno.api.tokenizer import configure_chat_template_enable_thinking
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


def _qwen35_moe_text_config() -> dict:
    config = _qwen35_text_config()
    config.update(
        {
            "model_type": "qwen3_5_moe",
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "norm_topk_prob": True,
        }
    )
    return config


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


def test_multimodal_prompt_passes_tools_to_processor_chat_template():
    image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP8z8BQDwAFgwJ/lwJw6QAAAABJRU5ErkJggg=="

    class Processor:
        image_token_id = 99

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None):
            self.messages = messages
            self.kwargs = {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "tools": tools,
            }
            return "<image> choose"

        def __call__(self, *, text, images, return_tensors):
            self.call_args = (text, len(images), return_tensors)
            return {
                "input_ids": torch.tensor([[1, 99, 2]]),
                "image_embeds": torch.ones(1, 4),
            }

    tools = [{"type": "function", "function": {"name": "choose_square"}}]
    processor = Processor()
    tokens, features = encode_multimodal_prompt(
        tokenizer=object(),
        processor=processor,
        record={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                        {"type": "text", "text": "choose"},
                    ],
                }
            ],
            "tools": tools,
            "images_base64": [image],
        },
    )

    assert tokens == [1, 99, 2]
    assert features["image_token_id"] == 99
    assert processor.kwargs["tools"] == tools
    assert processor.messages[0]["content"][0]["type"] == "image"
    assert processor.call_args == (["<image> choose"], 1, "pt")


def test_multimodal_prompt_passes_disable_thinking_to_processor_chat_template():
    image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP8z8BQDwAFgwJ/lwJw6QAAAABJRU5ErkJggg=="

    class Processor:
        image_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            del messages
            self.kwargs = dict(kwargs)
            return "<image> choose"

        def __call__(self, *, text, images, return_tensors):
            del text, images, return_tensors
            return {
                "input_ids": torch.tensor([[1, 99, 2]]),
                "image_embeds": torch.ones(1, 4),
            }

    processor = Processor()
    configure_chat_template_enable_thinking(processor, False)

    encode_multimodal_prompt(
        tokenizer=object(),
        processor=processor,
        record={
            "prompt": "choose",
            "image_base64": image,
        },
    )

    assert processor.kwargs["enable_thinking"] is False


def test_packed_train_collates_multimodal_features():
    first_embeds = torch.ones(1, 16)
    second_embeds = torch.ones(2, 16) * 2
    pack = {
        "input_ids": torch.tensor([[1, 99, 2, 0], [4, 99, 99, 5]]),
        "lengths": torch.tensor([3, 4], dtype=torch.int32),
        "prompt_mask": torch.tensor([[True, True, False, False], [True, True, True, False]]),
        "logprobs": torch.zeros(2, 4),
        "advantages": torch.zeros(2, 4),
        "features": [
            {
                "image_token_id": 99,
                "image_embeds": first_embeds,
                "mrope_position_ids": torch.tensor([[0, 1], [0, 2], [0, 3]]),
            },
            {
                "image_token_id": 99,
                "image_embeds": second_embeds,
                "mrope_position_ids": torch.tensor([[0, 1, 1], [0, 1, 2], [0, 1, 3]]),
            },
        ],
    }

    packed = _pack_train_data(pack)

    assert packed is not pack
    assert packed["input_ids"].tolist() == [[1, 99, 2, 4, 99, 99, 5]]
    assert packed["train_cu_seqlens"].tolist() == [0, 3, 7]
    assert packed["features"]["image_token_mask"].tolist() == [
        False,
        True,
        False,
        False,
        True,
        True,
        False,
    ]
    assert packed["features"]["image_feature_rows"][0]["image_embeds"] is first_embeds
    assert packed["features"]["image_feature_rows"][1]["image_embeds"] is second_embeds
    assert packed["features"]["mrope_position_ids"].tolist() == [
        [0, 1, 4, 0, 1, 1, 4],
        [0, 2, 4, 0, 1, 2, 4],
        [0, 3, 4, 0, 1, 3, 4],
    ]


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


def test_qwen35_moe_vl_adapter_matches_vision_moe_text_config():
    pytest.importorskip("triton")
    from areno.models.qwen3_5.model import (
        Qwen35MoeAdapter,
        Qwen35MoeForCausalLM,
        Qwen35MoeVLAdapter,
        Qwen35MoeVLForConditionalGeneration,
        Qwen35VLAdapter,
    )

    hf_config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5MoeVLForConditionalGeneration"],
        "text_config": _qwen35_moe_text_config(),
        "vision_config": _qwen35_vision_config(),
        "image_token_id": 99,
    }

    adapter = Qwen35MoeVLAdapter()
    assert adapter.match_hf_config(hf_config)
    assert not Qwen35VLAdapter().match_hf_config(hf_config)
    assert not Qwen35MoeAdapter().match_hf_config(hf_config)

    config = adapter.config_from_hf(hf_config)
    assert config.model_type == "qwen3_5_vl_moe"
    assert config.enable_moe_block is True
    assert config.num_experts == 4
    assert config.vision_config["out_hidden_size"] == 16
    assert config.image_token_id == 99

    model = adapter.build(config)
    assert isinstance(model, Qwen35MoeVLForConditionalGeneration)
    assert isinstance(model.language_model, Qwen35MoeForCausalLM)


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


def test_qwen35_image_grid_builds_expected_mrope_positions():
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


def test_qwen35_vision_merger_uses_exact_gelu_not_hidden_act():
    pytest.importorskip("triton")
    from areno.models.qwen3_5.model import Qwen35VisionMerger

    merger = Qwen35VisionMerger(_qwen35_vision_config(), torch.float32)
    hidden_states = torch.linspace(-1, 1, steps=32, dtype=torch.float32).view(4, 8)

    with torch.no_grad():
        normed = merger.norm(hidden_states).view(1, -1)
        expected = merger.linear_fc2(torch.nn.functional.gelu(merger.linear_fc1(normed), approximate="none"))
        tanh_variant = merger.linear_fc2(torch.nn.functional.gelu(merger.linear_fc1(normed), approximate="tanh"))

    actual = merger(hidden_states)

    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, tanh_variant)


def test_qwen35_vision_merger_uses_merged_hidden_size_by_default():
    pytest.importorskip("triton")
    from areno.models.qwen3_5.model import Qwen35VisionMerger

    config = dict(_qwen35_vision_config())
    config["hidden_size"] = 1152
    config["intermediate_size"] = 4304
    config["out_hidden_size"] = 2048
    config["spatial_merge_size"] = 2

    merger = Qwen35VisionMerger(config, torch.float32)

    assert merger.linear_fc1.weight.shape == (4608, 4608)
    assert merger.linear_fc2.weight.shape == (2048, 4608)


def test_qwen35_vision_rotary_uses_hw_axis_order():
    pytest.importorskip("triton")
    from areno.models.qwen3_5.model import Qwen35VisionTransformer, _apply_vision_rotary

    config = dict(_qwen35_vision_config())
    config["hidden_size"] = 128
    config["num_heads"] = 2
    vision = Qwen35VisionTransformer(config, torch.float32)

    cos, sin = vision._rot_pos_emb([(1, 2, 2)], torch.device("cpu"), torch.float32)

    assert cos.shape == (4, 32)
    assert sin.shape == (4, 32)
    assert torch.allclose(cos[1, :16], torch.ones(16))
    assert torch.allclose(sin[1, :16], torch.zeros(16))
    assert not torch.allclose(cos[1, 16:], torch.ones(16))

    q = torch.arange(4 * 2 * 64, dtype=torch.float32).view(4, 2, 64)
    rotated = _apply_vision_rotary(q, cos, sin)
    duplicated_cos = torch.cat((cos, cos), dim=-1)
    duplicated_sin = torch.cat((sin, sin), dim=-1)
    expected = (q * duplicated_cos[:, None, :]) + (
        torch.cat((-q[..., 32:], q[..., :32]), dim=-1) * duplicated_sin[:, None, :]
    )

    assert torch.equal(rotated, expected)


def test_qwen35_gdn_uses_projected_sequence_length_after_sp_gather():
    pytest.importorskip("triton")
    from areno.models.qwen3_5.model import Qwen35GatedDeltaNet

    class ConstantProjection(torch.nn.Module):
        def __init__(self, out_features: int):
            super().__init__()
            self.out_features = out_features

        def forward(self, hidden_states):
            return torch.zeros(hidden_states.shape[0], 12, self.out_features, dtype=hidden_states.dtype)

    class IdentityProjection(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    model = Qwen35GatedDeltaNet.__new__(Qwen35GatedDeltaNet)
    torch.nn.Module.__init__(model)
    model.local_key_heads = 2
    model.local_value_heads = 2
    model.head_k_dim = 4
    model.head_v_dim = 4
    model.local_key_dim = 8
    model.local_value_dim = 8
    model.in_proj_qkvz = ConstantProjection(32)
    model.in_proj_ba = ConstantProjection(4)
    model.out_proj = IdentityProjection()
    model.dt_bias = torch.nn.Parameter(torch.zeros(2))
    model.A_log = torch.nn.Parameter(torch.zeros(2))

    model._causal_conv = lambda x, train_meta, infer_meta: x
    model._forward_train = lambda query, key, value, g, beta, train_meta: value
    model._rmsnorm_gate = lambda out, z: out

    hidden_states = torch.zeros(2, 3, 16)
    out = model(hidden_states, torch.arange(12).unsqueeze(0).expand(2, -1), train_meta=None, infer_meta=None)

    assert out.shape == (2, 12, 8)


def test_qwen35_sequence_parallel_scatters_mrope_position_sequence_axis(monkeypatch):
    pytest.importorskip("triton")
    from types import SimpleNamespace

    import areno.models.qwen3_5.model as qwen35

    class RecorderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.position_ids = None

        def forward(self, hidden_states, position_ids, train_meta, infer_meta):
            del train_meta, infer_meta
            self.position_ids = position_ids
            return hidden_states

    def fake_scatter(x):
        return x[:, : x.shape[1] // 2].contiguous()

    monkeypatch.setattr(qwen35, "scatter_to_sequence_parallel_region", fake_scatter)

    model = qwen35.Qwen35ForCausalLM.__new__(qwen35.Qwen35ForCausalLM)
    torch.nn.Module.__init__(model)
    layer = RecorderLayer()
    model.layers = torch.nn.ModuleList([layer])
    model.norm = torch.nn.Identity()
    model.lm_head = torch.nn.Identity()

    hidden_states = torch.zeros(2, 8, 4)
    position_ids = torch.arange(3 * 2 * 8, dtype=torch.long).view(3, 2, 8)

    model.forward_from_embeddings(
        hidden_states,
        position_ids,
        train_meta=SimpleNamespace(sequence_parallel=True),
        infer_meta=None,
    )

    assert layer.position_ids.shape == (3, 2, 4)
    assert torch.equal(layer.position_ids, position_ids[:, :, :4])


def test_qwen35_full_attention_aligns_local_mrope_positions_to_projected_sequence(monkeypatch):
    pytest.importorskip("triton")
    from types import SimpleNamespace

    import areno.models.qwen3_5.model as qwen35

    monkeypatch.setattr(qwen35, "get_tp_context", lambda: SimpleNamespace(world_size=2))

    def fake_gather(x):
        return torch.cat([x, x + 100], dim=1)

    monkeypatch.setattr(qwen35, "gather_from_sequence_parallel_region", fake_gather)

    position_ids = torch.arange(3 * 2 * 4, dtype=torch.long).view(3, 2, 4)

    aligned = qwen35._align_position_ids_to_sequence_len(position_ids, 8)

    assert aligned.shape == (3, 2, 8)
    assert torch.equal(aligned[:, :, :4], position_ids)
    assert torch.equal(aligned[:, :, 4:], position_ids + 100)


def test_qwen35_feature_mrope_positions_continue_after_prompt_tail():
    pytest.importorskip("triton")
    import areno.models.qwen3_5.model as qwen35

    positions = torch.tensor(
        [
            [0, 1, 1, 2],
            [0, 1, 4, 5],
            [0, 1, 2, 3],
        ],
        dtype=torch.long,
    )

    full = qwen35._position_ids_from_features(
        [{"mrope_position_ids": positions}],
        batch=1,
        seqlen=7,
        device=torch.device("cpu"),
    )

    assert full.shape == (3, 1, 7)
    assert torch.equal(full[:, 0, :4], positions)
    assert full[:, 0, 4:].tolist() == [[6, 7, 8], [6, 7, 8], [6, 7, 8]]


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
