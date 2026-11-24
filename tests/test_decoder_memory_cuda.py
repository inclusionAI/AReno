"""Exercise prefill row selection through each bundled CUDA adapter's forward."""

from unittest.mock import patch

import pytest
import torch

from areno.engine.config import ModelConfig
from areno.engine.runtime.metadata import TrainMeta
from areno.models.registry import build_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

FAMILIES = (
    "bailing_moe_linear_v2",
    "bailing_moe_v3",
    "gemma4",
    "gpt_oss",
    "llama",
    "minicpmv46",
    "olmo2",
    "phi4mm",
    "qwen3",
    "qwen3_moe",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_vl",
    "qwen3_5_vl_moe",
)


def test_all_bundled_adapters_have_prefill_coverage():
    from areno.models.registry import _ADAPTERS, load_model_plugins

    load_model_plugins()
    assert {key for key, adapter in _ADAPTERS.items() if type(adapter).__module__.startswith("areno.models.")} == set(
        FAMILIES
    )


def _case(family):
    config = ModelConfig(
        model_type=family,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        dtype=torch.float32,
        layer_types=("full_attention",),
        max_position_embeddings=128,
        sequence_parallel=False,
    )
    ids = [1, 5, 3, 2]
    media = {}
    if family in {"qwen3_moe", "qwen3_5_moe", "qwen3_5_vl_moe", "gpt_oss"}:
        config.num_experts = 4
        config.num_experts_per_tok = 2
        config.moe_intermediate_size = config.shared_expert_intermediate_size = 64
        config.dtype = torch.bfloat16
    if family == "bailing_moe_linear_v2":
        config.qk_nope_head_dim = config.qk_rope_head_dim = config.head_dim // 2
    if family == "gpt_oss":
        config.hf_text_config = {}
        config.sliding_window = 4
    if family == "bailing_moe_v3":
        config.qk_nope_head_dim = config.qk_rope_head_dim = config.v_head_dim = config.kv_lora_rank = 16
    if family in {"qwen3_5_vl", "qwen3_5_vl_moe"}:
        from tests.test_qwen35_multimodal_cpu import _qwen35_vision_config

        config.vision_config = {**_qwen35_vision_config(), "out_hidden_size": config.hidden_size}
        config.image_token_id = 5
        media = {"image_token_id": 5, "pixel_values": torch.randn(4, 12), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    if family == "minicpmv46":
        from tests.test_minicpmv46_multimodal_cpu import _config

        config.vision_config = {**_config()["vision_config"], "insert_layer_id": 6}
        config.image_token_id = 5
        ids = [1, 5, 5, 5, 5, 2]
        media = {"image_token_id": 5, "pixel_values": torch.randn(1, 3, 2, 32), "target_sizes": torch.tensor([[4, 4]])}
    if family == "phi4mm":
        from tests.test_phi4mm_audio_cpu import _tiny_model_config

        config = _tiny_model_config()
        config.audio_token_id = 5
        media = {
            "audio_token_id": 5,
            "input_mode": 2,
            "input_audio_embeds": torch.randn(1, 8, 80),
            "audio_attention_mask": torch.ones(1, 8, dtype=torch.bool),
            "audio_embed_sizes": torch.tensor([1]),
        }
    if family == "gemma4":
        config.final_logit_softcapping = 0.5
        config.image_token_id, config.video_token_id = 5, 3
        config.hf_text_config = {"hidden_size": config.hidden_size}
        config.vision_config = dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            patch_size=2,
            pooling_kernel_size=2,
            position_embedding_size=16,
        )
        grid = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]])
        media = {
            "pixel_values": torch.randn(1, 4, 12),
            "image_position_ids": grid,
            "pixel_values_videos": torch.randn(1, 1, 4, 12),
            "video_position_ids": grid.unsqueeze(1),
        }
    kwargs = (
        {"features": {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in media.items()}} if media else {}
    )
    return build_model(config).to(device="cuda", dtype=config.dtype).eval(), torch.tensor([ids], device="cuda"), kwargs


@pytest.mark.parametrize("family", FAMILIES)
def test_selected_prefill_and_offloaded_decoder_backward(family, monkeypatch):
    torch.manual_seed(437)
    model, tokens, kwargs = _case(family)
    indices = torch.tensor([tokens.size(1) - 1, 1, 1], device=tokens.device)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.15, 0.15)
        if hasattr(model, "prepare_infer_weights"):
            model.prepare_infer_weights()
        assert model.supports_prefill_logits_indices
        full = model(tokens, **kwargs)
        selected = model(tokens, logits_indices=indices, **kwargs)
        torch.testing.assert_close(selected.logits_shard, full.logits_shard.index_select(1, indices))
        torch.testing.assert_close(selected.hidden_states, full.hidden_states)
        if "features" in kwargs:
            media = kwargs["features"]
            muted = {
                k: torch.zeros_like(v) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in media.items()
            }
            assert not torch.equal(model(tokens, features=muted).hidden_states, full.hidden_states)
        deferred = model(tokens, defer_lm_head=True, **kwargs)
        assert deferred.logits_shard is None
        torch.testing.assert_close(deferred.hidden_states, full.hidden_states)

    model.train()
    gradients = []
    for setting in ("0", "1"):
        monkeypatch.setenv("ARENO_OFFLOAD_ACTIVATIONS", setting)
        model.zero_grad(set_to_none=True)
        with patch.object(torch.autograd.graph, "save_on_cpu", wraps=torch.autograd.graph.save_on_cpu) as save:
            output = model(tokens, train_meta=TrainMeta(), **kwargs)
        assert save.call_count == int(setting)
        output.logits_shard[:, -1].float().square().mean().backward()
        gradients.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert gradients[0].keys() == gradients[1].keys()
    assert any(g.count_nonzero() for n, g in gradients[0].items() if "layers." in n)
    for n, expected in gradients[0].items():
        torch.testing.assert_close(gradients[1][n], expected)
