"""MST parity across bundled decoder families, using actual CUDA kernels.

Small random models keep this separate from full-checkpoint performance probes.
Every comparison uses packed rows (including a one-token row), a ragged final
chunk, a signed loss, and all participating parameter gradients.
"""

import json

import pytest
import torch

from areno.engine.config import ModelConfig
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.engine.runtime.logprobs import packed_next_token_logprobs, packed_next_token_logprobs_from_hidden
from areno.engine.runtime.metadata import TrainMeta
from areno.models.registry import build_model

FAMILIES = (
    "qwen3",
    "llama",
    "olmo2",
    "qwen3_moe",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_vl",
    "qwen3_5_vl_moe",
    "bailing_moe_v3",
    "gemma4",
    "gemma4_moe",
    "minicpmv46",
    "phi4mm",
    "phi4mm_vision",
    "qwen3_5_hybrid",
    "qwen3_5_moe_hybrid",
    "minicpmv46_hybrid",
    "bailing_moe_v3_hybrid",
    "qwen3_compiled",
    "qwen3_5_vl_pixels",
    "qwen3_5_vl_moe_pixels",
    "minicpmv46_pixels",
    "phi4mm_vision_pixels",
    "gemma4_pixels",
    "gemma4_audio",
)


def native_lora_config(config, *, qlora=False):
    """Use the supported HF-style native adapter topology for each fixture."""
    from areno.adapters import LoraConfig

    if config.model_type == "qwen3_moe":
        config.enable_moe_block = True
    if config.model_type == "bailing_moe_v3":
        config.no_kda_lora = True
        config.kv_lora_rank = 64
        config.num_key_value_heads = config.num_attention_heads
        return LoraConfig(
            rank=8, alpha=16, qlora=qlora, target_modules=("q_proj", "dense", "gate_proj", "up_proj", "down_proj")
        )
    return LoraConfig(rank=8, alpha=16, qlora=qlora)


def tiny_config(family):
    audio = family.endswith("_audio")
    family = family.removesuffix("_audio")
    pixels = family.endswith("_pixels")
    family = family.removesuffix("_pixels")
    family = family.removesuffix("_compiled")
    hybrid = family.endswith("_hybrid")
    family = family.removesuffix("_hybrid")
    phi_vision = family == "phi4mm_vision"
    if phi_vision:
        family = "phi4mm"
    config = ModelConfig(
        model_type="gemma4" if family == "gemma4_moe" else family,
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        qk_norm=family != "phi4mm",
        dtype=torch.float32,
        sequence_parallel=False,
        attn_backend="native",
        layer_types=("full_attention",) * 2,
        num_experts=4,
        moe_intermediate_size=128,
        num_experts_per_tok=2,
        group_norm_size=64,
        first_k_dense_replace=1,
        num_shared_experts=1,
        shared_expert_intermediate_size=128,
        enable_moe_block=family == "gemma4_moe",
        tie_word_embeddings=family == "phi4mm",
        final_logit_softcapping=5.0 if family.startswith("gemma4") else None,
        image_token_id=127,
        hf_text_config={
            "original_max_position_embeddings": 256,
            "rope_scaling": {
                "type": "longrope",
                "short_factor": (1.0,) * 32,
                "long_factor": (1.0,) * 32,
            },
        },
    )
    if "_vl" in family:
        config.vision_config = dict(
            depth=1,
            hidden_size=16,
            hidden_act="gelu_pytorch_tanh",
            in_channels=3,
            intermediate_size=32,
            num_heads=2,
            num_position_embeddings=16,
            out_hidden_size=128,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=1,
        )
    if hybrid:
        config.layer_types = ("linear_attention", "full_attention")
        config.layer_group_size = 2
        config.linear_key_head_dim = config.linear_value_head_dim = 64
        config.linear_num_key_heads = config.linear_num_value_heads = 2
    if phi_vision:
        config.vision_config = dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=8,
            patch_size=2,
            feature_layer=-2,
            crop_size=8,
        )
        config.hf_text_config["vision_lora"] = {"r": 4, "lora_alpha": 8, "dp": 0.0}
    if pixels and family == "minicpmv46":
        config.vision_config = dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_channels=3,
            image_size=8,
            patch_size=2,
            layer_norm_eps=1e-6,
            window_kernel_size=[2, 2],
        )
    if family == "gemma4" and (pixels or audio):
        config.hf_text_config = dict(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=128,
            hidden_size_per_layer_input=0,
        )
        if pixels:
            config.vision_config = dict(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                pooling_kernel_size=2,
                patch_size=2,
                position_embedding_size=16,
            )
        else:
            config.audio_token_id = 126
            config.audio_config = dict(
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                # The first channel count also defines the expected mel width
                # in Gemma's subsampling projection (128 input features).
                subsampling_conv_channels=(128, 4),
                conv_kernel_size=3,
                attention_chunk_size=4,
                attention_context_left=4,
                output_proj_dims=16,
                use_clipped_linears=False,
            )
    return config


def multimodal_inputs(family, config, tokens):
    """Use real image encoders for pixel cases, projected inputs otherwise."""
    device = tokens.device
    features = None
    count = 2
    if family == "gemma4_audio":
        features = dict(
            input_features=torch.randn(1, 32, 128, device=device),
            input_features_mask=torch.ones(1, 32, device=device, dtype=torch.bool),
            modality_token_ids={"audio": config.audio_token_id},
        )
        tokens.masked_fill_(tokens == config.audio_token_id, 0)
        tokens.masked_fill_(tokens == config.image_token_id, 0)
        tokens[:, 3:11] = config.audio_token_id
        return features
    if family.endswith("_pixels"):
        features = {"image_token_id": config.image_token_id}
        if family.startswith("qwen3_5"):
            features.update(
                pixel_values=torch.randn(4, 12, device=device), image_grid_thw=torch.tensor([[1, 2, 2]], device=device)
            )
            count = 1
        elif family.startswith("minicpm"):
            features.update(
                pixel_values=torch.randn(1, 3, 2, 32, device=device),
                target_sizes=torch.tensor([[4, 4]], device=device, dtype=torch.int32),
            )
            count = 4
        elif family.startswith("gemma4"):
            features.update(
                pixel_values=torch.randn(1, 4, 12, device=device),
                image_position_ids=torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]], device=device),
                modality_token_ids={"image": config.image_token_id},
            )
            count = 1
        else:
            features.update(
                input_image_embeds=torch.randn(1, 2, 3, 8, 8, device=device),
                image_sizes=torch.tensor([[8, 8]], device=device),
                image_attention_mask=torch.ones(1, 2, 4, 4, device=device, dtype=torch.bool),
            )
            count = 13
    elif "_vl" in family or family.startswith("minicpmv46") or family == "phi4mm_vision":
        features = {
            "image_token_id": config.image_token_id,
            "image_embeds": torch.randn(2, config.hidden_size, device=device, requires_grad=True),
        }
    if features is not None:
        tokens.masked_fill_(tokens == config.image_token_id, 0)
        tokens[:, 3 : 3 + count] = config.image_token_id
    return features


def enable_image_training(model, family):
    if family.endswith(("_pixels", "_audio")) and hasattr(model, "configure_multimodal_training"):
        model.configure_multimodal_training(
            unfreeze_tower=True,
            unfreeze_projector=True,
            tower_lr=None,
            projector_lr=None,
            base_lr=1e-5,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real model kernels require CUDA")
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mst_model_forward_and_gradients_cuda(family, dtype, record_property):
    _check_model_forward_and_gradients(family, dtype, record_property)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native LoRA MST requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    ("family", "qlora"),
    [
        ("qwen3", False),
        ("qwen3_moe", False),
        ("qwen3_compiled", False),
        ("bailing_moe_v3", False),
        ("bailing_moe_v3_hybrid", False),
        ("qwen3", True),
        ("bailing_moe_v3", True),
        ("bailing_moe_v3_hybrid", True),
    ],
)
def test_native_lora_mst_forward_and_gradients_cuda(family, dtype, qlora, record_property):
    _check_model_forward_and_gradients(family, dtype, record_property, lora=True, qlora=qlora)


def _check_model_forward_and_gradients(family, dtype, record_property, *, lora=False, qlora=False):
    previous = get_tp_context()
    set_tp_context(TPContext(0, 1, torch.device("cuda", 0), None))
    try:
        torch.manual_seed(31)
        torch.set_float32_matmul_precision("highest")
        config = tiny_config(family)
        lora_config = native_lora_config(config, qlora=qlora) if lora else None
        config.dtype = dtype
        # Native layers inherit the default dtype during construction, as in
        # engine.modeling.build_model_on_device; setting ModelConfig alone
        # would leave some weights FP32 in an invalid BF16 fixture.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            model = build_model(config).cuda().train()
        finally:
            torch.set_default_dtype(previous_dtype)
        enable_image_training(model, family)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "norm" in name and param.ndim == 1:
                    param.fill_(1.0)
                else:
                    param.normal_(0, 0.02)
        registry = None
        if lora:
            from areno.adapters.lora import initialize_lora

            registry = initialize_lora(model, lora_config, seed=0)
            with torch.no_grad():
                for slot in registry.slots.values():
                    slot.lora_B.normal_(0, 0.001)
            if qlora:
                from areno.adapters.qlora import initialize_qlora

                initialize_qlora(model)
        if family.endswith("_compiled"):
            model = torch.compile(model)
        tokens = torch.randint(128, (1, 37), device="cuda")
        features = multimodal_inputs(family, config, tokens)
        cu = torch.tensor([0, 1, 18, 37], device="cuda", dtype=torch.int32)
        results = []
        for size in (0, 0, 8) if dtype == torch.bfloat16 else (0, 8):
            # Match stochastic media-tower layers across the two schedules.
            torch.manual_seed(31)
            model.zero_grad(set_to_none=True)
            if features is not None and "image_embeds" in features:
                features["image_embeds"].grad = None
            meta = TrainMeta(
                cu_seqlens=cu, max_seqlen=19, packed=True, activation_checkpointing=True, mst_chunk_size=size
            )
            kwargs = {} if features is None else {"features": features}
            output = model(tokens, train_meta=meta, defer_lm_head=bool(size), **kwargs)
            output.hidden_states.retain_grad()
            if size:
                assert output.logits_shard is None
                logps = packed_next_token_logprobs_from_hidden(
                    output.hidden_states,
                    tokens,
                    cu,
                    model.lm_head,
                    logit_softcap=getattr(model, "final_logit_softcapping", None),
                    chunk_size=size,
                )
            else:
                logps = packed_next_token_logprobs(output.logits_shard, tokens, cu)
            upstream = torch.linspace(-1, 1, logps.numel(), device="cuda")
            (logps * upstream).mean().backward()
            grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
            if registry is not None:
                assert all(p.grad is not None for p in registry.parameters())
            if features is not None and "image_embeds" in features:
                assert features["image_embeds"].grad is not None
                grads["image_embeds"] = features["image_embeds"].grad.detach().clone()
            assert grads
            assert all(torch.isfinite(g).all() for g in grads.values())
            if family.endswith(("_pixels", "_audio")):
                media_prefixes = (
                    ("visual.",)
                    if family.startswith("qwen3_5")
                    else ("model.embed_tokens_extend.",)
                    if family.startswith("phi4mm")
                    else ("audio_tower.",)
                    if family.endswith("_audio")
                    else ("vision_tower.",)
                )
                media_grads = {n: g for n, g in grads.items() if n.startswith(media_prefixes)}
                assert media_grads, "raw media must exercise the encoder backward"
                assert any(g.count_nonzero().item() for g in media_grads.values())
                record_property(f"media_gradients_checked_chunk_{size}", len(media_grads))
            results.append((logps.detach(), grads, output.hidden_states.grad.detach().clone()))
        expected, actual = results[0], results[-1]
        record_property("family", family)
        record_property("dtype", str(dtype))
        record_property("training", "qlora" if qlora else "lora" if lora else "full_parameter")
        record_property("logprob_max_abs", (actual[0] - expected[0]).abs().max().item())
        record_property("head_hidden_gradient_max_abs", (actual[2] - expected[2]).abs().max().item())
        record_property("head_hidden_gradient_equal", torch.equal(actual[2], expected[2]))
        record_property("gradients_checked", len(actual[1]))
        record_property("gradient_max_abs", max((g - expected[1][n]).abs().max().item() for n, g in actual[1].items()))
        bf16 = dtype == torch.bfloat16
        relative_gradient_error = (
            sum((g.float() - expected[1][n].float()).square().sum().item() for n, g in actual[1].items())
            / max(sum(g.float().square().sum().item() for g in expected[1].values()), 1e-30)
        ) ** 0.5
        record_property("gradient_relative_l2", relative_gradient_error)
        if bf16:
            repeat_grads = results[1][1]
            record_property(
                "repeat_gradient_max_abs",
                max((g.float() - expected[1][n].float()).abs().max().item() for n, g in repeat_grads.items()),
            )
            failures = {}
            for name, value in actual[1].items():
                reference = expected[1][name].float()
                value = value.float()
                if not torch.isclose(value, reference, atol=2e-4, rtol=0.03).all():
                    failures[name] = dict(
                        max_abs=(value - reference).abs().max().item(),
                        relative_l2=(value - reference).norm().item() / max(reference.norm().item(), 1e-30),
                        repeat_max_abs=(repeat_grads[name].float() - reference).abs().max().item(),
                    )
            record_property("gradient_pointwise_failures", json.dumps(failures))
        # Two-layer BF16 models incur shape-dependent GEMM rounding. Keep
        # their forward/gradient bounds explicit and separate from FP32;
        # a global norm gate prevents the absolute floor hiding small grads.
        torch.testing.assert_close(actual[0], expected[0], atol=0.01 if bf16 else 1e-5, rtol=0.002 if bf16 else 1e-5)
        assert actual[1].keys() == expected[1].keys()
        for name, grad in actual[1].items():
            torch.testing.assert_close(
                grad, expected[1][name], atol=2e-4 if bf16 else 2e-6, rtol=0.03 if bf16 else 3e-4, msg=name
            )
        assert relative_gradient_error < (0.03 if bf16 else 3e-4)
    finally:
        set_tp_context(previous)
