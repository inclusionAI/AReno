from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import areno.engine.runtime.logprobs as logprob_ops
from areno.engine.config import ModelConfig
from areno.engine.runtime.logprobs import packed_mtp_token_logprobs
from areno.engine.runtime.metadata import TrainMeta
from tests.helpers import PatchedContext, single_tp_context


def _log_softmax_at(logits_row: torch.Tensor, label: int) -> torch.Tensor:
    return torch.log_softmax(logits_row, dim=-1)[label]


class TestPackedMtpTokenLogprobs:
    def test_targets_token_two_ahead_and_masks_row_tails(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 6, 5)
        tokens = torch.tensor([0, 1, 2, 0, 2, 1])
        cu_seqlens = torch.tensor([0, 4, 6])

        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            logprobs, valid = packed_mtp_token_logprobs(logits, tokens, cu_seqlens)

        # Action axis: positions 0,1,2 (row of length 4) and 4 (row of length 2).
        assert logprobs.shape == (4,)
        # Each row's final action site has no in-row t+2 target.
        assert valid.tolist() == [True, True, False, False]
        assert torch.allclose(logprobs[0], _log_softmax_at(logits[0, 0], int(tokens[2])), atol=1e-5)
        assert torch.allclose(logprobs[1], _log_softmax_at(logits[0, 1], int(tokens[3])), atol=1e-5)

    def test_length_one_row_contributes_no_action_sites(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 4, 5)
        tokens = torch.tensor([3, 0, 1, 2])
        cu_seqlens = torch.tensor([0, 1, 4])

        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            logprobs, valid = packed_mtp_token_logprobs(logits, tokens, cu_seqlens)

        assert logprobs.shape == (2,)
        assert valid.tolist() == [True, False]
        assert torch.allclose(logprobs[0], _log_softmax_at(logits[0, 1], int(tokens[3])), atol=1e-5)


class TestTrainingMtpLoss:
    def test_mtp_loss_masks_row_boundaries_and_prompt_targets(self):
        from areno.engine.training import TrainingManager

        # Uniform logits: every selected logprob is exactly -log(vocab).
        logits = torch.zeros(1, 6, 5)
        tokens = torch.tensor([0, 1, 2, 0, 2, 1])
        data_pack = {
            "train_cu_seqlens": torch.tensor([0, 4, 6]),
            "packed_response_mask": torch.tensor([False, True, True, True]),
            "packed_seq_ids": torch.tensor([0, 0, 0, 1]),
        }

        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            loss = TrainingManager._mtp_loss(None, logits, tokens, data_pack)

        # Trainable MTP sites: action 0 (target = response token at action 1)
        # and action 1; actions 2 and 3 are row-final sites.
        assert torch.allclose(loss, torch.log(torch.tensor(5.0)), atol=1e-5)

    def test_resolved_scale_is_runtime_opt_in_only(self):
        from areno.engine.training import TrainingManager

        config = SimpleNamespace(
            model=SimpleNamespace(num_nextn_predict_layers=1, mtp_loss_scaling_factor=0.3),
            runtime=SimpleNamespace(mtp_loss_scale=None),
        )
        manager = SimpleNamespace(worker=SimpleNamespace(config=config))
        # The checkpoint's pretraining scale must never be inherited: the
        # auxiliary NLL is wrong for preference objectives like DPO.
        assert TrainingManager._resolved_mtp_loss_scale(manager) == 0.0
        config.runtime.mtp_loss_scale = 0.1
        assert TrainingManager._resolved_mtp_loss_scale(manager) == pytest.approx(0.1)
        config.model.num_nextn_predict_layers = 0
        assert TrainingManager._resolved_mtp_loss_scale(manager) == 0.0


def _tiny_config(**overrides) -> ModelConfig:
    defaults = dict(
        model_type="bailing_v3",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        dtype=torch.float32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        kv_lora_rank=8,
        layer_group_size=2,
        num_nextn_predict_layers=1,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _tiny_moe_config(**overrides) -> ModelConfig:
    return _tiny_config(
        num_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=8,
        num_shared_experts=1,
        shared_expert_intermediate_size=8,
        first_k_dense_replace=0,
        **overrides,
    )


def test_mtp_layer_uses_softmax_attention_and_bypasses_routing_replay():
    pytest.importorskip("fla")
    from areno.models.bailing_v3.model import BailingMTPLayer, BailingSoftmaxAttention

    config = _tiny_moe_config()
    layer = BailingMTPLayer(config, config.num_hidden_layers)
    assert isinstance(layer.attention, BailingSoftmaxAttention)
    assert layer.mlp.gate.routing_layer_slot is None
    for name in ("enorm", "hnorm", "eh_proj", "final_layernorm"):
        assert isinstance(getattr(layer, name), torch.nn.Module)


def test_model_builds_mtp_layers_only_when_configured():
    pytest.importorskip("fla")
    from areno.models.bailing_v3.model import BailingMoeV3ForCausalLM

    model = BailingMoeV3ForCausalLM(_tiny_moe_config())
    assert model.mtp_layers is not None and len(model.mtp_layers) == 1
    assert model.layers[0].mlp.gate.routing_layer_slot == 0

    without_mtp = BailingMoeV3ForCausalLM(_tiny_moe_config(num_nextn_predict_layers=0))
    assert without_mtp.mtp_layers is None


def test_model_rejects_multi_layer_mtp():
    pytest.importorskip("fla")
    from areno.models.bailing_v3.model import BailingMoeV3ForCausalLM

    with pytest.raises(ValueError, match="at most one MTP layer"):
        BailingMoeV3ForCausalLM(_tiny_moe_config(num_nextn_predict_layers=2))


def test_forward_emits_mtp_logits_only_when_enabled(monkeypatch):
    pytest.importorskip("fla")
    from areno.models.bailing_v3 import model as bailing_v3

    config = _tiny_config()
    model = bailing_v3.BailingMoeV3ForCausalLM(config)
    # The decoder block itself is covered by trunk tests; identity-patch it so
    # the CPU test exercises only the MTP wiring (roll, fuse, norms, lm_head).
    monkeypatch.setattr(
        bailing_v3.BailingDecoderLayer, "forward", lambda self, hidden, pos, train_meta, infer_meta=None: hidden
    )
    # The vocab-parallel embedding/head and fused RMSNorm kernels are
    # CUDA-only; swap in plain CPU equivalents.
    model.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    model.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    monkeypatch.setattr(
        bailing_v3.RMSNorm,
        "forward",
        lambda self, x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight,
    )
    tokens = torch.randint(0, config.vocab_size, (1, 6))
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)

    meta_on = TrainMeta(cu_seqlens=cu_seqlens, max_seqlen=4, packed=True, mtp_enabled=True)
    out = model(tokens, train_meta=meta_on)
    assert out.mtp_logits_shard is not None
    assert out.mtp_logits_shard.shape == out.logits_shard.shape
    out.mtp_logits_shard.sum().backward()
    assert model.mtp_layers[0].eh_proj.weight.grad is not None

    meta_off = TrainMeta(cu_seqlens=cu_seqlens, max_seqlen=4, packed=True, mtp_enabled=False)
    assert model(tokens, train_meta=meta_off).mtp_logits_shard is None


def test_checkpoint_spec_declares_mtp_extra_layers():
    pytest.importorskip("fla")
    from areno.engine.checkpoints.common import _extra_layer_lists
    from areno.models.bailing_v3.checkpoint import CHECKPOINT_SPEC, MTP_LAYER_SPEC
    from areno.models.bailing_v3.model import BailingMoeV3ForCausalLM

    assert CHECKPOINT_SPEC.extra_layers[0].attr == "mtp_layers"
    replicated_keys = {spec.key for spec in MTP_LAYER_SPEC.replicated}
    assert {"{prefix}.enorm.weight", "{prefix}.hnorm.weight", "{prefix}.eh_proj.weight"} <= replicated_keys

    model = BailingMoeV3ForCausalLM(_tiny_moe_config())
    lists = _extra_layer_lists(model, CHECKPOINT_SPEC)
    assert len(lists) == 1
    layers, layer_spec = lists[0]
    assert len(layers) == 1 and layer_spec is MTP_LAYER_SPEC

    without_mtp = BailingMoeV3ForCausalLM(_tiny_moe_config(num_nextn_predict_layers=0))
    assert _extra_layer_lists(without_mtp, CHECKPOINT_SPEC) == []


@pytest.mark.parametrize("rows", [1, 2])
def test_kda_conv_verify_is_contiguous_and_matches_causal_conv(rows: int):
    """The recurrent kernel reads q/k/v with contiguous head axes; a single row
    must not leave the verify conv output as a strided transpose view."""
    pytest.importorskip("fla")
    from areno.engine.runtime.metadata import InferMeta
    from areno.models.bailing_v3.model import BailingKDAAttention

    with PatchedContext(__import__("areno.models.bailing_v3.model", fromlist=["x"]), get_tp_context=single_tp_context):
        attention = BailingKDAAttention(_tiny_config(), 0)
    steps, channels, kernel = 3, attention.local_proj_dim, attention.conv_kernel_size
    torch.manual_seed(0)
    weight = torch.randn(channels, 1, kernel)
    history = torch.randn(rows, channels, kernel - 1)
    x = torch.randn(1, rows * steps, channels)
    meta = InferMeta(mode="decode", tokens_per_seq=steps)
    out = attention._causal_conv_verify(x, weight, 0, history, meta)
    assert out.is_contiguous() and out.shape == x.shape
    window = torch.cat((history, x.view(rows, steps, channels).transpose(1, 2)), dim=-1)
    expected = torch.nn.functional.silu(
        torch.stack(
            [(window[:, :, j : j + kernel] * weight.view(1, channels, kernel)).sum(-1) for j in range(steps)], -1
        )
    )
    assert torch.allclose(out.view(rows, steps, channels), expected.transpose(1, 2), atol=1e-5)
    assert torch.equal(meta.speculative_conv_windows[0, :, 0], window)
