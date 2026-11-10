"""Native MLX full-parameter/QAT checkpoints, optimizer resume, and serving."""

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


def make_source(path):
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.utils import save_config, save_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    args = ModelArgs(
        model_type="qwen3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        head_dim=16,
        tie_word_embeddings=True,
        rope_theta=10000.0,
    )
    mx.random.seed(42)
    model = Model(args)
    mx.eval(model.parameters())
    save_model(path, model)
    save_config(vars(args), config_path=path / "config.json")
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "state": 3, "a": 4, "b": 5, "ANSWER": 6},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(path)
    return tokenizer


@pytest.mark.parametrize("q4", [False, True])
def test_full_backbone_head_checkpoint_resume_and_typed_serving(tmp_path, q4):
    from mlx.utils import tree_flatten

    from areno.api.backend.mlx.backend import MlxBackend
    from areno.api.backend.mlx.decision import train_questions
    from areno.api.context import Context
    from areno.api.jev.config import JevMlxConfig
    from areno.api.jev.data import encode_question
    from areno.api.jev.scorer import DecisionScorer

    source = tmp_path / "source"
    tokenizer = make_source(source)
    source_weights = (source / "model.safetensors").read_bytes()

    def load(path, checkpoint=None):
        config = JevMlxConfig(
            model_path=str(path),
            decision_checkpoint=str(checkpoint) if checkpoint else None,
            quantization_bits=4 if q4 and checkpoint is None else None,
            gradient_checkpointing=False,
            seed=42,
            optimizer={"lr": 1e-3, "grad_clip_norm": 1.0},
            head_lr=1e-3,
        )
        ctx = Context(1, str(path), tokenizer, config)
        backend = MlxBackend()
        backend.initialize(ctx)
        return backend, ctx

    backend, ctx = load(source)
    resumed = scorer = None
    try:
        rows = encode_question(
            {"prompt": "state", "candidates": ["a ANSWER", "b ANSWER"], "target": [1.0, 0.0], "question_id": "q"},
            tokenizer,
            64,
        )
        ctx.step()
        first = train_questions(backend, ctx, rows, mini_bs=1)
        assert first["head_update_norm"] > 0 and first["backbone_update_norm"] > 0
        assert first["lora_update_norm"] == 0
        checkpoint = tmp_path / "checkpoint"
        backend.save_checkpoint(ctx, str(checkpoint))
        assert not any(key.startswith("score_head.") for key in mx.load(str(checkpoint / "model.safetensors")))
        assert (checkpoint / "score_head.safetensors").is_file()
        assert (checkpoint / "jev_optimizer.safetensors").is_file()
        if q4:
            assert json.loads((checkpoint / "config.json").read_text())["areno_qat"]["bits"] == 4
        resumed, resumed_ctx = load(checkpoint, checkpoint)
        assert resumed_ctx.global_step == ctx.global_step
        for name, value in tree_flatten(backend.model.parameters()):
            np.testing.assert_array_equal(
                np.array(value), np.array(dict(tree_flatten(resumed.model.parameters()))[name])
            )
        ctx.step()
        resumed_ctx.step()
        expected = train_questions(backend, ctx, rows, mini_bs=1)
        actual = train_questions(resumed, resumed_ctx, rows, mini_bs=1)
        assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-7)
        assert actual["loss"] < first["loss"]
        for name, value in tree_flatten(backend.model.parameters()):
            np.testing.assert_allclose(
                np.array(value),
                np.array(dict(tree_flatten(resumed.model.parameters()))[name]),
                atol=1e-7,
                rtol=1e-6,
            )
        scorer = DecisionScorer(str(source), str(checkpoint))
        answers, tokens = scorer.decide(
            "state",
            {"q": {"type": "choice", "instructions": "state", "criteria": {"a": "a", "b": "b"}}},
        )
        assert answers["q"]["choice"] in {"a", "b"} and tokens > 0
        assert sum(answers["q"]["probabilities"].values()) == pytest.approx(1.0)
        assert (source / "model.safetensors").read_bytes() == source_weights
    finally:
        backend.close()
        if resumed is not None:
            resumed.close()
        if scorer is not None:
            scorer.close()


@pytest.mark.parametrize("kind", ["linear", "embedding", "switch"])
def test_qat_matches_packed_forward_and_updates_floating_master(kind):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.switch_layers import SwitchLinear

    from areno.api.backend.mlx.qat import configure_qat

    mx.random.seed(7)
    leaf = {
        "linear": lambda: nn.Linear(64, 64),
        "embedding": lambda: nn.Embedding(64, 64),
        "switch": lambda: SwitchLinear(64, 64, 4),
    }[kind]()

    def apply(module):
        if kind == "linear":
            return module(mx.ones((2, 64)))
        if kind == "embedding":
            return module(mx.array([1, 2])) + module.as_linear(mx.ones((2, 64)))
        return module(mx.ones((2, 1, 64)), mx.array([[0], [2]], dtype=mx.uint32))

    master = np.array(leaf.weight)
    reference = apply(leaf.to_quantized(bits=4, group_size=64))
    model = nn.Sequential(leaf)
    names = [name for name, _ in tree_flatten(model.parameters())]
    configure_qat(model, {}, bits=4, group_size=64)
    assert [name for name, _ in tree_flatten(model.parameters())] == names
    np.testing.assert_array_equal(np.array(model.layers[0].weight), master)
    np.testing.assert_allclose(np.array(apply(model.layers[0])), np.array(reference), atol=2e-5, rtol=2e-5)
    _, gradients = nn.value_and_grad(model, lambda module: apply(module.layers[0]).sum())(model)
    assert float(mx.abs(gradients["layers"][0]["weight"]).sum().item()) > 0
    assert mx.issubdtype(model.layers[0].weight.dtype, mx.floating)
