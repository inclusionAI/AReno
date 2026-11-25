"""Run on the MLX Mac; tiny real Qwen/Ling backbones, without release weights."""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from areno.api.jev.labels.mlx import label_loss, prepare_batch, prompt_logits  # noqa: E402
from areno.api.models import TrainSequence  # noqa: E402


def tiny_model(family):
    if family == "qwen":
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        return TextModel(
            TextModelArgs(
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=32,
                linear_num_key_heads=1,
                linear_num_value_heads=2,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                full_attention_interval=2,
                partial_rotary_factor=1.0,
            )
        )
    from mlx_lm.models.bailing_moe_linear import Model, ModelArgs

    return Model(
        ModelArgs(
            model_type="bailing_moe_linear",
            hidden_size=16,
            intermediate_size=24,
            max_position_embeddings=128,
            moe_intermediate_size=16,
            num_experts=2,
            num_shared_experts=0,
            norm_topk_prob=True,
            num_attention_heads=2,
            num_experts_per_tok=1,
            num_hidden_layers=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            vocab_size=32,
            first_k_dense_replace=2,
            layer_group_size=2,
            group_norm_size=2,
            topk_group=1,
        )
    )


@pytest.mark.parametrize("family", ["qwen", "ling"])
def test_bi_attn_mlx_real_model_padding_and_gradients(family, tmp_path):
    mx.random.seed(13)
    model = tiny_model(family)
    provider = SimpleNamespace(model=model)
    ids = mx.array([[1, 2, 3, 4], [5, 6, 0, 0]])
    out = prompt_logits(model, ids, [4, 2], bi_attn=True)
    # Padding does not change either real prompt; recurrent layers stay causal.
    solo = prompt_logits(model, mx.array([[5, 6]]), [2], bi_attn=True)
    np.testing.assert_allclose(np.array(out[1, 1]), np.array(solo[0, 1]), atol=2e-5, rtol=2e-4)
    changed = prompt_logits(model, mx.array([[1, 2, 3, 9]]), [4], bi_attn=True)
    assert not np.allclose(np.array(out[0, 0]), np.array(changed[0, 0]), atol=1e-6)
    causal = model(mx.array([[1, 2, 3, 4]]))
    causal_changed = model(mx.array([[1, 2, 3, 9]]))
    np.testing.assert_allclose(np.array(causal[0, 0]), np.array(causal_changed[0, 0]), atol=1e-6)
    rows = [
        TrainSequence(
            tokens=[1, 2, 3, 4],
            features={"label_jev": {"ids": [7, 8], "target": [0.3, 0.7], "temperature": 1.0, "bi_attn": True}},
        )
    ]
    batch = prepare_batch({"input_ids": ids[:1]}, rows)
    (loss, _), grads = nn.value_and_grad(model, lambda m: label_loss(provider, batch))(model)
    from mlx.utils import tree_flatten

    arrays = [v for _, v in tree_flatten(grads)]
    mx.eval(loss, arrays)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.array(a)).all() for a in arrays)
    assert any(np.any(np.array(a) != 0) for a in arrays)
    # A fresh call reconstructs prompt caches; no prior request KV survives.
    again = prompt_logits(model, ids, [4, 2], bi_attn=True)
    np.testing.assert_allclose(np.array(out), np.array(again), atol=1e-6)

    import mlx.optimizers as optim

    optimizer = optim.SGD(learning_rate=0.01)
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    updated = prompt_logits(model, ids, [4, 2], bi_attn=True)
    file = tmp_path / "model.safetensors"
    model.save_weights(str(file))
    restored = tiny_model(family)
    restored.load_weights(str(file))
    reloaded = prompt_logits(restored, ids, [4, 2], bi_attn=True)
    np.testing.assert_allclose(np.array(updated), np.array(reloaded), atol=1e-6)
