"""MLX math reference (runs on Apple Silicon, skips where MLX is unavailable)."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from areno.api.jev.labels.mlx import label_loss, prepare_batch  # noqa: E402
from areno.api.models import TrainSequence  # noqa: E402


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.array([[1.0, -2.0, 3.0, 0.5], [-1.0, 2.0, 0.2, 3.0]])

    def __call__(self, tokens):
        return self.weight[tokens]


def test_label_jev_mlx_reference_cpu():
    from types import SimpleNamespace

    model = Model()
    rows = [
        TrainSequence(tokens=[0], features={"label_jev": {"ids": [0, 2], "target": [0.25, 0.75], "temperature": 0.7}}),
        TrainSequence(
            tokens=[1], features={"label_jev": {"ids": [1, 0, 3], "target": [0.0, 1.0, 0.0], "temperature": 1.3}}
        ),
    ]
    batch = prepare_batch({"input_ids": mx.array([[0], [1]])}, rows)
    provider = SimpleNamespace(model=model)
    (loss, stats), grads = nn.value_and_grad(model, lambda m: label_loss(provider, batch))(model)
    scores = np.array([[1.0, -2.0, 3.0, 0.5], [-1.0, 2.0, 0.2, 3.0]], dtype=np.float32)
    expected_grad = np.zeros_like(scores)
    expected_loss = 0.0
    for i, (ids, target, temp) in enumerate([([0, 2], [0.25, 0.75], 0.7), ([1, 0, 3], [0.0, 1.0, 0.0], 1.3)]):
        values = scores[i, ids] / temp
        log_probs = values - np.logaddexp.reduce(values)
        expected_loss -= np.dot(target, log_probs) / 2
        expected_grad[i, ids] = (np.exp(log_probs) - np.array(target)) / temp / 2
    np.testing.assert_allclose(float(loss), expected_loss, atol=1e-6, rtol=1e-4)
    np.testing.assert_allclose(np.array(grads["weight"]), expected_grad, atol=1e-6, rtol=1e-4)
    assert 0 < float(stats["label_mass"]) < 1
