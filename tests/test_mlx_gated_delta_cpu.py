"""Parallel recurrence must preserve outputs, final states and every input VJP."""

import numpy as np
import pytest


@pytest.mark.parametrize("vector_gate", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("length", [7, 19, 37])
def test_parallel_delta_matches_reference_value_state_and_gradients(vector_gate, masked, length):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.gated_delta import gated_delta_ops as installed

    gated_delta_ops = getattr(installed, "_areno_reference", installed)

    from areno.api.backend.mlx.gated_delta import gated_delta_parallel

    with mx.stream(mx.cpu):
        mx.random.seed(19)
        q = mx.random.normal((2, length, 1, 5)) * 0.2
        k = mx.random.normal((2, length, 1, 5))
        k = k / mx.sqrt((k * k).sum(-1, keepdims=True) + 1e-8)
        v = mx.random.normal((2, length, 2, 4))
        gate_shape = (2, length, 2, 5) if vector_gate else (2, length, 2)
        g = mx.sigmoid(mx.random.normal(gate_shape))
        beta = mx.sigmoid(mx.random.normal((2, length, 2)))
        initial = mx.random.normal((2, 2, 4, 5)) * 0.1
        mask = (mx.arange(length)[None, :] % 3 != 1) & mx.ones((2, 1), dtype=mx.bool_) if masked else None
        weights = mx.random.normal((2, length, 2, 4))
        state_weights = mx.random.normal(initial.shape)
        inputs = (q, k, v, g, beta, initial)

        def objective(fn, *values):
            output, state = fn(*values, mask=mask)
            return (output * weights).sum() + (state * state_weights).sum()

        expected = gated_delta_ops(*inputs, mask=mask)
        actual = gated_delta_parallel(*inputs, mask=mask)
        mx.eval(expected, actual)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(np.array(a), np.array(b), atol=2e-5, rtol=2e-4)
        expected_grads = mx.grad(lambda *x: objective(gated_delta_ops, *x), argnums=tuple(range(6)))(*inputs)
        actual_grads = mx.grad(lambda *x: objective(gated_delta_parallel, *x), argnums=tuple(range(6)))(*inputs)
        mx.eval(expected_grads, actual_grads)
        for a, b in zip(actual_grads, expected_grads, strict=True):
            np.testing.assert_allclose(np.array(a), np.array(b), atol=4e-5, rtol=4e-4)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_parallel_delta_handles_zero_gates_and_extreme_decay(dtype):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.gated_delta import gated_delta_ops as installed

    gated_delta_ops = getattr(installed, "_areno_reference", installed)

    from areno.api.backend.mlx.gated_delta import gated_delta_parallel

    with mx.stream(mx.cpu):
        mx.random.seed(31)
        q = (mx.random.normal((1, 35, 2, 8)) * 0.2).astype(getattr(mx, dtype))
        k = mx.random.normal(q.shape)
        k = (k / mx.sqrt((k * k).sum(-1, keepdims=True) + 1e-8)).astype(q.dtype)
        v = mx.random.normal((1, 35, 2, 6)).astype(q.dtype)
        g = mx.exp(-mx.random.uniform(shape=(1, 35, 2, 8)) * 5)
        g[:, 4, :, 2] = 0
        beta = mx.sigmoid(mx.random.normal((1, 35, 2)))
        initial = mx.random.normal((1, 2, 6, 8))
        inputs = (q, k, v, g, beta, initial)
        expected = gated_delta_ops(*inputs)
        actual = gated_delta_parallel(*inputs, chunk_size=8)
        mx.eval(expected, actual)
        tolerance = 2e-5 if dtype == "float32" else 1e-2
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)), atol=tolerance, rtol=tolerance
            )

        def objective(fn, *x):
            y, state = fn(*x)
            return y.astype(mx.float32).sum() + state.sum()

        expected_grads = mx.grad(lambda *x: objective(gated_delta_ops, *x), argnums=tuple(range(6)))(*inputs)
        actual_grads = mx.grad(lambda *x: objective(gated_delta_parallel, *x), argnums=tuple(range(6)))(*inputs)
        mx.eval(expected_grads, actual_grads)
        for a, b in zip(actual_grads, expected_grads, strict=True):
            assert bool(mx.isfinite(a).all().item())
            np.testing.assert_allclose(
                np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)), atol=tolerance, rtol=tolerance
            )


def test_parallel_delta_is_stable_for_correlated_keys_and_beta_near_two():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.gated_delta import gated_delta_ops as installed

    gated_delta_ops = getattr(installed, "_areno_reference", installed)

    from areno.api.backend.mlx.gated_delta import gated_delta_parallel

    with mx.stream(mx.cpu):
        mx.random.seed(71)
        k = mx.ones((1, 45, 1, 8)) / (8**0.5)
        q = mx.random.normal(k.shape) * 0.2
        v = mx.random.normal((1, 45, 1, 6))
        g = mx.ones((1, 45, 1, 8))
        beta = mx.full((1, 45, 1), 1.9)
        expected = gated_delta_ops(q, k, v, g, beta)
        actual = gated_delta_parallel(q, k, v, g, beta)
        mx.eval(expected, actual)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-4, rtol=1e-4)


def test_operator_installation_is_idempotent_and_preserves_cpu_fallback(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models import gated_delta

    from areno.api.backend.mlx.gated_delta import install_parallel_gated_delta

    calls = []

    def original(q, k, v, g, beta, state=None, mask=None):
        calls.append((q, state, mask))
        return "reference"

    monkeypatch.setattr(gated_delta, "gated_delta_ops", original)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    install_parallel_gated_delta()
    installed = gated_delta.gated_delta_ops
    install_parallel_gated_delta()
    assert gated_delta.gated_delta_ops is installed
    assert installed._areno_reference is original
    assert installed("q", None, None, None, None, "state", "mask") == "reference"
    assert calls == [("q", "state", "mask")]


def test_training_allocator_restores_cache_policy_after_failure(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from areno.api.backend.mlx.training import training_allocator_cache

    current, cleared = [1234], []

    def set_limit(value):
        old, current[0] = current[0], value
        return old

    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"max_recommended_working_set_size": 32 * 1024**3})
    monkeypatch.setattr(mx, "set_cache_limit", set_limit)
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(True))
    with pytest.raises(RuntimeError, match="training failed"):
        with training_allocator_cache():
            assert current[0] == 512 * 1024**2
            raise RuntimeError("training failed")
    assert current[0] == 1234
    assert cleared == [True]
