"""Numerical checks for the MLX OPD losses (Apple Silicon only).

Reference values are NumPy closed forms of the CUDA formulas, so these tests
need MLX but not Torch. ``importorskip`` keeps the x86 CPU suite green.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

STUDENT = np.array([[-0.5, -1.0, -10.0, -0.1], [-0.3, -0.7, -1.5, -0.2]], dtype=np.float32)
OLD = np.array([[-0.4, -8.0, -1.8, -0.1], [-0.3, -0.6, -1.4, -0.2]], dtype=np.float32)
TEACHER = np.array([[-0.2, -0.5, -0.5, -0.1], [-2.0, -0.6, -1.4, -0.2]], dtype=np.float32)
MASK = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.float32)
COUNT = float(MASK.sum())


def _opd_loss():
    from areno.api.backend.mlx.losses import opd_loss

    return opd_loss


def _batch(teacher=None, old=None, mask=None, student=None):
    t = TEACHER if teacher is None else teacher
    o = OLD if old is None else old
    m = MASK if mask is None else mask
    s = STUDENT if student is None else student
    batch = {
        "response_mask": mx.array(m),
        "old_logprobs": mx.array(o),
        "ref_logprobs": mx.array(t),
    }
    return batch, mx.array(s)


def _as_numpy(array):
    if hasattr(array, "tolist"):
        return np.asarray(array.tolist(), dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def _scalar(array):
    mx.eval(array)
    return float(array.item())


def test_pg_loss_and_stats_match_closed_form():
    batch, student = _batch()
    loss, stats = _opd_loss()(batch, student, opd_loss_type="pg", opd_clip=5.0)

    advantage = np.clip(TEACHER - OLD, -5.0, 5.0)
    expected = float(np.sum(-advantage * STUDENT * MASK) / COUNT)

    assert _scalar(loss) == pytest.approx(expected, rel=1e-4, abs=1e-5)
    assert _scalar(stats["opd_loss"]) == pytest.approx(expected, rel=1e-4, abs=1e-5)
    assert _scalar(stats["total_loss"]) == pytest.approx(expected, rel=1e-4, abs=1e-5)
    assert _scalar(stats["opd_advantage_mean"]) == pytest.approx(
        float(np.sum(advantage * MASK) / COUNT), rel=1e-4, abs=1e-5
    )
    assert _scalar(stats["opd_clipfrac"]) == pytest.approx(1.0 / COUNT, rel=1e-4, abs=1e-5)
    assert _scalar(stats["opd_teacher_student_gap"]) == pytest.approx(
        float(np.sum((TEACHER - STUDENT) * MASK) / COUNT), rel=1e-4, abs=1e-5
    )


def test_pg_gradient_matches_closed_form():
    from areno.api.backend.mlx.losses import opd_loss

    batch, _ = _batch()
    x = mx.array(STUDENT)
    grad = mx.grad(lambda value: opd_loss(batch, value, opd_loss_type="pg", opd_clip=5.0)[0])(x)
    mx.eval(grad)

    advantage = np.clip(TEACHER - OLD, -5.0, 5.0)
    expected = -advantage * MASK / COUNT
    np.testing.assert_allclose(_as_numpy(grad), expected, rtol=1e-4, atol=1e-5)


def test_k3_loss_and_stats_match_closed_form():
    batch, student = _batch()
    loss, stats = _opd_loss()(batch, student, opd_loss_type="k3")

    delta = np.clip(TEACHER - STUDENT, -20.0, 5.0)
    expected = float(np.sum((np.exp(delta) - delta - 1.0) * MASK) / COUNT)

    assert _scalar(loss) == pytest.approx(expected, rel=1e-4, abs=1e-4)
    assert _scalar(stats["opd_loss"]) == pytest.approx(expected, rel=1e-4, abs=1e-4)
    assert _scalar(stats["opd_advantage_mean"]) == pytest.approx(0.0, abs=1e-6)
    assert _scalar(stats["opd_clipfrac"]) == pytest.approx(0.0, abs=1e-6)


def test_low_var_kl_alias_matches_k3():
    batch, student = _batch()
    k3, _ = _opd_loss()(batch, student, opd_loss_type="k3")
    alias, _ = _opd_loss()(batch, student, opd_loss_type="low_var_kl")
    assert _scalar(alias) == pytest.approx(_scalar(k3), rel=1e-5, abs=1e-5)


def test_k3_gradient_is_zero_where_delta_is_clamped():
    from areno.api.backend.mlx.losses import opd_loss

    batch, _ = _batch()
    x = mx.array(STUDENT)
    grad = mx.grad(lambda value: opd_loss(batch, value, opd_loss_type="k3")[0])(x)
    mx.eval(grad)

    raw_delta = TEACHER - STUDENT
    inside = (raw_delta > -20.0) & (raw_delta < 5.0)
    delta = np.clip(raw_delta, -20.0, 5.0)
    expected = (1.0 - np.exp(delta)) * MASK * inside / COUNT

    np.testing.assert_allclose(_as_numpy(grad), expected, rtol=1e-4, atol=1e-4)
    # Token [0, 2] has delta 9.5 > 5: clamping the exponent input must stop
    # the gradient there, while the loss itself stays unbounded.
    assert raw_delta[0, 2] > 5.0
    assert _as_numpy(grad)[0, 2] == pytest.approx(0.0, abs=1e-5)


def test_pg_clip_limits_advantage_and_coef_scales_loss():
    batch, student = _batch()
    clipped, stats = _opd_loss()(batch, student, opd_loss_type="pg", opd_clip=0.1)

    advantage = np.clip(TEACHER - OLD, -0.1, 0.1)
    expected = float(np.sum(-advantage * STUDENT * MASK) / COUNT)
    assert _scalar(clipped) == pytest.approx(expected, rel=1e-4, abs=1e-5)
    assert _scalar(stats["opd_clipfrac"]) == pytest.approx(
        float(np.sum((np.abs(TEACHER - OLD) > 0.1) * MASK) / COUNT), rel=1e-4, abs=1e-5
    )

    scaled, _ = _opd_loss()(batch, student, opd_loss_type="pg", opd_clip=0.1, opd_coef=2.5)
    assert _scalar(scaled) == pytest.approx(2.5 * expected, rel=1e-4, abs=1e-5)


def test_padding_tokens_are_ignored():
    from areno.api.backend.mlx.losses import opd_loss

    baseline, student = _batch()
    baseline_loss, _ = opd_loss(baseline, student, opd_loss_type="k3")

    noisy_teacher = TEACHER.copy()
    noisy_old = OLD.copy()
    noisy_teacher[~MASK.astype(bool)] = 50.0
    noisy_old[~MASK.astype(bool)] = -50.0
    noisy_batch, _ = _batch(teacher=noisy_teacher, old=noisy_old)
    noisy_loss, _ = opd_loss(noisy_batch, student, opd_loss_type="k3")

    assert _scalar(noisy_loss) == pytest.approx(_scalar(baseline_loss), rel=1e-4, abs=1e-5)

    x = mx.array(STUDENT)
    grad = mx.grad(lambda value: opd_loss(noisy_batch, value, opd_loss_type="k3")[0])(x)
    mx.eval(grad)
    assert np.all(_as_numpy(grad)[~MASK.astype(bool)] == 0.0)


def test_missing_teacher_logprobs_is_rejected():
    batch, student = _batch()
    del batch["ref_logprobs"]
    with pytest.raises(ValueError, match="requires teacher logprobs"):
        _opd_loss()(batch, student)


def test_misaligned_teacher_logprobs_is_rejected():
    batch, student = _batch(teacher=np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="must align"):
        _opd_loss()(batch, student)


@pytest.mark.parametrize("loss_type", ["pg", "k3"])
def test_gradient_works_under_compiled_transform_with_traced_teacher(loss_type):
    """The teacher batch is a tracer inside mx.compile / nn.value_and_grad.

    An eager ``.item()`` on the teacher array raises "Attempting to eval an
    array during function transformations" here. That is what broke the first
    MLX smoke runs; the loss must stay free of eager evaluation.
    """
    from areno.api.backend.mlx.losses import opd_loss

    def objective(teacher, student):
        batch = {
            "response_mask": mx.array(MASK),
            "old_logprobs": mx.array(OLD),
            "ref_logprobs": teacher,
        }
        return opd_loss(batch, student, opd_loss_type=loss_type)[0]

    compiled = mx.compile(mx.grad(objective, argnums=1))
    grad = compiled(mx.array(TEACHER), mx.array(STUDENT))
    mx.eval(grad)

    if loss_type == "pg":
        advantage = np.clip(TEACHER - OLD, -5.0, 5.0)
        expected = -advantage * MASK / COUNT
    else:
        raw_delta = TEACHER - STUDENT
        inside = (raw_delta > -20.0) & (raw_delta < 5.0)
        delta = np.clip(raw_delta, -20.0, 5.0)
        expected = (1.0 - np.exp(delta)) * MASK * inside / COUNT
    np.testing.assert_allclose(_as_numpy(grad), expected, rtol=1e-4, atol=1e-4)


def test_unknown_loss_type_is_rejected():
    batch, student = _batch()
    with pytest.raises(ValueError, match="unsupported opd_loss_type"):
        _opd_loss()(batch, student, opd_loss_type="mse")
