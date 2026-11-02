from __future__ import annotations

import copy
import io
from types import SimpleNamespace

import pytest
import torch

from areno.api.trainer_config import TrainerConfig
from areno.engine.config import OptimizerConfig
from areno.engine.layers.linear import mark_linear_weight
from areno.engine.modeling import build_optimizer
from areno.engine.optim import AdamWFP32Master, GaLoreAdamW
from areno.engine.optim.galore import galore_blocks, top_singular_basis

# Acceptance tolerance for GaLore updates against the independent reference.
ATOL = 1.0e-6
RTOL = 1.0e-4


def _reference_basis(gradient: torch.Tensor, rank: int, side: str) -> torch.Tensor:
    """Top singular vectors from a float64 SVD, a different solver than the optimizer's."""

    left, _values, right_t = torch.linalg.svd(gradient, full_matrices=False)
    basis = (left[:, :rank] if side == "left" else right_t[:rank].T).clone()
    for column in range(rank):
        pivot = int(basis[:, column].abs().argmax())
        if basis[pivot, column] < 0:
            basis[:, column] = -basis[:, column]
    return basis


class _ReferenceGaLoreAdamW:
    """Independent float64 GaLore AdamW for one weight split into row blocks."""

    def __init__(self, weight, rows, *, rank, update_proj_gap, scale, lr, betas, weight_decay, eps=1.0e-8):
        self.weight = weight.detach().double().clone()
        self.blocks = []
        start = 0
        for count in rows:
            self.blocks.append({"start": start, "rows": count, "basis": None, "m": None, "v": None})
            start += count
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.t = 0

    def step(self, gradient: torch.Tensor) -> None:
        beta1, beta2 = self.betas
        self.t += 1
        self.weight *= 1.0 - self.lr * self.weight_decay
        for block in self.blocks:
            grad = gradient[block["start"] : block["start"] + block["rows"]].double()
            side = "right" if grad.shape[0] >= grad.shape[1] else "left"
            if (self.t - 1) % self.update_proj_gap == 0:
                block["basis"] = _reference_basis(grad, self.rank, side)
            basis = block["basis"]
            projected = basis.T @ grad if side == "left" else grad @ basis
            if block["m"] is None:
                block["m"] = torch.zeros_like(projected)
                block["v"] = torch.zeros_like(projected)
            block["m"] = beta1 * block["m"] + (1.0 - beta1) * projected
            block["v"] = beta2 * block["v"] + (1.0 - beta2) * projected * projected
            m_hat = block["m"] / (1.0 - beta1**self.t)
            v_hat = block["v"] / (1.0 - beta2**self.t)
            normalized = m_hat / (v_hat.sqrt() + self.eps)
            update = basis @ normalized if side == "left" else normalized @ basis.T
            self.weight[block["start"] : block["start"] + block["rows"]] -= self.lr * self.scale * update


def _linear_param(rows: tuple[int, ...], cols: int, *, dtype=torch.float32, seed: int = 0) -> torch.nn.Parameter:
    generator = torch.Generator().manual_seed(seed)
    param = torch.nn.Parameter((0.1 * torch.randn(sum(rows), cols, generator=generator)).to(dtype))
    mark_linear_weight(param, rows)
    return param


def _spectral_gradient(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    """FP32 gradient with distinct, well-separated singular values."""

    left, _ = torch.linalg.qr(torch.randn(rows, rows, generator=generator, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(cols, cols, generator=generator, dtype=torch.float64))
    size = min(rows, cols)
    values = 0.8 ** torch.arange(size, dtype=torch.float64)
    return ((left[:, :size] * values) @ right[:, :size].T).float()


def _fused_gradient(rows: tuple[int, ...], cols: int, generator: torch.Generator) -> torch.Tensor:
    return torch.cat([_spectral_gradient(count, cols, generator) for count in rows])


def test_galore_matches_independent_reference_across_projection_refreshes() -> None:
    kwargs = {"lr": 3.0e-3, "betas": (0.9, 0.97), "weight_decay": 0.05}
    galore = {"rank": 3, "update_proj_gap": 3, "scale": 0.5}
    fused_rows, fused_cols = (12, 6), 10
    square_rows, square_cols = (8,), 8
    fused = _linear_param(fused_rows, fused_cols, seed=1)
    square = _linear_param(square_rows, square_cols, seed=2)
    bias = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 18))
    embedding = torch.nn.Parameter(0.1 * torch.randn(10, 6, generator=torch.Generator().manual_seed(3)))
    optimizer = GaLoreAdamW([fused, square, bias, embedding], **kwargs, **galore)
    fused_reference = _ReferenceGaLoreAdamW(fused, fused_rows, **kwargs, **galore)
    square_reference = _ReferenceGaLoreAdamW(square, square_rows, **kwargs, **galore)
    dense_reference = [torch.nn.Parameter(bias.detach().double()), torch.nn.Parameter(embedding.detach().double())]
    adamw_reference = torch.optim.AdamW(dense_reference, eps=1.0e-8, **kwargs)
    generator = torch.Generator().manual_seed(17)

    assert [block.side for block in optimizer._matrices[0].blocks] == ["right", "left"]
    assert [block.side for block in optimizer._matrices[1].blocks] == ["right"]
    # Ten steps cross refreshes at steps 1, 4, 7 and 10.
    for _ in range(10):
        fused_grad = _fused_gradient(fused_rows, fused_cols, generator)
        square_grad = _fused_gradient(square_rows, square_cols, generator)
        dense_grads = [torch.randn(18, generator=generator), torch.randn(10, 6, generator=generator)]
        fused.grad, square.grad = fused_grad.clone(), square_grad.clone()
        bias.grad, embedding.grad = dense_grads[0].clone(), dense_grads[1].clone()
        for reference, grad in zip(dense_reference, dense_grads, strict=True):
            reference.grad = grad.double()

        optimizer.step()
        fused_reference.step(fused_grad)
        square_reference.step(square_grad)
        adamw_reference.step()

        torch.testing.assert_close(fused.detach(), fused_reference.weight.float(), atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(square.detach(), square_reference.weight.float(), atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(bias.detach(), dense_reference[0].detach().float(), atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(embedding.detach(), dense_reference[1].detach().float(), atol=ATOL, rtol=RTOL)
        assert fused.grad is None and bias.grad is None


def _run_steps(optimizer, params, gradients) -> None:
    for step_gradients in gradients:
        for param, gradient in zip(params, step_gradients, strict=True):
            param.grad = gradient.to(param.dtype).clone()
        optimizer.step()


def _galore_tensors(optimizer: GaLoreAdamW) -> list[torch.Tensor]:
    tensors = []
    for matrix in optimizer._matrices:
        tensors.append(optimizer._matrix_master(matrix).detach().clone())
        for block in matrix.blocks:
            tensors.extend((block.basis, block.exp_avg, block.exp_avg_sq))
    return tensors


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("save_after", [2, 3, 4])
def test_galore_resume_matches_uninterrupted_run_at_refresh_boundaries(dtype, save_after) -> None:
    rows, cols = (12, 6), 10
    kwargs = {"lr": 2.0e-3, "betas": (0.9, 0.99), "weight_decay": 0.01, "rank": 3, "update_proj_gap": 3}

    def build(seed_offset: int = 0):
        weight = _linear_param(rows, cols, dtype=dtype, seed=5 + seed_offset)
        bias = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 18).to(dtype) + seed_offset)
        return [weight, bias], GaLoreAdamW([weight, bias], **kwargs)

    generator = torch.Generator().manual_seed(23)
    gradients = [[_fused_gradient(rows, cols, generator), torch.randn(18, generator=generator)] for _ in range(8)]
    continuous_params, continuous = build()
    _run_steps(continuous, continuous_params, gradients)

    first_params, first = build()
    _run_steps(first, first_params, gradients[:save_after])
    buffer = io.BytesIO()
    torch.save(copy.deepcopy(first.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)

    # Different initial weights prove the checkpoint restores the masters.
    resumed_params, resumed = build(seed_offset=1)
    resumed.load_state_dict(checkpoint)
    for resumed_param, first_param in zip(resumed_params, first_params, strict=True):
        assert torch.equal(resumed_param, first_param)
    _run_steps(resumed, resumed_params, gradients[save_after:])

    for resumed_param, continuous_param in zip(resumed_params, continuous_params, strict=True):
        assert torch.equal(resumed_param, continuous_param)
    for resumed_tensor, continuous_tensor in zip(_galore_tensors(resumed), _galore_tensors(continuous), strict=True):
        assert torch.equal(resumed_tensor, continuous_tensor)
    assert resumed._matrices[0].step == continuous._matrices[0].step == 8


def test_galore_checkpoint_records_settings_and_rejects_mismatches() -> None:
    weight = _linear_param((12,), 10)
    optimizer = GaLoreAdamW([weight], lr=1.0e-3, betas=(0.9, 0.99), weight_decay=0.0, rank=3, update_proj_gap=4)
    weight.grad = _fused_gradient((12,), 10, torch.Generator().manual_seed(0))
    optimizer.step()
    state = optimizer.state_dict()

    assert {key: state["galore"][key] for key in ("rank", "update_proj_gap", "scale")} == {
        "rank": 3,
        "update_proj_gap": 4,
        "scale": 1.0,
    }
    saved_matrix = state["galore"]["matrices"][0]
    assert saved_matrix["step"] == 1
    assert saved_matrix["blocks"][0]["basis"].shape == (10, 3)
    assert saved_matrix["blocks"][0]["exp_avg"].shape == (12, 3)

    for setting, value in (("rank", 2), ("update_proj_gap", 5), ("scale", 0.5)):
        other_weight = _linear_param((12,), 10)
        options = {"rank": 3, "update_proj_gap": 4, "scale": 1.0, setting: value}
        other = GaLoreAdamW([other_weight], lr=1.0e-3, betas=(0.9, 0.99), weight_decay=0.0, **options)
        with pytest.raises(ValueError, match=f"GaLore {setting} mismatch"):
            other.load_state_dict(state)
    plain = AdamWFP32Master([_linear_param((12,), 10)], lr=1.0e-3, betas=(0.9, 0.99), weight_decay=0.0)
    with pytest.raises(ValueError, match="no GaLore section"):
        GaLoreAdamW(
            [_linear_param((12,), 10)], lr=1.0e-3, betas=(0.9, 0.99), weight_decay=0.0, rank=3, update_proj_gap=4
        ).load_state_dict(plain.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_galore_state_for_1024_square_rank_128_is_smaller_than_same_precision_adamw(dtype) -> None:
    gradient = torch.randn(1024, 1024, generator=torch.Generator().manual_seed(0))
    galore_param = _linear_param((1024,), 1024, dtype=dtype)
    adamw_param = torch.nn.Parameter(galore_param.detach().clone())
    galore = GaLoreAdamW([galore_param], lr=1.0e-3, betas=(0.9, 0.999), weight_decay=0.0, rank=128, update_proj_gap=200)
    adamw = AdamWFP32Master([adamw_param], lr=1.0e-3, betas=(0.9, 0.999), weight_decay=0.0)
    galore_param.grad = gradient.to(dtype)
    adamw_param.grad = gradient.to(dtype)
    galore.step()
    adamw.step()

    metrics = galore.state_memory_metrics()
    adamw_moments = sum(bucket.exp_avg.nbytes + bucket.exp_avg_sq.nbytes for bucket in adamw.buckets)
    # Persistent FP32 state: basis (1024 x 128) plus two (1024 x 128) moments.
    assert metrics["projection_bytes"] == 1024 * 128 * 4
    assert metrics["moment_bytes"] == 2 * 1024 * 128 * 4
    assert metrics["full_rank_moment_bytes"] == adamw_moments == 2 * 1024 * 1024 * 4
    assert metrics["projection_bytes"] + metrics["moment_bytes"] < adamw_moments
    if dtype == torch.bfloat16:
        # Both optimizers keep the same lossless FP32 master for BF16 weights.
        adamw_master = sum(bucket.master_storage.nbytes for bucket in adamw.buckets)
        assert metrics["master_bytes"] == adamw_master
    assert metrics["state_bytes"] < adamw_moments + sum(bucket.master_storage.nbytes for bucket in adamw.buckets)


def test_galore_projects_fused_weights_like_separate_projections() -> None:
    kwargs = {"lr": 1.0e-2, "betas": (0.9, 0.95), "weight_decay": 0.1, "rank": 2, "update_proj_gap": 2}
    fused = _linear_param((6, 4), 8, seed=4)
    first = _linear_param((6,), 8)
    second = _linear_param((4,), 8)
    with torch.no_grad():
        first.copy_(fused[:6])
        second.copy_(fused[6:])
    fused_optimizer = GaLoreAdamW([fused], **kwargs)
    separate_optimizer = GaLoreAdamW([first, second], **kwargs)
    generator = torch.Generator().manual_seed(9)

    for _ in range(5):
        gradient = _fused_gradient((6, 4), 8, generator)
        fused.grad = gradient.clone()
        first.grad, second.grad = gradient[:6].clone(), gradient[6:].clone()
        fused_optimizer.step()
        separate_optimizer.step()

    torch.testing.assert_close(fused.detach(), torch.cat([first.detach(), second.detach()]), atol=0.0, rtol=0.0)


def test_galore_keeps_adamw_for_ineligible_tensors() -> None:
    rank = 4
    eligible = _linear_param((32, 16), 16)
    # Tagged, but a rank-4 basis plus moments would not shrink a 3 x 16 block.
    thin = _linear_param((3,), 16)
    untagged = torch.nn.Parameter(torch.zeros(32, 16))
    vector = torch.nn.Parameter(torch.zeros(16))
    frozen = _linear_param((32,), 16)
    frozen.requires_grad_(False)

    assert galore_blocks(eligible, rank) is not None
    assert galore_blocks(thin, rank) is None
    assert galore_blocks(untagged, rank) is None
    optimizer = GaLoreAdamW(
        [eligible, thin, untagged, vector, frozen],
        lr=1.0e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        rank=rank,
        update_proj_gap=10,
    )
    bucketed = {id(ref.model_param) for bucket in optimizer.buckets for ref in bucket.refs}

    assert [matrix.param for matrix in optimizer._matrices] == [eligible]
    assert bucketed == {id(thin), id(untagged), id(vector)}
    assert optimizer.state_memory_metrics()["matrices"] == 2


def test_top_singular_basis_is_orthonormal_sign_canonical_and_rejects_nonfinite() -> None:
    gradient = _spectral_gradient(9, 7, torch.Generator().manual_seed(2))
    for side, size in (("left", 9), ("right", 7)):
        basis = top_singular_basis(gradient, 3, side)
        assert basis.shape == (size, 3)
        torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=1.0e-6, rtol=0.0)
        pivots = basis.abs().argmax(dim=0)
        assert bool((basis[pivots, torch.arange(3)] > 0).all())
        flipped = top_singular_basis(-gradient, 3, side)
        torch.testing.assert_close(flipped, basis, atol=1.0e-6, rtol=0.0)
        reference = _reference_basis(gradient.double(), 3, side)
        torch.testing.assert_close(basis.double(), reference, atol=1.0e-5, rtol=0.0)
    gradient[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        top_singular_basis(gradient, 3, "left")


def test_galore_rejects_sharded_state_and_disk_offload(tmp_path) -> None:
    weight = _linear_param((12,), 10)
    kwargs = {"lr": 1.0e-3, "betas": (0.9, 0.99), "weight_decay": 0.0, "rank": 3, "update_proj_gap": 4}
    with pytest.raises(ValueError, match="dp_size=1"):
        GaLoreAdamW([weight], dp_size=2, **kwargs)
    optimizer = GaLoreAdamW([weight], **kwargs)
    with pytest.raises(ValueError, match="cpu only"):
        optimizer.configure_state_offload(mode="disk", directory=str(tmp_path))


def test_galore_cpu_offload_round_trip_preserves_next_update() -> None:
    kwargs = {"lr": 1.0e-3, "betas": (0.9, 0.99), "weight_decay": 0.0, "rank": 3, "update_proj_gap": 4}
    weights = [_linear_param((12,), 10, dtype=torch.bfloat16) for _ in range(2)]
    optimizers = [GaLoreAdamW([weight], **kwargs) for weight in weights]
    generator = torch.Generator().manual_seed(4)
    gradients = [_fused_gradient((12,), 10, generator) for _ in range(3)]
    for index, (weight, optimizer) in enumerate(zip(weights, optimizers, strict=True)):
        for step, gradient in enumerate(gradients):
            weight.grad = gradient.to(torch.bfloat16)
            optimizer.step()
            if index == 1 and step == 0:
                optimizer.configure_state_offload(mode="cpu")
                optimizer.offload_state(mode="cpu")
                optimizer.onload_state(torch.device("cpu"))

    assert torch.equal(weights[0], weights[1])


def test_build_optimizer_selects_galore_only_for_single_rank() -> None:
    config = OptimizerConfig(galore_rank=3, galore_update_proj_gap=7, galore_scale=0.25)
    single = SimpleNamespace(world_size=1, dp_rank=0, dp_size=1, dp_group=None)
    optimizer = build_optimizer([_linear_param((12,), 10)], config, single)

    assert isinstance(optimizer, GaLoreAdamW)
    assert (optimizer.rank, optimizer.update_proj_gap, optimizer.scale) == (3, 7, 0.25)
    with pytest.raises(ValueError, match="single-GPU"):
        build_optimizer([_linear_param((12,), 10)], config, SimpleNamespace(world_size=2, dp_size=1))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"galore_rank": 0}, "galore_rank must be positive"),
        ({"galore_rank": 4, "galore_update_proj_gap": 0}, "galore_update_proj_gap must be positive"),
        ({"galore_rank": 4, "galore_scale": 0.0}, "galore_scale must be positive"),
        ({"galore_rank": 4, "adam_8bit": True}, "cannot be combined"),
        ({"galore_rank": 4, "paged": True}, "cannot be combined"),
    ],
)
def test_engine_optimizer_config_validates_galore(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        OptimizerConfig(**overrides)


def test_trainer_config_propagates_galore_to_engine_optimizer_config() -> None:
    config = TrainerConfig(
        algo="sft",
        ckpt="unused",
        dataset_path="unused",
        backend="cuda",
        world_size=1,
        tp_size=1,
        galore_rank=128,
        galore_update_proj_gap=50,
        galore_scale=2.0,
    )
    optimizer = OptimizerConfig(**config.cuda_config().optimizer)

    assert (optimizer.galore_rank, optimizer.galore_update_proj_gap, optimizer.galore_scale) == (128, 50, 2.0)
    assert (
        OptimizerConfig(**TrainerConfig(algo="sft", ckpt="x", dataset_path="y").optimizer_config()).galore_rank is None
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"world_size": 8, "tp_size": 4}, "single-GPU"),
        ({"adam_8bit": True}, "cannot be combined"),
        ({"optimizer_state_offload": "disk", "optimizer_state_offload_dir": "/tmp/x"}, "none or cpu"),
    ],
)
def test_trainer_config_rejects_galore_outside_single_gpu_full_parameter_scope(overrides, message) -> None:
    options = {"world_size": 1, "tp_size": 1, **overrides}
    with pytest.raises(ValueError, match=message):
        TrainerConfig(algo="sft", ckpt="x", dataset_path="y", backend="cuda", galore_rank=8, **options)


def _fake_worker(optimizer, *, global_step: int = 0):
    return SimpleNamespace(
        optimizer=optimizer,
        _global_step=global_step,
        _prepare_actor_onloaded=lambda: None,
        _prepare_for_train=lambda: None,
        _optimizer_offload_options=lambda: ("none", None, 1),
        config=SimpleNamespace(runtime=SimpleNamespace(keep_rollout_state=True)),
    )


def test_worker_optimizer_state_files_restore_galore_state_and_step(tmp_path) -> None:
    from areno.engine.protocol import OptimizerStatePayload
    from areno.engine.worker import ArenoWorker

    kwargs = {"lr": 1.0e-3, "betas": (0.9, 0.99), "weight_decay": 0.01, "rank": 3, "update_proj_gap": 2}
    weight = _linear_param((12,), 10, dtype=torch.bfloat16)
    optimizer = GaLoreAdamW([weight], **kwargs)
    generator = torch.Generator().manual_seed(6)
    for _ in range(3):
        weight.grad = _fused_gradient((12,), 10, generator).to(torch.bfloat16)
        optimizer.step()
    payload = OptimizerStatePayload(path=str(tmp_path / "step_3"))

    saved = ArenoWorker.save_optimizer_state(_fake_worker(optimizer, global_step=3), payload)
    restored_weight = _linear_param((12,), 10, dtype=torch.bfloat16, seed=99)
    restored = _fake_worker(GaLoreAdamW([restored_weight], **kwargs))
    loaded = ArenoWorker.load_optimizer_state(restored, payload)

    assert saved == {"path": payload.path}
    assert (tmp_path / "step_3" / "areno_optimizer_rank00000.pt").is_file()
    assert loaded == {"global_step": 3} and restored._global_step == 3
    assert torch.equal(restored_weight, weight)
    for restored_tensor, tensor in zip(_galore_tensors(restored.optimizer), _galore_tensors(optimizer), strict=True):
        assert torch.equal(restored_tensor, tensor)
    plain = _fake_worker(AdamWFP32Master([_linear_param((12,), 10)], lr=1.0e-3, betas=(0.9, 0.99), weight_decay=0.0))
    with pytest.raises(ValueError, match="saved by GaLoreAdamW"):
        ArenoWorker.load_optimizer_state(plain, payload)


def test_trainer_load_optimizer_state_continues_step_numbering() -> None:
    from areno.api.trainer import Trainer

    calls = []

    class _Backend:
        def save_optimizer_state(self, ctx, path):
            calls.append(("save", path))
            return path

        def load_optimizer_state(self, ctx, path):
            calls.append(("load", path))
            return 7

    trainer = Trainer(1, "unused")
    trainer._backend = _Backend()
    trainer._ctx = SimpleNamespace(global_step=-1)

    assert trainer.save_optimizer_state("ckpt") == "ckpt"
    assert trainer.load_optimizer_state("ckpt") == 7
    assert trainer._ctx.global_step == 6
    assert calls == [("save", "ckpt"), ("load", "ckpt")]
