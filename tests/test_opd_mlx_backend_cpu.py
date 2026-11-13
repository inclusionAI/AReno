"""CPU-only wiring checks for the MLX OPD path."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from areno.api.algorithms import describe_loss_fn, list_algorithms
from areno.api.backend.cuda.backend import CudaBackend
from areno.api.backend.mlx import losses as mlx_losses
from areno.api.backend.mlx.backend import MlxBackend
from areno.api.roles import ModelRole


def test_mlx_declares_the_same_algorithms_and_roles_as_cuda():
    mlx = MlxBackend.capabilities()
    cuda = CudaBackend.capabilities()

    assert "opd" in mlx.algorithms
    assert "teacher" in mlx.model_roles
    assert mlx.algorithms == cuda.algorithms
    assert mlx.model_roles == cuda.model_roles


def test_opd_loss_is_registered_with_backend_neutral_kwargs():
    assert mlx_losses._LOSSES["opd"] is mlx_losses.opd_loss

    spec = list_algorithms()["opd"]
    config = SimpleNamespace(opd_coef=0.5, opd_loss_type="k3", opd_clip=2.0)
    described = describe_loss_fn(spec.make_loss_fn(config))

    assert described.name == "opd"
    assert described.kwargs == {"opd_coef": 0.5, "opd_loss_type": "k3", "opd_clip": 2.0}


def test_ensure_roles_loads_teacher_as_a_frozen_language_role():
    backend = MlxBackend()
    backend._roles = {}
    backend._lora_state = None
    role = ModelRole("teacher", "teacher-mlx", trainable=False)
    loaded = SimpleNamespace(tokenizer=object(), provider=object())

    with (
        patch.object(MlxBackend, "_require_runtime", return_value=None),
        patch.object(MlxBackend, "_validate_loaded_tokenizer", return_value=None),
        patch("areno.api.backend.mlx.backend.load_language_role", return_value=loaded) as load,
    ):
        backend.ensure_roles(SimpleNamespace(), {"teacher": role})

    load.assert_called_once_with("teacher-mlx")
    assert backend._roles["teacher"] is loaded


def test_teacher_role_is_rejected_under_qlora():
    backend = MlxBackend()
    backend._roles = {}
    backend._lora_state = SimpleNamespace(quantization=object())
    role = ModelRole("teacher", "teacher-mlx", trainable=False)

    with patch.object(MlxBackend, "_require_runtime", return_value=None):
        with pytest.raises(ValueError, match="QLoRA auxiliary roles"):
            backend.ensure_roles(SimpleNamespace(), {"teacher": role})

    assert backend._roles == {}
