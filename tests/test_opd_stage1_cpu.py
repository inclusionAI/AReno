import pytest

from areno.api.algorithms import get_algorithm
from areno.api.trainer_config import (
    OPDTrainerConfig,
    RolloutTrainerConfig,
)
from areno.api.trainers.opd import OPDTrainer


def test_opd_registration():
    spec = get_algorithm("opd")

    assert spec.requires_rollout is True
    assert spec.requires_reward is False
    assert spec.experimental is True
    assert spec.resolve_trainer_cls() is OPDTrainer


def test_opd_configuration():
    config = OPDTrainerConfig(
        algo="opd",
        ckpt="student",
        dataset_path="prompts",
        teacher_ckpt="teacher",
        opd_coef=1.0,
    )

    assert isinstance(
        config,
        RolloutTrainerConfig,
    )
    assert config.teacher_ckpt == "teacher"
    assert config.opd_coef == 1.0


@pytest.mark.parametrize(
    "teacher_ckpt",
    [None, "", "   "],
)
def test_missing_teacher_rejected(teacher_ckpt):
    with pytest.raises(
        ValueError,
        match="--teacher-ckpt is required",
    ):
        OPDTrainerConfig(
            algo="opd",
            ckpt="student",
            dataset_path="prompts",
            teacher_ckpt=teacher_ckpt,
        )


@pytest.mark.parametrize(
    "opd_coef",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_invalid_coefficient_rejected(opd_coef):
    with pytest.raises(
        ValueError,
        match="--opd-coef must be",
    ):
        OPDTrainerConfig(
            algo="opd",
            ckpt="student",
            dataset_path="prompts",
            teacher_ckpt="teacher",
            opd_coef=opd_coef,
        )