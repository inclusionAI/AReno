import pytest
import torch

from areno.api.algorithms import get_algorithm
from areno.api.backend.cuda.backend import CudaBackend
from areno.api.jev.labels.config import DenseLabelJevTrainerConfig
from areno.api.jev.labels.dense import CapabilityTarget, DenseLabelJevTrainer, dense_reward_weights


@pytest.mark.parametrize("coefficient", [0.0, 0.1, 0.8])
def test_dense_label_loss_and_gradients_cpu(coefficient):
    spec = get_algorithm("sgl-jev-dense")
    assert spec.resolve_trainer_cls() is DenseLabelJevTrainer
    assert spec.name in CudaBackend.capabilities().algorithms
    assert not spec.requires_rollout

    targets = [
        CapabilityTarget("choice", "hard", 1, (0.2, 0.8)),
        CapabilityTarget("noul", "standard", 0),
        CapabilityTarget("score", "hard", 1),
    ]
    ids = [[0, 2], [1, 2], [0, 1, 2]]
    teachers = [[0.2, 0.8], [0.9, 0.1], [0.1, 0.8, 0.1]]
    weights = list(dense_reward_weights(targets, [len(i) for i in ids]))
    questions = [
        {
            "ids": i,
            "target": q,
            "temperature": 1.0,
            "dense_reward": (target, weight),
            "dense_reward_weight": coefficient,
        }
        for i, q, target, weight in zip(ids, teachers, targets, weights, strict=True)
    ]
    logits = torch.tensor([[0.3, -0.7, 0.1, 2.0], [0.8, -0.5, 0.3, -1.0], [0.3, 0.5, -0.5, 1.0]], requires_grad=True)
    loss, stats = spec.default_loss_fn({"_label_questions": questions}, logits)
    anchors = []
    for row, labels, teacher in zip(logits, ids, teachers, strict=True):
        q = torch.tensor(teacher)
        lp = row[labels].log_softmax(0)
        anchors.append(-(q * lp).sum() + 0.5 * (lp.exp() - q).square().sum())
    anchor = torch.stack(anchors).mean()
    torch.testing.assert_close(stats["proper_loss"], anchor.detach())
    torch.testing.assert_close(loss, anchor - coefficient * stats["dense_reward"] / 100)
    (grad,) = torch.autograd.grad(loss, logits)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad[:, -1]) == 0


def test_dense_label_config_cpu(tmp_path):
    options = dict(
        algo="sgl-jev-dense",
        ckpt=str(tmp_path),
        dataset_path="train.jsonl",
        save_path=str(tmp_path),
        backend="cuda",
        world_size=1,
        tp_size=1,
        bi_attn=True,
        batch_size=32,
        mini_bs=4,
        gradient_accumulation_steps=8,
    )
    assert DenseLabelJevTrainerConfig(**options).dense_reward_weight == 0.1
    assert DenseLabelJevTrainerConfig(**options, dense_reward_weight=0.8).bi_attn
    with pytest.raises(ValueError, match="finite and nonnegative"):
        DenseLabelJevTrainerConfig(**options, dense_reward_weight=float("nan"))
    with pytest.raises(ValueError, match="requires CUDA"):
        DenseLabelJevTrainerConfig(**{**options, "backend": "mlx"})
    with pytest.raises(ValueError, match="one optimizer update"):
        DenseLabelJevTrainerConfig(**{**options, "gradient_accumulation_steps": 4})


def test_dense_question_gold_and_tier_cpu():
    prompt = {"type": "noul", "capability": {"tier": "hard", "gold": True}}
    assert CapabilityTarget.from_question(prompt, ["true", "false"]).gold == 0
    with pytest.raises(ValueError, match="capability tier"):
        CapabilityTarget.from_question({"type": "noul"}, ["true", "false"])


def test_dense_label_cli_cpu(tmp_path):
    from areno.cli.train import _trainer_config_from_options, train_command

    args = [
        "--algo",
        "sgl-jev-dense",
        "--bi-attn",
        "--backend",
        "cuda",
        "--ckpt",
        str(tmp_path),
        "--dataset-path",
        "train.jsonl",
        "--save-path",
        str(tmp_path),
        "--tp-size",
        "1",
        "--world-size",
        "1",
        "--dense-reward-weight",
        "0.8",
    ]
    with train_command.make_context("train", args) as ctx:
        config = _trainer_config_from_options(**ctx.params)
    assert isinstance(config, DenseLabelJevTrainerConfig)
    assert config.dense_reward_weight == 0.8 and config.bi_attn is True
