"""Backend-independent objectives and native Torch candidate gradients."""

from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from areno.api.backend.cuda.decision import (
    backward_questions,
    initialize_head,
    parameter_counts,
    question_logits,
    save_head,
)
from areno.api.jev import JevCudaConfig, JevTrainerConfig
from areno.api.jev.loss import jev_loss_fn
from areno.api.models import TrainSequence
from areno.engine.runtime.common import split_data_pack_by_dp
from areno.models.base import CausalLMOutput


class Backbone(nn.Module):
    def __init__(self, family, lora):
        super().__init__()
        self.family = family
        self.embedding = nn.Embedding(32, 8)
        self.projection = nn.Linear(8, 8)
        if lora:
            self.requires_grad_(False)
            self.lora_A = nn.Parameter(torch.randn(2, 8) * 0.1)
            self.lora_B = nn.Parameter(torch.randn(8, 2) * 0.1)

    def forward(self, input_ids, position_ids, train_meta, defer_lm_head):
        assert defer_lm_head
        values = self.embedding(input_ids)
        hidden = self.projection(values)
        if hasattr(self, "lora_A"):
            hidden = hidden + (values @ self.lora_A.T) @ self.lora_B.T
        sequences = []
        for a, b in zip(train_meta.cu_seqlens[:-1], train_meta.cu_seqlens[1:], strict=True):
            sequence = hidden[:, a:b].cumsum(dim=1)
            sequences.append(sequence.tanh() if self.family == "tanh" else sequence / (1 + sequence.abs()))
        return CausalLMOutput(hidden_states=torch.cat(sequences, dim=1))


def worker(family="tanh", lora=False):
    torch.manual_seed(13)
    result = SimpleNamespace(
        model=Backbone(family, lora),
        device=torch.device("cpu"),
        config=SimpleNamespace(
            lora_seed=17,
            model=SimpleNamespace(hidden_size=8),
            runtime=SimpleNamespace(
                decision_head={"head_lr": 2e-4, "checkpoint": None, "max_seq_len": 64}, activation_checkpointing=False
            ),
            effective_sequence_parallel=False,
            base_model_name_or_path="base",
            model_path="base",
        ),
    )
    initialize_head(result)
    return result


def groups():
    result = []
    for q, target in enumerate(([0.0, 1.0], [0.1, 0.2, 0.0, 0.3, 0.4], [1.0, 0.0, 0.0])):
        result.append(
            [
                TrainSequence(tokens=[1, q + 2, c + 3, 15 + c], features={"jev_target": probability})
                for c, probability in enumerate(target)
            ]
        )
    return result


@pytest.mark.parametrize("family", ["tanh", "rational"])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mini_bs", [1, 2, 4, 16])
def test_microbatch_loss_and_all_parameter_gradients_match_complete_questions(family, lora, mini_bs):
    candidate = worker(family, lora)
    reference = deepcopy(candidate)
    questions = groups()
    objective = partial(jev_loss_fn, brier_weight=0.5)
    logits = question_logits(reference, [row for question in questions for row in question])
    expected, offset = 0, 0
    for question in questions:
        target = logits.new_tensor([row.features["jev_target"] for row in question])
        expected += objective(target, logits[offset : offset + len(question)])[0] / len(questions)
        offset += len(question)
    expected.backward()
    actual, stats = backward_questions(
        candidate,
        {
            "_jev_groups": questions,
            "_jev_mini_bs": mini_bs,
            "_jev_weight": 1 / len(questions),
            "_jev_loss_fn": objective,
        },
        grad_scale=1,
    )
    torch.testing.assert_close(actual, expected.detach(), atol=2e-6, rtol=2e-5)
    assert stats["candidate_microbatches"] >= 1
    counts = parameter_counts(candidate)
    assert counts["head"] > 0 and counts["lora" if lora else "backbone"] > 0
    grads = dict(reference.model.named_parameters())
    for name, value in candidate.model.named_parameters():
        if value.requires_grad:
            torch.testing.assert_close(value.grad, grads[name].grad, atol=2e-6, rtol=2e-5)
        else:
            assert value.grad is None
    before = {name: value.detach().clone() for name, value in candidate.model.named_parameters()}
    optimizer = torch.optim.SGD((value for value in candidate.model.parameters() if value.requires_grad), lr=1e-2)
    optimizer.step()
    changed = [name for name, value in candidate.model.named_parameters() if not torch.equal(value, before[name])]
    assert any(name.startswith("score_head.") for name in changed)
    assert any(not name.startswith("score_head.") for name in changed)
    if lora:
        assert all(name.startswith("score_head.") or name.startswith("lora_") for name in changed)


def test_question_preserving_dp_normalization_including_uneven_shards():
    questions = groups()
    shards = split_data_pack_by_dp({"_jev_groups": questions, "_jev_weight": 1 / 3}, 2)
    assert [len(shard["_jev_groups"]) for shard in shards] == [2, 1]
    assert [q for shard in shards for q in shard["_jev_groups"]] == [questions[0], questions[2], questions[1]]
    assert sum(len(s["_jev_groups"]) * s["_jev_weight"] for s in shards) / 2 == pytest.approx(1)
    assert len(split_data_pack_by_dp({"_jev_groups": questions, "_jev_weight": 1 / 3}, 4)) == 4


def test_head_checkpoint_roundtrip_and_typed_cuda_runtime(tmp_path):
    source = worker()
    save_head(source, tmp_path)
    restored = worker("rational")
    restored.config.runtime.decision_head["checkpoint"] = str(tmp_path)
    initialize_head(restored)
    for name, value in source.model.score_head.state_dict().items():
        torch.testing.assert_close(value, restored.model.score_head.state_dict()[name], atol=0, rtol=0)
    config = JevTrainerConfig(
        algo="jev",
        ckpt=str(tmp_path),
        dataset_path="unused",
        backend="cuda",
        world_size=4,
        tp_size=4,
        save_path="unused",
        adam_4bit=True,
    )
    runtime = config.backend_config()
    assert isinstance(runtime, JevCudaConfig) and runtime.tp_size == 4 and runtime.optimizer["adam_4bit"]
    assert runtime.decision_checkpoint == str(tmp_path) and config.resume_checkpoint() == str(tmp_path)


@pytest.mark.parametrize("target", [[0.0, 1.0, 0.0], [0.1, 0.6, 0.3]])
def test_torch_mlx_loss_and_score_gradients_match(target):
    mx = pytest.importorskip("mlx.core")
    values = np.array([1.2, -3.1, 0.9], dtype=np.float32)
    logits = torch.tensor(values, requires_grad=True)
    loss, stats = jev_loss_fn(torch.tensor(target), logits)
    loss.backward()
    mlx_loss, mlx_grad = mx.value_and_grad(lambda score: jev_loss_fn(mx.array(target), score)[0])(mx.array(values))
    mx.eval(mlx_loss, mlx_grad)
    assert loss.item() == pytest.approx(mlx_loss.item(), abs=2e-6)
    np.testing.assert_allclose(logits.grad.numpy(), np.array(mlx_grad), atol=2e-6, rtol=2e-5)
    assert stats["jev_top1"].item() in (0.0, 1.0)


@pytest.mark.parametrize("mode", ["full", "lora", "qlora"])
@pytest.mark.parametrize("optimizer", ["default", "adam_4bit", "adam_8bit"])
def test_native_cuda_modes_preserve_optimizer_and_runtime_options(mode, optimizer):
    from areno import LoraConfig

    lora = None if mode == "full" else LoraConfig(qlora=mode == "qlora")
    options = {} if optimizer == "default" else {optimizer: True}
    config = JevTrainerConfig(
        algo="jev",
        ckpt="base",
        dataset_path="data",
        save_path="out",
        backend="cuda",
        world_size=4,
        tp_size=4,
        lora=lora,
        activation_checkpointing=True,
        **options,
    )
    runtime = config.backend_config()
    assert runtime.lora is lora
    assert runtime.optimizer["adam_4bit"] == (optimizer == "adam_4bit")
    assert runtime.optimizer["adam_8bit"] == (optimizer == "adam_8bit")
    assert runtime.runtime["activation_checkpointing"]
    assert runtime.runtime["decision_head"]["head_lr"] == config.head_lr


def test_head_weights_are_shared_between_torch_and_mlx(tmp_path):
    mx = pytest.importorskip("mlx.core")
    mlx_nn = pytest.importorskip("mlx.nn")
    source = worker()
    save_head(source, tmp_path)
    head = mlx_nn.Sequential(mlx_nn.Linear(8, 8), mlx_nn.GELU(), mlx_nn.Linear(8, 1))
    head.load_weights(str(tmp_path / "score_head.safetensors"))
    values = np.random.default_rng(17).normal(size=(3, 8)).astype(np.float32)
    expected = source.model.score_head(torch.tensor(values)).detach().numpy()
    np.testing.assert_allclose(expected, np.array(head(mx.array(values))), atol=2e-6, rtol=2e-5)


def test_training_respects_the_native_runtime_model_selection():
    candidate = worker()
    selected = candidate.model
    candidate.model = nn.Module()  # Must never forward through this wrapper.
    questions = groups()
    loss, _ = backward_questions(
        candidate,
        {
            "_jev_groups": questions,
            "_jev_mini_bs": 8,
            "_jev_weight": 1 / len(questions),
            "_jev_loss_fn": partial(jev_loss_fn, brier_weight=0.5),
        },
        grad_scale=1,
        model=selected,
    )
    assert torch.isfinite(loss)
    assert selected.score_head.layers[0].weight.grad.abs().sum() > 0


def test_scoring_does_not_run_the_training_mst_schedule(monkeypatch):
    candidate = worker()
    original = candidate.model.forward
    seen = []

    def record(**kwargs):
        seen.append(kwargs["train_meta"].mst_chunk_size)
        return original(**kwargs)

    monkeypatch.setattr(candidate.model, "forward", record)
    monkeypatch.setattr("areno.engine.runtime.mst.training_chunk_size", lambda *args: 256)
    rows = [row for question in groups() for row in question]
    question_logits(candidate, rows)
    with torch.no_grad():
        question_logits(candidate, rows)
    assert seen == [256, 0]


def test_mlx_inference_can_load_full_backbone_checkpoint_without_lora():
    from areno.api.jev import JevMlxConfig

    assert JevMlxConfig(model_path="full-checkpoint", inference_only=True).lora is None
    with pytest.raises(ValueError, match="requires LoRA"):
        JevMlxConfig(model_path="base")
