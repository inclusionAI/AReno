"""Stage 5: real AReno packing and accumulation-group OPD regression tests.

CPU-only. Exercises real TrainSequence -> _make_train_pack -> _pack_train_data,
loss/value/gradient equivalence, and a fake engine through ArenoBackend.train.
"""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from areno.api.backend.areno.backend import (
    ArenoBackend,
    _annotate_opd_token_mean_packs,
    _is_opd_loss_fn,
    _make_train_pack,
    _opd_response_token_count,
)
from areno.api.loss_fns.opd import opd_loss_fn
from areno.api.models import TrainSequence
from areno.engine.runtime.logprobs import next_token_logprobs, packed_next_token_logprobs
from areno.engine.runtime.train_step import _pack_train_data


def _sequences():
    # 1, 4, and 2 response tokens, respectively (deliberately uneven).
    return [
        TrainSequence(
            tokens=[11, 12, 31],
            prompt_mask=[True, True, False],
            logprobs=[0.0, 0.0, -1.2],
            ref_logprobs=[0.0, 0.0, -0.7],
            eos_token_id=0,
        ),
        TrainSequence(
            tokens=[21, 41, 42, 43, 44],
            prompt_mask=[True, False, False, False, False],
            logprobs=[0.0, -1.0, -1.4, -0.9, -1.1],
            ref_logprobs=[0.0, -0.5, -1.8, -0.4, -1.6],
            eos_token_id=0,
        ),
        TrainSequence(
            tokens=[51, 52, 61, 62],
            prompt_mask=[True, True, False, False],
            logprobs=[0.0, 0.0, -1.3, -0.8],
            ref_logprobs=[0.0, 0.0, -0.9, -0.4],
            eos_token_id=0,
        ),
    ]


def _current(packed, scalar):
    old = packed.get("packed_logprobs")
    if old is not None:
        return old.detach() + scalar
    return packed["logprobs"][:, 1:].detach() + scalar


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("mini_bs", [1, 2, 3])
def test_uneven_microbatches_match_full_token_mean_loss_and_gradient(packed, mini_bs):
    seqs = _sequences()
    entire = _make_train_pack(seqs)
    if packed:
        entire = _pack_train_data(entire)
    shift_full = torch.tensor(0.13, requires_grad=True)
    full_loss, _ = opd_loss_fn(entire, _current(entire, shift_full))
    full_grad = torch.autograd.grad(full_loss, shift_full)[0]

    packs = [
        _make_train_pack(seqs[i:i + mini_bs])
        for i in range(0, len(seqs), mini_bs)
    ]
    counts = [_opd_response_token_count(pack) for pack in packs]
    assert sum(counts) == 7
    _annotate_opd_token_mean_packs(packs, counts, gradient_accumulation_steps=None)
    shift_parts = torch.tensor(0.13, requires_grad=True)
    losses = []
    for pack in packs:
        if packed:
            pack = _pack_train_data(pack)
        local_loss, _ = opd_loss_fn(pack, _current(pack, shift_parts))
        losses.append(local_loss)
    # Emulate TrainingManager's loss/group_size division in backward.
    accumulated_loss = sum(losses) / len(losses)
    accumulated_grad = torch.autograd.grad(accumulated_loss, shift_parts)[0]
    torch.testing.assert_close(accumulated_loss, full_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(accumulated_grad, full_grad, atol=1e-6, rtol=1e-6)


def test_group_annotation_respects_accumulation_boundaries():
    seqs = _sequences()
    packs = [_make_train_pack([seq]) for seq in seqs]
    counts = [_opd_response_token_count(p) for p in packs]
    assert counts == [1, 4, 2]
    _annotate_opd_token_mean_packs(packs, counts, gradient_accumulation_steps=2)
    assert [(p["_opd_total_response_tokens"], p["_opd_grad_scale"]) for p in packs] == [
        (5, 2), (5, 2), (2, 1)
    ]


def test_masks_and_padding_excluded_from_group_count():
    seqs = _sequences()
    seqs[1].loss_mask = [False, True, False, True, True]
    pack = _make_train_pack(seqs)
    assert _opd_response_token_count(pack) == 6
    packed = _pack_train_data(pack)
    assert int(packed["packed_response_mask"].sum()) == 6
    current = _current(packed, torch.tensor(0.1, requires_grad=True))
    loss, stats = opd_loss_fn(packed, current)
    assert torch.isfinite(loss)
    assert stats["opd_response_tokens"].item() == 6


@pytest.mark.parametrize("bad_count,bad_scale", [(0, 1), (2, 0), (-1, 1), (2, -1)])
def test_invalid_accumulation_annotations_fail_fast(bad_count, bad_scale):
    pack = _make_train_pack([_sequences()[0]])
    pack["_opd_total_response_tokens"] = bad_count
    pack["_opd_grad_scale"] = bad_scale
    with pytest.raises(ValueError, match="normalization|global response"):
        opd_loss_fn(pack, torch.zeros((1, len(_sequences()[0].tokens) - 1)))


def test_annotation_rejects_empty_or_unaligned_token_counts():
    with pytest.raises(ValueError, match="align"):
        _annotate_opd_token_mean_packs([{}], [], gradient_accumulation_steps=None)
    with pytest.raises(ValueError, match="no response tokens"):
        _annotate_opd_token_mean_packs([{}], [0], gradient_accumulation_steps=None)
    _annotate_opd_token_mean_packs([], [], gradient_accumulation_steps=None)


def test_only_native_opd_loss_is_annotated():
    assert _is_opd_loss_fn(opd_loss_fn)
    assert _is_opd_loss_fn(partial(opd_loss_fn, opd_coef=2.0))
    assert _is_opd_loss_fn(partial(partial(opd_loss_fn, opd_coef=2.0), max_log_ratio=5.0))
    assert not _is_opd_loss_fn(lambda pack, lp: opd_loss_fn(pack, lp))


def test_packed_forward_selected_logprobs_and_gradients_equal_padded(monkeypatch):
    # Use AReno's REAL selected-token logprob kernels on CPU (TP world=1).
    import areno.engine.runtime.logprobs as lp_mod

    monkeypatch.setattr(lp_mod, "get_tp_context", lambda: SimpleNamespace(rank=0, group=None, world_size=1))
    seqs = _sequences()
    padded = _make_train_pack(seqs)
    packed = _pack_train_data(padded)
    torch.manual_seed(22)
    logits = torch.nn.Parameter(torch.randn(3, 5, 75))

    dense_selected = next_token_logprobs(logits, padded["input_ids"])
    packed_logits = torch.cat(
        [logits[i, :len(s.tokens)] for i, s in enumerate(seqs)], dim=0
    ).unsqueeze(0)
    flat_selected = packed_next_token_logprobs(packed_logits, packed["input_ids"], packed["train_cu_seqlens"])

    real_selected = torch.cat([dense_selected[i, :len(s.tokens)-1] for i, s in enumerate(seqs)])
    torch.testing.assert_close(flat_selected, real_selected, atol=1e-6, rtol=1e-6)
    dense_loss, _ = opd_loss_fn(padded, dense_selected)
    packed_loss, _ = opd_loss_fn(packed, flat_selected)
    torch.testing.assert_close(dense_loss, packed_loss, atol=1e-6, rtol=1e-6)
    dense_grad = torch.autograd.grad(dense_loss, logits, retain_graph=True)[0]
    packed_grad = torch.autograd.grad(packed_loss, logits)[0]
    torch.testing.assert_close(dense_grad, packed_grad, atol=1e-6, rtol=1e-6)


class FakeEngine:
    """Exercise real ArenoBackend.train() pack creation with a CPU optimizer."""

    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor(0.05))
        self.optimizer = torch.optim.SGD([self.parameter], lr=0.1)
        self.last_packs = None

    def step(self, packs, *, gradient_accumulation_steps=None):
        self.last_packs = packs
        results = []
        steps = len(packs) if gradient_accumulation_steps is None else gradient_accumulation_steps
        for i, pack in enumerate(packs):
            logprobs = _current(_pack_train_data(pack), self.parameter)
            packed = _pack_train_data(pack)
            loss, metrics = pack["_loss_fn"](packed, logprobs)
            group_start = (i // steps) * steps
            group_size = min(steps, len(packs) - group_start)
            (loss / group_size).backward()
            do_step = (i + 1) % steps == 0 or i == len(packs) - 1
            if do_step:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            results.append(SimpleNamespace(loss=float(loss.detach()), metrics={
                k: float(v.detach()) for k,v in metrics.items()}, stepped=do_step))
        return results


def test_backend_real_train_path_updates_student_and_keeps_teacher_frozen():
    from areno.api.backend.areno.backend import ArenoBackend

    backend = ArenoBackend()
    fake = FakeEngine()
    backend._train_engine = fake
    loss_fn = partial(opd_loss_fn, opd_coef=1.2)
    before = float(fake.parameter.detach())
    result = backend.train(None, _sequences(), loss_fn, mini_bs=1)
    assert float(fake.parameter.detach()) != pytest.approx(before)
    assert "loss" in result and "opd_response_tokens" in result
    assert len(fake.last_packs) == 3
    assert {pack["_opd_total_response_tokens"] for pack in fake.last_packs} == {7}
    assert {pack["_opd_grad_scale"] for pack in fake.last_packs} == {3}
    assert all("ref_logprobs" in pack for pack in fake.last_packs)
    assert all("values" not in pack and "returns" not in pack for pack in fake.last_packs)


def test_backend_optimizer_update_is_invariant_to_mini_batch_partition():
    outputs = []
    for mini_bs in (1, 2, 3):
        backend = ArenoBackend()
        engine = FakeEngine()
        backend._train_engine = engine
        backend.train(None, _sequences(), partial(opd_loss_fn, opd_coef=1.0), mini_bs=mini_bs)
        outputs.append(engine.parameter.detach().clone())
    for updated in outputs[1:]:
        torch.testing.assert_close(updated, outputs[0], atol=1e-7, rtol=1e-7)


def test_backend_separate_accumulation_groups_use_group_token_counts():
    backend = ArenoBackend()
    engine = FakeEngine()
    backend._train_engine = engine
    backend.train(
        None, _sequences(), partial(opd_loss_fn, opd_coef=1.0), mini_bs=1,
        gradient_accumulation_steps=2,
    )
    assert [p["_opd_total_response_tokens"] for p in engine.last_packs] == [5, 5, 2]
    assert [p["_opd_grad_scale"] for p in engine.last_packs] == [2, 2, 1]
