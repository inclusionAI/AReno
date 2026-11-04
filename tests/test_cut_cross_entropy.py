"""CPU + CUDA tests for the opt-in cut cross-entropy (CCE) LM-head path.

The reference implementation is pure PyTorch, so the numeric contract
(``F.cross_entropy`` equivalence, ignore-index handling, bf16 accumulation,
gradients) is pinned on CPU. The peak-memory assertion needs CUDA and is
skipped elsewhere.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from areno.accel.cut_cross_entropy import cut_cross_entropy, cut_cross_entropy_logprobs
from areno.engine.runtime import logprobs as logprob_ops
from tests.helpers import PatchedContext, single_tp_context


def _reference_logprobs(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = hidden @ weight.transpose(0, 1)
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets[:, None]).squeeze(-1)


def test_matches_cross_entropy_with_vocab_not_divisible_by_chunk():
    torch.manual_seed(0)
    hidden = torch.randn(5, 6)
    weight = torch.randn(37, 6)
    targets = torch.randint(37, (5,))

    expected = F.cross_entropy(hidden @ weight.T, targets)
    actual = cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=8)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_reductions_match_cross_entropy(reduction):
    torch.manual_seed(1)
    hidden = torch.randn(9, 4)
    weight = torch.randn(23, 4)
    targets = torch.randint(23, (9,))

    expected = F.cross_entropy(hidden @ weight.T, targets, reduction=reduction)
    actual = cut_cross_entropy(hidden, weight, targets, reduction=reduction, vocab_chunk_size=5)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_single_row_single_token():
    hidden = torch.tensor([[0.5, -1.5, 2.0]])
    weight = torch.randn(3, 3)
    targets = torch.tensor([2])

    expected = F.cross_entropy(hidden @ weight.T, targets)
    actual = cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=2)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_batched_leading_dims_are_flattened():
    torch.manual_seed(2)
    hidden = torch.randn(2, 3, 5)
    weight = torch.randn(11, 5)
    targets = torch.randint(11, (2, 3))

    expected = F.cross_entropy((hidden @ weight.T).reshape(-1, 11), targets.reshape(-1))
    actual = cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=4)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_ignore_index_rows_are_excluded():
    torch.manual_seed(3)
    hidden = torch.randn(7, 4)
    weight = torch.randn(19, 4)
    targets = torch.randint(19, (7,))
    targets[[0, 3, 6]] = -100

    expected = F.cross_entropy(hidden @ weight.T, targets, ignore_index=-100)
    actual = cut_cross_entropy(hidden, weight, targets, ignore_index=-100, vocab_chunk_size=6)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    expected_rows = F.cross_entropy(hidden @ weight.T, targets, ignore_index=-100, reduction="none")
    actual_rows = cut_cross_entropy(hidden, weight, targets, ignore_index=-100, reduction="none", vocab_chunk_size=6)
    torch.testing.assert_close(actual_rows, expected_rows, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual_rows[[0, 3, 6]], torch.zeros(3))


def test_custom_ignore_index():
    torch.manual_seed(4)
    hidden = torch.randn(4, 3)
    weight = torch.randn(7, 3)
    targets = torch.tensor([0, 3, 7, 7])

    expected = F.cross_entropy(hidden @ weight.T, targets, ignore_index=7)
    actual = cut_cross_entropy(hidden, weight, targets, ignore_index=7, vocab_chunk_size=2)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_all_targets_ignored_returns_zero_and_stays_finite():
    hidden = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    targets = torch.full((4,), -100)

    for reduction in ("sum", "mean"):
        loss = cut_cross_entropy(hidden, weight, targets, reduction=reduction, vocab_chunk_size=2)
        assert loss.item() == 0.0
        assert torch.isfinite(loss)
    loss = cut_cross_entropy(hidden, weight, targets, reduction="mean", vocab_chunk_size=2)
    loss.backward()
    assert hidden.grad is not None and weight.grad is not None
    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))


def test_empty_rows_are_allowed():
    hidden = torch.empty(0, 3)
    weight = torch.randn(5, 3)
    targets = torch.empty(0, dtype=torch.long)

    assert cut_cross_entropy(hidden, weight, targets, reduction="mean").item() == 0.0
    assert cut_cross_entropy(hidden, weight, targets, reduction="none").shape == (0,)


def test_logprobs_match_log_softmax_gather():
    torch.manual_seed(5)
    hidden = torch.randn(6, 5)
    weight = torch.randn(29, 5)
    targets = torch.randint(29, (6,))

    expected = _reference_logprobs(hidden, weight, targets)
    actual = cut_cross_entropy_logprobs(hidden, weight, targets, vocab_chunk_size=7)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert actual.dtype == torch.float32


def test_logit_softcap_matches_manual_softmax():
    torch.manual_seed(6)
    hidden = torch.randn(5, 4)
    weight = torch.randn(17, 4)
    targets = torch.randint(17, (5,))
    cap = 2.0

    logits = cap * torch.tanh((hidden @ weight.T) / cap)
    expected = F.cross_entropy(logits, targets)
    actual = cut_cross_entropy(hidden, weight, targets, logit_softcap=cap, vocab_chunk_size=5)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_gradients_match_cross_entropy():
    torch.manual_seed(7)
    reference_hidden = torch.randn(8, 5, requires_grad=True)
    reference_weight = torch.randn(31, 5, requires_grad=True)
    hidden = reference_hidden.detach().clone().requires_grad_(True)
    weight = reference_weight.detach().clone().requires_grad_(True)
    targets = torch.randint(31, (8,))

    F.cross_entropy(reference_hidden @ reference_weight.T, targets).backward()
    cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=6).backward()

    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=1e-5, rtol=1e-5)


def test_duplicate_targets_accumulate_weight_gradients():
    torch.manual_seed(8)
    hidden = torch.randn(6, 3)
    weight = torch.randn(9, 3, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    targets = torch.tensor([2, 2, 2, 5, 5, 8])

    F.cross_entropy(hidden @ reference_weight.T, targets).backward()
    cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=2).backward()

    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=1e-5, rtol=1e-5)


def test_bfloat16_accumulates_in_float32():
    torch.manual_seed(9)
    hidden = torch.randn(6, 4, dtype=torch.bfloat16)
    weight = torch.randn(41, 4, dtype=torch.bfloat16)
    targets = torch.randint(41, (6,))

    expected = F.cross_entropy((hidden @ weight.T).float(), targets)
    actual = cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=7)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=1e-2)


def test_float16_accumulates_in_float32():
    torch.manual_seed(10)
    hidden = torch.randn(4, 4, dtype=torch.float16)
    weight = torch.randn(13, 4, dtype=torch.float16)
    targets = torch.randint(13, (4,))

    expected = F.cross_entropy((hidden @ weight.T).float(), targets)
    actual = cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=3)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=1e-2)


def test_label_smoothing_is_rejected_instead_of_mis_scored():
    with pytest.raises(NotImplementedError, match="label_smoothing"):
        cut_cross_entropy(torch.randn(2, 3), torch.randn(5, 3), torch.tensor([0, 1]), label_smoothing=0.1)


def test_invalid_reduction_is_rejected():
    with pytest.raises(ValueError, match="reduction"):
        cut_cross_entropy(torch.randn(2, 3), torch.randn(5, 3), torch.tensor([0, 1]), reduction="batchmean")


@pytest.mark.parametrize("chunk", [0, -1, 1.5, True])
def test_invalid_vocab_chunk_size_is_rejected(chunk):
    with pytest.raises(ValueError, match="vocab_chunk_size"):
        cut_cross_entropy(torch.randn(2, 3), torch.randn(5, 3), torch.tensor([0, 1]), vocab_chunk_size=chunk)


def test_shape_mismatches_are_rejected():
    with pytest.raises(ValueError, match="target_ids"):
        cut_cross_entropy(torch.randn(3, 4), torch.randn(5, 4), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="hidden dim"):
        cut_cross_entropy(torch.randn(3, 3), torch.randn(5, 4), torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError, match="2-D"):
        cut_cross_entropy(torch.randn(3, 4), torch.randn(5, 4, 1), torch.tensor([0, 1, 2]))


def test_non_floating_weight_is_rejected():
    with pytest.raises(TypeError, match="floating-point"):
        cut_cross_entropy(torch.randn(2, 3), torch.zeros(5, 3, dtype=torch.int64), torch.tensor([0, 1]))


def _record_widths(widths, fn):
    def wrapped(hidden, weight_chunk, softcap):
        widths.append(weight_chunk.shape[0])
        return fn(hidden, weight_chunk, softcap)

    return wrapped


def test_large_vocab_stays_within_chunked_memory_bound():
    """Each projection block is bounded by rows * vocab_chunk, not rows * vocab."""

    from areno.accel import cut_cross_entropy as cce_module

    torch.manual_seed(11)
    hidden = torch.randn(128, 32)
    weight = torch.randn(1024, 32)
    targets = torch.randint(1024, (128,))

    seen_widths = []
    with PatchedContext(cce_module, _chunk_row_stats=_record_widths(seen_widths, cce_module._chunk_row_stats)):
        cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=100)

    assert seen_widths
    assert max(seen_widths) <= 100


def test_packed_head_flag_off_never_touches_cut_cross_entropy(monkeypatch):
    """The default path must not call into the new module at all."""

    from areno.accel import cut_cross_entropy as cce_module

    def boom(*args, **kwargs):
        raise AssertionError("cut_cross_entropy must stay off unless requested")

    monkeypatch.setattr(cce_module, "cut_cross_entropy_logprobs", boom)

    head = nn.Linear(8, 17, bias=False)
    hidden = torch.randn(1, 11, 8, requires_grad=True)
    tokens = torch.randint(17, (1, 11))
    with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
        logps = logprob_ops.packed_next_token_logprobs_from_hidden(
            hidden, tokens, torch.tensor([0, 11]), head, chunk_size=4
        )
    assert logps.shape == (10,)


def test_packed_head_flag_off_matches_legacy_bitwise():
    head = nn.Linear(8, 17, bias=False)
    hidden = torch.randn(1, 11, 8, requires_grad=True)
    tokens = torch.randint(17, (1, 11))
    cu = torch.tensor([0, 11])

    with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
        default = logprob_ops.packed_next_token_logprobs_from_hidden(hidden, tokens, cu, head, chunk_size=4)
        explicit_off = logprob_ops.packed_next_token_logprobs_from_hidden(
            hidden, tokens, cu, head, chunk_size=4, cut_cross_entropy=False
        )

    assert torch.equal(default, explicit_off)


def test_packed_head_flag_on_uses_cut_cross_entropy(monkeypatch):
    from areno.accel import cut_cross_entropy as cce_module

    calls = []
    original = cce_module.cut_cross_entropy_logprobs

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cce_module, "cut_cross_entropy_logprobs", spy)

    head = nn.Linear(8, 17, bias=False)
    hidden = torch.randn(1, 11, 8)
    tokens = torch.randint(17, (1, 11))
    with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
        logprob_ops.packed_next_token_logprobs_from_hidden(
            hidden, tokens, torch.tensor([0, 11]), head, cut_cross_entropy=True
        )

    assert calls == [1]


def test_packed_head_cut_cross_entropy_matches_full_projection():
    torch.manual_seed(12)
    head = nn.Linear(8, 37, bias=False)
    reference_head = nn.Linear(8, 37, bias=False)
    reference_head.load_state_dict(head.state_dict())
    hidden = torch.randn(1, 13, 8, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    tokens = torch.randint(37, (1, 13))
    cu = torch.tensor([0, 5, 13])

    positions = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
    labels = tokens.reshape(-1)[positions + 1]
    expected = _reference_logprobs(
        reference_hidden.reshape(-1, 8).index_select(0, positions), reference_head.weight, labels
    )
    upstream = torch.linspace(-1.0, 1.0, expected.numel())
    (expected * upstream).sum().backward()

    with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
        actual = logprob_ops.packed_next_token_logprobs_from_hidden(
            hidden, tokens, cu, head, cut_cross_entropy=True
        )
    (actual * upstream).sum().backward()

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(head.weight.grad, reference_head.weight.grad, atol=1e-5, rtol=1e-5)


def test_packed_head_cut_cross_entropy_honors_softcap():
    torch.manual_seed(13)
    head = nn.Linear(8, 23, bias=False)
    hidden = torch.randn(1, 9, 8)
    tokens = torch.randint(23, (1, 9))
    cap = 3.0

    positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    labels = tokens.reshape(-1)[positions + 1]
    logits = head(hidden.reshape(-1, 8).index_select(0, positions))
    expected = torch.log_softmax((cap * torch.tanh(logits / cap)).float(), dim=-1).gather(-1, labels[:, None])
    expected = expected.squeeze(-1)

    with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
        actual = logprob_ops.packed_next_token_logprobs_from_hidden(
            hidden, tokens, torch.tensor([0, 9]), head, logit_softcap=cap, cut_cross_entropy=True
        )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_packed_head_cut_cross_entropy_rejects_tensor_parallel():
    import types

    head = nn.Linear(8, 17, bias=False)
    hidden = torch.randn(1, 5, 8)
    tokens = torch.randint(17, (1, 5))

    def two_rank_context():
        return types.SimpleNamespace(rank=0, world_size=2, group=None)

    with PatchedContext(logprob_ops, get_tp_context=two_rank_context):
        with pytest.raises(NotImplementedError, match="tp_size=1"):
            logprob_ops.packed_next_token_logprobs_from_hidden(
                hidden, tokens, torch.tensor([0, 5]), head, cut_cross_entropy=True
            )


def test_backward_never_saves_a_full_logits_block():
    """The autograd graph keeps only row/chunk-sized tensors, never [rows, vocab]."""

    rows, hidden_size, vocab, chunk = 64, 8, 512, 64
    torch.manual_seed(15)
    hidden = torch.randn(rows, hidden_size, requires_grad=True)
    weight = torch.randn(vocab, hidden_size, requires_grad=True)
    targets = torch.randint(vocab, (rows,))

    saved_numel = []

    def pack(tensor):
        saved_numel.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=chunk)
        loss.backward()

    full_logits_numel = rows * vocab
    assert saved_numel, "expected the CCE graph to save small tensors for backward"
    assert max(saved_numel) * 4 < full_logits_numel, (
        f"largest saved tensor {max(saved_numel)} numel vs full logits {full_logits_numel}"
    )


def test_config_defaults_keep_cut_cross_entropy_off():
    try:
        from areno.api.trainer_config import TrainerConfig
    except RuntimeError as exc:  # dev boxes without a native backend
        pytest.skip(f"trainer config unavailable on this host: {exc}")

    from areno.engine.config import RuntimeConfig

    assert RuntimeConfig().cut_cross_entropy is False

    config = TrainerConfig(algo="sft", ckpt="dummy", dataset_path="dummy", backend="cuda")
    assert config.cut_cross_entropy is False
    assert config.cuda_config().runtime["cut_cross_entropy"] is False

    enabled = TrainerConfig(
        algo="sft", ckpt="dummy", dataset_path="dummy", backend="cuda", cut_cross_entropy=True
    )
    assert enabled.cuda_config().runtime["cut_cross_entropy"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="peak-memory assertions require CUDA")
def test_peak_memory_below_materialized_logits_cuda():
    torch.manual_seed(14)
    rows, hidden_size, vocab = 1024, 256, 32768
    hidden = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(vocab, (rows,), device="cuda")

    def peak_bytes(run):
        hidden.grad = None
        weight.grad = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        run().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    materialized = peak_bytes(lambda: F.cross_entropy(hidden @ weight.T, targets))
    fused = peak_bytes(lambda: cut_cross_entropy(hidden, weight, targets, vocab_chunk_size=2048))

    assert fused < materialized / 2, f"cce peak {fused} vs materialized {materialized}"
