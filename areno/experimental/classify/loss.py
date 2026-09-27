"""Grouped-softmax objective for sequence-scoring (classification) training.

Each training row is one candidate of one question. The actor's score head
turns every row into a scalar logit; the candidates of a question share one
softmax, which is fitted to the question's target distribution with
cross-entropy plus an optional Brier term (JevForge's `question_loss`).

Rows carry three `sequence_labels`:

- `group`: question id inside the optimizer step; `-1` marks padding rows.
- `target`: target probability of this candidate.
- `weight`: per-question loss weight chosen by the trainer so the DP- and
  microbatch-averaged gradient equals the mean over all questions in a step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def classify_loss_fn(data_pack, scores, *, brier_weight: float = 0.5):
    """Worker-side entry: `scores` holds one logit per packed sequence."""

    labels = data_pack.get("sequence_labels")
    if not isinstance(labels, dict) or not {"group", "target", "weight"} <= set(labels):
        raise ValueError("classify loss requires sequence_labels with group, target, and weight")
    device = scores.device
    return grouped_softmax_loss(
        scores,
        labels["group"].to(device=device),
        labels["target"].to(device=device),
        labels["weight"].to(device=device),
        brier_weight=brier_weight,
    )


def grouped_softmax_loss(
    scores: torch.Tensor,
    group: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    brier_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum over questions of `CE + brier_weight * Brier`."""

    import torch

    scores = scores.float()
    # Keeps the graph connected when this rank holds only padding rows.
    anchor = scores.sum() * 0.0
    valid = group >= 0
    if not bool(valid.any()):
        zero = anchor.detach()
        return anchor, {
            "classify_ce": zero,
            "classify_brier": zero,
            "classify_top1": zero,
            "classify_questions_per_rank": zero,
        }
    logits = scores[valid]
    target = target[valid].float()
    weight = weight[valid].float()
    _, index = torch.unique(group[valid].long(), return_inverse=True)
    num_groups = int(index.max()) + 1

    def segment_max(values: torch.Tensor) -> torch.Tensor:
        out = values.new_full((num_groups,), float("-inf"))
        return out.scatter_reduce(0, index, values, reduce="amax", include_self=True)

    def segment_sum(values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros(num_groups).index_add(0, index, values)

    shifted = logits - segment_max(logits.detach())[index]
    log_probs = shifted - torch.log(segment_sum(torch.exp(shifted)))[index]
    probs = torch.exp(log_probs)
    ce = segment_sum(-(target * log_probs))
    brier = segment_sum((probs - target).square())
    group_weight = weight.new_zeros(num_groups).scatter_reduce(0, index, weight, reduce="amax", include_self=False)
    loss = (group_weight * (ce + brier_weight * brier)).sum() + anchor

    with torch.no_grad():
        predicted_top = log_probs >= segment_max(log_probs)[index]
        target_top = target >= segment_max(target)[index]
        top1 = segment_sum((predicted_top & target_top).float()) > 0
    return loss, {
        "classify_ce": ce.detach().mean(),
        "classify_brier": brier.detach().mean(),
        "classify_top1": top1.float().mean(),
        "classify_questions_per_rank": torch.tensor(float(num_groups), device=scores.device),
    }


__all__ = ["classify_loss_fn", "grouped_softmax_loss"]
