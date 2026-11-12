"""Restricted-label cross entropy on differentiable answer-position LM logits."""


def label_distribution(logits, label_ids, temperature=1.0):
    import torch

    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("expected finite one-dimensional vocabulary logits")
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")
    if (
        len(label_ids) < 2
        or len(set(label_ids)) != len(label_ids)
        or min(label_ids) < 0
        or max(label_ids) >= logits.numel()
    ):
        raise ValueError("expected distinct valid label IDs")
    selected = logits.float()[label_ids]
    return (
        selected - logits.float().logsumexp(0),
        (selected / temperature).softmax(0),
        (selected.logsumexp(0) - logits.float().logsumexp(0)).exp(),
    )


def label_jev_loss_fn(pack, logits):
    import torch

    losses, masses, accuracies = [], [], []
    questions = pack["_label_questions"]
    if len(questions) != logits.shape[0]:
        raise ValueError("one answer-position logit row is required per question")
    for row, question in zip(logits, questions, strict=True):
        ids, temperature = question["ids"], question["temperature"]
        _, probabilities, mass = label_distribution(row, ids, temperature)
        target = row.new_tensor(question["target"], dtype=torch.float32)
        if (
            target.shape != probabilities.shape
            or not torch.isfinite(target).all()
            or (target < 0).any()
            or not torch.isclose(target.sum(), target.new_tensor(1.0), atol=1e-5)
        ):
            raise ValueError("invalid target distribution")
        log_probs = (row.float()[ids] / temperature).log_softmax(0)
        losses.append(-(target * log_probs).sum())
        masses.append(mass.detach())
        accuracies.append((probabilities.argmax() == target.argmax()).float().detach())
    ce = torch.stack(losses).mean()
    loss = ce * pack.get("_label_weight", 1.0)
    return loss, {
        "label_ce": ce.detach(),
        "label_mass": torch.stack(masses).mean(),
        "label_accuracy": torch.stack(accuracies).mean(),
    }
