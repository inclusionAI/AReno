"""Differentiable label readout from the existing MLX LM head."""

import math


def prepare_batch(batch, rows):
    import mlx.core as mx

    questions = [row.features["label_jev"] for row in rows]
    width = max(len(q["ids"]) for q in questions)
    ids, targets, valid, temperatures = [], [], [], []
    for question in questions:
        labels, target, temperature = question["ids"], question["target"], question["temperature"]
        if len(labels) < 2 or len(set(labels)) != len(labels) or len(labels) != len(target) or min(labels) < 0:
            raise ValueError("invalid label IDs or target shape")
        if any(not math.isfinite(p) or p < 0 for p in target) or not math.isclose(sum(target), 1, abs_tol=1e-5):
            raise ValueError("invalid target distribution")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        padding = width - len(labels)
        ids.append(labels + [0] * padding)
        targets.append(target + [0.0] * padding)
        valid.append([True] * len(labels) + [False] * padding)
        temperatures.append(temperature)
    batch.update(
        label_ids=mx.array(ids),
        label_targets=mx.array(targets, dtype=mx.float32),
        label_valid=mx.array(valid),
        label_temperature=mx.array(temperatures, dtype=mx.float32),
        label_positions=mx.array([len(row.tokens) - 1 for row in rows]),
    )
    return batch


def label_loss(provider, batch):
    import mlx.core as mx

    logits = provider.model(batch["input_ids"])
    logits = logits.logits if hasattr(logits, "logits") else logits
    rows = mx.arange(logits.shape[0])
    logits = logits[rows, batch["label_positions"]].astype(mx.float32)
    selected = mx.take_along_axis(logits, batch["label_ids"], axis=-1)
    selected = mx.where(batch["label_valid"], selected, -float("inf"))
    scaled = selected / batch["label_temperature"][:, None]
    log_probs = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
    loss = -mx.sum(batch["label_targets"] * mx.where(batch["label_valid"], log_probs, 0), axis=-1).mean()
    mass = mx.exp(mx.logsumexp(selected, axis=-1) - mx.logsumexp(logits, axis=-1)).mean()
    accuracy = (mx.argmax(log_probs, axis=-1) == mx.argmax(batch["label_targets"], axis=-1)).astype(mx.float32).mean()
    return loss, {"label_ce": loss, "label_mass": mass, "label_accuracy": accuracy}
