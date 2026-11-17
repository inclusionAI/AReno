"""Differentiable label readout from the existing MLX LM head."""

import math


def prepare_batch(batch, rows):
    import mlx.core as mx

    questions = [row.features["label_jev"] for row in rows]
    modes = {bool(q.get("bi_attn", False)) for q in questions}
    if len(modes) != 1:
        raise ValueError("cannot mix causal and bidirectional label rows")
    batch["bi_attn"] = modes.pop()
    batch["label_lengths"] = [len(row.tokens) for row in rows]
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

    logits = prompt_logits(provider.model, batch["input_ids"], batch["label_lengths"], bi_attn=batch["bi_attn"])
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


def prompt_logits(model, input_ids, lengths, *, bi_attn=False):
    """One-shot prompt logits. Only full-attention caches get a noncausal mask."""
    if not bi_attn:
        return model(input_ids)
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    if len(lengths) != input_ids.shape[0] or any(n < 1 or n > input_ids.shape[1] for n in lengths):
        raise ValueError("invalid bidirectional prompt lengths")
    key_mask = mx.arange(input_ids.shape[1])[None, None, None, :] < mx.array(lengths)[:, None, None, None]

    class PromptCache(KVCache):
        def make_mask(self, n, *, return_array=False, window_size=None):
            if window_size is not None:
                raise ValueError("bi-attn supports full attention, not sliding-window attention")
            return key_mask

        def update_and_fetch(self, keys, values):
            # No in-place writes or cache reuse in a differentiable one-shot prompt.
            return keys, values

    original = make_prompt_cache(model)
    cache = [PromptCache() if isinstance(c, KVCache) else None for c in original]
    if not any(c is not None for c in cache):
        raise ValueError("model has no supported full-attention layers for --bi-attn")
    return model(input_ids, cache=cache)
