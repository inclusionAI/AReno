"""Proper probability losses over all candidates of exactly one question."""


def jev_loss_fn(target, logits, *, brier_weight: float = 0.5):
    import mlx.core as mx

    logits = logits.astype(mx.float32)
    logp = logits - mx.logsumexp(logits)
    probabilities = mx.exp(logp)
    ce = -(target * logp).sum()
    brier = ((probabilities - target) ** 2).sum()
    return ce + brier_weight * brier, {
        "jev_ce": ce,
        "jev_brier": brier,
        "jev_top1": (mx.argmax(probabilities) == mx.argmax(target)).astype(mx.float32),
    }
