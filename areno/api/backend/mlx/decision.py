"""MLX decision head, joint head/LoRA updates, and checkpoint persistence."""

import json
import math
from pathlib import Path
from typing import Any


def initialize_head(backend: Any) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from areno.api.backend.mlx.roles import _hidden_size

    hidden = _hidden_size(backend.model_config)
    backend.model.score_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
    # FP32 head and loss stabilize initially small differences between candidates.
    checkpoint = backend.config.decision_checkpoint
    if checkpoint:
        backend.model.score_head.load_weights(str(Path(checkpoint) / "score_head.safetensors"))
    head_config = {"lr": backend.config.head_lr}
    head_optimizer = optim.AdamW(learning_rate=backend.config.head_lr, weight_decay=0.01)
    backend.optimizer = optim.MultiOptimizer(
        [head_optimizer, backend.optimizer],
        filters=[lambda path, _: path.startswith("score_head.")],
    )
    backend._optimizer_groups.append(("head", head_optimizer, head_config))
    mx.eval(backend.model.score_head.parameters())


def question_logits(provider: Any, rows: list[Any]) -> Any:
    """Right padding cannot affect the gathered final valid causal token."""
    import mlx.core as mx

    pooled = provider.pooled_candidate_hidden_states([row.tokens for row in rows])
    return provider.model.score_head(pooled.astype(mx.float32)).squeeze(-1)


def candidate_packs(groups, mini_bs):
    """Fill candidate microbatches while keeping each question's softmax intact."""
    # Sorting only changes execution within this optimizer update. Dataset
    # iteration and per-question loss weights remain unchanged.
    ordered = sorted(groups, key=lambda group: max(len(row.tokens) for row in group))
    pack, count, tokens, longest = [], 0, 0, 0
    for group in ordered:
        length = max(len(row.tokens) for row in group)
        padded_tokens = (count + len(group)) * max(longest, length)
        useful_tokens = tokens + len(group) * length
        if pack and (count + len(group) > mini_bs or padded_tokens > 1.25 * useful_tokens):
            yield pack
            pack, count, tokens, longest = [], 0, 0, 0
        if len(group) > mini_bs:
            yield [group]
            continue
        pack.append(group)
        count += len(group)
        tokens += len(group) * length
        longest = max(longest, length)
    if pack:
        yield pack


def train_questions(
    backend: Any, ctx: Any, rows: list[Any], *, brier_weight: float = 0.5, mini_bs: int = 1
) -> dict[str, float]:
    import time

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map

    from areno.api.backend.mlx.optimizer import apply_optimizer_update, set_group_learning_rates
    from areno.api.backend.mlx.training import clip_grad_norm
    from areno.api.jev.data import group_questions
    from areno.api.jev.loss import jev_loss_fn

    if not getattr(backend.config, "decision_head", False):
        raise ValueError("jev loss requires JevMlxConfig")
    if mini_bs < 1:
        raise ValueError("mini_bs must be positive")
    groups = group_questions(rows)
    backend.model.train()
    started = time.perf_counter()
    rates = set_group_learning_rates(backend._optimizer_groups, ctx.global_step)
    accumulated = None
    stats_sum = {}
    pending = None
    candidate_microbatches = 0

    def record_stats(loss, stats):
        for key, value in {"loss": loss, **stats}.items():
            stats_sum[key] = stats_sum.get(key, 0.0) + float(value.item()) / len(groups)

    def finish_pending():
        nonlocal pending
        if pending is not None:
            loss, stats, gradients = pending
            mx.eval(loss, stats, gradients)
            record_stats(loss, stats)
            pending = None
            mx.clear_cache()

    for pack in candidate_packs(groups, mini_bs):
        group = pack[0]
        target = mx.array([row.features["jev_target"] for row in group], dtype=mx.float32)
        if len(pack) > 1 or len(group) <= mini_bs:
            # The complete question already fits the activation budget. Its
            # exact loss can backpropagate directly, without a scoring replay.
            def objective(model):
                del model
                all_rows = [row for question in pack for row in question]
                logits = question_logits(backend.provider, all_rows)
                losses, sums, offset = [], {}, 0
                for question in pack:
                    probabilities = mx.array([row.features["jev_target"] for row in question], dtype=mx.float32)
                    value, values = jev_loss_fn(
                        probabilities, logits[offset : offset + len(question)], brier_weight=brier_weight
                    )
                    losses.append(value)
                    for key, item in values.items():
                        sums[key] = sums.get(key, 0.0) + item
                    offset += len(question)
                return sum(losses), sums

            (loss, stats), grads = nn.value_and_grad(backend.model, objective)(backend.model)
            accumulated = grads if accumulated is None else tree_map(lambda a, b: a + b, accumulated, grads)
            # Build at most one next graph while the previous microbatch runs
            # on the GPU. Join it before submitting this one to bound activation
            # memory; model weights stay fixed until the final optimizer update.
            finish_pending()
            mx.async_eval(loss, stats, accumulated)
            pending = (loss, stats, accumulated)
            candidate_microbatches += 1
            del grads
            continue
        else:
            finish_pending()
            # First compute the COMPLETE question distribution, without retaining
            # backbone activations. Then apply dL/dscore to small candidate batches.
            # Chain rule gives the same joint gradient as a batched forward, while
            # bounding activation memory by mini_bs rather than candidate count.
            scores = []
            for start in range(0, len(group), mini_bs):
                score = question_logits(backend.provider, group[start : start + mini_bs])
                mx.eval(score)
                scores.append(mx.stop_gradient(score))
                mx.clear_cache()
            (loss, stats), score_grads = mx.value_and_grad(
                lambda logits: jev_loss_fn(target, logits, brier_weight=brier_weight)
            )(mx.concatenate(scores))
            mx.eval(loss, stats, score_grads)
            for start in range(0, len(group), mini_bs):
                candidate_rows = group[start : start + mini_bs]
                weights = score_grads[start : start + len(candidate_rows)]

                def objective(model):
                    del model  # The provider references this same nn.Module.
                    return (question_logits(backend.provider, candidate_rows) * weights).sum()

                _, grads = nn.value_and_grad(backend.model, objective)(backend.model)
                accumulated = grads if accumulated is None else tree_map(lambda a, b: a + b, accumulated, grads)
                mx.eval(accumulated)
                candidate_microbatches += 1
                mx.clear_cache()
        record_stats(loss, stats)
        mx.clear_cache()
    finish_pending()
    grads = tree_map(lambda value: value / len(groups), accumulated)
    leaves = tree_flatten(grads)
    norms = {}
    for name, predicate in (
        ("head", lambda p: p.startswith("score_head.")),
        ("lora", lambda p: p.endswith(("lora_a", "lora_b"))),
    ):
        norms[name + "_grad_norm"] = mx.sqrt(sum((g.astype(mx.float32) ** 2).sum() for p, g in leaves if predicate(p)))
    grads, grad_norm = clip_grad_norm(grads, backend.config.optimizer.get("grad_clip_norm", 1.0))
    mx.eval(grad_norm, norms)
    if not all(math.isfinite(value) for value in stats_sum.values()) or not bool(mx.isfinite(grad_norm).item()):
        raise FloatingPointError("nonfinite JEV loss or gradient; update was not applied")
    before = dict(tree_flatten(backend.model.trainable_parameters()))
    apply_optimizer_update(backend.model, backend.optimizer, grads)
    after = dict(tree_flatten(backend.model.trainable_parameters()))
    for name, predicate in (
        ("head", lambda p: p.startswith("score_head.")),
        ("lora", lambda p: p.endswith(("lora_a", "lora_b"))),
    ):
        norms[name + "_update_norm"] = mx.sqrt(
            sum(
                ((after[p].astype(mx.float32) - value.astype(mx.float32)) ** 2).sum()
                for p, value in before.items()
                if predicate(p)
            )
        )
    mx.eval(norms)
    backend._policy_version += 1
    backend._rollout_version = -1
    return {
        **stats_sum,
        **{key: float(value.item()) for key, value in norms.items()},
        "grad_norm": float(grad_norm.item()),
        "lr": rates["model"],
        "head_lr": rates["head"],
        "questions": len(groups),
        "candidate_microbatches": candidate_microbatches,
        "policy_train_wall_time_s": time.perf_counter() - started,
    }


def save_head(backend: Any, path: str | Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    destination = Path(path)
    mx.save_safetensors(
        str(destination / "score_head.safetensors"), dict(tree_flatten(backend.model.score_head.parameters()))
    )
    (destination / "jev_config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "head": "linear-gelu-linear",
                "pooling": "last_valid_token",
                "temperature": 1.0,
                "base_model": backend.config.base_model_name_or_path or backend.config.model_path,
                "max_seq_len": getattr(backend.config, "max_seq_len", 1024),
            },
            indent=2,
        )
        + "\n"
    )
