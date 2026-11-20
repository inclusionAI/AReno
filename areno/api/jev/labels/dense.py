"""Teacher-anchored dense Capability proxy on candidate-label logits."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean

from areno.api.jev.labels.config import DEFAULT_DENSE_REWARD_WEIGHT
from areno.api.jev.labels.protocol import encode_question, training_row
from areno.api.jev.labels.trainer import LabelJevTrainer

_TIER_WEIGHTS = {"easy": 0.1, "standard": 0.2, "hard": 0.4, "judge": 0.3}


@dataclass(frozen=True)
class CapabilityTarget:
    kind: str
    tier: str
    gold: int
    gold_probs: tuple[float, ...] | None = None

    @classmethod
    def from_question(cls, question, names):
        spec = question.get("capability")
        if not isinstance(spec, dict) or spec.get("tier") not in _TIER_WEIGHTS:
            raise ValueError("sgl-jev-dense requires a capability tier on each question")
        gold = spec.get("gold")
        if isinstance(gold, bool):
            gold = str(gold).lower()
        if str(gold) not in names or gold is None:
            raise ValueError("capability gold must name a valid candidate")
        distribution = spec.get("gold_probs")
        if distribution is not None:
            if question["type"] != "choice" or not isinstance(distribution, dict) or set(distribution) != set(names):
                raise ValueError("capability gold_probs must map all choice candidates to probabilities")
            distribution = tuple(float(distribution[name]) for name in names)
            if any(not math.isfinite(p) or p < 0 for p in distribution) or not math.isclose(
                sum(distribution), 1, abs_tol=1e-5
            ):
                raise ValueError("invalid capability gold_probs")
        return cls(question["type"], spec["tier"], names.index(str(gold)), distribution)


def dense_reward_weights(targets, widths):
    """Weights and ordinal chance baselines for one full optimizer batch."""
    types = Counter(target.kind for target in targets)
    cells = Counter((target.kind, target.tier) for target in targets)
    tiers = defaultdict(float)
    score_chances = defaultdict(list)
    for kind, tier in cells:
        tiers[kind] += _TIER_WEIGHTS[tier]
    for target, width in zip(targets, widths, strict=True):
        if target.kind == "score":
            gap = abs((width - 1) / 2 - target.gold) / (width - 1)
            score_chances[target.tier].append(gap)
    chances = {tier: fmean(gaps) for tier, gaps in score_chances.items()}
    for tier, chance in chances.items():
        if not chance:
            chances[tier] = fmean(
                fmean(abs(i - t.gold) for i in range(width)) / (width - 1)
                for t, width in zip(targets, widths, strict=True)
                if t.kind == "score" and t.tier == tier
            )
    scale = len(targets) / len(types)
    for target in targets:
        i_weight = _TIER_WEIGHTS[target.tier] / (tiers[target.kind] * cells[target.kind, target.tier])
        yield scale * i_weight, scale / types[target.kind], chances.get(target.tier)


class DenseLabelJevTrainer(LabelJevTrainer):
    def _encode(self, question, tokenizer):
        state, prompt, target = question
        encoded = encode_question(state, prompt, tokenizer, target=target, max_seq_len=self.config.max_seq_len)
        row = training_row(encoded, temperature=self.config.label_temperature)
        row.features["capability_target"] = CapabilityTarget.from_question(prompt, encoded["names"])
        return row

    @staticmethod
    def _batch_targets(rows):
        targets = [row.features.get("capability_target") if isinstance(row.features, dict) else None for row in rows]
        if not all(isinstance(target, CapabilityTarget) for target in targets):
            raise ValueError("sgl-jev-dense requires raw questions with capability gold and tier")
        return targets

    def _train_batch(self, rows, max_updates):
        targets = self._batch_targets(rows)
        questions = [row.features["label_jev"] for row in rows]
        widths = [len(question["ids"]) for question in questions]
        for question, target, weight in zip(questions, targets, dense_reward_weights(targets, widths), strict=True):
            question["dense_reward"] = target, weight
            question["dense_reward_weight"] = self.config.dense_reward_weight
        yield from super()._train_batch(rows, max_updates)


def _dense_components_tensor(target, p, score_chance):
    import torch

    gold_p = p[target.gold]
    if target.kind == "choice":
        chance = 1 / p.numel()
        intelligence = (gold_p - chance) / (1 - chance)
        gold = p.new_tensor(target.gold_probs or tuple(i == target.gold for i in range(p.numel())))
        calibration = 1 - 0.5 * (p - gold).square().sum()
        if target.gold_probs is not None:
            calibration = 0.5 * (calibration + 1 - 0.5 * (p - gold).abs().sum())
    elif target.kind == "noul":
        decisive = torch.sigmoid((gold_p - 0.8) / 0.1)
        intelligence = gold_p + decisive - 1
        calibration = 1 - (1 - gold_p).square()
    else:
        levels = torch.arange(p.numel(), device=p.device, dtype=p.dtype)
        error = ((levels * p).sum() - target.gold).abs() / (p.numel() - 1)
        intelligence = 1 - error / score_chance
        gold_cdf = (levels[:-1] >= target.gold).to(p.dtype)
        calibration = 1 - (p[:-1].cumsum(0) - gold_cdf).square().mean()
    return intelligence, calibration


def dense_label_jev_loss(pack, logits):
    import torch

    intelligence, calibration, ce, brier = [], [], [], []
    for row, question in zip(logits, pack["_label_questions"], strict=True):
        target, (i_weight, c_weight, chance) = question["dense_reward"]
        log_probs = (row.float()[question["ids"]] / question["temperature"]).log_softmax(0)
        probabilities = log_probs.exp()
        teacher = probabilities.new_tensor(question["target"])
        ce.append(-(teacher * log_probs).sum())
        brier.append((probabilities - teacher).square().sum())
        i_score, c_score = _dense_components_tensor(target, probabilities, chance)
        intelligence.append(i_weight * i_score)
        calibration.append(c_weight * c_score)
    weight = pack.get("_label_weight", 1.0)
    i_mean, c_mean = torch.stack(intelligence).mean() * weight, torch.stack(calibration).mean() * weight
    reward = 0.5 * (i_mean + c_mean)
    metrics = {
        "dense_reward": 100 * reward.detach(),
        "dense_intelligence": 100 * i_mean.detach(),
        "dense_calibration": 100 * c_mean.detach(),
    }
    ce_mean, brier_mean = torch.stack(ce).mean() * weight, torch.stack(brier).mean() * weight
    proper = ce_mean + 0.5 * brier_mean
    metrics.update(teacher_ce=ce_mean.detach(), teacher_brier=brier_mean.detach(), proper_loss=proper.detach())
    coefficient = pack["_label_questions"][0].get("dense_reward_weight", DEFAULT_DENSE_REWARD_WEIGHT)
    metrics["dense_reward_weight"] = proper.new_tensor(coefficient)
    return proper - coefficient * reward, metrics
