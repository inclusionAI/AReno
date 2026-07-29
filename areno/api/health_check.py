"""Startup-window health checks for the first training updates (Issue #249).

This module is a thin, side-effect-free core that inspects a snapshot of the
signals already produced by a normal rollout→train step and classifies them as
``pass`` / ``warn`` / ``fail``. It deliberately reuses the trainer-layer signal
contract (`collect_train_batch_stats` response lengths / rewards / skipped-long
counts plus the backend train result's ``loss`` and ``grad_zero_ratio``) rather
than building a parallel collection path.

The pure functions here are CPU-testable and fault-injectable without a GPU.
The coordinator-side wiring that feeds them lives on `Trainer`.

Design rules (from the issue):
- ``enabled=False`` is the default and must produce no artifact / metric / log.
- Failures annotate ``stage`` + ``input`` (config field or batch id) + a
  ``metric_ref`` into the existing TensorBoard namespace; they never embed
  training-sample text and never swallow the original error.
- NaN/Inf in loss or reward is a dedicated ``fail`` rather than a threshold
  comparison that silently masks it.
- Thresholds ship conservative (warn-prone, not fail-prone) defaults.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

# Status ordering used by the aggregator: higher = more severe.
_STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}
_VALID_STATUS = ("pass", "warn", "fail")

# Metric references into the existing TensorBoard namespaces written by
# `MetricsRecorder.record_training_stats`. Keeping these stable lets the
# coordinator summary "link to the underlying metrics" as the issue requires.
_METRIC_REF = {
    "effective_tokens": "metrics/rollout/response_len_mean",
    "reward_variance": "metrics/rollout/rewards_std",
    "loss_change": "metrics/train/loss",
    "skipped_batches": "metrics/rollout/skipped_long",
}

# Stage labels reused by every check; ``trainer`` covers backend-reported
# signals (loss / grad-zero), ``rollout`` covers sample-side signals.
_STAGE = {
    "effective_tokens": "trainer",
    "reward_variance": "trainer",
    "loss_change": "trainer",
    "skipped_batches": "rollout",
}


class HealthCheckConfigError(ValueError):
    """Raised when a `HealthCheckConfig` violates the input contract.

    Subclassing `ValueError` keeps CLI validation surfacing consistent with the
    rest of the trainer-config family (see `TrainerConfig.__post_init__`).
    """


class HealthCheckError(RuntimeError):
    """Raised when a finalized report is ``fail`` and ``on_fail == 'fail'``.

    The message carries the affected stage(s) + triggering input(s) only; it
    never contains sample text. The originating check messages are attached so
    the original signal context is preserved (not swallowed).

    This exception is raised by
    ``areno.api.trainer.TrainingHealthChecker._evaluate()``, not by the pure
    check functions in this module.
    """


@dataclass(slots=True)
class EffectiveTokensCheckConfig:
    """Thresholds for the effective-token check.

    ``min_per_batch`` is a soft floor: windows whose per-batch mean effective
    token count drops below it warn. A window with zero effective tokens across
    every batch fails when ``fail_if_zero`` is set (the default).
    """

    min_per_batch: int = 0
    fail_if_zero: bool = True


@dataclass(slots=True)
class RewardVarianceCheckConfig:
    """Thresholds for the reward-variance check.

    ``require_variation`` is task-declared: set ``False`` for tasks whose
    reward is legitimately constant (the check then short-circuits to ``pass``),
    ``True`` when reward must vary. With variation required, ``std == 0`` fails
    and ``std < min_std_warn`` warns.
    """

    enabled: bool = True
    require_variation: bool = True
    min_std_warn: float = 1.0e-6
    min_std_fail: float = 0.0


@dataclass(slots=True)
class LossChangeCheckConfig:
    """Thresholds for the loss-change check.

    A window whose first/last loss delta is exactly zero (with at least two
    samples) fails. A delta below ``min_abs_delta_warn`` warns. ``mode`` selects
    absolute (``|last - first|``) or relative (``|last - first| / max(|first|, eps)``)
    comparison.
    """

    enabled: bool = True
    min_abs_delta_warn: float = 1.0e-8
    min_abs_delta_fail: float = 0.0
    mode: str = "absolute"


@dataclass(slots=True)
class SkippedBatchesCheckConfig:
    """Thresholds for the skipped-batch check.

    Skipped is approximated from two existing signals: rollout-side
    ``skipped_long`` (overlong prompts dropped before training) and the backend
    ``grad_zero_ratio`` (a NaN / empty-gradient proxy). The ratio is
    ``skipped_long / total_batches``; a separate ``max_grad_zero_ratio_fail``
    guards the gradient proxy. A zero-denominator window (no batches at all) is
    itself a fail pointing at the data/input contract.
    """

    enabled: bool = True
    max_ratio_warn: float = 0.1
    max_ratio_fail: float = 0.5
    max_grad_zero_ratio_warn: float = 0.5
    max_grad_zero_ratio_fail: float = 1.0


@dataclass(slots=True)
class HealthCheckConfig:
    """Top-level health-check configuration.

    Defaults are conservative: ``enabled=False`` (full backward compatibility),
    a 20-update startup window, and ``on_fail='warn'`` so a degenerate signal
    surfaces without aborting a run the user may want to inspect.
    """

    enabled: bool = False
    startup_window_updates: int = 20
    on_fail: str = "warn"
    effective_tokens: EffectiveTokensCheckConfig = field(default_factory=EffectiveTokensCheckConfig)
    reward_variance: RewardVarianceCheckConfig = field(default_factory=RewardVarianceCheckConfig)
    loss_change: LossChangeCheckConfig = field(default_factory=LossChangeCheckConfig)
    skipped_batches: SkippedBatchesCheckConfig = field(default_factory=SkippedBatchesCheckConfig)

    def __post_init__(self) -> None:
        # Early validation runs at config construction time — on the CLI path
        # this is before `Trainer.init()` spawns workers, satisfying the issue's
        # "validate those inputs before expensive initialization".
        if self.startup_window_updates < 1:
            raise HealthCheckConfigError(f"startup_window_updates must be >= 1, got {self.startup_window_updates!r}")
        if self.on_fail not in ("warn", "fail"):
            raise HealthCheckConfigError(f"on_fail must be one of: warn, fail, got {self.on_fail!r}")
        etc = self.effective_tokens
        if etc.min_per_batch < 0:
            raise HealthCheckConfigError(f"effective_tokens.min_per_batch must be >= 0, got {etc.min_per_batch!r}")
        rvc = self.reward_variance
        if rvc.min_std_warn < 0 or rvc.min_std_fail < 0:
            raise HealthCheckConfigError("reward_variance thresholds must be non-negative")
        if rvc.min_std_warn < rvc.min_std_fail:
            raise HealthCheckConfigError("reward_variance.min_std_warn must be >= min_std_fail")
        lcc = self.loss_change
        if lcc.mode not in ("absolute", "relative"):
            raise HealthCheckConfigError(f"loss_change.mode must be one of: absolute, relative, got {lcc.mode!r}")
        if lcc.min_abs_delta_warn < 0 or lcc.min_abs_delta_fail < 0:
            raise HealthCheckConfigError("loss_change thresholds must be non-negative")
        if lcc.min_abs_delta_warn < lcc.min_abs_delta_fail:
            raise HealthCheckConfigError("loss_change.min_abs_delta_warn must be >= min_abs_delta_fail")
        sbc = self.skipped_batches
        for name, value in (
            ("max_ratio_warn", sbc.max_ratio_warn),
            ("max_ratio_fail", sbc.max_ratio_fail),
            ("max_grad_zero_ratio_warn", sbc.max_grad_zero_ratio_warn),
            ("max_grad_zero_ratio_fail", sbc.max_grad_zero_ratio_fail),
        ):
            if not 0.0 <= value <= 1.0:
                raise HealthCheckConfigError(f"skipped_batches.{name} must be in [0, 1], got {value!r}")
        if sbc.max_ratio_warn > sbc.max_ratio_fail:
            raise HealthCheckConfigError("skipped_batches.max_ratio_warn must be <= max_ratio_fail")
        if sbc.max_grad_zero_ratio_warn > sbc.max_grad_zero_ratio_fail:
            raise HealthCheckConfigError("skipped_batches.max_grad_zero_ratio_warn must be <= max_grad_zero_ratio_fail")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON-compatible consumers (dashboard).

        Uses ``dataclasses.asdict`` so new fields are picked up automatically
        without a manual serialization mirror.
        """
        return asdict(self)


@dataclass(slots=True)
class WindowSignals:
    """Accumulated signal snapshot for one startup window.

    ``effective_tokens_per_batch`` is the per-batch count of non-prompt
    response tokens (already prompt-mask-filtered by the caller). ``rewards``
    is the flat reward sequence across the window. ``losses`` is the per-step
    loss sequence. ``skipped_long`` / ``total_batches`` come from rollout
    overlong-prompt filtering; ``grad_zero_ratios`` is the backend per-step
    gradient-zero proxy.
    """

    effective_tokens_per_batch: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    skipped_long: int = 0
    total_batches: int = 0
    grad_zero_ratios: list[float] = field(default_factory=list)
    # Original error context (NaN/Inf detections) accumulated by checks; kept
    # on the snapshot so the final report can echo it without sample content.
    original_errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one check: status + localized, sample-free diagnostics.

    ``errors`` carries non-fatal diagnostic strings (e.g. NaN detection context)
    that the aggregator collects into ``HealthReport.original_errors``.  Keeping
    errors on the *result* (rather than mutating the input ``WindowSignals``)
    preserves the pure-function contract of the check functions.
    """

    name: str
    stage: str
    status: str
    message: str
    metric_ref: str
    input: str
    errors: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "status": self.status,
            "message": self.message,
            "metric_ref": self.metric_ref,
            "input": self.input,
        }


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Aggregated report for one startup window."""

    run_id: str
    summary: str
    checks: tuple[CheckResult, ...]
    window_updates: int
    completed_at_step: int
    original_errors: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "window": {
                "updates": self.window_updates,
                "completed_at_step": self.completed_at_step,
            },
            "summary": self.summary,
            "checks": [c.to_json() for c in self.checks],
            "original_errors": list(self.original_errors),
        }


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _any_non_finite(values: list[float]) -> bool:
    return any(not _is_finite(v) for v in values)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    """Sample standard deviation (Bessel-corrected, /n-1).

    For startup windows with few samples the correction matters: at n=2 the
    population std underestimates by ~29 %.  ``len < 2`` returns 0.0 so the
    division by ``n-1`` is always safe.
    """
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def check_effective_tokens(cfg: HealthCheckConfig, signals: WindowSignals) -> CheckResult:
    """Classify the effective-token signal.

    fail: every batch in the window had zero effective tokens (when
    ``fail_if_zero``). warn: per-batch mean below ``min_per_batch``. NaN-free
    by construction (integer token counts).
    """

    etc = cfg.effective_tokens
    per_batch = signals.effective_tokens_per_batch
    total = sum(per_batch)
    name = "effective_tokens"
    if not per_batch:
        # No batches observed — flag as a data/config anomaly pointing at the
        # input rather than silently passing.
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="fail",
            message="no batches observed in window (effective_tokens empty)",
            metric_ref=_METRIC_REF[name],
            input="effective_tokens",
        )
    if total == 0 and etc.fail_if_zero:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="fail",
            message=(f"zero effective tokens across the whole window (batches={len(per_batch)})"),
            metric_ref=_METRIC_REF[name],
            input="effective_tokens.fail_if_zero",
        )
    mean_tokens = total / len(per_batch)
    if etc.min_per_batch > 0 and mean_tokens < etc.min_per_batch:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="warn",
            message=(f"low effective tokens: mean={mean_tokens:.1f} < min_per_batch={etc.min_per_batch}"),
            metric_ref=_METRIC_REF[name],
            input="effective_tokens.min_per_batch",
        )
    return CheckResult(
        name=name,
        stage=_STAGE[name],
        status="pass",
        message=f"effective tokens ok (min_batch={min(per_batch)}, mean={mean_tokens:.1f})",
        metric_ref=_METRIC_REF[name],
        input="-",
    )


def check_reward_variance(cfg: HealthCheckConfig, signals: WindowSignals) -> CheckResult:
    """Classify the reward-variance signal.

    require_variation=False → pass (legitimately constant reward), but an empty
    reward list still warns (data-pipeline anomaly). NaN reward → fail (error
    carried on the result). std==0 with variation required → fail.
    std < min_std_warn → warn.
    """

    rvc = cfg.reward_variance
    name = "reward_variance"
    if not rvc.enabled:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="pass",
            message="reward_variance check disabled",
            metric_ref=_METRIC_REF[name],
            input="reward_variance.enabled",
        )
    rewards = signals.rewards
    if _any_non_finite(rewards):
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="fail",
            message="non-finite (NaN/Inf) reward detected in window",
            metric_ref=_METRIC_REF[name],
            input="reward_variance",
            errors=("reward_variance: non-finite reward value",),
        )
    if not rewards:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="warn",
            message="no rewards observed in window (data-pipeline anomaly)",
            metric_ref=_METRIC_REF[name],
            input="reward_variance",
        )
    if not rvc.require_variation:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="pass",
            message="reward variation not required (constant reward allowed)",
            metric_ref=_METRIC_REF[name],
            input="reward_variance.require_variation=false",
        )
    std = _std(rewards)
    if std == 0.0 and rvc.min_std_fail == 0.0:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="fail",
            message=f"constant reward (std=0.0, variation required, n={len(rewards)})",
            metric_ref=_METRIC_REF[name],
            input="reward_variance.require_variation=true",
        )
    if std < rvc.min_std_warn:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="warn",
            message=f"low reward std={std:.3e} < min_std_warn={rvc.min_std_warn:.3e}",
            metric_ref=_METRIC_REF[name],
            input="reward_variance.min_std_warn",
        )
    return CheckResult(
        name=name,
        stage=_STAGE[name],
        status="pass",
        message=f"reward variation ok (std={std:.3e})",
        metric_ref=_METRIC_REF[name],
        input="-",
    )


def check_loss_change(cfg: HealthCheckConfig, signals: WindowSignals) -> CheckResult:
    """Classify the loss-change signal.

    NaN loss → fail (error carried on the result). Single-step window → warn
    (unreliable delta). delta <= min_abs_delta_fail (default 0) → fail.
    delta below min_abs_delta_warn → warn.
    """

    lcc = cfg.loss_change
    name = "loss_change"
    if not lcc.enabled:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="pass",
            message="loss_change check disabled",
            metric_ref=_METRIC_REF[name],
            input="loss_change.enabled",
        )
    losses = signals.losses
    if _any_non_finite(losses):
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="fail",
            message="non-finite (NaN/Inf) loss detected in window",
            metric_ref=_METRIC_REF[name],
            input="loss_change",
            errors=("loss_change: non-finite loss value",),
        )
    if len(losses) < 2:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="warn",
            message=(f"loss change unreliable: window has <2 loss samples (n={len(losses)})"),
            metric_ref=_METRIC_REF[name],
            input="startup_window_updates",
        )
    first, last = losses[0], losses[-1]
    if lcc.mode == "relative":
        denom = max(abs(first), abs(last), 1.0e-8)
        delta = abs(last - first) / denom
    else:
        delta = abs(last - first)
    if delta <= lcc.min_abs_delta_fail:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="fail",
            message=(f"loss unchanged or below fail threshold (first={first}, last={last}, "
                     f"delta={delta:.3e}, min_abs_delta_fail={lcc.min_abs_delta_fail:.3e}, n={len(losses)})"),
            metric_ref=_METRIC_REF[name],
            input="loss_change.min_abs_delta_fail",
        )
    if delta < lcc.min_abs_delta_warn:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="warn",
            message=(f"loss barely changed: delta={delta:.3e} < min_abs_delta_warn={lcc.min_abs_delta_warn:.3e}"),
            metric_ref=_METRIC_REF[name],
            input="loss_change.min_abs_delta_warn",
        )
    return CheckResult(
        name=name,
        stage=_STAGE[name],
        status="pass",
        message=f"loss change ok (delta={delta:.3e}, mode={lcc.mode})",
        metric_ref=_METRIC_REF[name],
        input="-",
    )


def check_skipped_batches(cfg: HealthCheckConfig, signals: WindowSignals) -> CheckResult:
    """Classify the skipped-batch signal.

    Ratio = skipped_long / total_batches. Zero denominator → fail (data/input
    anomaly). Ratio >= fail/warn thresholds → fail/warn (threshold-inclusive).
    The grad-zero proxy is evaluated independently and the more severe status
    wins.
    """

    sbc = cfg.skipped_batches
    name = "skipped_batches"
    if not sbc.enabled:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="pass",
            message="skipped_batches check disabled",
            metric_ref=_METRIC_REF[name],
            input="skipped_batches.enabled",
        )
    total = signals.total_batches
    skipped = signals.skipped_long
    if total <= 0:
        return CheckResult(
            name=name,
            stage=_STAGE[name],
            status="fail",
            message="no batches in window (denominator=0, data/input anomaly)",
            metric_ref=_METRIC_REF[name],
            input="total_batches",
        )
    ratio = skipped / total
    grad_ratios = signals.grad_zero_ratios
    grad_max = max(grad_ratios) if grad_ratios else 0.0
    # Determine the status from each sub-signal independently; the more severe
    # one wins so a healthy skip ratio cannot mask a total grad-zero collapse.
    # Comparisons are threshold-inclusive (>=): reaching 50 % skip is the same
    # as exceeding it.
    status = "pass"
    detail = f"skip_ratio={ratio:.3f} ({skipped}/{total})"
    if ratio >= sbc.max_ratio_fail:
        status = "fail"
    elif ratio >= sbc.max_ratio_warn:
        status = _worse(status, "warn")
    if grad_ratios:
        detail += f", grad_zero_max={grad_max:.3f}"
        if grad_max >= sbc.max_grad_zero_ratio_fail:
            status = _worse(status, "fail")
        elif grad_max >= sbc.max_grad_zero_ratio_warn:
            status = _worse(status, "warn")
    if status == "fail":
        if ratio >= sbc.max_ratio_fail:
            message = f"skipped ratio too high: {ratio:.3f} >= max_ratio_fail={sbc.max_ratio_fail:.3f}"
            input_ref = "skipped_batches.max_ratio_fail"
        else:
            message = (
                f"gradient-zero ratio too high: grad_zero_max={grad_max:.3f} "
                f">= max_grad_zero_ratio_fail={sbc.max_grad_zero_ratio_fail:.3f}"
            )
            input_ref = "skipped_batches.max_grad_zero_ratio_fail"
    elif status == "warn":
        # Distinguish which sub-signal triggered the warn so the user can
        # locate the cause without re-deriving thresholds.
        warn_parts = []
        if ratio >= sbc.max_ratio_warn:
            warn_parts.append(f"skip_ratio={ratio:.3f} >= max_ratio_warn={sbc.max_ratio_warn:.3f}")
        if grad_ratios and grad_max >= sbc.max_grad_zero_ratio_warn:
            warn_parts.append(f"grad_zero_max={grad_max:.3f} >= max_grad_zero_ratio_warn={sbc.max_grad_zero_ratio_warn:.3f}")
        message = "; ".join(warn_parts) if warn_parts else detail
        input_ref = "skipped_batches"
    else:
        message = f"skipped batches ok ({detail})"
        input_ref = "-"
    return CheckResult(
        name=name,
        stage=_STAGE[name],
        status=status,
        message=message,
        metric_ref=_METRIC_REF[name],
        input=input_ref,
    )


def _worse(a: str, b: str) -> str:
    """Return the more severe of two statuses."""

    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


def run_checks(cfg: HealthCheckConfig, signals: WindowSignals) -> list[CheckResult]:
    """Run all four checks against a window snapshot.

    The checks are pure: they read from *signals* but never mutate it.
    Diagnostic errors (e.g. NaN detections) are returned on each
    ``CheckResult.errors`` field for the aggregator to collect.
    """

    return [
        check_effective_tokens(cfg, signals),
        check_reward_variance(cfg, signals),
        check_loss_change(cfg, signals),
        check_skipped_batches(cfg, signals),
    ]


def aggregate(
    cfg: HealthCheckConfig,
    results: list[CheckResult],
    *,
    window_updates: int,
    completed_at_step: int,
    run_id: str | None = None,
    original_errors: list[str] | None = None,
) -> HealthReport:
    """Aggregate per-check results into one report.

    The overall status is the most severe individual status. ``on_fail`` does
    not change the recorded summary — it only governs whether the coordinator
    raises on a ``fail`` summary (handled by the `Trainer` wiring).

    ``original_errors`` is collected from each ``CheckResult.errors`` field so
    the report is self-contained and no external mutation is needed.
    """

    summary = "pass"
    for r in results:
        summary = _worse(summary, r.status)
    # Collect errors from each check result; fall back to the explicit
    # ``original_errors`` argument for backward compatibility.
    collected: list[str] = list(original_errors or [])
    for r in results:
        collected.extend(r.errors)
    return HealthReport(
        run_id=run_id or _new_run_id(),
        summary=summary,
        checks=tuple(results),
        window_updates=window_updates,
        completed_at_step=completed_at_step,
        original_errors=tuple(collected),
    )


def run_health_check(
    cfg: HealthCheckConfig,
    signals: WindowSignals,
    *,
    completed_at_step: int,
    run_id: str | None = None,
) -> HealthReport | None:
    """Evaluate a window snapshot.

    Returns ``None`` when disabled so the caller can short-circuit without
    producing any artifact / metric / log (the backward-compatibility guard).
    """

    if not cfg.enabled:
        return None
    results = run_checks(cfg, signals)
    return aggregate(
        cfg,
        results,
        window_updates=cfg.startup_window_updates,
        completed_at_step=completed_at_step,
        run_id=run_id,
    )


def _new_run_id() -> str:
    """Generate a short, collision-resistant run id for artifact filenames.

    `uuid.uuid4` is used (rather than wall-clock time) so the id is stable under
    the test harness's frozen-clock constraints and unique across DP ranks.
    """

    return uuid.uuid4().hex[:12]


def configure_health_check_if_supported(instance: Any, config: HealthCheckConfig | None) -> None:
    """Attach the health checker to a trainer instance if it supports it.

    Trainer implementations call this from ``fit()``. The SDK ``Trainer`` exposes
    ``configure_health_check``; stubs/backends used in tests that do not carry
    this method are silently skipped so the health check is opt-in and never
    breaks a trainer that runs without it.  A debug-level log records whether
    the checker was attached so the caller can confirm without inspecting the
    instance.
    """

    configure = getattr(instance, "configure_health_check", None)
    if callable(configure):
        configure(config)
        logging.getLogger("areno.health_check").debug(
            "health checker attached to %s", type(instance).__name__
        )
    else:
        logging.getLogger("areno.health_check").debug(
            "health checker skipped: %s has no configure_health_check", type(instance).__name__
        )
