"""Non-finite value detection and actionable reporting.

Issue #238: Produce an actionable non-finite-value training report with
optional skip-update and controlled termination.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist

from areno.engine.runtime.train_step import _param_grad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class NonFiniteTrainingError(RuntimeError):
    """Raised when non-finite values are detected and termination is enabled."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NonFiniteEvent:
    """One parameter or gradient that contains NaN / Inf."""
    name: str
    layer: str
    is_gradient: bool = False
    nan_count: int = 0
    inf_count: int = 0
    total_elements: int = 0
    max_value: float = float("nan")
    min_value: float = float("nan")
    grad_norm: Optional[float] = None


@dataclass
class NonFiniteReport:
    """Actionable report produced when non-finite values are detected."""
    step: int
    loss_value: float
    phase: str
    events: list[NonFiniteEvent] = field(default_factory=list)
    learning_rate: float = 0.0
    global_grad_norm: float = 0.0
    recent_losses: list[float] = field(default_factory=list)
    last_checkpoint: Optional[str] = None
    gpu_memory_gb: Optional[float] = None

    causes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def analyse(self) -> None:
        self.causes = _infer_causes(self)
        self.suggestions = _infer_suggestions(self)

    def to_dict(self) -> dict[str, Any]:
        """Return numeric-only metrics dict safe for _merge_metrics."""
        total_nan = sum(e.nan_count for e in self.events)
        total_inf = sum(e.inf_count for e in self.events)
        affected_layers = len(set(e.layer for e in self.events if e.layer not in ("loss", "optimizer")))
        return {
            "non_finite_step": float(self.step),
            "non_finite_loss": self.loss_value if not math.isnan(self.loss_value) else -1.0,
            "non_finite_total_nan": float(total_nan),
            "non_finite_total_inf": float(total_inf),
            "non_finite_affected_layers": float(affected_layers),
            "non_finite_event_count": float(len(self.events)),
            "non_finite_lr": self.learning_rate,
            "non_finite_grad_norm": self.global_grad_norm if not math.isnan(self.global_grad_norm) else -1.0,
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Full structured dict for JSON file export (not limited to numeric types)."""
        return {
            "step": self.step,
            "phase": self.phase,
            "loss": self.loss_value,
            "global_grad_norm": self.global_grad_norm,
            "learning_rate": self.learning_rate,
            "gpu_memory_gb": self.gpu_memory_gb,
            "last_checkpoint": self.last_checkpoint,
            "events": [
                {
                    "name": e.name,
                    "layer": e.layer,
                    "is_gradient": e.is_gradient,
                    "nan_count": e.nan_count,
                    "inf_count": e.inf_count,
                    "total_elements": e.total_elements,
                    "nan_ratio": round(e.nan_count / e.total_elements, 4) if e.total_elements > 0 else 0.0,
                    "inf_ratio": round(e.inf_count / e.total_elements, 4) if e.total_elements > 0 else 0.0,
                    "grad_norm": e.grad_norm,
                }
                for e in self.events
            ],
            "causes": self.causes,
            "suggestions": self.suggestions,
            "total_nan": sum(e.nan_count for e in self.events),
            "total_inf": sum(e.inf_count for e in self.events),
            "affected_layers": sorted(set(e.layer for e in self.events if e.layer not in ("loss", "optimizer"))),
        }

    def to_json_file(self, output_dir: str = "non_finite_reports") -> str:
        """Write full report to a JSON file. Returns the file path."""
        os.makedirs(output_dir, exist_ok=True)
        fname = f"step_{self.step}_{self.phase}.json"
        fpath = os.path.join(output_dir, fname)
        data = self.to_json_dict()
        # Replace NaN/Inf with null for valid JSON
        def _sanitize(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj
        with open(fpath, "w") as f:
            json.dump(_sanitize(data), f, indent=2)
        return fpath

    def format_terminal(self) -> str:
        lines = [
            "=" * 56,
            "  WARNING  Non-Finite Value Training Report",
            "=" * 56,
            "",
            "LOCATION",
            f"  Step: {self.step}  |  Phase: {self.phase}",
            f"  Last checkpoint: {self.last_checkpoint or 'N/A'}",
            "",
            "ANOMALIES DETECTED",
        ]
        max_show = 5
        for i, evt in enumerate(self.events):
            if i >= max_show:
                break

            pct = (evt.nan_count + evt.inf_count) / max(evt.total_elements, 1) * 100
            tag = "GRAD" if evt.is_gradient else "PARAM"
            lines.append(f"  [{tag}] {evt.name}")
            if evt.nan_count > 0:
                lines.append(f"    -> {evt.nan_count} NaN ({pct:.2f}%)")
            if evt.inf_count > 0:
                lines.append(f"    -> {evt.inf_count} Inf ({pct:.2f}%)")
            if evt.grad_norm is not None:
                lines.append(f"    -> grad_norm = {evt.grad_norm:.2e}")
            if not evt.is_gradient and not math.isnan(evt.max_value):
                lines.append(f"    -> max={evt.max_value:.4e}  min={evt.min_value:.4e}")
        lines.append("")
        lines.append("CONTEXT")
        lines.append(f"  Loss: {self.loss_value}")
        lines.append(f"  LR: {self.learning_rate:.2e}")
        lines.append(f"  Global grad_norm: {self.global_grad_norm:.2e}")
        if self.gpu_memory_gb is not None:
            lines.append(f"  GPU memory: {self.gpu_memory_gb:.2f} GB")
        if self.recent_losses:
            shown = [f"{v:.4f}" if not math.isnan(v) else "NaN"
                     for v in self.recent_losses[-5:]]
            lines.append(f"  Recent losses: [{', '.join(shown)}]")
        if len(self.events) > max_show:
            lines.append(f"  ... and {len(self.events) - max_show} more events (showing first {max_show})")
            # Summary statistics
            grad_events = sum(1 for e in self.events if e.is_gradient)
            param_events = sum(1 for e in self.events if not e.is_gradient)
            total_nan = sum(e.nan_count for e in self.events)
            total_inf = sum(e.inf_count for e in self.events)
            affected_layers = set(e.layer for e in self.events if e.layer not in ("loss", "optimizer"))
            lines.append(f"  SUMMARY: {grad_events} gradient + {param_events} parameter events")
            lines.append(f"  Total NaN: {total_nan:,}  Total Inf: {total_inf:,}")
            lines.append(f"  Affected layers: {len(affected_layers)} ({', '.join(list(affected_layers)[:3])}{'...' if len(affected_layers) > 3 else ''})")
        lines.append("")
        lines.append("LIKELY CAUSES")
        for i, c in enumerate(self.causes, 1):
            lines.append(f"  {i}. {c}")
        lines.append("")
        lines.append("SUGGESTED FIXES")
        for i, s in enumerate(self.suggestions, 1):
            lines.append(f"  {i}. {s}")
        lines.append("")
        # Write JSON file
        try:
            json_path = self.to_json_file()
            lines.append(f"")
            lines.append(f"JSON REPORT: {json_path}")
        except Exception:
            pass
        lines.append("=" * 56)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fast per-step loss check (nearly zero overhead)
# ---------------------------------------------------------------------------

def check_loss_non_finite(loss: torch.Tensor) -> bool:
    """Return True if loss is NaN or Inf. Call this every step."""
    return bool(torch.isnan(loss) or torch.isinf(loss))


# ---------------------------------------------------------------------------
# Deep detection (call on stepped steps or when loss is non-finite)
# ---------------------------------------------------------------------------

def detect_non_finite(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    grad_norm: float,
    step: int,
    lr: float,
    phase: str = "actor",
    recent_losses: list[float] | None = None,
    last_checkpoint: str | None = None,
    grad_norm_threshold: float = 1e6,
) -> NonFiniteReport | None:
    """Scan parameters, gradients, and optimizer state for NaN/Inf."""
    events: list[NonFiniteEvent] = []
    loss_is_bad = check_loss_non_finite(loss)
    loss_value = loss.item() if not loss_is_bad else float("nan")

    for name, param in model.named_parameters():
        if param.data is not None:
            nan_c = int(torch.isnan(param.data).sum().item())
            inf_c = int(torch.isinf(param.data).sum().item())
            if nan_c > 0 or inf_c > 0:
                events.append(NonFiniteEvent(
                    name=name,
                    layer=_extract_layer(name),
                    is_gradient=False,
                    nan_count=nan_c,
                    inf_count=inf_c,
                    total_elements=param.data.numel(),
                    max_value=_safe_max(param.data),
                    min_value=_safe_min(param.data),
                ))

        grad = _param_grad(param)
        if grad is not None:
            g_nan = int(torch.isnan(grad).sum().item())
            g_inf = int(torch.isinf(grad).sum().item())
            g_norm = float(grad.data.norm(2).item())
            if g_nan > 0 or g_inf > 0 or g_norm > grad_norm_threshold:
                events.append(NonFiniteEvent(
                    name=name + ".grad",
                    layer=_extract_layer(name),
                    is_gradient=True,
                    nan_count=g_nan,
                    inf_count=g_inf,
                    total_elements=grad.numel(),
                    grad_norm=g_norm,
                ))

    for group_idx, p_idx, sname, sval in _safe_optimizer_state(optimizer):
        s_nan = int(torch.isnan(sval).sum().item())
        s_inf = int(torch.isinf(sval).sum().item())
        if s_nan > 0 or s_inf > 0:
            events.append(NonFiniteEvent(
                name=f"opt.state[{group_idx}][{p_idx}].{sname}",
                layer="optimizer",
                nan_count=s_nan,
                inf_count=s_inf,
                total_elements=sval.numel(),
            ))

    if not loss_is_bad and not events:
        return None

    if loss_is_bad and not events:
        events.append(NonFiniteEvent(
            name="loss",
            layer="loss",
            nan_count=1,
            total_elements=1,
        ))

    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9

    report = NonFiniteReport(
        step=step,
        loss_value=loss_value,
        phase=phase,
        events=events,
        learning_rate=lr,
        global_grad_norm=grad_norm,
        recent_losses=list(recent_losses or []),
        last_checkpoint=last_checkpoint,
        gpu_memory_gb=gpu_mem,
    )
    report.analyse()
    return report


# ---------------------------------------------------------------------------
# Cause / suggestion inference
# ---------------------------------------------------------------------------

def _infer_causes(report: NonFiniteReport) -> list[str]:
    causes: list[str] = []

    if report.global_grad_norm > 1e4:
        causes.append(
            f"[HIGH] Gradient explosion -> grad_norm={report.global_grad_norm:.2e}")

    if len(report.recent_losses) >= 3:
        valid = [l for l in report.recent_losses[-5:] if not math.isnan(l)]
        if valid and (math.isnan(report.loss_value) or report.loss_value > max(valid) * 10):
            causes.append(
                f"[HIGH] Loss spike -> from {valid[-1]:.4f} to {report.loss_value}")

    layers = {}
    for e in report.events:
        if not e.is_gradient and e.layer not in ("loss", "optimizer"):
            layers.setdefault(e.layer, 0)
            layers[e.layer] += 1
    if len(layers) == 1:
        causes.append(f"[MID] Single-layer anomaly -> {list(layers.keys())[0]}")

    if any(e.layer == "optimizer" for e in report.events):
        causes.append("[MID] Optimizer state pollution -> momentum/variance NaN")

    if report.global_grad_norm < 1e-7 and not math.isnan(report.global_grad_norm):
        causes.append(f"[LOW] Vanishing gradients -> grad_norm={report.global_grad_norm:.2e}")

    if not causes:
        causes.append("[UNKNOWN] Cannot auto-diagnose; check data and model")

    return causes


def _infer_suggestions(report: NonFiniteReport) -> list[str]:
    suggestions: list[str] = []
    text = " ".join(report.causes)

    if "Gradient explosion" in text or "explosion" in text.lower():
        suggestions.append("Enable gradient clipping: clip_grad_norm(max_norm=1.0)")
        suggestions.append(f"Reduce learning rate: {report.learning_rate:.2e} -> 1/5 or 1/10")

    if "Loss spike" in text or "spike" in text.lower():
        suggestions.append("Reduce learning rate to avoid loss oscillation")
        suggestions.append("Check batch data for anomalous samples")

    if "Single-layer" in text:
        suggestions.append("Check input data range/distribution for that layer")
        suggestions.append("Consider adding LayerNorm or reducing init variance")

    if "Optimizer" in text or "optimizer" in text.lower():
        suggestions.append("Restore from checkpoint and reset optimizer state")

    if "Vanishing" in text or "vanishing" in text.lower():
        suggestions.append("Check model depth; consider residual connections or different activation")

    if report.last_checkpoint:
        suggestions.append(f"Restore from checkpoint: {report.last_checkpoint}")

    if not suggestions:
        suggestions.append("Check input data for NaN/Inf values")
        suggestions.append("Try reducing learning rate and batch size")

    return suggestions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_optimizer_state(optimizer) -> list[tuple[int, int, str, torch.Tensor]]:
    """Yield (group_idx, param_idx, state_name, tensor) from any optimizer."""
    entries = []
    try:
        # Standard PyTorch optimizer
        for group_idx, group in enumerate(optimizer.param_groups):
            for p_idx, p in enumerate(group["params"]):
                if p not in optimizer.state:
                    continue
                state = optimizer.state[p]
                for sname, sval in state.items():
                    if isinstance(sval, torch.Tensor):
                        entries.append((group_idx, p_idx, sname, sval))
    except AttributeError:
        # Custom optimizer (e.g. AdamWFP32Master)
        try:
            state = getattr(optimizer, "state", {})
            for pid, sdict in state.items():
                if not isinstance(sdict, dict):
                    continue
                for sname, sval in sdict.items():
                    if isinstance(sval, torch.Tensor):
                        entries.append((0, 0, sname, sval))
        except Exception:
            pass
    return entries

def _extract_layer(name: str) -> str:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p.isdigit():
            return ".".join(parts[: i + 1])
    return parts[0]


def _safe_max(t: torch.Tensor) -> float:
    try:
        return float(t[~torch.isinf(t) & ~torch.isnan(t)].max().item())
    except (RuntimeError, ValueError):
        return float("nan")


def _safe_min(t: torch.Tensor) -> float:
    try:
        return float(t[~torch.isinf(t) & ~torch.isnan(t)].min().item())
    except (RuntimeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Cross-rank coordination and report output
# ---------------------------------------------------------------------------

def all_reduce_non_finite_flag(local_non_finite: bool, *, tp_group=None, dp_group=None) -> bool:
    """Return True if *any* rank detected non-finite values.

    Uses MAX all-reduce so that a single ``True`` (1) on any rank propagates
    to all ranks. When no distributed group is active, returns the local flag.
    """
    if not dist.is_available() or not dist.is_initialized():
        return local_non_finite
    flag = torch.tensor(1.0 if local_non_finite else 0.0, dtype=torch.float32)
    if tp_group is not None:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=tp_group)
    if dp_group is not None:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=dp_group)
    return bool(flag.item())


def emit_non_finite_report(
    report: NonFiniteReport | None,
    *,
    skip_update: bool = False,
    terminate: bool = False,
) -> None:
    """Log the report and optionally raise NonFiniteTrainingError.

    Shared by actor and critic training paths so both use logger.warning
    instead of bare ``print``.
    """
    if report is None:
        return
    logger.warning("Non-finite values detected (step %d, phase %s):\n%s", report.step, report.phase, report.format_terminal())
    if terminate:
        raise NonFiniteTrainingError(
            f"Training terminated at step {report.step} (phase={report.phase}) "
            f"due to non-finite values. See JSON report for details."
        )
