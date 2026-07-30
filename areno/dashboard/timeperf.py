"""Run-level timing aggregation for the ``areno timing-summary`` CLI.

This module builds a per-phase timing summary (latest update + whole run) on
top of the segment-tag normalization that the dashboard already owns. To keep
this change minimal and avoid any risk to other contributors, the four
segment helpers below are **reused in place** from
``areno.dashboard.server`` rather than moved:

* ``TIME_SEGMENT_ORDER`` — canonical, display-ordered phase vocabulary
* ``tensorboard_event_sources`` — locate ``events.out.tfevents.*`` for a run
* ``tensorboard_time_segment_name`` — map a TB scalar tag to a phase segment
  (which internally delegates to the dashboard's ``normalize_time_segment_name``)

``server.py`` is therefore left untouched. The aggregation logic defined here
(``load_step_segments`` / ``load_run_status`` / ``summarize`` / ``format_*``)
is the only new code; it only reads run artifacts and writes nothing.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

# Re-exported in place from the dashboard so this module is the single import
# target for the CLI, without duplicating or relocating the source of truth.
from areno.dashboard.server import (
    TIME_SEGMENT_ORDER,
    tensorboard_event_sources,
    tensorboard_time_segment_name,
)

# --------------------------------------------------------------------------- #
# Run-level aggregation (used by the ``areno timing-summary`` CLI).
#
# The functions below read a run's TensorBoard event files once (a snapshot),
# assemble per-step segment dicts using the same tag->segment semantics the
# dashboard uses, and then aggregate them into a run-level summary covering the
# latest update and the whole run. Reconciliation (reported vs reconstructed
# totals) and explicit overlap/missing annotation are produced here.
# --------------------------------------------------------------------------- #

# Tags that carry a step-level *total* (end-to-end wall time) rather than a
# single phase. These are the "ground truth" a reconstructed total is compared
# against during reconciliation.
_TOTAL_TAGS = {"train/step_e2e_time_s", "time/total", "time/e2e"}

# The dedicated step-level rollup tags. These are the authoritative source for
# the ``rollout`` / ``train`` phase values; ``time/rollout`` / ``time/train``
# carry the same numbers in current trainers but are treated as sub-phase
# echoes (see ``_rollup_tags``). Keeping rollups and ``time/*`` phases separate
# avoids an order-dependent dict overwrite when the two ever diverge.
_ROLLUP_TAGS = {
    "train/step_rollout_time_s": "rollout",
    "train/step_train_time_s": "train",
    "train/policy_train_wall_time_s": "train",
}


def _dashboard_state_file(run_dir: Path, pid: int | None) -> Path | None:
    """Locate the low-latency ``dashboard_state.<pid>.json`` for a run.

    Minimal re-implementation of ``areno.dashboard.server.dashboard_state_source``
    (kept local so this module stays free of a reverse import of the server).
    Returns the most recently modified ``dashboard_state.*.json`` when ``pid``
    is unknown.
    """
    if pid is not None:
        state_file = run_dir / f"dashboard_state.{pid}.json"
        return state_file if state_file.exists() else None
    candidates = sorted(run_dir.glob("dashboard_state.*.json"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def _pid_alive(pid: int) -> bool:
    """Return ``True`` if process ``pid`` currently exists.

    Uses ``os.kill(pid, 0)``. ``ProcessLookupError`` means the process is gone
    (run completed); ``PermissionError`` means it exists but is owned by another
    user (treated as alive). Any other ``OSError`` is treated conservatively as
    "not alive" so a stale/unknown pid resolves to ``completed``.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def load_run_status(run_dir: Path, pid: int | None = None) -> str:
    """Return ``"active"`` or ``"completed"`` for a run, with a status basis.

    The dashboard_state file written by ``MetricsRecorder.record_dashboard_state``
    always carries ``status == "running"`` (the trainer never writes a terminal
    status — see issue #256 review, finding 1), so ``status`` alone **cannot**
    distinguish a finished run from one in progress. This function therefore
    resolves the run's pid (from the caller or from the state file's ``pid``
    field) and checks process liveness via ``os.kill(pid, 0)``:

    * process alive  -> ``"active"``
    * process gone   -> ``"completed"`` (regardless of the stale ``running`` status)
    * pid unknown / file missing or unreadable -> ``"completed"`` (conservative
      default: do not over-warn about partial steps that are actually final)

    The returned status (``"active"`` / ``"completed"``) reflects this
    liveness check rather than the stale ``running`` field, so a finished run
    is reported as ``completed`` once its process is gone.

    Limitation (heuristic, not a guarantee): pid liveness only proves that
    *some* process with that pid exists now — not that it is the original
    training run. It is reliable for the intended local usage (reading a run's
    own metrics dir on the same host shortly after/while it runs). It can be
    wrong when (a) the metrics dir is read on a different host than where the
    run executed (the pid refers to that host's pid space), or (b) the run has
    exited and its pid has been reused by an unrelated process. The correct
    fix is for the trainer to write a terminal ``status`` on close (tracked
    separately, out of this issue's scope); until then this heuristic is the
    best available signal and falls back to ``"completed"`` conservatively.
    """
    state_file = _dashboard_state_file(run_dir, pid)
    if state_file is None:
        return "completed"
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "completed"
    resolved_pid = pid if pid is not None else payload.get("pid")
    if isinstance(resolved_pid, int) and _pid_alive(resolved_pid):
        return "active"
    return "completed"


def _accumulate_event(
    bucket: dict[str, float],
    step: int,
    tag: str,
    value: float,
    rollup_seen: dict[int, dict[str, float]],
    phase_seen: dict[int, dict[str, float]],
) -> str | None:
    """Fold one TensorBoard scalar event into a step's segment ``bucket``.

    Pure (no I/O) so the classification below — which is where findings 3 and 4
    live — can be unit-tested without the ``tensorboard`` package. Returns a
    divergence message string when the event conflicts with a prior value for
    the same phase, else ``None``.

    Classification:
    * ``_TOTAL_TAGS``           -> rollup key ``"total"``
    * ``_ROLLUP_TAGS``          -> rollup ``rollout``/``train`` (authoritative)
    * normalizer maps to a vocab segment -> that phase (echo; rollup value wins)
    * normalizer maps outside vocab     -> ``"other"`` (kept, not dropped)
    * normalizer returns ``None``       -> ignored (e.g. e2e/total leaves)
    """
    if tag in _TOTAL_TAGS:
        bucket["total"] = value
        return None
    if tag in _ROLLUP_TAGS:
        rollup_name = _ROLLUP_TAGS[tag]
        rollup_seen.setdefault(step, {})
        if (
            rollup_name in rollup_seen[step]
            and abs(rollup_seen[step][rollup_name] - value) > 1e-9
        ):
            return f"step {step} {rollup_name} rollup diverges"
        rollup_seen[step][rollup_name] = value
        bucket[rollup_name] = value
        return None
    time_name = tensorboard_time_segment_name(tag)
    if time_name is None:
        return None
    if time_name in TIME_SEGMENT_ORDER:
        phase_seen.setdefault(step, {})
        # An echo (time/*) that disagrees with the authoritative rollup value
        # (train/step_*_time_s) for the same phase is a real divergence
        # (finding 4) — surface it instead of silently overwriting.
        rollup_value = rollup_seen.get(step, {}).get(time_name)
        if rollup_value is not None and abs(rollup_value - value) > 1e-9:
            return f"step {step} {time_name} phase echo diverges (rollup={rollup_value} echo={value})"
        if (
            time_name in phase_seen[step]
            and abs(phase_seen[step][time_name] - value) > 1e-9
        ):
            return f"step {step} {time_name} phase echo diverges"
        phase_seen[step][time_name] = value
        if time_name not in bucket:
            bucket[time_name] = value
        return None
    # Out-of-vocab sub-phase (e.g. critic value forward): fold into "other"
    # so the time is not silently dropped (finding 3).
    bucket["other"] = bucket.get("other", 0.0) + value
    return None


def load_step_segments(
    run_dir: Path, pid: int | None = None
) -> tuple[dict[int, dict[str, float]], list[str]]:
    """Read TensorBoard scalars and assemble per-step segment dicts.

    Returns ``(by_step, divergences)``.

    ``by_step`` is ``{step: {segment_name: seconds}}`` with two kinds of keys:

    * rollup keys ``"total"`` (step end-to-end, from ``_TOTAL_TAGS``),
      ``"rollout"`` and ``"train"`` — taken **only** from the dedicated
      ``train/step_*_time_s`` tags in ``_ROLLUP_TAGS`` so the rollup source is
      unambiguous (finding 4);
    * phase keys — every ``time/*`` and ``train/*_time_s`` tag mapped through
      ``tensorboard_time_segment_name``. Segment names that fall **outside**
      ``TIME_SEGMENT_ORDER`` (e.g. PPO's ``critic value forward`` /
      ``critic train`` / ``reward score``) are folded into ``"other"`` and
      **kept in the reconstructed sum**, so recorded time is never silently
      dropped and ``missing`` no longer contradicts reality (finding 3).

    ``divergences`` lists steps where the ``rollout``/``train`` rollup value
    disagreed with the ``time/*`` echo of the same phase (finding 4) — currently
    empty for all real trainers, since both sources read the same
    ``_metric_timings`` value, but surfaced instead of silently overwritten.

    ``NaN`` values are skipped. Raises ``RuntimeError`` if ``tensorboard`` is
    not importable.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:  # pragma: no cover - depends on env
        raise RuntimeError("The 'tensorboard' package is required to read run metrics but is not installed.") from exc

    by_step: dict[int, dict[str, float]] = {}
    rollup_seen: dict[int, dict[str, float]] = {}
    phase_seen: dict[int, dict[str, float]] = {}
    divergences: list[str] = []
    # Track which (step, tag) pairs we've already accepted, so a re-run that
    # rewrites step 0 into a second event file does not double-count it.
    # ``tensorboard_event_sources`` returns files in ascending mtime order, so
    # we iterate newest-first: the first time we see a (step, tag) we keep that
    # value (from the most recent run) and skip older duplicates. See review
    # finding: real GSPO run restarted 3x and step 0 was summed 3x.
    processed: set[tuple[int, str]] = set()

    sources = tensorboard_event_sources(run_dir, pid)
    for accumulator_path in reversed(sources):
        try:
            accumulator = EventAccumulator(str(accumulator_path), size_guidance={"scalars": 10000})
            accumulator.Reload()
            tags = accumulator.Tags().get("scalars", [])
        except Exception:
            continue
        for tag in tags:
            try:
                events = accumulator.Scalars(tag)[-500:]
            except Exception:
                continue
            for event in events:
                value = float(event.value)
                if math.isnan(value):
                    continue
                step = int(event.step)
                key = (step, tag)
                if key in processed:
                    continue
                processed.add(key)
                bucket = by_step.setdefault(step, {})
                msg = _accumulate_event(bucket, step, tag, value, rollup_seen, phase_seen)
                if msg:
                    divergences.append(msg)
    return by_step, divergences


def _reconcile(step_values: dict[str, float]) -> dict[str, Any]:
    """Compute the reconciliation columns for a single step's segment dict.

    Returns ``{reported_total, reconstructed_total, diff, total_source}``.
    ``reconstructed_total`` sums the canonical phases present in
    ``TIME_SEGMENT_ORDER`` (including ``"other"``, which holds out-of-vocab
    sub-phase time) but excludes the synthetic ``"total"`` rollup. This keeps
    reconstructed totals honest even when PPO sub-phases fold into ``"other"``
    (finding 3).
    """
    phases = {name: step_values[name] for name in TIME_SEGMENT_ORDER if name in step_values}
    reconstructed = sum(v for v in phases.values() if v > 0)
    reported = step_values.get("total")
    if reported is None:
        return {
            "reported_total": reconstructed,
            "reconstructed_total": reconstructed,
            "diff": 0.0,
            "total_source": "reconstructed",
        }
    return {
        "reported_total": reported,
        "reconstructed_total": reconstructed,
        "diff": reported - reconstructed,
        "total_source": "reported",
    }


def _is_partial(step_values: dict[str, float]) -> bool:
    """A step is partial when it has phase data but no reported ``total``.

    This is the signature of a still-running step whose rollout segment has
    landed but whose train/e2e segment has not been written yet.
    """
    has_phase = any(name in step_values for name in TIME_SEGMENT_ORDER if name != "other")
    return has_phase and "total" not in step_values


def summarize(run_dir: Path, pid: int | None = None) -> dict[str, Any]:
    """Build a run-level timing summary for ``run_dir``.

    Produces a plain dict (so it serializes directly to JSON) with:

    * ``run_status``: ``"active"`` | ``"completed"`` (resolved from process
      liveness, not the stale ``running`` status field — finding 1)
    * ``run_dir``: the resolved directory
    * ``num_steps``: number of steps with any timing data
    * ``latest_update``: per-segment breakdown for the highest step, with
      reconciliation columns and a ``partial`` flag
    * ``whole_run``: per-segment sums across all steps (including ``"other"``,
      which holds out-of-vocab sub-phase time), with reconciliation
    * ``overlap``: always ``[]`` — current trainers record no overlapping
      sub-phase timers; ``overlap_note`` explains why (finding 2)
    * ``missing``: canonical segments (excluding ``"other"``) never seen
    * ``divergences``: steps where the rollup and ``time/*`` echo of a phase
      disagreed (finding 4; empty for current trainers)

    The function only reads files; it writes nothing and never mutates the run.
    """
    run_dir = Path(run_dir).resolve()
    run_status = load_run_status(run_dir, pid)
    by_step, divergences = load_step_segments(run_dir, pid)

    steps = sorted(by_step)
    # Union of canonical segments actually seen across all steps (excluding the
    # synthetic "total" rollup key). "other" does not count toward missing.
    seen_segments: set[str] = set()
    for values in by_step.values():
        seen_segments.update(name for name in values if name in TIME_SEGMENT_ORDER)
    seen_segments.discard("total")

    # Whole-run sums: sum each canonical phase across steps (including "other",
    # which carries out-of-vocab sub-phase time so reconstructed totals match).
    whole_phases = {
        name: sum(by_step[s].get(name, 0.0) for s in steps if by_step[s].get(name, 0.0) > 0)
        for name in TIME_SEGMENT_ORDER
    }
    whole_reported = sum(by_step[s].get("total", 0.0) for s in steps if by_step[s].get("total", 0.0) > 0)
    whole_reconstructed = sum(v for v in whole_phases.values() if v > 0)
    if whole_reported > 0:
        whole_total_source = "reported"
        whole_diff = whole_reported - whole_reconstructed
        whole_reported_total = whole_reported
    else:
        whole_total_source = "reconstructed"
        whole_diff = 0.0
        whole_reported_total = whole_reconstructed

    latest_update: dict[str, Any] = {}
    partial = False
    if steps:
        last = steps[-1]
        last_values = by_step[last]
        partial = _is_partial(last_values)
        recon = _reconcile(last_values)
        latest_phases = {
            name: (last_values[name] if name in last_values else None) for name in TIME_SEGMENT_ORDER
        }
        latest_update = {
            "step": last,
            # Keep 0.0 distinct from missing: a phase that was recorded with
            # zero wall time (e.g. SFT's rollout, which has no generation)
            # shows as 0.0, while a phase absent from this step shows as null.
            # This matches whole_run.segments, which never collapses 0 to null.
            "segments": dict(latest_phases),
            "partial": partial,
            **recon,
        }

    missing = [name for name in TIME_SEGMENT_ORDER if name not in seen_segments and name != "other"]

    # The trainers currently record no overlapping sub-phase timers (the only
    # candidate, PPO's critic value/train, normalizes outside the segment
    # vocabulary and is folded into "other"). So there are no containment
    # relations to declare; ``overlap`` is always empty and reported honestly
    # rather than via a hardcoded pair that never triggers (finding 2).
    overlap: list[Any] = []

    def _share(value: float | None, total: float) -> float | None:
        if value is None or value <= 0 or total <= 0:
            return None
        return round(100.0 * value / total, 2)

    return {
        "run_status": run_status,
        "run_dir": str(run_dir),
        "num_steps": len(steps),
        "latest_update": latest_update,
        "whole_run": {
            "segments": whole_phases,
            "reported_total": whole_reported_total,
            "reconstructed_total": whole_reconstructed,
            "diff": whole_diff,
            "total_source": whole_total_source,
        },
        "overlap": overlap,
        "overlap_note": "no overlapping sub-phase timers are recorded by current trainers; overlap is always empty",
        "missing": missing,
        "divergences": divergences,
    }


def format_json(summary: dict[str, Any]) -> str:
    """Serialize a summary to a pretty JSON string."""
    return json.dumps(summary, indent=2, sort_keys=False)


def format_table(summary: dict[str, Any]) -> str:
    """Render a summary as a human-readable fixed-width table.

    Lists each canonical phase with two columns (latest update / whole run),
    each showing seconds and share-of-total, then a reconciliation footer with
    reported vs reconstructed totals and the diff. Missing phases and declared
    overlaps are shown explicitly.
    """
    lines: list[str] = []
    lines.append(
        f"Run status: {summary['run_status']}    Run dir: {summary['run_dir']}    Steps: {summary['num_steps']}"
    )
    lines.append("")

    latest = summary.get("latest_update") or {}
    whole = summary["whole_run"]
    latest_total = latest.get("reported_total") or 0.0
    whole_total = whole["reported_total"] or 0.0
    latest_segs = latest.get("segments", {})
    whole_segs = whole["segments"]

    def _share(value: float | None, total: float) -> float | None:
        if value is None or value <= 0 or total <= 0:
            return None
        return round(100.0 * value / total, 2)

    def _cell(value: float | None) -> str:
        if value is None:
            return "missing"
        if value <= 0:
            return f"{value:.2f}"
        share = _share(value, latest_total) or 0.0
        return f"{value:.2f}  {share:5.1f}%"

    def _cell_whole(value: float | None) -> str:
        if value is None or value == 0:
            return "missing" if value is None else f"{value:.2f}"
        share = _share(value, whole_total) or 0.0
        return f"{value:.2f}  {share:5.1f}%"

    header = f"{'Phase':<22}{'latest update':>22}{'whole run':>22}"
    lines.append(header)
    lines.append("-" * len(header))
    for name in TIME_SEGMENT_ORDER:
        lv = latest_segs.get(name)
        wv = whole_segs.get(name)
        lines.append(f"{name:<22}{_cell(lv):>22}{_cell_whole(wv):>22}")
    lines.append("-" * len(header))
    lines.append(f"{'reported_total':<22}{latest_total:>22.2f}{whole_total:>22.2f}")
    lines.append(
        f"{'reconstructed_total':<22}"
        f"{(latest.get('reconstructed_total') or 0.0):>22.2f}"
        f"{whole['reconstructed_total']:>22.2f}"
    )
    lines.append(f"{'diff':<22}{(latest.get('diff') or 0.0):>22.2f}{whole['diff']:>22.2f}")
    lines.append(f"{'total_source':<22}{str(latest.get('total_source', '-')):>22}{whole['total_source']:>22}")
    if latest.get("partial"):
        lines.append("")
        lines.append(f"Note: latest update (step {latest.get('step')}) is partial — some phases not yet recorded.")
    if summary.get("missing"):
        lines.append("")
        lines.append("Missing phases: " + ", ".join(summary["missing"]))
    note = summary.get("overlap_note")
    if note:
        lines.append("")
        lines.append(f"Overlap: none — {note}")
    if summary.get("divergences"):
        lines.append("")
        lines.append("Divergences (rollup vs time/* echo): " + "; ".join(summary["divergences"]))
    return "\n".join(lines)
