"""Optional dashboard integration helpers."""

from __future__ import annotations

from typing import Any


def record_dashboard_state(instance: Any, **kwargs: Any) -> None:
    """Record dashboard state when the backend supports it.

    Unit-test fakes and third-party backend-like objects do not need to
    implement dashboard reporting. Keeping this optional avoids coupling the
    core trainer loops to the dashboard feature.
    """

    recorder = getattr(instance, "record_dashboard_state", None)
    if recorder is not None:
        recorder(**kwargs)


# ---------------------------------------------------------------------------
# Phase waterfall
# ---------------------------------------------------------------------------

# Map the 12 fine-grained segment names from ``TIME_SEGMENT_ORDER`` (see
# ``areno/dashboard/server.py``) into the 5 coarse RL phases the waterfall
# visualises. ``save`` is grouped under ``training`` because checkpoint I/O is
# part of the training flow; leaving it unmapped would silently drop its
# seconds and break ``Σ phases.duration_s == total_s``.
PHASE_ORDER = ["rollout", "reward", "training", "synchronization", "waiting"]

PHASE_GROUPS: dict[str, list[str]] = {
    "rollout": ["rollout", "make_sample"],
    "reward": ["reward"],
    "training": [
        "train",
        "save",
        "advantages",
        "value",
        "old policy log probs",
        "actor log probs",
        "ref log probs",
    ],
    "synchronization": ["sync weight"],
    "waiting": ["other"],
}


def phase_waterfall(
    timeperf_rows: list[dict[str, Any]],
    *,
    slow_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive per-update phase waterfall data from existing ``timeperf`` rows.

    Each *row* in ``timeperf_rows`` follows the structure produced by
    ``DashboardState._append_timeperf_row``::

        {step, segments: [{name, seconds}], total_s, ...}

    The function groups the fine-grained *segments* into 5 coarse RL phases
    (rollout, reward, training, synchronization, waiting), then derives
    ``start_s`` / ``end_s`` by sequential accumulation — the waterfall layout.

    Parameters
    ----------
    timeperf_rows
        List of ``timeperf`` dicts (as stored on ``Job.timeperf``).
    slow_threshold
        Updates whose ``total_s`` strictly exceeds this value are flagged
        ``is_slow = True``.  ``0.0`` (default) disables the flag, preserving
        current dashboard behaviour.

    Returns
    -------
    (updates, errors)
        ``updates`` — one dict per healthy row::

            {step, total_s, is_slow,
             phases: [{name, start_s, end_s, duration_s, segments: [...]}]}

        ``errors`` — one dict per skipped/invalid row::

            {step, reason}

    Raises
    ------
    ValueError
        If *slow_threshold* is negative or non-numeric.
    """

    # --- validation --------------------------------------------------------
    try:
        threshold = float(slow_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"slow_threshold must be a non-negative number, got {slow_threshold!r}"
        ) from exc
    if threshold < 0:
        raise ValueError(
            f"slow_threshold must be a non-negative number, got {threshold}"
        )

    # Build a reverse lookup: segment_name -> phase_name
    seg_to_phase: dict[str, str] = {}
    for phase, segs in PHASE_GROUPS.items():
        for seg in segs:
            seg_to_phase[seg] = phase

    updates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for row in timeperf_rows:
        if not isinstance(row, dict):
            errors.append({"step": None, "reason": f"row is not a dict: {type(row).__name__}"})
            continue

        step = row.get("step")
        if step is None:
            errors.append({"step": None, "reason": "missing 'step' field"})
            continue

        # --- in-progress detection ----------------------------------------
        total = row.get("total_s")
        segments = row.get("segments", [])
        if not isinstance(segments, list):
            segments = []

        accounted = 0.0
        valid_segments: list[dict[str, Any]] = []
        row_errors: list[str] = []
        for seg in segments:
            if not isinstance(seg, dict):
                row_errors.append(f"segment is not a dict: {type(seg).__name__}")
                continue
            name = seg.get("name")
            seconds = seg.get("seconds", 0)
            if not isinstance(name, str) or not name:
                row_errors.append(f"segment missing name: {seg!r}")
                continue
            try:
                seconds = float(seconds)
            except (TypeError, ValueError):
                row_errors.append(f"segment '{name}' has non-numeric seconds: {seg.get('seconds')!r}")
                seconds = 0.0
            if seconds < 0:
                seconds = 0.0
            accounted += seconds
            valid_segments.append({"name": name, "seconds": seconds})

        if total is None:
            errors.append({"step": step, "reason": "in-progress: total_s missing"})
            continue
        try:
            total = float(total)
        except (TypeError, ValueError):
            errors.append({"step": step, "reason": f"total_s non-numeric: {row.get('total_s')!r}"})
            continue
        if total <= 0:
            errors.append({"step": step, "reason": "in-progress: total_s <= 0"})
            continue
        if total < accounted:
            errors.append({"step": step, "reason": "in-progress: total_s < sum(segments)"})
            continue

        # Record non-fatal per-row issues
        for msg in row_errors:
            errors.append({"step": step, "reason": msg})

        # --- group segments into phases -----------------------------------
        phase_seconds: dict[str, float] = {p: 0.0 for p in PHASE_ORDER}
        phase_segments: dict[str, list[dict[str, Any]]] = {p: [] for p in PHASE_ORDER}
        for seg in valid_segments:
            phase = seg_to_phase.get(seg["name"])
            if phase is None:
                # Unmapped segment — don't silently drop; route to waiting
                # and record an error so callers know.
                errors.append({"step": step, "reason": f"unmapped segment: {seg['name']}"})
                phase = "waiting"
            phase_seconds[phase] += seg["seconds"]
            phase_segments[phase].append(seg)

        # --- waiting: if 'other' segment is absent, compute residual -------
        has_other = any(seg["name"] == "other" for seg in valid_segments)
        if not has_other:
            non_waiting = sum(
                phase_seconds[p] for p in PHASE_ORDER if p != "waiting"
            )
            phase_seconds["waiting"] = max(total - non_waiting, 0.0)

        # --- sequential accumulation → waterfall layout --------------------
        cursor = 0.0
        phases: list[dict[str, Any]] = []
        for phase_name in PHASE_ORDER:
            duration = phase_seconds[phase_name]
            phases.append({
                "name": phase_name,
                "start_s": round(cursor, 6),
                "end_s": round(cursor + duration, 6),
                "duration_s": round(duration, 6),
                "segments": phase_segments[phase_name],
            })
            cursor += duration

        is_slow = threshold > 0 and total > threshold

        updates.append({
            "step": step,
            "total_s": round(total, 6),
            "is_slow": is_slow,
            "phases": phases,
        })

    return updates, errors
