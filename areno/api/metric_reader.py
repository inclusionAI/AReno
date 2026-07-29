"""Read-side helpers for querying metric history from local run artifacts.

Issue #254 extracts the dashboard's TensorBoard scalar reading into a set of
pure, side-effect-free functions so the ``areno metrics`` CLI and the dashboard
share one fact source. Reading is CPU-only; the heavy ``tensorboard`` import is
deferred to inside :func:`read_scalar_points`.

Scope of the first version (issue #254 plan):
  - First TLS data source only: ``events.out.tfevents.*`` scalars.
  - ``--pid`` selects a run by filename pid suffix.
  - jsonl fallback and ``--run <id>`` are follow-ups (tracked separately).

Reading semantics mirror ``areno/dashboard/server.py``'s
``_load_tensorboard_scalars`` byte-for-byte so a dashboard switch is
behavior-preserving:
  - ``EventAccumulator(size_guidance={"scalars": 10000})``
  - ``accumulator.Scalars(tag)[-500:]``
  - NaN values skipped
  - ``(name, step, value)`` de-duplication
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any


def now() -> str:
    """UTC timestamp string, matching ``areno/dashboard/server.py``'s ``now``."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def number_like(value: Any) -> bool:
    """True if ``value`` is a usable finite number (mirrors server's check)."""
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def tensorboard_event_sources(path: Path, pid: int | None) -> list[Path]:
    """Locate ``events.out.tfevents.*`` files under ``path``.

    ``pid`` filters by the filename pid suffix when given; ``None`` merges every
    event file in the directory. Falls back to ``[path]`` when no event files
    exist so a fresh directory degrades to an empty read rather than a crash.
    """
    event_files = sorted(path.rglob("events.out.tfevents.*"), key=lambda item: item.stat().st_mtime)
    if not event_files:
        return [path]
    if pid is None:
        return event_files
    pid_marker = f".{pid}."
    return [file for file in event_files if pid_marker in file.name or file.parent.name == f"pid-{pid}"]


def locate_event_files(metrics_dir: str | Path, pid: int | None = None) -> list[Path]:
    """Public locator: resolve ``metrics_dir`` and return event file paths."""
    return tensorboard_event_sources(Path(metrics_dir), pid)


def read_scalar_points(metrics_dir: str | Path, pid: int | None = None) -> list[dict[str, Any]]:
    """Read TensorBoard scalars into de-duplicated ``[{name, value, step, time}]``.

    Iteration order, truncation (:code:`[-500:]`), NaN skip, and
    ``(name, step, value)`` dedup match the dashboard read path exactly; callers
    that feed these points back through the dashboard's ``_add_metric`` get
    byte-identical ``job.metrics`` ordering.
    """
    path = Path(metrics_dir)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return []

    points: list[dict[str, Any]] = []
    seen: set[tuple[str, int, float]] = set()
    for accumulator_path in tensorboard_event_sources(path, pid):
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
                step = int(event.step)
                value = float(event.value)
                if math.isnan(value):
                    continue
                key = (tag, step, value)
                if key in seen:
                    continue
                seen.add(key)
                points.append({"name": tag, "value": value, "step": step, "time": now()})
    return points


def list_available_tags(points: list[dict[str, Any]]) -> list[str]:
    """Return the distinct metric names found in ``points``, sorted for display."""
    names: set[str] = set()
    for point in points:
        name = str(point.get("name") or "")
        if name:
            names.add(name)
    return sorted(names)


def summarize_metric(
    points: list[dict[str, Any]], name: str, *, recent_n: int = 20
) -> dict[str, Any]:
    """Aggregate one metric into a summary dict.

    Existing contract (unchanged, additive fields below):
    ``{name, count, last, min, max, recent, trend}`` plus the new step-aware
    ``last_step``/``min_step``/``max_step``/``mean``/``recent_steps``.

    - ``count``: number of finite points for ``name`` (NaN-free, since the reader
      already skips NaN).
    - ``last`` / ``last_step``: value and step at the highest step.
    - ``min``/``max``/``mean``: streaming single-pass over all retained points
      (O(1) memory, independent of the ``[-500:]`` truncation).
    - ``min_step``/``max_step``: the step range of the **retained window**
      (the reader keeps ``accumulator.Scalars(tag)[-500:]``). For runs longer
      than 500 steps ``min_step`` is the earliest step still in that tail, **not**
      the start of training; ``max_step`` is always the latest available step.
    - ``recent`` / ``recent_steps``: the last ``recent_n`` values and the matching
      steps, in step order (parallel arrays of equal length).
    - ``trend``: ``recent`` normalized to ``[0, 1]`` -- the same window as
      ``recent`` -- so ``--limit`` bounds the sparkline length; ``render_table``
      maps it to a UTF-8 sparkline, ``render_json`` returns it verbatim.
    """
    series = sorted(
        (point for point in points if point.get("name") == name and number_like(point.get("value"))),
        key=lambda point: int(point.get("step") or 0),
    )
    steps = [int(point.get("step") or 0) for point in series]
    values = [float(point.get("value")) for point in series]
    count = len(values)
    if count == 0:
        last = min_v = max_v = mean = None
        last_step = min_step = max_step = None
        recent: list[float] = []
        recent_steps: list[int] = []
        trend: list[float] = []
    else:
        last = values[-1]
        min_v = max_v = values[0]
        total = 0.0
        for value in values:
            if value < min_v:
                min_v = value
            if value > max_v:
                max_v = value
            total += value
        mean = total / count
        last_step = max(steps)
        min_step = min(steps)
        max_step = last_step
        for idx, step in enumerate(steps):
            if step == last_step:
                last = values[idx]
        window = values[-recent_n:] if recent_n > 0 else []
        window_steps = steps[-recent_n:] if recent_n > 0 else []
        recent = window
        recent_steps = window_steps
        trend = _normalize(window)

    return {
        "name": name,
        "count": count,
        "last": last,
        "last_step": last_step,
        "min": min_v,
        "max": max_v,
        "mean": mean,
        "min_step": min_step,
        "max_step": max_step,
        "recent": recent,
        "recent_steps": recent_steps,
        "trend": trend,
    }


def _normalize(values: list[float]) -> list[float]:
    """Scale ``values`` to ``[0, 1]``; a flat series maps to ``0.5`` everywhere."""
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5 for _ in values]
    span = high - low
    return [(value - low) / span for value in values]


_SPARKS = "▁▂▃▄▅▆▇█"


def render_table(summary: dict[str, Any]) -> str:
    """Render a summary as text: header rows + a hand-written sparkline.

    No external dependency (``rich``/``sparklines``) -- the trend is mapped to the
    8-glyph ``▁▂▃▄▅▆▇█`` rung by normalized value. Step-aware fields are shown so a
    human can read training progress (``steps``/``last``/``recent``) at a glance.
    """
    name = summary.get("name", "")
    count = summary.get("count", 0)
    trend = summary.get("trend") or []
    sparkline = _sparkline(trend)
    last = _fmt(summary.get("last"))
    last_step = summary.get("last_step")
    if last_step is not None and summary.get("last") is not None:
        last = f"{last} (step {last_step})"
    steps = _fmt_range(summary.get("min_step"), summary.get("max_step"))
    recent = summary.get("recent") or []
    recent_steps = summary.get("recent_steps") or []
    if recent_steps and len(recent_steps) == len(recent):
        recent_str = ", ".join(f"step {s}: {_fmt(v)}" for s, v in zip(recent_steps, recent))
    else:
        recent_str = ", ".join(_fmt(value) for value in recent)
    lines = [
        f"metric   {name}",
        f"count    {count}",
        f"steps    {steps}",
        f"last     {last}",
        f"min      {_fmt(summary.get('min'))}",
        f"max      {_fmt(summary.get('max'))}",
        f"mean     {_fmt(summary.get('mean'))}",
        f"trend    {sparkline}",
        f"recent   {recent_str}",
    ]
    return "\n".join(lines)


def _fmt_range(low: Any, high: Any) -> str:
    """Format a step range as ``low -> high`` (``-`` when either end is missing)."""
    if low is None or high is None:
        return "-"
    if low == high:
        return str(low)
    return f"{low} -> {high}"


def _sparkline(trend: list[float]) -> str:
    if not trend:
        return ""
    glyphs = []
    for value in trend:
        index = min(len(_SPARKS) - 1, max(0, int(value * len(_SPARKS))))
        glyphs.append(_SPARKS[index])
    return "".join(glyphs)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_json(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the summary as a JSON-serializable dict (trend = normalized array).

    The step-aware fields (``last_step``/``min_step``/``max_step``/``mean``/
    ``recent_steps``) are exposed so machine consumers can read training progress
    and pair recent values with their steps (``jq '.recent_steps, .recent'``).
    Existing keys keep their shape; the additions are purely append-only.
    """
    return {
        "name": summary.get("name"),
        "count": summary.get("count"),
        "last": summary.get("last"),
        "last_step": summary.get("last_step"),
        "min": summary.get("min"),
        "max": summary.get("max"),
        "mean": summary.get("mean"),
        "min_step": summary.get("min_step"),
        "max_step": summary.get("max_step"),
        "recent": summary.get("recent"),
        "recent_steps": summary.get("recent_steps"),
        "trend": summary.get("trend"),
    }