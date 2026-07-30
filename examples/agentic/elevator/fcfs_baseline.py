"""First-come-first-served baseline runner for the elevator dispatch demo.

Runs the deterministic FCFS policy (defined in ``game.fcfs_policy``) over a
JSONL dataset and reports delivered passengers, mean wait, invalid actions,
and overload refusals. Use it to compare a trained agent against a simple
hand-coded dispatcher. Output is both human-readable and available as JSON via
``--json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def run_dataset(records: list[dict]) -> dict:
    """Run FCFS over all records and return aggregated metrics."""

    per_episode: list[dict] = []
    for record in records:
        per_episode.append(game.run_fcfs_episode(record))
    return _aggregate(per_episode)


def _aggregate(per_episode: list[dict]) -> dict:
    if not per_episode:
        return {"episodes": 0}
    delivered = sum(m["delivered"] for m in per_episode)
    total = sum(m["total_passengers"] for m in per_episode)
    total_wait = sum(m["total_wait"] for m in per_episode)
    invalid = sum(m["invalid_actions"] for m in per_episode)
    refused = sum(m["overload_refused"] for m in per_episode)
    return {
        "episodes": len(per_episode),
        "total_delivered": delivered,
        "total_passengers": total,
        "delivery_rate": delivered / max(total, 1),
        "mean_wait_per_passenger": total_wait / max(total, 1),
        "total_invalid_actions": invalid,
        "total_overload_refused": refused,
        "by_scenario": _by_scenario(per_episode),
    }


def _by_scenario(per_episode: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for m in per_episode:
        grouped.setdefault(m["scenario"], []).append(m)
    out: dict[str, dict] = {}
    for scenario, items in grouped.items():
        delivered = sum(m["delivered"] for m in items)
        pax = sum(m["total_passengers"] for m in items)
        wait = sum(m["total_wait"] for m in items)
        out[scenario] = {
            "episodes": len(items),
            "delivery_rate": delivered / max(pax, 1),
            "mean_wait_per_passenger": wait / max(pax, 1),
            "invalid_actions": sum(m["invalid_actions"] for m in items),
            "overload_refused": sum(m["overload_refused"] for m in items),
        }
    return out


def _load_records(dataset_path: str | None) -> list[dict]:
    """Load records from a JSONL file or fall back to the default generator."""

    if not dataset_path:
        return dataset_generator.generate_records(count=64, seed=2026, scenario="mixed")
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_report(metrics: dict) -> str:
    """Render a human-readable summary string."""

    lines = [
        "Elevator FCFS baseline",
        f"  episodes:           {metrics.get('episodes', 0)}",
        f"  delivered:          {metrics.get('total_delivered', 0)}/{metrics.get('total_passengers', 0)}",
        f"  delivery_rate:      {metrics.get('delivery_rate', 0.0):.4f}",
        f"  mean_wait/pax:      {metrics.get('mean_wait_per_passenger', 0.0):.3f}",
        f"  invalid_actions:    {metrics.get('total_invalid_actions', 0)}",
        f"  overload_refused:   {metrics.get('total_overload_refused', 0)}",
    ]
    by_scenario = metrics.get("by_scenario", {})
    if by_scenario:
        lines.append("  by_scenario:")
        for name, m in sorted(by_scenario.items()):
            lines.append(
                f"    {name:12s} ep={m['episodes']:3d} rate={m['delivery_rate']:.3f} "
                f"wait={m['mean_wait_per_passenger']:.2f} invalid={m['invalid_actions']} refused={m['overload_refused']}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FCFS elevator dispatch baseline over a dataset.")
    parser.add_argument("--dataset", "-d", default=None, help="JSONL dataset path; omit to use the default generator.")
    parser.add_argument("--json", action="store_true", help="Emit metrics as JSON instead of human-readable text.")
    args = parser.parse_args()

    records = _load_records(args.dataset)
    metrics = run_dataset(records)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print(_format_report(metrics))


if __name__ == "__main__":
    main()
