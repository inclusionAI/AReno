"""Convert public typed-decision datasets into JevForge records.

Output is a directory of `<split>.jsonl` files that `dataset_loader.load_questions`
(and therefore `train.py` / `tmp.sh`) reads directly. State objects are rendered
the way `serve_decisions.py` renders a JSON `state` from a request
(`json.dumps(..., ensure_ascii=False, sort_keys=True)`), so training text matches
what the decisions API feeds the model.

Open-Jev v1.1 (ModelScope `ZefanCai/Open-Jev-v1.1`, config
`community-hard-mix-v2-redistributable`); one row is one question:

    python examples/classify/jev/convert_datasets.py open-jev \
        --src ~/data/open-jev-v1.1 --out ~/data/jev-records/open-jev-v1.1

  splits: train, validation -> dev, calibration, test, ood. Choice options are
  bare ids in this release, so each option doubles as its own description.

typed-decisions (`LocalLLaMA/typed-decisions`, config `all`); one row is one
case with several questions over a shared state:

    python examples/classify/jev/convert_datasets.py typed-decisions \
        --src ~/data/typed-decisions --out ~/data/jev-records/typed-decisions

  splits: train, test. `source_group` is the workflow name.

Both read `<src>/<split>.parquet` (pyarrow).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OPEN_JEV_SPLITS = {"train": "train", "validation": "dev", "calibration": "calibration", "test": "test", "ood": "ood"}
TYPED_DECISIONS_SPLITS = {"train": "train", "test": "test"}


def render_state(state) -> str:
    """Same text `serve_decisions.render_state` produces for a JSON state."""

    if isinstance(state, str):
        try:
            state = json.loads(state)
        except ValueError:
            return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def normalized(values: dict[str, float]) -> dict[str, float]:
    """Rescale to sum 1 (sources round to ~6 decimals)."""

    total = sum(float(v) for v in values.values())
    if total <= 0:
        raise ValueError(f"target has no probability mass: {values}")
    return {key: float(value) / total for key, value in values.items()}


def open_jev_question(row: dict) -> tuple[dict, dict[str, float]]:
    """Map one Open-Jev row to a Jev question and its target distribution."""

    kind = row["kind"]
    options = [str(option) for option in row["options"]]
    target = [float(value) for value in row["target"]]
    if len(options) != len(target):
        raise ValueError(f"{row['id']}: {len(options)} options vs {len(target)} targets")
    question = {"type": kind, "instructions": str(row["question"])}
    if kind == "choice":
        question["criteria"] = {option: option for option in options}
        return question, normalized(dict(zip(options, target, strict=True)))
    if kind == "noul":
        if options != ["no", "yes"]:
            raise ValueError(f"{row['id']}: unexpected noul options {options}")
        return question, normalized({"false": target[0], "true": target[1]})
    if kind == "score":
        question["criteria"] = options
        return question, normalized({str(index): value for index, value in enumerate(target)})
    raise ValueError(f"{row['id']}: unknown kind {kind!r}")


def convert_open_jev(src: Path, out: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    columns = ["id", "group_id", "source", "kind", "question", "options", "target", "state_json", "metadata_json"]
    counts = {}
    for source_split, split in OPEN_JEV_SPLITS.items():
        rows = pq.read_table(src / f"{source_split}.parquet", columns=columns).to_pylist()
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                question, target = open_jev_question(row)
                basis = (json.loads(row["metadata_json"] or "{}") or {}).get("target_basis") or "unspecified"
                record = {
                    "id": row["id"],
                    "split": split,
                    "source_group": row["group_id"],
                    "request": {"state": render_state(row["state_json"]), "questions": {"q": question}},
                    "targets": {"q": target},
                    "target_kinds": {"q": str(basis)},
                    "meta": {"source": row["source"], "dataset": "ZefanCai/Open-Jev-v1.1"},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    return counts


def convert_typed_decisions(src: Path, out: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    counts = {}
    for source_split, split in TYPED_DECISIONS_SPLITS.items():
        rows = pq.read_table(src / f"{source_split}.parquet", columns=["id", "workflow", "state", "questions", "gold"])
        rows = rows.to_pylist()
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                questions = json.loads(row["questions"])
                gold = json.loads(row["gold"])
                record = {
                    "id": row["id"],
                    "split": split,
                    "source_group": row["workflow"],
                    "request": {"state": render_state(row["state"]), "questions": questions},
                    "targets": {qid: normalized(gold[qid]["probabilities"]) for qid in questions},
                    "target_kinds": {qid: "teacher_mean" for qid in questions},
                    "meta": {"dataset": "LocalLLaMA/typed-decisions", "workflow": row["workflow"]},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["open-jev", "typed-decisions"])
    parser.add_argument("--src", required=True, help="directory holding <split>.parquet")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    convert = convert_open_jev if args.dataset == "open-jev" else convert_typed_decisions
    counts = convert(src, out)
    (out / "manifest.json").write_text(
        json.dumps({"dataset": args.dataset, "source": str(src), "records": counts}, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
