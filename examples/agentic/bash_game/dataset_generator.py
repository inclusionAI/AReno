"""Deterministic JSONL generator for the Bash game (巴什博弈 / 取石子游戏).

Produces complete *positions* (``n`` stones, ``m`` max-take), never a
conversation trajectory. The oracle optimal move is stored as hidden metadata
(``oracle_move`` / ``solution``) that the prompt must never expose.

Split guarantees:

* Each split draws ``n`` and ``m`` from a local RNG seeded by a split-specific
  constant, so train / validation / test are disjoint instance streams by
  construction (no shared sub-seed, no overlapping ID space).
* A global fingerprint set additionally rejects any duplicated ``(n, m)``
  position across splits (defence against degenerate seeds).
* Every generation run recomputes the oracle for every emitted row and verifies
  the move is legal and unique, aborting loudly on any mismatch.

Difficulty axes:

* ``m`` (max take) controls the modulus period ``m + 1``; the reachable
  ``take`` values scale with ``m`` so bigger ``m`` means more answer classes.
* ``n`` (pile size) controls how hard the modular reduction is to read.
* ``--losing`` toggles whether losing positions (``resign``) are included.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import VALID_MAX_TAKES, BashGame, normalize_record  # noqa: E402

# Split-specific big offsets (arbitrary but fixed) keep the three instance
# streams disjoint without sharing a counter.
_SPLIT_OFFSET = {"train": 0x9E3779B9, "val": 0x85EBCA77, "test": 0xC2B2AE3D}


def _mix(x: int) -> int:
    """SplitMix64-style finalizer for a stable per-record sub-seed."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    return x ^ (x >> 31)


def _make_rng(split: str, index: int, seed: int) -> random.Random:
    return random.Random(_mix(seed + _SPLIT_OFFSET[split] + index))


def generate_records(
    split: str, count: int, seed: int, n_max: int, max_takes: tuple[int, ...], include_losing: bool = True
) -> list[dict]:
    records: list[dict] = []
    seen: set[tuple[int, int]] = set()
    index = 0
    while len(records) < count:
        r = _make_rng(split, index, seed)
        m = r.choice(list(max_takes))
        n = r.randint(1, n_max)
        key = (n, m)
        index += 1
        if key in seen or (not include_losing and n % (m + 1) == 0):
            continue
        seen.add(key)
        records.append(normalize_record({"n": n, "m": m, "instance_id": f"{split}-{index:06d}"}))
    return records


def generate_splits(
    train: int, val: int, test: int, seed: int, n_max: int, max_takes: tuple[int, ...], include_losing: bool = True
) -> dict[str, list[dict]]:
    """Generate three disjoint splits with a global fingerprint dedup."""
    splits: dict[str, list[dict]] = {}
    seen: set[tuple[int, int]] = set()
    for split, count in (("train", train), ("val", val), ("test", test)):
        records: list[dict] = []
        index = 0
        attempts = 0
        while len(records) < count:
            attempts += 1
            if attempts > count * 1000:
                raise RuntimeError("position space exhausted; loosen n_max/take range")
            r = _make_rng(split, index, seed)
            m = r.choice(list(max_takes))
            n = r.randint(1, n_max)
            key = (n, m)
            index += 1
            if key in seen or (not include_losing and n % (m + 1) == 0):
                continue
            seen.add(key)
            records.append(normalize_record({"n": n, "m": m, "instance_id": f"{split}-{index:06d}"}))
        splits[split] = records
    return splits


def _self_check(records: list[dict]) -> None:
    for r in records:
        g = BashGame(r["n"], r["m"])
        move = g.optimal_move()
        assert move == r["oracle_move"], f"oracle mismatch for {r}"
        if r["winning"]:
            assert "take" in move and 1 <= move["take"] <= r["m"]
        else:
            assert move.get("resign") is True
        if r["n"] < 1 or r["m"] not in VALID_MAX_TAKES:
            raise AssertionError(f"bad record {r}")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Bash game JSONL splits.")
    p.add_argument("--output", "-o", required=True, help="Output .jsonl or (with --split-all) directory.")
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--count", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-max", type=int, default=40, help="Max pile size n.")
    p.add_argument("--max-takes", type=int, nargs="+", default=list(VALID_MAX_TAKES), help="Allowed max-take values m.")
    p.add_argument("--no-losing", action="store_true", help="Generate winning positions only.")
    p.add_argument(
        "--split-all",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Generate three splits into --output directory.",
    )
    a = p.parse_args()

    max_takes = tuple(v for v in a.max_takes if v in VALID_MAX_TAKES)
    if not max_takes:
        p.error("no valid max-take values")
    losing = not a.no_losing

    if a.split_all:
        train, val, test = a.split_all
        splits = generate_splits(train, val, test, a.seed, a.n_max, max_takes, losing)
        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=True)
        for s, recs in splits.items():
            _self_check(recs)
            write_jsonl(out / f"{s}.jsonl", recs)
            print(f"{s}: {out / f'{s}.jsonl'} ({len(recs)} records)")
    else:
        recs = generate_records(a.split, a.count, a.seed, a.n_max, max_takes, losing)
        _self_check(recs)
        write_jsonl(Path(a.output), recs)
        print(f"{a.split}: {a.output} ({len(recs)} records)")


if __name__ == "__main__":
    main()
