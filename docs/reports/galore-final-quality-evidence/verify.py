"""Recompute final-source SFT evidence using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
CONTAINER = load_module("capacity_container", HERE.parent / "galore-capacity-evidence/verify.py")
AUDIT = load_module("final_quality_auditor", HERE / "audit.py")


def load_bundle(directory):
    return CONTAINER.load_bundle(directory, schema="areno-final-quality-evidence-v1")


def audit_bundle(manifest, files, repo=None):
    if not __debug__:
        raise RuntimeError("Evidence auditing requires Python assertions; do not use -O")
    CONTAINER.check(manifest["schema"] == "areno-final-quality-evidence-v1", "unexpected schema")
    if repo is not None:
        repo = Path(repo)
        paths = sorted((repo / "areno").rglob("*.py"))
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.relative_to(repo).as_posix().encode())
            digest.update(path.read_bytes())
        CONTAINER.check(
            len(paths) == 221 and digest.hexdigest() == AUDIT.SOURCE, "current runtime differs from measured code"
        )
        for name in (
            "scripts/galore_deferred/run.py",
            "scripts/galore_sft/common.py",
            "scripts/galore_deferred/model-manifest.json",
            "scripts/galore_sft/data-manifest.json",
        ):
            key = "repo/" + name + (".txt" if name.endswith(".py") else "")
            CONTAINER.check((repo / name).read_bytes() == files[key], "public reproduction source/manifest differs")
    with tempfile.TemporaryDirectory(prefix="areno-quality-audit-") as temp:
        root = Path(temp)
        # Container validation rejects paths/links. These are data-only copies;
        # no archived code is imported or executed, including the .py.txt files.
        for name, raw in files.items():
            CONTAINER.safe_name(name)
            relocated = name[:-4] if name.endswith(".py.txt") else name
            path = root / relocated
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        previous = AUDIT.ROOT
        AUDIT.ROOT = root
        try:
            records = json.loads(files["stage3/CASE_RESULTS.json"])
            fresh = AUDIT.analyze_quality(root / "stage3", records)
        finally:
            AUDIT.ROOT = previous
    for record in records:
        CONTAINER.check(fresh["cases"][record["case_id"]]["gpu"] == str(record["gpu"]), "device assignment differs")
    saved = json.loads(files["stage3/ANALYSIS.json"])
    CONTAINER.check(json.loads(json.dumps(fresh)) == saved, "fresh analysis differs from recorded analysis")
    status = json.loads(files["stage3/STATUS.json"])
    CONTAINER.check(
        status["status"] == "complete" and not status["errors"] and not status["active"], "queue incomplete"
    )
    CONTAINER.check(status["completed"] == records, "queue inventory differs")
    comparisons = []
    for seed in range(3):
        control = fresh["cases"][f"quality_adam8_s{seed}"]
        candidate = fresh["cases"][f"quality_deferred_s{seed}"]
        comparisons.append(
            {
                "seed": seed,
                "gpu": seed + 1,
                "memory_reduction_percent": 100
                * (1 - candidate["all"]["allocated_peak_bytes"] / control["all"]["allocated_peak_bytes"]),
                "full_wall_overhead_percent": 100
                * (candidate["full_wall_mean_seconds"] / control["full_wall_mean_seconds"] - 1),
            }
        )
    return {
        "status": "passed",
        "cases": 9,
        "training_updates": 9 * 4096,
        "examples_per_validation": 708,
        "targets_per_validation": 52181,
        "runtime_sha256": AUDIT.SOURCE,
        "current_repo_checked": repo is not None,
        "three_seed_results": fresh["three_seed_results"],
        "primary_quality_gate": fresh["primary_quality_gate"],
        "response_nll_comparisons": fresh["response_nll_comparisons"],
        "same_card_training_observations": comparisons,
        "token_payload_replayed_offline": False,
        "raw_response_nll_recomputed": True,
        "new_checkpoint_saved": False,
    }


def verify(directory, repo=None):
    return audit_bundle(*load_bundle(directory), repo=repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory, args.repo)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.json_output:
        args.json_output.write_text(text)
    print(text, end="")
