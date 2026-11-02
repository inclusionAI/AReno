"""Audit P1 raw endpoints with the standard library; never execute archived code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
CONTAINER = load_module("p1_container", HERE.parent / "galore-capacity-evidence/verify.py")
AUDIT = load_module("p1_quality_audit", HERE / "audit.py")
SOURCE = "230fd97515ad044ebe075a011fed96f29ef7fa93aaac31147d8ec3fca3e4d668"
CORE = "47c75ef8a4e76cdbda5770a3875830d56e1a5420faf9c41ea9fd07f1972776ff"
NATIVE = "f2d49f7b8ffd2fccbfce0f6eeb3108088be7834627b88c9ad71a0eb486f4d226"
PROTOCOL = "7deed04e9721421558065c60d90992b2a7b61b08fa4c2c346b966a3316fca095"
COMMIT = "799070f6e1bd180ab98ea2a7722268f13536ee33"


def load_bundle(directory):
    return CONTAINER.load_bundle(directory, schema="areno-p1-controlled-quality-v1")


def json_at(files, name):
    return json.loads(files[name])


def audit_sources(files):
    prepared = json_at(files, "records/queue/harness/PREPARED.json")
    check = CONTAINER.check
    check(prepared["runtime_sha256"] == SOURCE and prepared["core_runtime_sha256"] == CORE, "runtime differs")
    check(prepared["native_sha256"] == NATIVE and prepared["commit"] == COMMIT, "native/source differs")
    check(prepared["model_revision"] == "4855588ea1a12789f2e965e5f52a9e4a24c94b2a", "model revision differs")
    full, core, count = hashlib.sha256(), hashlib.sha256(), 0
    prefix = "sources/runtime/"
    port = json_at(files, "records/PORT_MANIFEST.json")
    for name in sorted(files):
        if name.startswith(prefix + "areno/") and name.endswith(".py.txt"):
            relative = name[len(prefix) : -4]
            raw = files[name]
            full.update(relative.encode())
            full.update(raw)
            count += 1
            check(CONTAINER.sha(raw) == port["runtime_files"][relative]["sha256"], "runtime file differs")
            if not relative.startswith("areno/experimental/controlled_galore640/"):
                core.update(relative.encode())
                core.update(raw)
    check(count == 225 and full.hexdigest() == SOURCE and core.hexdigest() == CORE, "runtime snapshot differs")
    check(
        CONTAINER.sha(files[prefix + "areno/experimental/controlled_galore640/original.py.txt"])
        == "9234126fd49706ff9259d2fae7ae9f2122d101e131643e5c64ff7cc7df4a022c",
        "original640 algorithm differs",
    )
    protocol = files["records/queue/PREREGISTERED_QUALITY_PROTOCOL.json"]
    check(protocol == files["records/original/PREREGISTERED_QUALITY_PROTOCOL.json"], "scientific protocol changed")
    check(CONTAINER.sha(protocol) == PROTOCOL == prepared["preregistered_protocol_sha256"], "protocol identity differs")
    for name, expected in prepared["files"].items():
        raw = files["sources/queue/harness/" + name + ".txt"]
        check(CONTAINER.sha(raw) == expected["sha256"], "frozen scientific file differs")
        check(raw == files["sources/original/harness/" + name + ".txt"], "retry scientific code changed")
    for key in ("quality_helper", "capacity_helper", "capture_helper"):
        check(
            CONTAINER.sha(files[f"sources/{key}.py.txt"]) == prepared["dependencies"][prepared[key]]["sha256"],
            "historical helper differs",
        )
    return prepared


def audit_selection(files):
    check = CONTAINER.check
    cases = json_at(files, "records/cases.json")
    wanted = {(method, seed) for method in ("port640", "deferred647") for seed in range(3)}
    check(len(cases) == 6 and {(r["method"], r["seed"]) for r in cases} == wanted, "six fixed cases required")
    reused = {("port640", 0), ("deferred647", 1), ("port640", 2)}
    status = json_at(files, "records/queue/control/STATUS.json")
    check(status["status"] == "complete" and not status["active"] and not status["errors"], "queue incomplete")
    records = json_at(files, "records/queue/control/stage3/CASE_RESULTS.json")
    check(status["completed"] == records and len(records) == 3, "queued completion inventory differs")
    original = json_at(files, "records/original/control/stage3/CASE_RESULTS.json")
    interrupted = [r for r in original if r["outcome"] == "error"]
    check(
        len(interrupted) == 3 and all(r["interruption"] == "foreign_PID" for r in interrupted),
        "original interruption attribution differs",
    )
    for case in cases:
        method, seed, gpu = case["method"], case["seed"], case["gpu"]
        check(gpu == seed + 1 and case["portable_output"] == f"cases/{method}_s{seed}", "case mapping differs")
        check(case["reused"] is ((method, seed) in reused), "old/new endpoint selection differs")
        if case["reused"]:
            check(
                any(
                    r["case_id"] == f"q4096_gpu{gpu}_{method}_s{seed}"
                    and r["outcome"] == "complete"
                    and r["return_code"] == 0
                    for r in original
                ),
                "old completion missing",
            )
        else:
            q = status["queue"][str(gpu)]
            check(
                q["status"] == "complete" and q["dispatches"] == 1 and q["resource_retries"] == 0,
                "queued attempt differs",
            )
            check(q["accepted_output"] == case["original_output"], "accepted output differs")
            check(
                any(
                    r["case_id"] == q["accepted_case_id"] and r["outcome"] == "complete" and r["return_code"] == 0
                    for r in records
                ),
                "new completion missing",
            )
    inventory = json_at(files, "records/queue/REUSE_INPUT_INVENTORY.json")
    for case in inventory["cases"]:
        check((case["method"], case["seed"]) in reused, "unexpected reused case")
        for name, expected in case["files"].items():
            raw = files[f"cases/{case['method']}_s{case['seed']}/{name}"]
            check(
                len(raw) == expected["bytes"] and CONTAINER.sha(raw) == expected["sha256"],
                "old completed evidence changed",
            )
    for method, seed in (("deferred647", 0), ("port640", 1), ("deferred647", 2)):
        rows = [json.loads(line) for line in files[f"interrupted/{method}_s{seed}/steps.jsonl"].splitlines()]
        check(0 < len(rows) < 4096, "interrupted job cannot become full endpoint")
        check(
            all(row["step"] == i and row["seed"] == seed for i, row in enumerate(rows)),
            "partial update ordering differs",
        )
    return cases


def audit_bundle(manifest, files):
    if not __debug__:
        raise RuntimeError("Evidence auditing requires assertions; do not use -O")
    CONTAINER.check(manifest["schema"] == "areno-p1-controlled-quality-v1", "unexpected schema")
    prepared = audit_sources(files)
    cases = audit_selection(files)
    with tempfile.TemporaryDirectory(prefix="areno-p1-quality-") as temporary:
        root = Path(temporary)
        for name, raw in files.items():
            if name.startswith("cases/"):
                CONTAINER.safe_name(name)
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
        mapping = {(r["method"], r["seed"]): root / r["portable_output"] for r in cases}
        fresh = AUDIT.audit_group(mapping, prepared)
    stored = json_at(files, "records/queue/FROZEN_SIX_CASE_QUALITY_ANALYSIS.json")
    CONTAINER.check(fresh == stored, "fresh raw quality analysis differs")
    # A separate summation implementation verifies every actual scalar again.
    for case in cases:
        rows = [
            json.loads(line)
            for line in files[case["portable_output"] + "/validation-response-logprobs.jsonl"].splitlines()
        ]
        values = [value for row in rows for value in row["response_logprobs"]]
        actual = -math.fsum(values) / len(values)
        report = next(r for r in fresh["reports"] if r["method"] == case["method"] and r["seed"] == case["seed"])
        CONTAINER.check(
            math.isclose(actual, report["validation"]["loss"], rel_tol=0, abs_tol=1e-12), "fsum raw NLL differs"
        )
    return dict(
        status="passed",
        cases=6,
        training_updates=24576,
        updates_per_case=4096,
        examples_per_validation=708,
        response_targets_per_validation=52181,
        runtime_sha256=SOURCE,
        native_sha256=NATIVE,
        raw_response_nll_recomputed=True,
        all_completed_worker_score_train_records_checked=True,
        old_completed3_unchanged=True,
        paired_seeds=fresh["paired_seeds"],
        mean_response_token_nll=fresh["mean_response_token_nll"],
        relative_mean_nll_difference_percent=fresh["relative_mean_nll_difference_percent"],
        untouched_PR640_end_to_end=False,
        historical_adamw_gate_retested=False,
        fresh_8bit_quality_comparison=False,
        token_payload_replayed_offline=False,
        archived_code_executed=False,
    )


def verify(directory):
    return audit_bundle(*load_bundle(directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.json_output:
        args.json_output.write_text(output)
    print(output, end="")
