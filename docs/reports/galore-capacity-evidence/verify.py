"""Portable standard-library audit of the capacity evidence; never runs harnesses."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import statistics
import tarfile
from pathlib import Path, PurePosixPath

SOURCE = "337c7612282e8fc5254d890ac7659c88c2f38a8e8430d5b3cdc84f6dc96b28b8"
NATIVE = "f2d49f7b8ffd2fccbfce0f6eeb3108088be7834627b88c9ad71a0eb486f4d226"
METHODS = ("adamw", "adam8", "deferred")
GPUS = (1, 2, 3)
BATCHES = (1, 2, 4, 8, 12, 16, 24, 32)
CONTEXTS = (512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
UUIDS = {
    1: "GPU-d78c0eae-9ae7-3849-fa3d-f092f7502900",
    2: "GPU-833bb0e5-2efb-bd4f-8f7f-21f792359487",
    3: "GPU-3afb1149-27c9-9142-da5b-9738e1520649",
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    check(bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name, "unsafe archive path")
    return name


def read_tar(raw, compressed=False):
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz" if compressed else "r:") as archive:
        for member in archive:
            name = safe_name(member.name)
            check(member.isfile() and name not in files, "nonregular or duplicate archive member")
            check(member.size <= 128 * 1024**2, "oversized archive member")
            files[name] = archive.extractfile(member).read()
    return files


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    check(manifest["schema"] == "areno-capacity-evidence-v1", "unsupported schema")
    check(manifest["chunk_bytes"] == 512 * 1024 and manifest["max_archive_bytes"] == 900_000, "unexpected size policy")
    chunks, files, part_names = {}, {}, set()
    for part in manifest["parts"]:
        name = safe_name(part["archive"])
        check(len(PurePosixPath(name).parts) == 1 and name not in part_names, "duplicate/unsafe archive")
        part_names.add(name)
        raw = (directory / name).read_bytes()
        check(len(raw) == part["bytes"] < 900_000 and sha(raw) == part["sha256"], "archive identity mismatch")
        members = read_tar(raw, True)
        check(
            set(members) == set(part["members"]) and len(members) == len(part["members"]),
            "partition inventory mismatch",
        )
        for key, data in members.items():
            check(key not in chunks, "duplicate chunk")
            expected = manifest["chunks"][key]
            check(
                expected["archive"] == name
                and len(data) == expected["bytes"] <= 512 * 1024
                and sha(data) == expected["sha256"],
                "chunk identity mismatch",
            )
            chunks[key] = data
    check(set(chunks) == set(manifest["chunks"]), "missing/extra chunks")
    used, units = set(), set()
    for unit in manifest["units"]:
        check(unit["name"] not in units, "duplicate logical unit")
        units.add(unit["name"])
        check(
            not used.intersection(unit["chunks"]) and len(set(unit["chunks"])) == len(unit["chunks"]),
            "reused logical chunk",
        )
        used.update(unit["chunks"])
        raw = b"".join(chunks[key] for key in unit["chunks"])
        check(len(raw) == unit["bytes"] and sha(raw) == unit["sha256"], "logical tar identity mismatch")
        members = read_tar(raw)
        index = json.loads(members.pop("INDEX.json"))
        check(set(members) == set(index) and len(index) == unit["files"], "logical file inventory mismatch")
        for key, data in members.items():
            check(key not in files, "duplicate global evidence path")
            check(
                len(data) == index[key]["bytes"] and sha(data) == index[key]["sha256"],
                "evidence file identity mismatch",
            )
            check(PurePosixPath(key).suffix in (".json", ".jsonl", ".txt", ".log", ".md"), "unexpected payload type")
            data.decode("utf-8")
            files[key] = data
    check(used == set(chunks), "unreferenced chunk")
    return manifest, files


def json_at(files, name):
    return json.loads(files[name])


def lines_at(files, name):
    return [json.loads(line) for line in files.get(name, b"").splitlines()]


def axis_cells(axis):
    return [(1024, batch) for batch in BATCHES] if axis == "batch" else [(context, 4) for context in CONTEXTS]


def screen_cells():
    return list(dict.fromkeys(axis_cells("batch") + axis_cells("context")))


def frontiers(records, method):
    passed = {
        (r["context"], r["batch"])
        for r in records
        if r["stage"] == "frontier" and r["method"] == method and r["outcome"] == "complete"
    }
    return {
        axis: next((cell for cell in reversed(axis_cells(axis)) if cell in passed), None)
        for axis in ("batch", "context")
    }


def audit_schedule(records):
    check(len(records) == 69 and len({r["case_id"] for r in records}) == 69, "69 unique valid cases required")
    check(
        all(
            r["method"] in METHODS and r["gpu"] in GPUS and r["stage"] in ("screen", "frontier", "boundary")
            for r in records
        ),
        "unknown case",
    )
    stages = [("screen", "frontier", "boundary").index(r["stage"]) for r in records]
    check(stages == sorted(stages), "overlapping/reordered stages")
    screens = [r for r in records if r["stage"] == "screen"]
    check(
        len(screens) == 45
        and {(r["method"], r["context"], r["batch"]) for r in screens}
        == {(m, c, b) for m in METHODS for c, b in screen_cells()},
        "screen grid incomplete",
    )
    check(
        all(r["gpu"] == METHODS.index(r["method"]) + 1 and r["phase"] == "screen" for r in screens),
        "screen assignment mismatch",
    )
    for method, gpu in (("adamw", 2), ("adam8", 3), ("deferred", 1)):
        confirmations = [r for r in records if r["stage"] == "frontier" and r["method"] == method]
        check(
            all(r["gpu"] == gpu and r["phase"] == "confirm" for r in confirmations), "confirmation assignment mismatch"
        )
        observed = {(r["context"], r["batch"]): r for r in confirmations}
        check(len(observed) == len(confirmations), "duplicate confirmation")
        passed = {(r["context"], r["batch"]) for r in screens if r["method"] == method and r["outcome"] == "complete"}
        visited = set()
        for axis in ("batch", "context"):
            for cell in [cell for cell in reversed(axis_cells(axis)) if cell in passed][:2]:
                check(cell in observed, "missing preregistered confirmation")
                visited.add(cell)
                if observed[cell]["outcome"] == "complete":
                    break
        check(visited == set(observed), "unregistered confirmation/fallback")
    cells = list(dict.fromkeys(cell for cell in frontiers(records, "deferred").values() if cell))
    boundary = [r for r in records if r["stage"] == "boundary"]
    expected = {(g, m, c, b) for g in GPUS for m in METHODS for c, b in cells}
    check(
        len(boundary) == 18
        and len(expected) == 18
        and {(r["gpu"], r["method"], r["context"], r["batch"]) for r in boundary} == expected,
        "all-three-card boundary grid incomplete",
    )
    check(all(r["phase"] == "confirm" for r in boundary), "boundary budget mismatch")


def typed_oom(failure):
    if not isinstance(failure, dict) or failure.get("status") != "cuda_oom":
        return False
    for native in (failure, failure.get("worker_failure")):
        if not isinstance(native, dict):
            continue
        message, trace = native.get("exception", ""), native.get("traceback", "")
        if (
            native.get("status") == "cuda_oom"
            and native.get("exception_type") == "OutOfMemoryError"
            and native.get("stage") in ("init", "score", "train")
            and isinstance(message, str)
            and re.match(r"^CUDA(?: error:)? out of memory(?:[.!:]|\s|$)", message, re.I)
            and isinstance(trace, str)
            and "OutOfMemoryError:" in trace
            and message in trace
        ):
            return True
    return False


def measure(rows):
    if not rows:
        return None
    wall = sum(r["score_plus_train_seconds"] for r in rows)
    result = {
        "updates": len(rows),
        "target_tokens": sum(r["target_tokens"] for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "complete_wall_seconds": wall,
        "complete_wall_mean_seconds": wall / len(rows),
        "complete_wall_median_seconds": statistics.median(r["score_plus_train_seconds"] for r in rows),
        "input_tokens_per_second": sum(r["input_tokens"] for r in rows) / wall,
        "optimizer_event_mean_seconds": statistics.mean(r["update_optimizer_time_s"] for r in rows),
    }
    for output, field in (
        ("allocated", "score_plus_train_peak_allocated_bytes"),
        ("reserved", "score_plus_train_peak_reserved_bytes"),
        ("scoring_allocated", "scoring_peak_allocated_bytes"),
        ("scoring_reserved", "scoring_peak_reserved_bytes"),
        ("training_allocated", "training_call_peak_allocated_bytes"),
        ("training_reserved", "training_call_peak_reserved_bytes"),
    ):
        result[f"{output}_peak_bytes"] = max(r[field] for r in rows)
    return result


def matches(actual, expected, name):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        check(
            isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8),
            f"numeric summary mismatch: {name}",
        )
    else:
        check(actual == expected, f"summary mismatch: {name}")


def audit_input(row, record, data, orders):
    context, batch, step = record["context"], record["batch"], row["step"]
    size = data["token_count"] // context
    indices = []
    for cursor in range(step * batch, (step + 1) * batch):
        epoch, offset = divmod(cursor, size)
        key = (size, epoch)
        if key not in orders:
            order = list(range(size))
            random.Random(1000003 * epoch).shuffle(order)
            orders[key] = order
        indices.append(orders[key][offset])
    expected = {
        "seed": 0,
        "context": context,
        "batch": batch,
        "block_count": size,
        "block_indices": indices,
        "input_tokens": batch * context,
        "target_tokens": batch * (context - 1),
        "tail_tokens_dropped": data["token_count"] % context,
        "source_stream_sha256": data["stream_sha256"],
        "packing_policy": data["packing_policy"],
        "capacity_only_not_quality": True,
    }
    for key, value in expected.items():
        check(row[key] == value, f"input identity mismatch: {record['case_id']} {step} {key}")
    check(re.fullmatch(r"[0-9a-f]{64}", row["batch_sha256"]) is not None, "invalid batch hash")


def audit_rows(rows, record, data, orders):
    deferred = record["method"] == "deferred"
    for step, row in enumerate(rows):
        check(row["step"] == step and row["warmup"] == (step < 2), "row clock/warmup mismatch")
        audit_input(row, record, data, orders)
        check(row["sequence_lengths"] == [record["context"]] * record["batch"], "non-full-length input")
        expected = {
            "update_worker_step": step + 1,
            "update_optimizer_steps": 1,
            "logp_abs_diff_mean": 0,
            "sft_pack_count": 1,
            "sft_dp_shards_per_pack": 1,
            "sft_total_target_tokens": row["target_tokens"],
            "sft_grad_scale": 1,
            "optimizer_model_owned_master_enabled": 0,
            "optimizer_deferred_projection_enabled": int(deferred),
            "optimizer_fused_scale_writeback_enabled": int(deferred),
            "update_refresh_any": int(deferred and step % 200 == 0),
        }
        if deferred:
            expected.update(
                deferred_owner_count=226,
                deferred_owners_once=1,
                deferred_barrier_versions_unchanged=1,
                deferred_full_leaf_bytes_after_backward=0,
                deferred_cache_empty_after_step=1,
            )
            check(row["galore_master_bytes"] > 0, "strict FP32 master missing")
        for key, value in expected.items():
            check(row[key] == value, f"native lifecycle/normalization mismatch: {key}")
        check(
            all(
                math.isfinite(row[key])
                for key in ("loss", "grad_norm", "clipped_grad_norm", "score_plus_train_seconds")
            ),
            "nonfinite loss/norm/time",
        )
        check(row["score_plus_train_seconds"] > 0 and row["scoring_seconds"] > 0, "invalid wall time")
        check(0 <= row["clipped_grad_norm"] <= 1.00001 and row["grad_norm"] >= 0, "gradient clipping guard")
        check(
            math.isclose(row["clipped_grad_norm"], min(row["grad_norm"], 1), rel_tol=1e-5, abs_tol=1e-6),
            "current global gradient clipping mismatch",
        )
        for metric in ("allocated", "reserved"):
            score, train = row[f"scoring_peak_{metric}_bytes"], row[f"training_call_peak_{metric}_bytes"]
            check(
                score > 0
                and train > 0
                and row[f"update_peak_{metric}_bytes"] <= train
                and row[f"score_plus_train_peak_{metric}_bytes"] == max(score, train),
                "whole-call allocator mismatch",
            )


def audit_case(files, label, record, prepared, data, orders, invalid=False):
    prefix = f"{label}/{record['output_dir']}/"
    metadata = json_at(files, prefix + "metadata.json")
    args = metadata["arguments"]
    for key in ("method", "context", "batch", "phase"):
        check(args[key] == record[key], f"case metadata mismatch: {key}")
    check(
        args["seed"] == 0 and args["model"] == prepared["model_path"] and args["data_dir"] == prepared["data_dir"],
        "model/data/seed mismatch",
    )
    check(
        metadata["source_sha256"] == SOURCE and metadata["native_extension_sha256"] == NATIVE,
        "measured runtime mismatch",
    )
    check(
        PurePosixPath(metadata["source_root"]) == PurePosixPath(prepared["data_dir"]).parent / "repo",
        "imported source tree mismatch",
    )
    check(metadata["CUDA_VISIBLE_DEVICES"] == str(record["gpu"]), "physical card mismatch")
    check(metadata["data_manifest_sha256"] == sha(files[f"{label}/data/DATA_MANIFEST.json"]), "input manifest mismatch")
    check(
        metadata["read_validation_or_test"] is False and metadata["saved_checkpoint"] is False,
        "unexpected validation/checkpoint",
    )
    flag = str(int(record["method"] == "deferred"))
    check(
        metadata["flags"]
        == {"ARENO_EXPERIMENT_DEFERRED_PROJECTION": flag, "ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK": flag},
        "optimizer flags mismatch",
    )
    monitors = lines_at(files, prefix + "monitor.jsonl")
    check(bool(monitors) and record["uuid"] == UUIDS[record["gpu"]], "missing monitor/device identity")
    for sample in monitors:
        check(not sample["foreign_pids"] and sample["free_disk_bytes"] >= 8 * 1024**3, "resource-conflicted case")
        gpu = [part.strip() for part in sample["gpu"].split(",")]
        check(int(gpu[0]) == record["gpu"] and gpu[1] == record["uuid"], "monitor UUID mismatch")
    rows = lines_at(files, prefix + "steps.jsonl")
    audit_rows(rows, record, data, orders)
    budget = 16 if record["phase"] == "screen" else 203
    outcome = record["outcome"]
    if invalid:
        check(
            outcome == "error"
            and record["interruption"] in ("case_timeout", "controller_stop_requested")
            and len(rows) < budget,
            "interrupted attempt relabeled",
        )
    else:
        check(record["interruption"] is None, "interrupted valid case")
        evidence = json_at(files, prefix + ("summary.json" if outcome == "complete" else "failure.json"))
        check(outcome in ("complete", "cuda_oom"), "unclassified case")
        for key in ("method", "context", "batch", "phase"):
            check(evidence[key] == record[key], "evidence case mismatch")
        check(
            evidence["completed_updates"] == len(rows) and evidence["source_sha256"] == SOURCE,
            "evidence row count/source mismatch",
        )
        check(
            evidence["saved_checkpoint"] is False and evidence["read_validation_or_test"] is False,
            "unexpected saved/evaluated state",
        )
        if outcome == "cuda_oom":
            check(
                record["return_code"] == 42 and typed_oom(evidence) and len(rows) < budget, "not actual typed CUDA OOM"
            )
            if evidence["stage"] in ("score", "train"):
                failed = evidence["failed_batch_identity"]
                check(failed["step"] == len(rows), "failed input clock mismatch")
                audit_input(failed, record, data, orders)
        else:
            check(
                record["return_code"] == 0 and evidence["status"] == "complete" and len(rows) == budget,
                "incomplete success",
            )
            check(
                evidence["warmup_updates"] == 2
                and evidence["measured_updates"] == budget - 2
                and evidence["quality_claim"] is False,
                "update budget/claim mismatch",
            )
            refresh = [row for row in rows[2:] if row["update_refresh_any"]]
            check(evidence["refresh_steps"] == [row["step"] for row in refresh], "refresh summary mismatch")
            check(
                [row["step"] for row in refresh] == ([200] if record["method"] == "deferred" and budget == 203 else []),
                "production refresh200 missing",
            )
            groups = {
                "all": rows[2:],
                "cold": rows[:1],
                "ordinary": [r for r in rows[2:] if not r["update_refresh_any"]],
                "refresh": refresh,
            }
            for group, values in groups.items():
                expected = measure(values)
                if expected is None:
                    check(evidence[group] is None, "unexpected refresh statistics")
                else:
                    for key, value in expected.items():
                        matches(evidence[group][key], value, group + "." + key)
    if rows:
        inventory = json_at(files, prefix + "worker-model-inventory.json")
        expected = json_at(files, f"{label}/META_MODEL_EXPECTATION.json")
        for key in (
            "model_parameter_owners",
            "optimizer_parameter_owners",
            "unique_parameters",
            "dtype_elements",
            "dtype_owners",
            "shape_eligible_basis_owners_rank128",
            "shape_order_dtype_sha256",
        ):
            check(inventory[key] == expected[key], "actual model inventory mismatch")
        check(
            inventory["optimizer_order_matches_named_owners"] is True
            and all(owner["device"].startswith("cuda:") for owner in inventory["owners"]),
            "model ownership mismatch",
        )
        state = json_at(files, prefix + "worker-step0-state-inventory.json")
        check(
            state["worker_step"] == 1 and state["basis_owners"] == (113 if record["method"] == "deferred" else 0),
            "basis inventory mismatch",
        )
    for row in rows:
        score = json_at(files, prefix + f"worker-score-{row['step']:04d}.json")
        check(score["stage"] == "score" and score["step"] == row["step"], "worker score clock mismatch")
        for metric in ("allocated", "reserved"):
            check(
                score[f"peak_{metric}_bytes"] == row[f"scoring_peak_{metric}_bytes"],
                "worker scoring allocator mismatch",
            )
    return {
        "case_id": record["case_id"],
        "evidence_round": label,
        "outcome": outcome,
        "rows": len(rows),
        "monitor_samples": len(monitors),
        "all": measure(rows[2:]) if outcome == "complete" else None,
        "ordinary": measure([r for r in rows[2:] if not r["update_refresh_any"]]) if outcome == "complete" else None,
        "refresh": measure([r for r in rows[2:] if r["update_refresh_any"]]) if outcome == "complete" else None,
    }, rows


def capacity_gate(records, control):
    candidate, baseline = frontiers(records, "deferred"), frontiers(records, control)
    axes = {}
    for axis, position in (("batch", 1), ("context", 0)):
        cell, base = candidate[axis], baseline[axis]
        ratio = cell[position] / base[position] if cell and base else None
        cards = {}
        for gpu in GPUS:
            matched = {
                r["method"]: r
                for r in records
                if r["stage"] == "boundary" and r["gpu"] == gpu and (r["context"], r["batch"]) == cell
            }
            cards[str(gpu)] = {
                "candidate_confirmed": matched.get("deferred", {}).get("outcome") == "complete",
                "control_typed_oom": matched.get(control, {}).get("outcome") == "cuda_oom",
            }
        axes[axis] = {
            "candidate_confirmed_cell": list(cell) if cell else None,
            "control_confirmed_cell": list(base) if base else None,
            "confirmed_grid_ratio": ratio,
            "passed": ratio is not None
            and ratio >= 1.25
            and all(r["candidate_confirmed"] and r["control_typed_oom"] for r in cards.values()),
            "same_cell_three_card_results": cards,
        }
    return {"control": control, "passed": any(value["passed"] for value in axes.values()), "axes": axes}


def audit_bundle(manifest, files, repo=None):
    prepared = {label: json_at(files, f"{label}/PREPARED.json") for label in ("screen", "repair")}
    hosts = {label: str(PurePosixPath(value["data_dir"]).parent) for label, value in prepared.items()}
    for label, value in prepared.items():
        check(value["source_sha256"] == SOURCE and value["native_sha256"] == NATIVE, "frozen runtime mismatch")
        freeze = json_at(files, f"{label}/FREEZE.json")
        check(
            freeze["parent_approved"] is True and freeze["prepared_sha256"] == sha(files[f"{label}/PREPARED.json"]),
            "freeze identity mismatch",
        )
        check(freeze["source_sha256"] == SOURCE, "freeze runtime mismatch")
        for path, identity in value["files"].items():
            bundled = (
                f"{label}/harness/{path}.txt"
                if "/" not in path and path.endswith(".py")
                else f"{label}/{path}" + (".txt" if path.startswith("data/") and path.endswith(".py") else "")
            )
            if bundled in files:
                check(
                    len(files[bundled]) == identity["bytes"] and sha(files[bundled]) == identity["sha256"],
                    "frozen bundled file mismatch: " + bundled,
                )
        data = json_at(files, f"{label}/data/DATA_MANIFEST.json")
        replay = json_at(files, f"{label}/data/INDEPENDENT_REPLAY.json")
        check(
            data["model_revision"] == value["model_revision"]
            and replay["status"] == "independent_full_train_replay_passed"
            and replay["stream_sha256"] == data["stream_sha256"]
            and replay["all_row_boundaries_and_hashes_exact"] is True,
            "input replay proof mismatch",
        )
    original = json_at(files, "screen/CASE_RESULTS.json")
    records = json_at(files, "repair/CASE_RESULTS.json")
    valid = {r["case_id"]: r for r in original if r["outcome"] in ("complete", "cuda_oom")}
    invalid = [r for r in original if r["outcome"] == "error"]
    check(len(original) == 51 and len(valid) == 49 and len(invalid) == 2, "original49/2 separation mismatch")
    check(
        {r["interruption"] for r in invalid} == {"case_timeout", "controller_stop_requested"},
        "original errors not distinct",
    )
    reused = [r for r in records if r.get("evidence_root") == hosts["screen"]]
    check(len(reused) == 49 and {r["case_id"] for r in reused} == set(valid), "49 reuse identities missing")
    for record in reused:
        check(
            {key: value for key, value in record.items() if key != "evidence_root"} == valid[record["case_id"]],
            "reused record was changed",
        )
    check(all(r.get("evidence_root", hosts["repair"]) in hosts.values() for r in records), "unknown evidence root")
    proofs = prepared["repair"]["proofs"]
    for name in ("CASE_RESULTS.json", "PARTIAL_ANALYSIS.json"):
        identity = proofs[hosts["screen"] + "/" + name]
        check(
            (identity["sha256"] if isinstance(identity, dict) else identity) == sha(files["screen/" + name]),
            "original proof identity mismatch",
        )
    audit_schedule(records)
    data = {label: json_at(files, f"{label}/data/DATA_MANIFEST.json") for label in prepared}
    orders, paired, reports = {}, {}, []
    for record in records:
        label = "screen" if record.get("evidence_root") == hosts["screen"] else "repair"
        report, rows = audit_case(files, label, record, prepared[label], data[label], orders)
        reports.append(report)
        key = (record["context"], record["batch"])
        identities = [(row["step"], row["batch_sha256"], row["target_tokens"]) for row in rows]
        if key in paired:
            count = min(len(identities), len(paired[key]))
            check(identities[:count] == paired[key][:count], "paired batch identities differ")
        if key not in paired or len(identities) > len(paired[key]):
            paired[key] = identities
    excluded = [
        audit_case(files, "screen", record, prepared["screen"], data["screen"], orders, True)[0] for record in invalid
    ]
    saved = json_at(files, "repair/ANALYSIS.json")
    primary, stronger = capacity_gate(records, "adamw"), capacity_gate(records, "adam8")
    for name, gate in (("primary_adamw_capacity_gate", primary), ("stronger_8bit_capacity_gate", stronger)):
        check(saved[name]["passed"] == gate["passed"], "saved capacity gate differs")
        for axis in ("batch", "context"):
            for key in (
                "passed",
                "confirmed_grid_ratio",
                "candidate_confirmed_cell",
                "control_confirmed_cell",
                "same_cell_three_card_results",
            ):
                check(saved[name]["axes"][axis][key] == gate["axes"][axis][key], "saved axis gate differs")
    runtime = None
    if repo is not None:
        repo = Path(repo)
        paths = sorted((repo / "areno").rglob("*.py"))
        canonical = hashlib.sha256()
        for path in paths:
            canonical.update(path.relative_to(repo).as_posix().encode())
            canonical.update(path.read_bytes())
        runtime = canonical.hexdigest()
        check(len(paths) == 221 and runtime == SOURCE, "current repository runtime differs from measured337")
    boundaries = []
    lookup = {r["case_id"]: r for r in reports}
    for context, batch in dict.fromkeys((r["context"], r["batch"]) for r in records if r["stage"] == "boundary"):
        for gpu in GPUS:
            methods = {
                r["method"]: lookup[r["case_id"]]
                for r in records
                if r["stage"] == "boundary" and r["gpu"] == gpu and (r["context"], r["batch"]) == (context, batch)
            }
            ours, eight = methods["deferred"]["all"], methods["adam8"]["all"]
            boundaries.append(
                {
                    "context": context,
                    "batch": batch,
                    "gpu": gpu,
                    "methods": methods,
                    "vs8bit_memory_reduction_percent": (
                        1 - ours["allocated_peak_bytes"] / eight["allocated_peak_bytes"]
                    )
                    * 100,
                    "vs8bit_wall_overhead_percent": (
                        ours["complete_wall_mean_seconds"] / eight["complete_wall_mean_seconds"] - 1
                    )
                    * 100,
                }
            )
    row_count, monitor_count = sum(r["rows"] for r in reports), sum(r["monitor_samples"] for r in reports)
    review = json_at(files, "repair/FINAL_REVIEW.json")
    check(
        row_count == review["valid_completed_step_rows_replayed"] == 4128
        and monitor_count == review["monitor_samples"] == 26448,
        "final raw audit counts differ",
    )
    check(sum(r["rows"] for r in excluded) == 370, "excluded partial update count differs")
    return {
        "status": "passed",
        "valid_cases": len(records),
        "complete_cases": sum(r["outcome"] == "complete" for r in records),
        "typed_cuda_oom_cases": sum(r["outcome"] == "cuda_oom" for r in records),
        "reused_cases": len(reused),
        "step_rows": row_count,
        "monitor_samples": monitor_count,
        "excluded_interrupted_cases": excluded,
        "primary_adamw_capacity_gate": primary,
        "stronger_8bit_capacity_gate": stronger,
        "boundary_comparisons": boundaries,
        "runtime_sha256": runtime or SOURCE,
        "current_repo_checked": repo is not None,
        "archive_parts": len(manifest["parts"]),
        "evidence_files": len(files),
        "token_payload_replayed_offline": False,
        "limitation": "Token payload omitted; token values cannot be regenerated from batch digests offline. Preserved original replay proofs attest the on-host full replay. Finite grid; no quality or exact global-maximum claim.",
    }


def verify(directory=None, repo=None):
    manifest, files = load_bundle(Path(directory) if directory else Path(__file__).resolve().parent)
    return audit_bundle(manifest, files, repo)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory, args.repo)
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "valid_cases",
                    "complete_cases",
                    "typed_cuda_oom_cases",
                    "step_rows",
                    "monitor_samples",
                    "current_repo_checked",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
