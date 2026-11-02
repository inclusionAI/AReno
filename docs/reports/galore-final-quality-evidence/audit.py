"""Standard-library audit for the unchanged published runner and public restore."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = None  # Assigned to a temporary data-only evidence tree by verify.py.
METHODS = ("adamw", "adam8", "deferred")
SOURCE = "337c7612282e8fc5254d890ac7659c88c2f38a8e8430d5b3cdc84f6dc96b28b8"
NATIVE = "f2d49f7b8ffd2fccbfce0f6eeb3108088be7834627b88c9ad71a0eb486f4d226"


def read(path):
    return json.loads(Path(path).read_text())


def rows_at(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_rows(rows, method, *, warmup, expected_updates):
    assert len(rows) == expected_updates
    for index, row in enumerate(rows):
        assert row["step"] == index and row["warmup"] == (index < warmup)
        assert row["update_worker_step"] == index + 1 and row["update_optimizer_steps"] == 1
        assert row["target_tokens"] > 0 and row["sft_total_target_tokens"] == row["target_tokens"]
        assert row["sft_grad_scale"] == 1 and row["sft_pack_count"] == row["sft_dp_shards_per_pack"] == 1
        assert row["logp_abs_diff_mean"] == 0 and math.isfinite(row["loss"])
        assert row["score_plus_train_seconds"] > 0 and math.isfinite(row["score_plus_train_seconds"])
        assert row["optimizer_model_owned_master_enabled"] == 0
        assert row["optimizer_deferred_projection_enabled"] == int(method == "deferred")
        assert row["optimizer_fused_scale_writeback_enabled"] == int(method == "deferred")
        assert bool(row["update_refresh_any"]) == (method == "deferred" and index % 200 == 0)
        assert len(row["sample_indices"]) == 4 and len(row["batch_sha256"]) == 64
        assert all(0 <= value < 13544 for value in row["sample_indices"])
        if method == "deferred":
            assert row["galore_master_bytes"] > 0
            for key, value in {
                "deferred_owner_count": 226,
                "deferred_owners_once": 1,
                "deferred_barrier_versions_unchanged": 1,
                "deferred_full_leaf_bytes_after_backward": 0,
                "deferred_cache_empty_after_step": 1,
            }.items():
                assert row[key] == value, key


def measured(rows):
    return {
        "updates": len(rows),
        "score_plus_train_seconds": sum(r["score_plus_train_seconds"] for r in rows),
        "update_cuda_time_s": sum(r["update_cuda_time_s"] for r in rows),
        "allocated_peak_bytes": max((r["update_peak_allocated_bytes"] for r in rows), default=0),
        "reserved_peak_bytes": max((r["update_peak_reserved_bytes"] for r in rows), default=0),
        "target_tokens": sum(r["target_tokens"] for r in rows),
    }


def audit_metadata(out, method, mode, seed):
    meta = read(out / "metadata.json")
    args = meta["arguments"]
    assert args["method"] == method and args["mode"] == mode and args["seed"] == seed
    assert args["context"] == 512 and args["batch"] == args["micro_batch"] == 4
    assert args["rank"] == 128 and args["gap"] == 200 and args["scale"] == 1 and args["retained"] is False
    assert args["fused_scale_writeback"] == (method == "deferred")
    assert meta["source_sha256"] == SOURCE and meta["native_extension_sha256"] == NATIVE
    assert meta["wrapper_sha256"]["run.py"] == digest(ROOT / "repo/scripts/galore_deferred/run.py")
    assert meta["wrapper_sha256"]["helper"] == digest(ROOT / "repo/scripts/galore_sft/common.py")
    expected = read(ROOT / "repo/scripts/galore_deferred/model-manifest.json")
    assert meta["model_revision"] == expected["revision"] and meta["model_assets"] == expected["files"]
    splits = ("train",) if mode == "performance" else ("train", "validation")
    data = read(ROOT / "repo/scripts/galore_sft/data-manifest.json")
    assert set(meta["data_sha256"]) == set(splits)
    assert all(meta["data_sha256"][split] == data["files"][split + ".jsonl"]["sha256"] for split in splits)
    assert meta["read_test"] is False and meta["save_checkpoint"] is False
    assert meta["deferred_flag"] == str(int(method == "deferred"))
    assert meta["fused_scale_writeback_flag"] == str(int(method == "deferred"))
    return meta


def audit_public_case(out, method, mode, seed, updates):
    meta = audit_metadata(out, method, mode, seed)
    rows = rows_at(out / "steps.jsonl")
    warmup = 2 if mode == "performance" else 0
    audit_rows(rows, method, warmup=warmup, expected_updates=updates)
    summary = read(out / "summary.json")
    assert summary["optimizer_updates"] == updates and summary["warmup_updates"] == warmup
    assert summary["read_test"] is False and summary["saved_checkpoint"] is False
    values = rows[warmup:]
    groups = {
        "all": values,
        "ordinary": [r for r in values if not r["update_refresh_any"]],
        "refresh": [r for r in values if r["update_refresh_any"]],
    }
    for group, rows_in_group in groups.items():
        for key, expected in measured(rows_in_group).items():
            assert math.isclose(summary[group][key], expected, rel_tol=1e-12, abs_tol=1e-8), (group, key)
    encoding = read(out / "encoding.json")
    assert encoding["train"]["rows"] == 13544
    if mode == "performance":
        assert set(encoding) == {"train"}
        assert [r["step"] for r in groups["refresh"]] == ([200, 400] if method == "deferred" else [])
        assert "validation" not in summary
    else:
        assert updates == 4096 and set(encoding) == {"train", "validation"}
        assert encoding["validation"]["rows"] == 708 and encoding["validation"]["targets"] == 52181
        assert summary["validation"]["examples"] == 708 and summary["validation"]["target_tokens"] == 52181
        assert math.isfinite(summary["validation"]["loss"])
        assert [r["step"] for r in groups["refresh"]] == (list(range(0, 4096, 200)) if method == "deferred" else [])
    identity = [(r["step"], r["sample_indices"], r["batch_sha256"], r["target_tokens"]) for r in rows]
    return {
        "method": method,
        "mode": mode,
        "seed": seed,
        "updates": updates,
        "gpu": meta["cuda_visible_devices"],
        "all": summary["all"],
        "ordinary": summary["ordinary"],
        "refresh": summary["refresh"],
        "full_wall_mean_seconds": sum(r["score_plus_train_seconds"] for r in values) / len(values),
        "validation": summary.get("validation"),
        "allocator_scope": "unchanged public runner native training-microbatch peaks; full score+train wall time",
    }, identity


QUALITY_RELATIVE_LIMIT = 0.02


def relative_comparison(candidate, control):
    assert len(candidate) == len(control) == 3
    assert all(math.isfinite(value) and value >= 0 for value in candidate)
    assert all(math.isfinite(value) and value > 0 for value in control)
    relative = [(value - baseline) / baseline for value, baseline in zip(candidate, control, strict=True)]
    candidate_mean, control_mean = statistics.mean(candidate), statistics.mean(control)
    return {
        "seeds": [0, 1, 2],
        "candidate_response_nll": candidate,
        "control_response_nll": control,
        "per_seed_relative_nll_increase": relative,
        "candidate_mean_response_nll": candidate_mean,
        "control_mean_response_nll": control_mean,
        "mean_relative_nll_increase": (candidate_mean - control_mean) / control_mean,
        "mean_of_paired_relative_nll_increases": statistics.mean(relative),
    }


def quality_comparisons(results):
    candidate = results["deferred"]["response_nll"]
    comparisons = {
        method: relative_comparison(candidate, results[method]["response_nll"]) for method in ("adamw", "adam8")
    }
    baseline = results["adamw"]["response_nll"]
    # Multiplication retains the exact <=2% boundary without a division-rounding
    # artifact such as (1.02 - 1) / 1 > .02. There is no added tolerance.
    paired_pass = [
        value <= control * (1 + QUALITY_RELATIVE_LIMIT) for value, control in zip(candidate, baseline, strict=True)
    ]
    mean_pass = statistics.mean(candidate) <= statistics.mean(baseline) * (1 + QUALITY_RELATIVE_LIMIT)
    gate = {
        "candidate": "deferred",
        "control": "adamw",
        "relative_nll_increase_limit": QUALITY_RELATIVE_LIMIT,
        "per_seed_at_most_limit": paired_pass,
        "all_paired_seeds_at_most_limit": all(paired_pass),
        "mean_at_most_limit": mean_pass,
        "mean_definition": "relative increase of the three-seed mean response NLLs",
        "passed": all(paired_pass) and mean_pass,
    }
    return comparisons, gate


def audit_capture(out, report):
    raw = (out / "validation-response-logprobs.jsonl").read_bytes()
    metadata = json.loads((out / "validation-capture-metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["sdk_global_step"] == 4095
    assert metadata["examples"] == metadata["expected_examples"] == 708
    assert metadata["target_tokens"] == metadata["expected_target_tokens"] == 52181
    assert (
        metadata["score_calls"] == 177 and metadata["method"] == report["method"] and metadata["seed"] == report["seed"]
    )
    assert metadata["stream_sha256"] == hashlib.sha256(raw).hexdigest() and metadata["stream_bytes"] == len(raw)
    for key, path in (
        ("adapter_sha256", ROOT / "stage3/quality_capture.py"),
        ("public_runner_sha256", ROOT / "repo/scripts/galore_deferred/run.py"),
        ("helper_sha256", ROOT / "repo/scripts/galore_sft/common.py"),
    ):
        assert metadata[key] == hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 708
    total, count, identities = 0.0, 0, []
    for index, row in enumerate(rows):
        assert row["example_index"] == index and row["score_call"] == index // 4
        values = row["response_logprobs"]
        assert len(values) == row["target_tokens"] == row["sequence_tokens"] - row["prompt_len"] > 0
        assert all(type(value) in (int, float) and math.isfinite(value) for value in values)
        assert len(row["sample_sha256"]) == 64
        total -= sum(values)
        count += len(values)
        identities.append((row["sample_sha256"], row["prompt_len"], row["sequence_tokens"], row["target_tokens"]))
    assert count == 52181
    assert metadata["original_evaluation_result"] == report["validation"]
    assert math.isclose(total / count, report["validation"]["loss"], rel_tol=1e-13, abs_tol=1e-13)
    return {
        "status": "passed",
        "examples": 708,
        "target_tokens": count,
        "independent_response_nll": total / count,
        "stream_sha256": metadata["stream_sha256"],
        "original_evaluate_once": True,
    }, identities


def analyze_quality(directory, records):
    assert len(records) == 9 and all(r["outcome"] == "complete" and r["return_code"] == 0 for r in records)
    assert {r["case_id"] for r in records} == {f"quality_{method}_s{seed}" for method in METHODS for seed in (0, 1, 2)}
    cases, pairing, validation_identity = {}, {}, None
    for seed in (0, 1, 2):
        for method in METHODS:
            record = next(r for r in records if r["case_id"] == f"quality_{method}_s{seed}")
            assert record["gpu"] == seed + 1
            report, identity = audit_public_case(ROOT / record["output_dir"], method, "quality", seed, 4096)
            assert seed not in pairing or identity == pairing[seed]
            pairing[seed] = identity
            capture, val_identity = audit_capture(ROOT / record["output_dir"], report)
            assert validation_identity is None or validation_identity == val_identity
            validation_identity = val_identity
            report["final_scalar_capture"] = capture
            cases[record["case_id"]] = report
    results = {}
    for method in METHODS:
        losses = [cases[f"quality_{method}_s{seed}"]["validation"]["loss"] for seed in (0, 1, 2)]
        results[method] = {
            "seeds": [0, 1, 2],
            "response_nll": losses,
            "mean_response_nll": statistics.mean(losses),
            "sample_std_response_nll": statistics.stdev(losses),
        }
    comparisons, gate = quality_comparisons(results)
    return {
        "status": "passed",
        "source_sha256": SOURCE,
        "cases": cases,
        "three_seed_results": results,
        "response_nll_comparisons": comparisons,
        "primary_quality_gate": gate,
        "paired_inputs_by_seed": True,
        "updates_per_job": 4096,
        "validation_examples_per_job": 708,
        "validation_targets_per_job": 52181,
        "interim_validation": False,
        "saved_checkpoint": False,
        "test_read": False,
        "selection_from_validation": False,
    }
