"""Frozen P1 scientific checks with a static, reviewed training schedule.

Functions come from the measured protocol/eval sources. Only audit_case's
historical absolute-path helper loader is replaced by the identical static
batch_indices below. No archived Python is executed; no numerical check or
tolerance is changed. Original bytes remain separately archived.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

METHODS = ("port640", "deferred647")
SEEDS = (0, 1, 2)
UPDATES = 4096


def check(condition, message):
    if not condition:
        raise ValueError(message)


def optimizer_options(method, seed=0):
    check(type(seed) is int and seed in SEEDS, "unregistered training/projector seed")
    check(method in METHODS, "unregistered method")
    options = dict(
        lr=1e-5,
        min_lr=1e-5,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        grad_clip_norm=1.0,
        save_training_state=False,
        resume_training_state=False,
        galore_deferred_projection=method == "deferred647",
        galore_fused_scale_writeback=method == "deferred647",
    )
    if method == "port640":
        options.update(galore_rank=128, galore_update_proj_gap=200, galore_scale=1.0)
    elif method == "deferred647":
        options.update(
            galore_rank=128,
            galore_update_proj_gap=200,
            galore_scale=1.0,
            galore_projector_backend="recycled",
            galore_projector_oversampling=16,
            galore_projector_iterations=1,
            galore_projector_seed=seed,
            galore_projector_final_qr=True,
            galore_compact_master=True,
            galore_tile_rows=16384,
            galore_release_gradients=True,
            galore_fused_writeback=True,
            galore_transport_moments=True,
            galore_exclude_vocab=False,
        )
    return options


def runtime_options():
    return dict(
        attn_backend="native",
        compile_model=False,
        activation_checkpointing=True,
        keep_rollout_state=True,
        profile_training=True,
    )


def audit_update(result, *, method, index, targets, sdk_step):
    check(method in METHODS and 0 <= index < UPDATES, "unregistered method/update")
    check(sdk_step == index and math.isfinite(result["loss"]), "public clock or finite loss failed")
    expected = {
        "update_optimizer_steps": 1,
        "update_worker_step": index + 1,
        "update_refresh_any": int(method in ("port640", "deferred647") and index % 200 == 0),
        "optimizer_deferred_projection_enabled": int(method == "deferred647"),
        "optimizer_fused_scale_writeback_enabled": int(method == "deferred647"),
        "optimizer_model_owned_master_enabled": 0,
        "sft_pack_count": 1,
        "sft_dp_shards_per_pack": 1,
        "sft_total_target_tokens": targets,
        "sft_grad_scale": 1,
        "logp_abs_diff_mean": 0,
    }
    if method == "deferred647":
        expected.update(
            deferred_owner_count=226,
            deferred_owners_once=1,
            deferred_barrier_versions_unchanged=1,
            deferred_full_leaf_bytes_after_backward=0,
            deferred_cache_empty_after_step=1,
        )
    for key, value in expected.items():
        check(key in result and result[key] == value, f"native guard failed: {key}")
    if method in ("port640", "deferred647"):
        check(result["galore_master_bytes"] > 0, "strict logical FP32 master authority missing")
    for key in ("grad_norm", "clipped_grad_norm"):
        check(key in result and math.isfinite(result[key]), f"nonfinite native {key}")
    check(
        result["grad_norm"] >= 0 and abs(result["clipped_grad_norm"] - min(1.0, result["grad_norm"])) < 1e-5,
        "native global clipping differs",
    )
    for kind in ("allocated", "reserved"):
        check(
            result[f"training_call_peak_{kind}_bytes"] >= result[f"update_peak_{kind}_bytes"] > 0,
            "owning-worker peak scope mismatch",
        )


def summarize(rows):
    check(bool(rows), "empty measurement group")
    wall = sum(r["score_plus_train_seconds"] for r in rows)
    return {
        "updates": len(rows),
        "target_tokens": sum(r["target_tokens"] for r in rows),
        "score_plus_train_seconds": wall,
        "full_wall_mean_seconds": wall / len(rows),
        **{
            f"{prefix}_peak_{kind}_bytes": max(r[f"{prefix}_peak_{kind}_bytes"] for r in rows)
            for prefix in ("score_plus_train", "scoring", "training_call", "update")
            for kind in ("allocated", "reserved")
        },
    }


def batch_indices(step, seed, size, batch, orders):
    result = []
    for cursor in range(step * batch, (step + 1) * batch):
        epoch, offset = divmod(cursor, size)
        if epoch not in orders:
            order = list(range(size))
            random.Random(seed + 1000003 * epoch).shuffle(order)
            orders[epoch] = order
        result.append(orders[epoch][offset])
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute_capture(directory, *, expected_examples=708, expected_targets=52181, expected_step=UPDATES - 1):
    directory = Path(directory)
    metadata = json.loads((directory / "validation-capture-metadata.json").read_text())
    path = directory / "validation-response-logprobs.jsonl"
    check(
        metadata["status"] == "complete" and metadata["sdk_global_step"] == expected_step,
        "incomplete/wrong-clock final capture",
    )
    check(
        metadata["stream_sha256"] == digest(path) and metadata["stream_bytes"] == path.stat().st_size,
        "raw scalar stream identity differs",
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    check(len(rows) == expected_examples, "raw evaluation example coverage differs")
    total, targets, identities = 0.0, 0, []
    for index, row in enumerate(rows):
        values = row["response_logprobs"]
        check(row["example_index"] == index and row["score_call"] == index // 4, "raw evaluation order/call differs")
        check(
            row["target_tokens"] == len(values) == row["sequence_tokens"] - row["prompt_len"] > 0,
            "raw target/mask coverage differs",
        )
        check(0 < row["prompt_len"] < row["sequence_tokens"] <= 512, "raw response bounds differ")
        check(
            all(type(value) in (float, int) and math.isfinite(value) for value in values),
            "raw nonfinite response scalar",
        )
        check(len(row["sample_sha256"]) == 64, "raw sample identity missing")
        total -= sum(values)
        targets += len(values)
        identities.append(row["sample_sha256"])
    check(targets == expected_targets, "raw target coverage differs")
    expected_calls = (expected_examples + 3) // 4
    check(
        metadata["examples"] == expected_examples
        and metadata["target_tokens"] == expected_targets
        and metadata["score_calls"] == expected_calls,
        "capture metadata coverage differs",
    )
    result = {"loss": total / targets, "target_tokens": targets, "examples": len(rows)}
    check(result == metadata["original_evaluation_result"], "independent response-NLL reconstruction differs")
    return result, identities


def audit_case(directory, method, seed, prepared):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    check(method in METHODS and seed in SEEDS, "unregistered method/seed")
    for key in (
        "commit",
        "runtime_sha256",
        "core_runtime_sha256",
        "native_sha256",
        "model_revision",
        "preregistered_protocol_sha256",
    ):
        check(metadata[key] == prepared[key], "measured identity differs: " + key)
    check(metadata["method"] == method and metadata["seed"] == seed, "measured method/seed differs")
    check(
        metadata["optimizer"] == json.loads(json.dumps(optimizer_options(method, seed)))
        and metadata["runtime"] == runtime_options(),
        "registered recipe differs",
    )
    check(
        not metadata["saved_checkpoint"]
        and metadata["read_test"] is False
        and metadata["optimizer_environment_flags"] == {},
        "forbidden scope",
    )
    check((metadata["controlled_port"] is not None) == (method == "port640"), "controlled factory route differs")
    inventory = json.loads((directory / "worker-model-inventory.json").read_text())
    check(
        inventory["model_parameter_owners"] == inventory["optimizer_parameter_owners"] == 226
        and inventory["unique_parameters"] == 1720574976,
        "model/optimizer owner coverage differs",
    )
    check(
        inventory["shape_order_dtype_sha256"] == "e4b08380197f3b86f2dab1916f0e947cbb204fc949366daaaa0b84f1b4168311",
        "shape/order/dtype identity differs",
    )
    check(
        inventory["dtype_elements"] == {"torch.bfloat16": 1720451072, "torch.float32": 123904},
        "model dtype policy differs",
    )
    check(
        all(
            inventory[key] is True
            for key in (
                "tied_vocabulary_parameter_identity",
                "tied_vocabulary_storage_identity",
                "optimizer_unique_owner_coverage",
                "optimizer_order_matches_named_owners",
            )
        ),
        "storage alias/unique-owner proof differs",
    )
    aliases = json.loads((directory / "worker-parameter-storage-aliases.json").read_text())
    check(
        aliases["unique_parameter_owners"] == 226 and aliases["unique_storages"] > 0,
        "actual parameter/storage owner inventory differs",
    )
    check(
        any(
            "embed_tokens.weight" in group and "lm_head.weight" in group for group in aliases["parameter_alias_groups"]
        ),
        "actual tied vocabulary alias names missing",
    )
    check(
        sorted(row["owner_index"] for group in aliases["storage_groups"] for row in group) == list(range(226)),
        "actual storage inventory owner coverage differs",
    )
    worker_native = json.loads((directory / "worker-native-after-first-update.json").read_text())
    check(
        worker_native["native_sha256"] == prepared["native_sha256"] and worker_native["worker_step"] == 1,
        "actual owning-worker native identity differs",
    )
    check(bool(worker_native["native_path"]) and worker_native["native_bytes"] > 0, "owning-worker native path missing")
    for precision in (
        metadata["main_precision_policy"],
        worker_native["precision"],
        json.loads((directory / "worker-precision-before-first-update.json").read_text()),
    ):
        check(
            all(
                key in precision
                for key in (
                    "float32_matmul_precision",
                    "cuda_matmul_allow_tf32",
                    "cuda_matmul_allow_bf16_reduced_precision_reduction",
                )
            ),
            "actual precision observations missing",
        )
    state = json.loads((directory / "worker-state-policy.json").read_text())
    check(
        state["method"] == method and state["model_parameter_owners"] == 226 and state["fp32_projected_state"] is True,
        "optimizer state policy differs",
    )
    if method == "port640":
        check(
            state["optimizer_class"].split(".")[-1] == "ControlledGaLore640"
            and state["projected_matrices"] == 112
            and state["logical_blocks"] == 196
            and state["vocabulary_fallback"] is True,
            "original640 geometry/fallback differs",
        )
        check(
            state["provenance"]["vendored_sha256"]
            == metadata["controlled_port"]["vendored_sha256"]
            == prepared["port_provenance"]["vendored_sha256"],
            "original640 mathematical source differs",
        )
    else:
        check(
            state["optimizer_class"].split(".")[-1] == "GaLoreAdamW"
            and state["projected_matrices"] == 113
            and state["vocabulary_fallback"] is False,
            "candidate geometry differs",
        )
    check(json.loads((directory / "cleanup.json").read_text())["status"] == "passed", "worker cleanup failed")
    rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    check(len(rows) == UPDATES, "incomplete4096-update case")
    identities = []
    scheduling = batch_indices
    orders = {}
    for index, row in enumerate(rows):
        check(row["step"] == index and row["seed"] == seed and not row["warmup"], "training row order/seed differs")
        check(
            len(row["sample_indices"]) == 4 and all(type(i) is int and 0 <= i < 13544 for i in row["sample_indices"]),
            "training sample schedule differs",
        )
        check(
            row["sample_indices"] == scheduling(index, seed, 13544, 4, orders),
            "original seeded training schedule differs",
        )
        check(len(row["batch_sha256"]) == 64, "training token/mask identity missing")
        audit_update(row, method=method, index=index, targets=row["target_tokens"], sdk_step=row["sdk_global_step"])
        score = json.loads((directory / f"worker-score-{index:04d}.json").read_text())
        native = json.loads((directory / f"worker-train-{index:04d}.json").read_text())
        check(score["step"] == native["step"] == index, "owning-worker clock differs")
        for key, value in native.items():
            if key not in ("step", "scope"):
                check(row[key] == value, "native training counter differs: " + key)
        for kind in ("allocated", "reserved"):
            check(row[f"scoring_peak_{kind}_bytes"] == score[f"peak_{kind}_bytes"], "native scoring counter differs")
            check(
                row[f"score_plus_train_peak_{kind}_bytes"]
                == max(row[f"scoring_peak_{kind}_bytes"], row[f"training_call_peak_{kind}_bytes"]),
                "full-call peak arithmetic differs",
            )
        identities.append([row["sample_indices"], row["batch_sha256"], row["target_tokens"]])
    refresh = [row for row in rows if row["update_refresh_any"]]
    check([row["step"] for row in refresh] == list(range(0, UPDATES, 200)), "actual production refresh clock differs")
    validation, validation_ids = recompute_capture(directory)
    capture_metadata = json.loads((directory / "validation-capture-metadata.json").read_text())
    check(
        capture_metadata["method"] == method
        and capture_metadata["seed"] == seed
        and capture_metadata["training_updates"] == UPDATES,
        "final capture job identity differs",
    )
    for key in ("runtime_sha256", "preregistered_protocol_sha256"):
        check(capture_metadata[key] == prepared[key], "final capture source differs")
    check(
        capture_metadata["source_commit"] == prepared["commit"]
        and capture_metadata["interim_evaluation_count"] == 0
        and capture_metadata["read_test"] is False,
        "final evaluation scope differs",
    )
    groups = {
        "all": summarize(rows),
        "ordinary": summarize([r for r in rows if not r["update_refresh_any"]]),
        "refresh": summarize(refresh),
    }
    summary = json.loads((directory / "summary.json").read_text())
    check(
        summary["status"] == "passed"
        and summary["optimizer_updates"] == UPDATES
        and summary["warmup_updates"] == 0
        and summary["method"] == method
        and summary["seed"] == seed,
        "saved completion differs",
    )
    check(
        summary["final_validation"] == validation and all(summary[key] == value for key, value in groups.items()),
        "saved summary arithmetic differs",
    )
    check(
        summary["read_validation"] is True and summary["read_test"] is False and summary["saved_checkpoint"] is False,
        "saved scope differs",
    )
    return {
        "gpu": metadata["admission"]["gpu"],
        "method": method,
        "seed": seed,
        "validation": validation,
        "training_identities": identities,
        "validation_identities": validation_ids,
        **groups,
    }


def audit_group(cases, prepared):
    check(set(cases) == {(method, seed) for method in METHODS for seed in SEEDS}, "all six preregistered jobs required")
    reports = {key: audit_case(Path(path), *key, prepared) for key, path in cases.items()}
    pairs = []
    for seed in SEEDS:
        port, candidate = reports[("port640", seed)], reports[("deferred647", seed)]
        check(port["gpu"] == candidate["gpu"], "paired seed methods used different physical cards")
        check(
            port["training_identities"] == candidate["training_identities"],
            "paired training indices/tokens/masks differ",
        )
        check(
            port["validation_identities"] == candidate["validation_identities"],
            "paired final validation tokens/masks/order differ",
        )
        pairs.append(
            {
                "seed": seed,
                "gpu": port["gpu"],
                "port640_nll": port["validation"]["loss"],
                "deferred647_nll": candidate["validation"]["loss"],
                "relative_nll_difference_percent": 100
                * (candidate["validation"]["loss"] / port["validation"]["loss"] - 1),
            }
        )
    check(len({row["gpu"] for row in pairs}) == 3, "three physical cards required")
    reference_validation = reports[("port640", 0)]["validation_identities"]
    check(
        all(report["validation_identities"] == reference_validation for report in reports.values()),
        "fixed development tokens/masks differ across seeds",
    )
    compact = [
        {key: value for key, value in report.items() if key not in ("training_identities", "validation_identities")}
        for report in reports.values()
    ]
    means = {
        method: sum(reports[(method, seed)]["validation"]["loss"] for seed in SEEDS) / len(SEEDS) for method in METHODS
    }
    return {
        "status": "passed",
        "source_commit": prepared["commit"],
        "runtime_sha256": prepared["runtime_sha256"],
        "reports": compact,
        "paired_seeds": pairs,
        "mean_response_token_nll": means,
        "relative_mean_nll_difference_percent": 100 * (means["deferred647"] / means["port640"] - 1),
        "historical_adamw_gate_retested": False,
        "read_test": False,
    }
