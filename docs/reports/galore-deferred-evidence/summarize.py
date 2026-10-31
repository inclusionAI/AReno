"""Recompute headline tables and paired training guards from packaged raw records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

from verify import verify


def summarize(directory: Path, repo: Path | None = None) -> dict:
    verified = verify(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    entries = {entry["path"]: entry for entry in manifest["files"]}
    cache = {}

    def read(name):
        entry = entries[name]
        pieces = []
        for index in entry["chunks"]:
            part = manifest["parts"][index]
            if part["archive"] not in cache:
                with tarfile.open(directory / part["archive"], "r:xz") as archive:
                    cache[part["archive"]] = {
                        member.name: archive.extractfile(member).read() for member in archive.getmembers()
                    }
            pieces.append(cache[part["archive"]][part["member"]])
        return b"".join(pieces)

    def obj(name):
        return json.loads(read(name))

    def rows(name, count):
        result = [json.loads(line) for line in read(name).splitlines()]
        assert len(result) == count, name
        for step, row in enumerate(result):
            assert row["step"] == step, name
            assert row["update_optimizer_steps"] == 1 and row["update_worker_step"] == step + 1, name
            assert row["logp_abs_diff_mean"] == 0 and math.isfinite(row["loss"]), name
            assert row["sft_total_target_tokens"] == row["target_tokens"], name
            assert row["sft_grad_scale"] == 1, name
            if "/deferred/" in name:
                assert row["deferred_owner_count"] == 226, name
                assert row["deferred_owners_once"] == 1, name
                assert row["deferred_full_leaf_bytes_after_backward"] == 0, name
                assert row["deferred_cache_empty_after_step"] == 1, name
                assert row["deferred_barrier_versions_unchanged"] == 1, name
        return result

    def paired(group):
        for index in range(len(group[0])):
            for key in ("step", "sample_indices", "batch_sha256", "target_tokens"):
                assert len({json.dumps(method[index][key]) for method in group}) == 1, key

    quality, quality_rows = {}, {}
    for budget in (1024, 4096):
        label = f"quality{budget}"
        losses = {method: [] for method in ("adamw", "adam8", "deferred")}
        for seed in range(3):
            group = []
            for method in losses:
                prefix = f"{label}/outputs/seed{seed}/{method}"
                raw = rows(prefix + "/steps.jsonl", budget)
                group.append(raw)
                quality_rows[budget, seed, method] = raw
                val = obj(prefix + "/validation-live.json")
                assert val["target_tokens"] == 52181 and val["examples"] == 708
                assert val["loss"] == val["negative_logprob_sum"] / val["target_tokens"]
                losses[method].append(val["loss"])
            paired(group)
        means = {method: statistics.mean(values) for method, values in losses.items()}
        quality[str(budget)] = {
            "per_seed_nll": losses,
            "means": means,
            "relative_to_8bit_percent": (means["deferred"] / means["adam8"] - 1) * 100,
            "relative_to_full_adamw_percent": (means["deferred"] / means["adamw"] - 1) * 100,
            "all_seeds_within_full_adamw_plus2percent": all(
                c <= a * 1.02 for c, a in zip(losses["deferred"], losses["adamw"], strict=True)
            ),
            "all_seeds_below_8bit": all(c < a for c, a in zip(losses["deferred"], losses["adam8"], strict=True)),
            "validation_replay_limit": "loss totals/counts verified; original scalar streams unavailable",
        }
    for seed in range(3):
        for method in ("adamw", "adam8", "deferred"):
            paired([quality_rows[1024, seed, method], quality_rows[4096, seed, method][:1024]])

    timing = {}
    for gpu in range(3):
        group, records = [], {}
        for method in ("adam8", "original", "deferred"):
            raw = rows(f"performance/outputs/gpu{gpu}/{method}/steps.jsonl", 402)
            group.append(raw)
            measured = raw[2:]
            assert all(row["warmup"] == (index < 2) for index, row in enumerate(raw))
            refreshes = [row["step"] for row in measured if row["update_refresh_any"]]
            assert refreshes == ([] if method == "adam8" else [200, 400])
            records[method] = {
                "full_wall_ms": statistics.mean(r["score_plus_train_seconds"] for r in measured) * 1000,
                "allocated_gib": max(r["update_peak_allocated_bytes"] for r in measured) / 2**30,
                "reserved_gib": max(r["update_peak_reserved_bytes"] for r in measured) / 2**30,
                "refresh_indices": refreshes,
            }
        paired(group)
        a, c = records["adam8"], records["deferred"]
        change = (c["full_wall_ms"] / a["full_wall_ms"] - 1) * 100
        reduction = (1 - c["allocated_gib"] / a["allocated_gib"]) * 100
        timing[str(gpu)] = {
            "methods": records,
            "wall_change_percent": change,
            "allocated_reduction_percent": reduction,
            "original_primary_passed": reduction >= 10 and change <= 3,
            "original_tradeoff_passed": reduction >= 20 and change <= 10,
        }
    source = hashlib.sha256()
    for name in sorted(n for n in entries if n.startswith("measured-runtime/")):
        source.update(name.removeprefix("measured-runtime/").encode())
        source.update(read(name))
    assert source.hexdigest() == manifest["source_sha256"]
    final_matches_measured = None
    if repo is not None:
        final = hashlib.sha256()
        for path in sorted((repo / "areno").rglob("*.py")):
            final.update(path.relative_to(repo).as_posix().encode())
            final.update(path.read_bytes())
        final_matches_measured = final.hexdigest() == source.hexdigest()
        if not final_matches_measured:
            bridge = json.loads((directory / "final-checks/FINAL_SOURCE_BRIDGE.json").read_text())
            assert bridge["historical_runtime_sha256"] == source.hexdigest()
            assert bridge["final_runtime_sha256"] == final.hexdigest()
            actual_changes = {}
            for name in sorted(n for n in entries if n.startswith("measured-runtime/")):
                relative = name.removeprefix("measured-runtime/")
                old, new = read(name), (repo / relative).read_bytes()
                if old != new:
                    actual_changes[relative] = {
                        "historical_sha256": hashlib.sha256(old).hexdigest(),
                        "final_sha256": hashlib.sha256(new).hexdigest(),
                    }
            assert actual_changes == bridge["changed_runtime_files"]
    return {
        "status": "passed",
        "verified_evidence": verified,
        "source_sha256": source.hexdigest(),
        "final_runtime_checked": repo is not None,
        "final_runtime_byte_identical_to_historical": final_matches_measured,
        "quality": quality,
        "performance": timing,
        "quality_raw_updates": 46080,
        "performance_raw_updates": 3618,
        "scope": "raw training guards and paired inputs; validation arithmetic, not missing scalar replay",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.repo), indent=2), flush=True)
