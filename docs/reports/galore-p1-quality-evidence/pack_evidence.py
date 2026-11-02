"""Pack the six fixed P1 endpoints and interrupted-attempt context losslessly."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path


def capacity_packer():
    path = Path(__file__).resolve().parents[1] / "galore-capacity-evidence/pack_evidence.py"
    spec = importlib.util.spec_from_file_location("capacity_packer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing/nonregular evidence: {path}")
    raw = path.read_bytes()
    raw.decode("utf-8")
    return raw


def collect(root, old, runtime):
    prepared = json.loads(text(root / "harness/PREPARED.json"))
    status = json.loads(text(root / "control/STATUS.json"))
    original = json.loads(text(old / "control/stage3/CASE_RESULTS.json"))
    cases = []
    for method, seed, gpu in (("port640", 0, 1), ("deferred647", 1, 2), ("port640", 2, 3)):
        record = next(r for r in original if r["case_id"] == f"q4096_gpu{gpu}_{method}_s{seed}")
        if record["outcome"] != "complete" or record["return_code"] != 0:
            raise ValueError("original accepted case did not complete")
        cases.append(
            dict(
                method=method,
                seed=seed,
                gpu=gpu,
                reused=True,
                original_output=str(old / "control" / record["output_dir"]),
            )
        )
    for method, seed, gpu in (("deferred647", 0, 1), ("port640", 1, 2), ("deferred647", 2, 3)):
        q = status["queue"][str(gpu)]
        if q["status"] != "complete" or q["dispatches"] != 1 or q["resource_retries"] != 0:
            raise ValueError("unexpected queued case selection")
        cases.append(dict(method=method, seed=seed, gpu=gpu, reused=False, original_output=q["accepted_output"]))
    cases.sort(key=lambda row: (row["method"], row["seed"]))
    for case in cases:
        case["portable_output"] = f"cases/{case['method']}_s{case['seed']}"
    protocol = {"records/cases.json": (json.dumps(cases, indent=2) + "\n").encode()}
    for name in (
        "SESSION.json",
        "QUEUE_PROTOCOL.json",
        "PREREGISTERED_QUALITY_PROTOCOL.json",
        "REUSE_PROOF.json",
        "REUSE_INPUT_INVENTORY.json",
        "PARENT_HARNESS_APPROVAL.json",
        "PARENT_COMPLETE_QUALITY_REVIEW_20261102.json",
        "PARENT_CONTROL_CPU_FINAL.log",
        "HARNESS_QUEUE_CPU_READY.json",
        "BACKGROUND_QUEUE_LAUNCH.json",
    ):
        protocol["records/queue/" + name] = text(root / name)
    for base, label in ((root, "queue"), (old, "original")):
        for name in (
            "SESSION.json",
            "PREREGISTERED_QUALITY_PROTOCOL.json",
            "harness/PREPARED.json",
            "control/STATUS.json",
            "control/stage3/STATUS.json",
            "control/stage3/CASE_RESULTS.json",
        ):
            protocol[f"records/{label}/{name}"] = text(base / name)
        for path in sorted((base / "harness").iterdir()):
            if path.suffix != ".py" and path.name != "CHANGES.md":
                continue
            protocol[f"sources/{label}/harness/{path.name}.txt"] = text(path)
        for path in sorted((base / "control").glob("*.py")):
            protocol[f"sources/{label}/control/{path.name}.txt"] = text(path)
        for path in sorted((base / "control/stage3/control").rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
                protocol[f"records/{label}/control/{path.relative_to(base / 'control/stage3/control').as_posix()}"] = (
                    text(path)
                )
    for name in ("ANALYSIS.json", "FROZEN_SIX_CASE_QUALITY_ANALYSIS.json", "WATCHDOG_EXIT.json", "eval.stdout.log"):
        protocol["records/queue/" + name] = text(root / "control/stage3" / name)
    for key in ("quality_helper", "capture_helper", "capacity_helper"):
        protocol[f"sources/{key}.py.txt"] = text(Path(prepared[key]))
    for name in ("PORT_MANIFEST.json", "PORT_PARENT_REVIEW.json"):
        path = runtime.parent / name
        if path.is_file():
            protocol["records/" + name] = text(path)
    units = [("protocol", protocol)]
    snapshot = {}
    for path in sorted((runtime / "areno").rglob("*.py")):
        snapshot["sources/runtime/" + path.relative_to(runtime).as_posix() + ".txt"] = text(path)
    snapshot["sources/runtime/LICENSE.txt"] = text(runtime / "LICENSE")
    units.append(("runtime-python-snapshot", snapshot))
    fixed = {
        "cleanup.json",
        "metadata.json",
        "progress.json",
        "steps.jsonl",
        "summary.json",
        "validation-capture-metadata.json",
        "validation-response-logprobs.jsonl",
        "worker-model-inventory.json",
        "worker-native-after-first-update.json",
        "worker-parameter-storage-aliases.json",
        "worker-precision-before-first-update.json",
        "worker-state-policy.json",
    }
    for case in cases:
        directory = Path(case["original_output"])
        files = {}
        for path in sorted(directory.iterdir()):
            if path.name not in fixed and not re.fullmatch(r"worker-(?:score|train|eval-score)-\d{4}\.json", path.name):
                raise ValueError(f"unexpected completed output: {path.name}")
            files[case["portable_output"] + "/" + path.name] = text(path)
        if not fixed.issubset({p.name for p in directory.iterdir()}):
            raise ValueError("missing completed case records")
        units.append((case["portable_output"], files))
    # Partial attempts cannot become endpoints. Retain their exact update rows,
    # metadata/cursor and all monitor/failure context; redundant worker mirrors
    # remain at the original experiment root and are explicitly excluded.
    for method, seed, gpu in (("deferred647", 0, 1), ("port640", 1, 2), ("deferred647", 2, 3)):
        directory = old / f"control/stage3/outputs/gpu{gpu}/{method}_s{seed}"
        files = {}
        for name in (
            "steps.jsonl",
            "metadata.json",
            "progress.json",
            "cleanup.json",
            "failure.json",
            "worker-failure.json",
        ):
            if (directory / name).is_file():
                files[f"interrupted/{method}_s{seed}/{name}"] = text(directory / name)
        units.append((f"interrupted/{method}_s{seed}", files))
    return units


def pack(root, old, runtime, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or any(output.glob("part-*.tar.xz")):
        raise FileExistsError("preserve existing package")
    helper = capacity_packer()
    chunks, units = [], []
    for index, (name, files) in enumerate(collect(root, old, runtime)):
        raw = helper.logical_tar(files)
        keys = []
        for offset in range(0, len(raw), helper.CHUNK_BYTES):
            key = f"chunks/u{index:03d}-{offset // helper.CHUNK_BYTES:04d}.bin"
            chunks.append((key, raw[offset : offset + helper.CHUNK_BYTES]))
            keys.append(key)
        units.append(dict(name=name, bytes=len(raw), sha256=helper.digest(raw), files=len(files), chunks=keys))
    parts, inventory = [], {}

    def emit(group):
        raw = helper.make_tar(dict(group), compressed=True)
        if len(raw) >= helper.MAX_ARCHIVE_BYTES:
            if len(group) == 1:
                raise ValueError("single chunk exceeds archive limit")
            emit(group[: len(group) // 2])
            emit(group[len(group) // 2 :])
            return
        name = f"part-{len(parts):03d}.tar.xz"
        (output / name).write_bytes(raw)
        parts.append(dict(archive=name, bytes=len(raw), sha256=helper.digest(raw), members=[k for k, _ in group]))
        for key, value in group:
            inventory[key] = dict(archive=name, bytes=len(value), sha256=helper.digest(value))

    for offset in range(0, len(chunks), 8):
        emit(chunks[offset : offset + 8])
    manifest = dict(
        schema="areno-p1-controlled-quality-v1",
        chunk_bytes=helper.CHUNK_BYTES,
        max_archive_bytes=helper.MAX_ARCHIVE_BYTES,
        parts=parts,
        chunks=inventory,
        units=units,
        exclusions=[
            "weights",
            "checkpoints",
            "native binaries",
            "dataset/token payloads",
            "credentials",
            "redundant worker mirrors from interrupted attempts",
        ],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return dict(
        parts=len(parts),
        files=sum(u["files"] for u in units),
        compressed_bytes=sum(p["bytes"] for p in parts),
        largest_part=max(p["bytes"] for p in parts),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.root, args.old_root, args.runtime_repo, args.output), indent=2), flush=True)
