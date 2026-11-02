"""Deterministically pack the allowlisted, tensor-free final-quality records."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def capacity_packer():
    path = Path(__file__).resolve().parents[1] / "galore-capacity-evidence/pack_evidence.py"
    spec = importlib.util.spec_from_file_location("capacity_packer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(root):
    protocol = {}
    for name in (
        "PLAN.md",
        "CHANGES.md",
        "SESSION.json",
        "PREPARED_quality.json",
        "FREEZE_quality.json",
        "QUALITY_DECISION.json",
        "QUALIFICATION_quality.json",
        "stage3/PLAN.md",
        "stage3/CHANGES.md",
        "stage3/READY.json",
        "stage3/STATUS.json",
        "stage3/CASE_RESULTS.json",
        "stage3/CASE_EVENTS.jsonl",
        "stage3/ANALYSIS.json",
        "stage3/PARENT_RESULT_REVIEW.json",
        "stage3/parent-cpu.log",
        "stage3/quality-gate-tests.log",
        "stage3/quality-gate-lint.log",
        "stage3/controller.py",
        "stage3/analyze.py",
        "stage3/quality_capture.py",
        "stage3/test_capture.py",
        "stage3/test_quality_gate.py",
        "stage1/analyze.py",
        "contract.py",
        "guards.py",
        "launch_registered.py",
        "prepare_phase.py",
        "watchdog.py",
        "repo/scripts/galore_deferred/run.py",
        "repo/scripts/galore_deferred/model-manifest.json",
        "repo/scripts/galore_sft/common.py",
        "repo/scripts/galore_sft/data-manifest.json",
    ):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing/nonregular protocol: {name}")
        protocol[name + (".txt" if path.suffix == ".py" else "")] = path.read_bytes()
    units = [("protocol", protocol)]
    names = {
        "metadata.json",
        "encoding.json",
        "steps.jsonl",
        "summary.json",
        "validation-response-logprobs.jsonl",
        "validation-capture-metadata.json",
    }
    for method in ("adamw", "adam8", "deferred"):
        for seed in range(3):
            case = f"{method}_s{seed}"
            files = {}
            for name in sorted(names):
                path = root / "stage3/outputs" / case / name
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"missing/nonregular output: {path}")
                files[f"stage3/outputs/{case}/{name}"] = path.read_bytes()
            control = root / "stage3/control" / f"quality_{case}"
            for path in sorted(control.rglob("*")):
                if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".jsonl", ".log"):
                    files[path.relative_to(root).as_posix()] = path.read_bytes()
            units.append((case, files))
    return units


def pack(root, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or any(output.glob("part-*.tar.xz")):
        raise FileExistsError("evidence package already exists")
    helper = capacity_packer()
    chunks, units = [], []
    for index, (name, files) in enumerate(collect(root)):
        raw = helper.logical_tar(files)
        names = []
        for offset in range(0, len(raw), helper.CHUNK_BYTES):
            key = f"chunks/u{index:03d}-{offset // helper.CHUNK_BYTES:04d}.bin"
            chunks.append((key, raw[offset : offset + helper.CHUNK_BYTES]))
            names.append(key)
        units.append(
            {"name": name, "bytes": len(raw), "sha256": helper.digest(raw), "files": len(files), "chunks": names}
        )
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
        parts.append(
            {"archive": name, "bytes": len(raw), "sha256": helper.digest(raw), "members": [k for k, _ in group]}
        )
        for key, value in group:
            inventory[key] = {"archive": name, "bytes": len(value), "sha256": helper.digest(value)}

    for offset in range(0, len(chunks), 8):
        emit(chunks[offset : offset + 8])
    manifest = {
        "schema": "areno-final-quality-evidence-v1",
        "chunk_bytes": helper.CHUNK_BYTES,
        "max_archive_bytes": helper.MAX_ARCHIVE_BYTES,
        "parts": parts,
        "chunks": inventory,
        "units": units,
        "exclusions": ["weights", "checkpoints", "native binaries", "raw/tokenized dataset payloads", "credentials"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "parts": len(parts),
        "files": sum(u["files"] for u in units),
        "compressed_bytes": sum(p["bytes"] for p in parts),
        "largest_part": max(p["bytes"] for p in parts),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.root, args.output), indent=2))
