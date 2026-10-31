"""Package immutable small evidence; model/checkpoint tensors remain external."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

BLOCKED = {".pt", ".pth", ".safetensors", ".so", ".pyc", ".bin", ".lock"}
MAX_BYTES = 25 * 1024 * 1024
CONFIRM = Path("runs/frontier_confirm_20261029")
PERF = Path("runs/frontier_round_20261029")
SOURCE = Path("runs/experiments/frontier_precision_controls/repo")


def selected(root: Path) -> list[Path]:
    chosen: set[Path] = set()

    def add(path: Path) -> None:
        if path.is_file() and not path.is_symlink() and path.suffix not in BLOCKED:
            if path.stat().st_size > MAX_BYTES:
                raise ValueError(f"Unexpected oversized evidence file: {path.relative_to(root)}")
            chosen.add(path)

    def tree(path: Path) -> None:
        if not path.is_dir():
            raise FileNotFoundError(path)
        for f in sorted(path.rglob("*")):
            if "__pycache__" not in f.parts:
                add(f)

    for name in (
        "COMPLETED_AUDIT.json",
        "GRAM_DIAGNOSTIC.json",
        "analysis.json",
        "status.json",
        "jobs.json",
        "SELECTION.json",
        "READY.json",
        "ROOT_REVIEW.json",
        "PREPARED.json",
        "SUPERVISOR_CONTRACT.json",
        "LAUNCH.json",
        "events.jsonl",
        "MORNING_REPORT.md",
        "analyze.py",
        "audit_completed_results.py",
        "PLAN.md",
        "PREPARATION.json",
    ):
        required = root / CONFIRM / name
        if not required.is_file():
            raise FileNotFoundError(required)
        add(required)
    for sub in ("outputs", "harness", "acceptance"):
        tree(root / CONFIRM / sub)
    # Tensor-free checkpoint inventories/configs identify retained external weights.
    for file in (root / CONFIRM / "checkpoints").rglob("*"):
        if file.name in {
            "optimizer-retention.json",
            "training_state.json",
            "config.json",
            "model.safetensors.index.json",
            "configuration.json",
        }:
            add(file)
    for relative in ("performance-selected-s1", "performance-model-owned-s2"):
        directory = root / PERF / relative
        for file in directory.iterdir():
            if file.suffix in {".json", ".py", ".md"}:
                add(file)
        methods = ["galore"] if relative == "performance-selected-s1" else ["adamw", "adam8"]
        for method in methods:
            tree(directory / "outputs" / method)
    # Store exact experimental Python runtime, including unselected flag branch,
    # so its measured source digest can be reconstructed without claiming final-head identity.
    for file in (root / SOURCE / "areno").rglob("*.py"):
        add(file)
    for relative in (
        "environment.json",
        "model-source.json",
        "model-revisions.json",
        "model-sha256.txt",
        "requirements.lock",
        "runs/experiments/overnight_20261029_data/data/summary.json",
        "runs/experiments/overnight_20261029_data/data/LICENSE-ATTRIBUTION.md",
    ):
        add(root / relative)
    return sorted(chosen, key=lambda path: path.relative_to(root).as_posix())


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(root: Path, output: Path) -> dict:
    paths = selected(root)
    if len(paths) < 150:
        raise ValueError("Evidence selection unexpectedly incomplete")
    files = []
    total = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:xz") as archive:
        for path in paths:
            data = path.read_bytes()
            name = path.relative_to(root).as_posix()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            archive.addfile(info, io.BytesIO(data))
            files.append({"path": name, "bytes": len(data), "sha256": sha(data)})
            total += len(data)
        manifest = {
            "schema": "galore-frontier-evidence-v1",
            "files": files,
            "raw_bytes": total,
            "tensor_payloads": "excluded; retained externally",
            "scope": "measured experimental source; final-head acceptance separate",
        }
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size, info.mtime, info.mode = len(data), 0, 0o644
        archive.addfile(info, io.BytesIO(data))
    return manifest


def verify(path: Path) -> dict:
    with tarfile.open(path, "r:xz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        if any(
            not member.isfile() or PurePosixPath(member.name).is_absolute() or ".." in PurePosixPath(member.name).parts
            for member in members
        ):
            raise ValueError("Unsafe archive member")
        manifest = json.load(archive.extractfile("MANIFEST.json"))
        expected = {entry["path"] for entry in manifest["files"]} | {"MANIFEST.json"}
        if expected != set(names):
            raise ValueError("Archive/manifest membership mismatch")
        for entry in manifest["files"]:
            data = archive.extractfile(entry["path"]).read()
            if len(data) != entry["bytes"] or sha(data) != entry["sha256"]:
                raise ValueError(f"Evidence mismatch: {entry['path']}")
    return {
        "status": "passed",
        "verified_files": len(manifest["files"]),
        "raw_bytes": manifest["raw_bytes"],
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha(path.read_bytes()),
        "tensor_payloads": "excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        if args.root is None:
            parser.error("--root is required for packaging")
        build(args.root.resolve(), args.output)
    print(json.dumps(verify(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
