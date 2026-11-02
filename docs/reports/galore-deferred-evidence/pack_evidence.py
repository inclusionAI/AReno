"""Package immutable tensor-free study records with portable chunk verification."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

CHUNK = 512 * 1024
SUFFIXES = {".json", ".jsonl", ".py", ".md", ".txt"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    studies = {
        "performance": "runs/deferred_scale_confirm_20261030",
        "quality1024": "runs/deferred_scale_quality_20261030",
        "quality4096": "runs/experiments/deferred_17b_long4096",
        "context1024": "runs/experiments/deferred_17b_ctx1024",
        "resume": "runs/experiments/deferred_17b_resume",
    }
    chosen: dict[str, Path] = {}
    for label, relative in studies.items():
        root = args.project_root / relative
        paths = list(root.iterdir()) + list((root / "outputs").rglob("*"))
        for path in paths:
            if path.is_file() and not path.is_symlink() and path.suffix in SUFFIXES:
                chosen[f"{label}/{path.relative_to(root).as_posix()}"] = path
    runtime = args.project_root / studies["performance"] / "repo"
    for path in sorted((runtime / "areno").rglob("*.py")):
        chosen[f"measured-runtime/{path.relative_to(runtime).as_posix()}"] = path
    manifest = {
        "schema": "galore-strict-deferred-evidence-v1",
        "scope": "historical unchanged studies; final runtime identity verified separately",
        "tensor_payloads": "excluded; checkpoint bytes retained externally",
        "source_sha256": "80a70ff0f855fd5ec742ee985a507bc8adcbfd0ee3dafc23cc51da9708981c79",
        "files": [],
        "parts": [],
        "chunk_bytes": CHUNK,
        "max_archive_bytes": 900000,
    }
    payloads = []
    for number, (name, path) in enumerate(sorted(chosen.items())):
        raw = path.read_bytes()
        entry = {"path": name, "bytes": len(raw), "sha256": sha(raw), "chunks": []}
        for offset in range(0, max(1, len(raw)), CHUNK):
            data = raw[offset : offset + CHUNK]
            member = name + f".chunk{offset // CHUNK:04d}"
            index = len(manifest["parts"])
            manifest["parts"].append({"member": member, "raw_bytes": len(data), "raw_sha256": sha(data)})
            payloads.append((index, member, data))
            entry["chunks"].append(index)
        manifest["files"].append(entry)

    archive_count = 0

    def pack_group(group):
        nonlocal archive_count
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:xz", preset=6) as archive:
            for _, member, data in group:
                info = tarfile.TarInfo(member)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                archive.addfile(info, io.BytesIO(data))
        packed = buffer.getvalue()
        if len(packed) > manifest["max_archive_bytes"]:
            if len(group) == 1:
                raise ValueError("Oversized singleton chunk")
            mid = len(group) // 2
            pack_group(group[:mid])
            pack_group(group[mid:])
            return
        piece = f"part-{archive_count:03d}.tar.xz"
        archive_count += 1
        (args.output / piece).write_bytes(packed)
        for index, _, _ in group:
            manifest["parts"][index].update(archive=piece, bytes=len(packed), sha256=sha(packed))

    group, size = [], 0
    for payload in payloads:
        if group and size + len(payload[2]) > 8 * 1024 * 1024:
            pack_group(group)
            group, size = [], 0
        group.append(payload)
        size += len(payload[2])
    if group:
        pack_group(group)
    manifest["archive_count"] = archive_count
    manifest["file_count"] = len(manifest["files"])
    manifest["raw_bytes"] = sum(entry["bytes"] for entry in manifest["files"])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"files": manifest["file_count"], "parts": len(manifest["parts"]), "raw_bytes": manifest["raw_bytes"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
