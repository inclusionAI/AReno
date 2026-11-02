"""Pack immutable text evidence into deterministic, bounded archive parts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

CHUNK_BYTES = 512 * 1024
MAX_ARCHIVE_BYTES = 900_000
TEXT = {".json", ".jsonl", ".txt", ".log"}
TOP_TEXT = TEXT | {".md"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_tar(files, *, compressed=False):
    stream = io.BytesIO()
    options = {"preset": 6} if compressed else {}
    with tarfile.open(
        fileobj=stream, mode="w:xz" if compressed else "w", format=tarfile.PAX_FORMAT, **options
    ) as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def logical_tar(files):
    inventory = {name: {"bytes": len(data), "sha256": digest(data)} for name, data in sorted(files.items())}
    return make_tar(
        {**files, "INDEX.json": (json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n").encode()}
    )


def collect(root, label):
    """Allowlisted text only; never traverse model/repo/cache/raw-data trees."""
    units = []
    top = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in TOP_TEXT:
            top[f"{label}/{path.name}"] = path.read_bytes()
        elif path.is_file() and path.suffix == ".py":
            top[f"{label}/harness/{path.name}.txt"] = path.read_bytes()
    for directory in ("postrun_audit", "independent_review"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in TOP_TEXT:
                top[f"{label}/{path.relative_to(root).as_posix()}"] = path.read_bytes()
    data_names = {
        "DATA_MANIFEST.json",
        "DATA_READY.json",
        "INDEPENDENT_REPLAY.json",
        "INPUT_PREFIXES.json",
        "LICENSE-ATTRIBUTION.md",
        "PLAN.md",
        "CHANGES.md",
        "prepare_data.py",
        "verify_data.py",
    }
    for name in sorted(data_names):
        path = root / "data" / name
        if path.is_file():
            suffix = ".txt" if path.suffix == ".py" else ""
            top[f"{label}/data/{name}{suffix}"] = path.read_bytes()
    units.append((f"{label}/protocols", top))
    for case in sorted((root / "outputs").iterdir()):
        if not case.is_dir():
            continue
        files = {}
        for path in sorted(case.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix in TEXT:
                data = path.read_bytes()
                data.decode("utf-8")
                files[f"{label}/{path.relative_to(root).as_posix()}"] = data
        if files:
            units.append((f"{label}/cases/{case.name}", files))
    return units


def pack(screen_root, repair_root, output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("part-*.tar.xz")) or (output / "manifest.json").exists():
        raise FileExistsError("output already contains a package")
    chunks, units = [], []
    for index, (name, files) in enumerate(collect(screen_root, "screen") + collect(repair_root, "repair")):
        raw = logical_tar(files)
        names = []
        for offset in range(0, len(raw), CHUNK_BYTES):
            chunk_name = f"chunks/u{index:03d}-{offset // CHUNK_BYTES:04d}.bin"
            chunks.append((chunk_name, raw[offset : offset + CHUNK_BYTES]))
            names.append(chunk_name)
        units.append({"name": name, "bytes": len(raw), "sha256": digest(raw), "files": len(files), "chunks": names})
    parts, chunk_index = [], {}

    def emit(group):
        raw = make_tar(dict(group), compressed=True)
        if len(raw) >= MAX_ARCHIVE_BYTES:
            if len(group) == 1:
                raise ValueError("single chunk exceeds archive budget")
            middle = len(group) // 2
            emit(group[:middle])
            emit(group[middle:])
            return
        name = f"part-{len(parts):03d}.tar.xz"
        (output / name).write_bytes(raw)
        parts.append({"archive": name, "bytes": len(raw), "sha256": digest(raw), "members": [key for key, _ in group]})
        for key, data in group:
            chunk_index[key] = {"archive": name, "bytes": len(data), "sha256": digest(data)}

    for offset in range(0, len(chunks), 8):
        emit(chunks[offset : offset + 8])
    manifest = {
        "schema": "areno-capacity-evidence-v1",
        "chunk_bytes": CHUNK_BYTES,
        "max_archive_bytes": MAX_ARCHIVE_BYTES,
        "parts": parts,
        "chunks": chunk_index,
        "units": units,
        "rounds": {"screen": {"prepared": "screen/PREPARED.json"}, "repair": {"prepared": "repair/PREPARED.json"}},
        "exclusions": [
            "token-stream.u32",
            "raw training text",
            "weights",
            "native binaries",
            "event binaries",
            "caches",
            "credentials",
        ],
        "offline_limit": "Token payload omitted: input token values cannot be regenerated from digests offline. Original full replay proofs are preserved.",
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if len(data) >= 1_000_000:
        raise ValueError("manifest exceeds repository file limit")
    (output / "manifest.json").write_bytes(data)
    return {
        "parts": len(parts),
        "units": len(units),
        "files": sum(unit["files"] for unit in units),
        "compressed_bytes": sum(part["bytes"] for part in parts),
        "largest_part": max(part["bytes"] for part in parts),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.screen_root, args.repair_root, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
