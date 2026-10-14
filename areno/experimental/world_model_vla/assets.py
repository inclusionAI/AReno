"""Manifest verification for large world-model and VLA snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AssetVerification:
    """Result of checking one snapshot against its manifest."""

    name: str
    verified_bytes: int
    expected_bytes: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors and self.verified_bytes == self.expected_bytes


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Summary of a newly written snapshot manifest."""

    path: Path
    files: int
    total_bytes: int
    includes_sha256: bool


def create_snapshot_manifest(
    snapshot: str | Path, *, include_hashes: bool = True, overwrite: bool = False
) -> SnapshotManifest:
    """Create a deterministic size and optional SHA-256 snapshot manifest."""

    root = Path(snapshot).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {root}")
    destination = root / "snapshot_manifest.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"snapshot manifest already exists: {destination}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and destination != path
        and not path.name.startswith("snapshot_manifest.tmp.")
        and ".git" not in path.relative_to(root).parts
    )
    if not files:
        raise ValueError(f"snapshot contains no files: {root}")
    entries: list[dict[str, str | int]] = []
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        entry: dict[str, str | int] = {
            "path": path.relative_to(root).as_posix(),
            "size": size,
        }
        if include_hashes:
            entry["sha256"] = file_sha256(path)
        entries.append(entry)
        total_bytes += size

    temporary = destination.with_suffix(f".tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps({"schema_version": 1, "files": entries}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return SnapshotManifest(destination, len(entries), total_bytes, include_hashes)


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(snapshot: str | Path, *, check_hashes: bool = False) -> AssetVerification:
    """Verify declared snapshot files by size and, when requested, hash."""

    root = Path(snapshot)
    manifest_path = root / "snapshot_manifest.json"
    if not manifest_path.is_file():
        return AssetVerification(root.name, 0, 0, (f"missing manifest: {manifest_path}",))

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return AssetVerification(root.name, 0, 0, (f"invalid manifest {manifest_path}: {error}",))
    if not isinstance(manifest, dict):
        return AssetVerification(root.name, 0, 0, (f"invalid manifest object: {manifest_path}",))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return AssetVerification(root.name, 0, 0, (f"invalid manifest files list: {manifest_path}",))
    if not entries:
        return AssetVerification(root.name, 0, 0, (f"empty manifest files list: {manifest_path}",))

    verified = 0
    expected_total = 0
    errors: list[str] = []
    seen: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict) or "path" not in entry or "size" not in entry:
            errors.append(f"invalid manifest entry: {entry!r}")
            continue
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe manifest path: {relative}")
            continue
        if relative in seen:
            errors.append(f"duplicate manifest path: {relative}")
            continue
        seen.add(relative)
        try:
            expected_size = int(entry["size"])
        except (TypeError, ValueError):
            errors.append(f"invalid size for {relative}: {entry['size']!r}")
            continue
        if expected_size < 0:
            errors.append(f"invalid size for {relative}: {expected_size}")
            continue
        expected_total += expected_size
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            errors.append(f"size mismatch: {path}: expected {expected_size}, got {actual_size}")
            continue
        expected_hash = entry.get("sha256")
        if check_hashes and expected_hash and file_sha256(path) != str(expected_hash).lower():
            errors.append(f"sha256 mismatch: {path}")
            continue
        verified += expected_size
    return AssetVerification(root.name, verified, expected_total, tuple(errors))


__all__ = [
    "AssetVerification",
    "SnapshotManifest",
    "create_snapshot_manifest",
    "file_sha256",
    "verify_snapshot",
]
