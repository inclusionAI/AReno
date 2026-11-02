"""Verify portable strict-deferred records without model assets or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import tarfile
from pathlib import Path, PurePosixPath


def verify(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    final_manifest = json.loads((directory / "FINAL_CHECKS_MANIFEST.json").read_text())
    final_files = final_manifest["files"]
    for name, entry in final_files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("Unsafe final-check filename")
        data = (directory / "final-checks" / name).read_bytes()
        if entry.get("encoding") == "xz":
            if len(data) != entry["stored_bytes"] or hashlib.sha256(data).hexdigest() != entry["stored_sha256"]:
                raise ValueError(f"Compressed final check identity mismatch: {name}")
            data = lzma.decompress(data)
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Final check identity mismatch: {name}")
    runtime_acceptance = json.loads((directory / "final-checks/RUNTIME_ACCEPTANCE.json").read_text())
    originals = {entry["original_name"]: entry for entry in final_files.values()}
    for name, entry in runtime_acceptance["proofs"].items():
        if originals[name]["bytes"] != entry["bytes"] or originals[name]["sha256"] != entry["sha256"]:
            raise ValueError(f"Runtime acceptance proof mismatch: {name}")
    for bundle in final_manifest.get("bundles", []):
        name = PurePosixPath(bundle["archive"])
        if name.is_absolute() or len(name.parts) != 1:
            raise ValueError("Unsafe supplementary archive name")
        path = directory / str(name)
        packed = path.read_bytes()
        if len(packed) != bundle["bytes"] or hashlib.sha256(packed).hexdigest() != bundle["sha256"]:
            raise ValueError(f"Supplementary archive identity mismatch: {name}")
        expected = {entry["path"]: entry for entry in bundle["files"]}
        if len(expected) != len(bundle["files"]):
            raise ValueError("Duplicate supplementary member")
        with tarfile.open(path, "r:xz") as archive:
            members = archive.getmembers()
            if len(members) != len(expected) or {m.name for m in members} != set(expected):
                raise ValueError("Supplementary archive membership mismatch")
            for member in members:
                relative = PurePosixPath(member.name)
                if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe supplementary member")
                raw = archive.extractfile(member).read()
                entry = expected[member.name]
                if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ValueError("Supplementary member identity mismatch")
    seen_names, seen_parts = set(), set()
    archive_records = {}
    for part in manifest["parts"]:
        archive_records.setdefault(part["archive"], {})[part["member"]] = part
    archive_cache = {}
    for entry in manifest["files"]:
        name = PurePosixPath(entry["path"])
        if name.is_absolute() or ".." in name.parts or str(name) in seen_names:
            raise ValueError("Unsafe or duplicate logical file")
        seen_names.add(str(name))
        digest, length = hashlib.sha256(), 0
        for chunk_number, index in enumerate(entry["chunks"]):
            if index in seen_parts:
                raise ValueError("Duplicate chunk reference")
            seen_parts.add(index)
            part = manifest["parts"][index]
            archive_name = PurePosixPath(part["archive"])
            if archive_name.is_absolute() or len(archive_name.parts) != 1:
                raise ValueError("Unsafe archive path")
            path = directory / str(archive_name)
            if str(archive_name) not in archive_cache:
                packed = path.read_bytes()
                for record in archive_records[str(archive_name)].values():
                    if (
                        len(packed) != record["bytes"]
                        or len(packed) > manifest["max_archive_bytes"]
                        or hashlib.sha256(packed).hexdigest() != record["sha256"]
                    ):
                        raise ValueError(f"Archive identity mismatch: {archive_name}")
                with tarfile.open(path, "r:xz") as archive:
                    members = archive.getmembers()
                    if (
                        len(members) != len(archive_records[str(archive_name)])
                        or {m.name for m in members} != set(archive_records[str(archive_name)])
                        or not all(m.isfile() for m in members)
                    ):
                        raise ValueError("Unsafe or mismatched archive membership")
                    archive_cache[str(archive_name)] = {m.name: archive.extractfile(m).read() for m in members}
            expected_member = str(name) + f".chunk{chunk_number:04d}"
            raw = archive_cache[str(archive_name)][expected_member]
            if (
                len(raw) != part["raw_bytes"]
                or expected_member != part["member"]
                or hashlib.sha256(raw).hexdigest() != part["raw_sha256"]
            ):
                raise ValueError("Chunk identity mismatch")
            digest.update(raw)
            length += len(raw)
        if length != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Logical file identity mismatch: {name}")
    if (
        seen_parts != set(range(len(manifest["parts"])))
        or len(seen_names) != manifest["file_count"]
        or sum(entry["bytes"] for entry in manifest["files"]) != manifest["raw_bytes"]
    ):
        raise ValueError("Incomplete global inventory")
    if len(archive_cache) != manifest["archive_count"]:
        raise ValueError("Archive count mismatch")
    return {
        "status": "passed",
        "files": len(seen_names),
        "archives": len(archive_cache),
        "final_check_files": len(final_files),
        "supplementary_archives": len(final_manifest.get("bundles", [])),
        "raw_bytes": manifest["raw_bytes"],
        "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2), flush=True)
