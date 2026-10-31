"""Verify all archive/member identities from a portable partition manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

from pack_evidence import verify


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    expected = {entry["path"]: entry for entry in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate global member")
    observed = {}
    for part in manifest["parts"]:
        name = PurePosixPath(part["archive"])
        if len(name.parts) != 1 or name.is_absolute():
            raise ValueError("Unsafe partition filename")
        archive = args.directory / str(name)
        data = archive.read_bytes()
        if len(data) != part["bytes"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
            raise ValueError(f"Partition identity mismatch: {name}")
        if len(data) > manifest["max_archive_bytes"]:
            raise ValueError(f"Oversized partition: {name}")
        report = verify(archive)
        with tarfile.open(archive, "r:xz") as stream:
            entries = json.load(stream.extractfile("MANIFEST.json"))["files"]
        if {entry["path"] for entry in entries} != set(part["members"]):
            raise ValueError(f"Partition membership mismatch: {name}")
        if report["verified_files"] != part["verified_members"]:
            raise ValueError(f"Partition member count mismatch: {name}")
        for entry in entries:
            if entry["path"] in observed:
                raise ValueError(f"Duplicate partitioned member: {entry['path']}")
            observed[entry["path"]] = entry
    if observed != expected or len(observed) != manifest["member_count"]:
        raise ValueError("Global/partition member inventory mismatch")
    print(
        json.dumps(
            {
                "status": "passed",
                "verified_archives": len(manifest["parts"]),
                "verified_members": len(observed),
                "largest_archive_bytes": max(part["bytes"] for part in manifest["parts"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
