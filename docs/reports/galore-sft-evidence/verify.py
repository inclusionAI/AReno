"""Verify every archive and complete raw member without extracting any paths."""

import hashlib
import json
import tarfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "areno-galore-review-evidence-v2"
    seen = set()
    total = 0
    for name, expected in manifest["archives"].items():
        assert Path(name).name == name and name.endswith(".tar.xz")
        data = (root / name).read_bytes()
        assert len(data) == expected["bytes"] and sha(data) == expected["sha256"], name
        assert len(data) <= 1000 * 1024, name
        total += len(data)
        members = set()
        with tarfile.open(root / name, "r:xz") as archive:
            for item in archive:
                assert item.isfile() and not Path(item.name).is_absolute() and ".." not in Path(item.name).parts
                assert item.name not in seen
                record = manifest["raw_files"][item.name]
                assert record["archive"] == name
                raw = archive.extractfile(item).read()
                assert len(raw) == record["bytes"] and sha(raw) == record["sha256"], item.name
                seen.add(item.name)
                members.add(item.name)
        assert members == set(expected["members"]), name
    assert seen == set(manifest["raw_files"])
    assert total == manifest["archive_bytes"]
    print(f"verified {len(seen)} complete raw files across {len(manifest['archives'])} archives")


if __name__ == "__main__":
    main()
