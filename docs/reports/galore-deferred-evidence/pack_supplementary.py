"""Package the bounded ABBA and separately frozen scalar repair without tensors."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--round-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if list(args.output.glob("writeback-*.tar.xz")):
        raise FileExistsError("Supplementary archives already exist")
    records = []

    def pack(paths, label):
        buffer, files = io.BytesIO(), []
        with tarfile.open(fileobj=buffer, mode="w:xz", preset=6) as archive:
            for path in sorted(paths):
                raw = path.read_bytes()
                name = path.relative_to(args.round_root).as_posix()
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(raw), 0, 0o644
                archive.addfile(info, io.BytesIO(raw))
                files.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        raw = buffer.getvalue()
        if len(raw) > 900000:
            if len(paths) < 2:
                raise ValueError("Oversized singleton evidence")
            middle = len(paths) // 2
            pack(paths[:middle], label + "a")
            pack(paths[middle:], label + "b")
            return
        name = "writeback-" + label + ".tar.xz"
        (args.output / name).write_bytes(raw)
        records.append({"archive": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "files": files})

    native = args.round_root / "writeback"
    for method in ("strict_first", "candidate_first", "candidate_second", "strict_second"):
        pack(
            [p for p in (native / "outputs" / method).rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"}],
            method,
        )
    chosen = []
    for base in (native, args.round_root / "writeback_scalarfix"):
        for path in base.rglob("*"):
            relative = path.relative_to(base)
            if not path.is_file() or path.is_symlink():
                continue
            if any(part in {"repo", "outputs", "__pycache__", ".pytest_cache"} for part in relative.parts):
                continue
            if path.suffix in {".py", ".json", ".jsonl", ".log", ".md", ".ptx"}:
                chosen.append(path)
        for name in ("compact_writeback.py", "deferred_projection.py"):
            chosen.append(base / "repo/areno/engine/optim" / name)
    pack(sorted(chosen), "qualification")
    manifest_path = args.output / "FINAL_CHECKS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bundles"] = records
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "archives": len(records),
                "files": sum(len(p["files"]) for p in records),
                "bytes": sum(p["bytes"] for p in records),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
