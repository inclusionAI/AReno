#!/usr/bin/env python3
"""Compare non-weight checkpoint assets needed for HF-compatible reload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def assets(root: Path) -> dict[str, str]:
    result = {}
    for path in root.iterdir():
        if not path.is_file() or path.suffix in WEIGHT_SUFFIXES or path.name.endswith(".index.json"):
            continue
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("saved", type=Path)
    args = parser.parse_args()
    source, saved = assets(args.source), assets(args.saved)
    missing = sorted(set(source) - set(saved))
    changed = sorted(name for name in source.keys() & saved.keys() if source[name] != saved[name])
    result = {
        "ok": not missing and not changed,
        "missing": missing,
        "changed": changed,
        "extra": sorted(set(saved) - set(source)),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
