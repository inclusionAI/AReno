#!/usr/bin/env python3
"""Group rank-tagged exceptions and report earliest distinct failure signatures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

RANK = re.compile(r"\[rank(?P<rank>\d+)\]")
ERROR = re.compile(r"^(?:\[rank\d+\]:\s*)?(?P<kind>[\w.]+(?:Error|Exception|Exit|Failure)):\s*(?P<message>.*)$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    groups: dict[str, dict] = {}
    current_rank: int | None = None
    for line_number, line in enumerate(args.log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        rank_match = RANK.search(line)
        if rank_match:
            current_rank = int(rank_match.group("rank"))
        match = ERROR.match(line.strip())
        if not match:
            continue
        signature = f"{match.group('kind')}: {match.group('message')}"
        key = hashlib.sha1(signature.encode()).hexdigest()[:12]
        item = groups.setdefault(
            key,
            {"signature": signature, "first_line": line_number, "ranks": [], "occurrences": 0},
        )
        item["occurrences"] += 1
        if current_rank is not None and current_rank not in item["ranks"]:
            item["ranks"].append(current_rank)
    ordered = sorted(groups.values(), key=lambda item: item["first_line"])
    print(json.dumps({"ok": bool(ordered), "groups": ordered}, indent=2, sort_keys=True))
    return 0 if ordered else 1


if __name__ == "__main__":
    raise SystemExit(main())
