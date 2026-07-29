#!/usr/bin/env python3
"""Group rank-tagged exceptions and report earliest distinct failure signatures."""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import Result, build_parser, skill_main

RANK = re.compile(r"\[rank(?P<rank>\d+)\]")
ERROR = re.compile(r"^(?:\[rank\d+\]:\s*)?(?P<kind>[\w.]+(?:Error|Exception|Exit|Failure)):\s*(?P<message>.*)$")


@skill_main
def main() -> Result:
    parser = build_parser("Group rank-tagged exceptions and report earliest distinct failure signatures.")
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    groups: dict[str, dict] = {}
    current_rank = None  # type: int | None
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
    return Result(ok=bool(ordered), data={"groups": ordered})


if __name__ == "__main__":
    raise SystemExit(main())