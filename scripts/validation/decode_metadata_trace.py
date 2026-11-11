"""Summarize an Nsight Systems SQLite export of a bounded decode rollout."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def summarize(path):
    db = sqlite3.connect(path)
    ranges = db.execute("select start,end from NVTX_EVENTS where text='model_and_sample' order by start").fetchall()
    if not ranges:
        raise ValueError("trace has no model_and_sample ranges")
    start, stop = ranges[0][0], ranges[-1][1]
    transfers = db.execute(
        """select copyKind,bytes,count(*),sum(end-start)
        from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<=?
        group by copyKind,bytes order by copyKind,bytes""",
        (start, stop),
    ).fetchall()
    api = db.execute(
        """select s.value,count(*),sum(r.end-r.start)
        from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id
        where r.start>=? and r.start<=? group by s.value order by sum(r.end-r.start) desc""",
        (start, stop),
    ).fetchall()
    # Include GPU work queued by the final iteration, even when it executes
    # after the host's NVTX range closes. Final delivery/reset work is included.
    intervals = db.execute(
        """select start,end from CUPTI_ACTIVITY_KIND_KERNEL where end>=?
        union all select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? order by start""",
        (start, start),
    ).fetchall()
    gaps, busy, end = [], 0, None
    for a, b in intervals:
        a = max(a, start)
        if end is None:
            end = a
        if a > end:
            gaps.append(a - end)
        busy += max(0, b - max(a, end))
        end = max(end, b)
    result = {
        "trace": str(path),
        "decode_iterations": len(ranges),
        "metadata_window": "first model enqueue to final model enqueue; excludes initial admission and final output D2H",
        "memcpy": [
            {"kind": kind, "bytes": size, "count": count, "time_ms": ns / 1e6} for kind, size, count, ns in transfers
        ],
        "cuda_api": [{"name": name, "count": count, "time_ms": ns / 1e6} for name, count, ns in api],
        "gpu_window": "first decode enqueue through final GPU work, including completion/output",
        "gpu_active_ms": busy / 1e6,
        "gpu_idle_gap_ms": sum(gaps) / 1e6,
        "gpu_idle_gap_max_ms": max(gaps, default=0) / 1e6,
        "d2h_count": sum(row[2] for row in transfers if row[0] == 2),
        "h2d_count": sum(row[2] for row in transfers if row[0] == 1),
    }
    db.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [summarize(path) for path in args.traces]
    args.output.write_text(json.dumps(results, indent=2))
    for row in results:
        print(json.dumps({k: v for k, v in row.items() if k not in {"memcpy", "cuda_api"}}))


if __name__ == "__main__":
    main()
