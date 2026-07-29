#!/usr/bin/env python3
"""Measure TTFT and total latency from an OpenAI-compatible streaming API."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def request_once(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_content = None
    chunks = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            chunks += 1
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if first_content is None and (delta.get("content") or delta.get("reasoning_content")):
                first_content = time.perf_counter()
    ended = time.perf_counter()
    return {
        "ttft_seconds": None if first_content is None else first_content - started,
        "total_seconds": ended - started,
        "chunks": chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Reply with one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
    }
    try:
        for _ in range(max(args.warmup, 0)):
            request_once(args.base_url, payload, args.timeout)
        request_count = max(args.requests, 1)
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
            measured = list(
                executor.map(
                    lambda _: request_once(args.base_url, payload, args.timeout),
                    range(request_count),
                )
            )
        ttft = [row["ttft_seconds"] for row in measured if row["ttft_seconds"] is not None]
        total = [row["total_seconds"] for row in measured]
        result = {
            "ok": bool(ttft),
            "concurrency": max(args.concurrency, 1),
            "requests": measured,
            "summary": {
                "ttft_mean_seconds": statistics.fmean(ttft) if ttft else None,
                "ttft_median_seconds": statistics.median(ttft) if ttft else None,
                "total_mean_seconds": statistics.fmean(total),
                "total_median_seconds": statistics.median(total),
            },
        }
        if not ttft:
            result["error"] = "stream contained no content or reasoning delta"
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
