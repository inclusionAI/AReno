#!/usr/bin/env python3
"""Deterministic mock probe of a serving endpoint (no network).

Part of the ``serve_to_summary`` example workflow. Instead of issuing real
HTTP requests, it checks that the upstream ``base_url`` and ``model_id``
are present and non-empty, simulating a probe result.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock-probe a serving endpoint.")
    parser.add_argument("--base-url", required=True, help="Base URL of the endpoint.")
    parser.add_argument("--model-id", required=True, help="Model ID to check.")
    args = parser.parse_args()

    probe_ok = bool(args.base_url) and bool(args.model_id)
    probe = {
        "ok": probe_ok,
        "base_url": args.base_url,
        "model_id": args.model_id,
        "latency_ms": 12,
    }
    print(json.dumps(probe, indent=2))
    return 0 if probe_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())