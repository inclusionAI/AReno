#!/usr/bin/env python3
"""Produce a deterministic mock serving endpoint description (no network).

Part of the ``serve_to_summary`` example workflow. Instead of starting a real
server, it returns a mock ``base_url`` and ``model_id`` that the downstream
probe step can use. This keeps the example fully local and deterministic.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock a serving endpoint.")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", help="Model checkpoint to advertise."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for the mock endpoint."
    )
    args = parser.parse_args()

    serve = {
        "ok": True,
        "base_url": f"http://127.0.0.1:{args.port}",
        "model_id": args.model,
        "port": args.port,
    }
    print(json.dumps(serve, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())