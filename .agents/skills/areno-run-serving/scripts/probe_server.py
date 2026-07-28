#!/usr/bin/env python3
"""Run bounded OpenAI-compatible AReno server probes."""

from __future__ import annotations

import argparse
import json
import urllib.request


def request_json(url: str, payload: dict | None, timeout: float) -> object:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--prompt", default="Reply with the word ready.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    result: dict[str, object] = {"base_url": base, "checks": {}}
    try:
        models = request_json(f"{base}/v1/models", None, args.timeout)
        result["checks"]["models"] = {"ok": True, "response": models}
        model = args.model
        if model is None and isinstance(models, dict):
            rows = models.get("data")
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                model = rows[0].get("id")
        if not model:
            raise ValueError("no model supplied or advertised")
        chat = request_json(
            f"{base}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.max_tokens,
                "stream": False,
            },
            args.timeout,
        )
        result["checks"]["chat"] = {"ok": True, "response": chat}
    except Exception as exc:
        response_detail = ""
        if hasattr(exc, "read"):
            try:
                response_detail = f"; response: {exc.read().decode('utf-8')}"
            except Exception:
                pass
        result["error"] = f"{type(exc).__name__}: {exc}{response_detail}"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
