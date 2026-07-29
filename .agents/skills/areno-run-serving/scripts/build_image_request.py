#!/usr/bin/env python3
"""Write an OpenAI-compatible image chat request as JSON."""

from __future__ import annotations

import base64
import json
import mimetypes
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import build_parser


def main() -> int:
    parser = build_parser("Write an OpenAI-compatible image chat request as JSON.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mime = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(args.image.read_bytes()).decode("ascii")
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "max_tokens": args.max_tokens,
    }
    text = json.dumps(payload, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "output": str(args.output), "bytes": len(text)}))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
