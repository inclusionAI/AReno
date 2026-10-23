"""Verify that Ollama imports and generates from the direct-export smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-report", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    native = json.loads(Path(args.native_report).read_text())
    if native.get("status") != "native_training_and_export_passed" or len(native["steps"]) != 1:
        raise ValueError("expected a successful one-update native export report")

    def post(endpoint, values):
        request = Request(
            args.api_url.rstrip("/") + endpoint,
            data=json.dumps(values).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=300) as response:
            return json.load(response)

    info = post("/api/show", {"model": args.model})
    if info["model_info"].get("general.architecture") != "qwen3":
        raise ValueError("Ollama did not identify the exported Qwen3 architecture")
    if "tools" in info.get("capabilities", []):
        raise ValueError("Ollama did not retain the exported text-only chat profile")
    options = {"temperature": 0, "seed": 123, "num_predict": 32, "num_ctx": 512, "num_thread": 8}
    raw = post(
        "/api/generate",
        {
            "model": args.model,
            "prompt": native["probe_prompt"],
            "raw": True,
            "stream": False,
            "options": options,
        },
    )
    chat = post(
        "/api/chat",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "What is 1 + 1? Answer briefly."}],
            "stream": False,
            "options": options,
        },
    )
    for result, text in ((raw, raw.get("response", "")), (chat, chat.get("message", {}).get("content", ""))):
        if not result.get("done") or result.get("eval_count", 0) < 1 or not text.strip():
            raise RuntimeError(f"Ollama did not produce a completed nonempty response: {result}")
    report = {
        "status": "ollama_import_and_generation_passed",
        "model": args.model,
        "gguf_path": native["gguf_path"],
        "details": info["details"],
        "native_after": native["after"]["text"],
        "raw": raw,
        "chat": chat,
        "raw_matches_native": raw["response"] == native["after"]["text"],
    }
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
