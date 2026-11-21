"""Daily serving helper for the competition agentic example."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_loader import make_prompt  # noqa: E402
from game import GENERATE_CONTENT_TOOL, load_profile  # noqa: E402

RATING_TO_SCORE = {"😭": 1, "😡": 2, "😐": 3, "😊": 4, "🤩": 5}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate two daily sandwich-feedback candidates.")
    parser.add_argument("--diary", required=True, help="Diary text for today.")
    parser.add_argument("--mood", default="unknown", help="Mood label for today.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--model", default="policy", help="Model name served by AReno.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "areno-agentic"))
    parser.add_argument("--output", help="Optional JSONL file to append candidates and rating.")
    parser.add_argument("--rating", choices=sorted(RATING_TO_SCORE), help="Optional user emoji rating for the winner.")
    parser.add_argument("--winner", type=int, choices=[0, 1], help="Optional winning candidate index.")
    args = parser.parse_args()

    profile = load_profile()
    record = {"diary": args.diary, "mood": args.mood, "user_profile": profile}
    prompt = make_prompt(record, profile)
    candidates = [_generate_feedback(args, prompt, agent_index) for agent_index in range(2)]

    for index, candidate in enumerate(candidates):
        print(f"\nAgent {index}\n{'=' * 7}\n{candidate}")

    if args.output:
        event = {
            "diary": args.diary,
            "mood": args.mood,
            "candidates": candidates,
            "winner": args.winner,
            "rating": args.rating,
            "rating_score": RATING_TO_SCORE.get(args.rating or ""),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _generate_feedback(args: argparse.Namespace, prompt: str, agent_index: int) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install `openai` to use serve_daily.py.") from exc

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are competing to write the most helpful sandwich feedback. "
                    f"You are agent {agent_index}."
                ),
            },
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": "Call generate_content with the feedback you would show the user.",
            },
        ],
        tools=[GENERATE_CONTENT_TOOL],
        tool_choice={"type": "function", "function": {"name": "generate_content"}},
        stream=False,
    )
    message = response.choices[0].message
    calls = [call for call in (message.tool_calls or []) if call.function.name == "generate_content"]
    if not calls:
        return message.content or ""
    try:
        arguments = json.loads(calls[0].function.arguments or "{}")
    except json.JSONDecodeError:
        return ""
    return str(arguments.get("content", ""))


if __name__ == "__main__":
    main()
