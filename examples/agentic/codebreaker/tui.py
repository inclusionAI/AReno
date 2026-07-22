"""Interactive terminal UI for the Codebreaker rule engine."""

from __future__ import annotations

import argparse
import json
import random

from game import DEFAULT_CODE_LENGTH, DEFAULT_MAX_GUESSES, GUESS_TOOL, make_prompt, score_guess

SYSTEM_PROMPT = (
    "You are a rigorous codebreaker. Call guess_code exactly once per turn, use all prior clues, "
    "never repeat a guess, and do not answer in plain text."
)


def _secret(seed: int | None) -> str:
    return "".join(random.Random(seed).sample("0123456789", DEFAULT_CODE_LENGTH))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-guesses", type=int, default=DEFAULT_MAX_GUESSES)
    parser.add_argument("--agent", action="store_true", help="Let an OpenAI-compatible model crack the code")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    args = parser.parse_args()
    secret = _secret(args.seed)
    print("\033[1;36mCODEBREAKER // TERMINAL LOCK\033[0m")
    print("Crack 4 unique digits. exact=right place, present=wrong place.\n")
    if args.agent:
        _run_llm(secret, args)
    else:
        _run_human(secret, args.max_guesses)


def _run_human(secret: str, max_guesses: int) -> None:
    for turn in range(1, max_guesses + 1):
        guess = input(f"[{turn}/{max_guesses}] code> ").strip()
        result = score_guess(secret, guess)
        if _render_result(result):
            return
    print(f"\n\033[1;31mLOCKED OUT\033[0m  code was {secret}")


def _run_llm(secret: str, args: argparse.Namespace) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("LLM mode requires `openai`. Install it with `pip install openai`.") from exc

    max_guesses = min(max(int(args.max_guesses), 1), DEFAULT_MAX_GUESSES)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    record = {"code_length": DEFAULT_CODE_LENGTH, "max_guesses": max_guesses}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": make_prompt(record)}]
    tool_choice = {"type": "function", "function": {"name": "guess_code"}}
    for turn in range(1, max_guesses + 1):
        turn_prompt = {"role": "user", "content": f"Guess {turn} of {max_guesses}: call guess_code now."}
        response = client.chat.completions.create(
            model=args.model,
            messages=[*messages, turn_prompt],
            tools=[GUESS_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        message = response.choices[0].message
        calls = list(message.tool_calls or [])
        if len(calls) != 1 or calls[0].function.name != "guess_code":
            print("\033[31mMODEL ERROR\033[0m expected exactly one guess_code call")
            break
        try:
            arguments = json.loads(calls[0].function.arguments or "")
        except json.JSONDecodeError:
            print("\033[31mMODEL ERROR\033[0m invalid JSON tool arguments")
            break
        if not isinstance(arguments, dict) or "code" not in arguments:
            print("\033[31mMODEL ERROR\033[0m missing code argument")
            break
        result = score_guess(secret, arguments["code"])
        print(f"[{turn}/{max_guesses}] model> {arguments['code']}")
        assistant_message = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": calls[0].id,
                    "type": calls[0].type,
                    "function": {"name": "guess_code", "arguments": calls[0].function.arguments},
                }
            ],
        }
        messages.extend(
            [
                turn_prompt,
                assistant_message,
                {
                    "role": "tool",
                    "tool_call_id": calls[0].id,
                    "name": "guess_code",
                    "content": json.dumps(result),
                },
            ]
        )
        if _render_result(result):
            return
        if not result["valid"]:
            break
    print(f"\n\033[1;31mLOCKED OUT\033[0m  code was {secret}")


def _render_result(result: dict) -> bool:
    if not result["valid"]:
        print(f"\033[31mINVALID\033[0m {result['error']}")
        return False
    print(f"  exact: \033[32m{result['exact']}\033[0m  present: \033[33m{result['present']}\033[0m")
    if result["solved"]:
        print("\n\033[1;32mACCESS GRANTED\033[0m")
        return True
    return False


if __name__ == "__main__":
    main()
