"""Interactive terminal UI for the Codebreaker rule engine."""

from __future__ import annotations

import argparse
import random

from game import DEFAULT_CODE_LENGTH, DEFAULT_MAX_GUESSES, score_guess


def _secret(seed: int | None) -> str:
    return "".join(random.Random(seed).sample("0123456789", DEFAULT_CODE_LENGTH))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-guesses", type=int, default=DEFAULT_MAX_GUESSES)
    args = parser.parse_args()
    secret = _secret(args.seed)
    history: list[dict] = []
    print("\033[1;36mCODEBREAKER // TERMINAL LOCK\033[0m")
    print("Crack 4 unique digits. exact=right place, present=wrong place.\n")
    for turn in range(1, args.max_guesses + 1):
        guess = input(f"[{turn}/{args.max_guesses}] code> ").strip()
        result = score_guess(secret, guess)
        history.append(result)
        if not result["valid"]:
            print(f"\033[31mINVALID\033[0m {result['error']}")
            continue
        print(f"  exact: \033[32m{result['exact']}\033[0m  present: \033[33m{result['present']}\033[0m")
        if result["solved"]:
            print("\n\033[1;32mACCESS GRANTED\033[0m")
            return
    print(f"\n\033[1;31mLOCKED OUT\033[0m  code was {secret}")


if __name__ == "__main__":
    main()
