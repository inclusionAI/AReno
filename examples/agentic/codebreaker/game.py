"""Deterministic Bulls and Cows rules for the Codebreaker example."""

from __future__ import annotations

from typing import Any

DIGITS = "0123456789"
DEFAULT_CODE_LENGTH = 4
DEFAULT_MAX_GUESSES = 6

GUESS_TOOL = {
    "type": "function",
    "function": {
        "name": "guess_code",
        "description": "Guess the hidden unique-digit code and receive Bulls and Cows clues.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "pattern": "^[0-9]{4}$",
                    "description": "Exactly four distinct digits; a leading zero is allowed.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


def normalize_code(value: object, *, code_length: int = DEFAULT_CODE_LENGTH) -> str:
    """Return a validated unique-digit code."""

    code = str(value)
    if len(code) != code_length or any(digit not in DIGITS for digit in code):
        raise ValueError(f"code must contain exactly {code_length} digits")
    if len(set(code)) != len(code):
        raise ValueError("code digits must be unique")
    return code


def score_guess(secret: str, guess: object) -> dict[str, Any]:
    """Score one guess without revealing the secret."""

    secret = normalize_code(secret, code_length=len(secret))
    try:
        normalized_guess = normalize_code(guess, code_length=len(secret))
    except ValueError as exc:
        return {"valid": False, "error": str(exc), "guess": str(guess)}
    exact = sum(left == right for left, right in zip(secret, normalized_guess, strict=True))
    present = len(set(secret) & set(normalized_guess)) - exact
    return {
        "valid": True,
        "guess": normalized_guess,
        "exact": exact,
        "present": present,
        "solved": exact == len(secret),
    }


def make_prompt(record: dict[str, Any]) -> str:
    """Build one prompt without leaking the hidden code."""

    code_length = int(record.get("code_length", DEFAULT_CODE_LENGTH))
    max_guesses = int(record.get("max_guesses", DEFAULT_MAX_GUESSES))
    return (
        "Crack the terminal lock's hidden code. "
        f"It has {code_length} unique digits and may begin with 0. "
        f"You have at most {max_guesses} guesses. After each guess, exact is the number of digits "
        "in the correct position and present is the number of other correct digits in wrong positions. "
        "Call guess_code once per turn and use every clue to deduce the code."
    )


def score_episode(secret: str, guesses: list[object], *, max_guesses: int = DEFAULT_MAX_GUESSES) -> float:
    """Reward solving efficiently, partial valid progress, and invalid paths separately."""

    if not guesses:
        return -1.0
    valid_results = [score_guess(secret, guess) for guess in guesses[:max_guesses]]
    if any(not result["valid"] for result in valid_results):
        return -1.0
    for index, result in enumerate(valid_results, start=1):
        if result["solved"]:
            efficiency = (max_guesses - index) / max(max_guesses - 1, 1)
            return 0.8 + 0.2 * efficiency
    best_information = max(result["exact"] + result["present"] for result in valid_results)
    return 0.1 * best_information / len(secret)
