"""Deterministic Wordle rules for the agentic RL example."""

from __future__ import annotations

from typing import Any

WORDLE_LENGTH = 5
DEFAULT_MAX_GUESSES = 6

# Public-domain 5-letter English words (common usage, no proper nouns).
WORDLE_WORDS = [
    "about", "above", "abuse", "actor", "acute",
    "admit", "adopt", "adore", "adult", "after",
    "again", "agent", "agree", "ahead", "alarm",
    "album", "alert", "alike", "alive", "allow",
    "alone", "along", "alter", "among", "anger",
    "angle", "angry", "apart", "apple", "apply",
    "arena", "argue", "arise", "array", "aside",
    "asset", "audio", "audit", "avoid", "award",
    "aware", "badly", "baker", "bases", "basic",
    "beach", "began", "begin", "begun", "being",
    "below", "bench", "billy", "birth", "black",
    "blame", "blank", "blast", "blind", "block",
    "blood", "board", "boost", "booth", "bound",
    "brain", "brand", "brass", "bread", "break",
    "breed", "brick", "brief", "bring", "broad",
    "broke", "brown", "build", "built", "buyer",
    "cable", "calif", "carry", "catch", "chain",
    "chair", "chart", "chase", "cheap", "check",
    "chest", "chief", "child", "china", "chose",
    "civic", "civil", "claim", "class", "clean",
    "clear", "click", "clock", "close", "coach",
    "coast", "could", "court", "cover", "craft",
    "crash", "cream", "crime", "cross", "crowd",
    "crown", "curve", "cycle", "daily", "dance",
    "dated", "dealt", "death", "debut", "delay",
    "depth", "doing", "doubt", "dozen", "draft",
    "drama", "drawn", "dream", "dress", "drill",
    "drink", "drive", "drove", "dying", "eager",
    "early", "earth", "eight", "elite", "empty",
    "enemy", "enjoy", "enter", "entry", "equal",
    "error", "event", "every", "exact", "exist",
    "extra", "faith", "false", "fault", "fiber",
    "field", "fifth", "fifty", "fight", "final",
    "first", "fixed", "flash", "fleet", "floor",
    "fluid", "focus", "force", "forth", "forty",
    "forum", "found", "frame", "fraud", "fresh",
    "front", "fruit", "fully", "funny", "ghost",
    "giant", "given", "glass", "globe", "going",
    "grace", "grade", "grand", "grant", "grass",
    "great", "green", "gross", "group", "grown",
    "guard", "guess", "guest", "guide", "happy",
    "harsh", "harry", "heart", "heavy", "hence",
    "henry", "horse", "hotel", "house", "human",
    "ideal", "image", "index", "inner", "input",
    "issue", "japan", "jimmy", "joint", "jones",
    "judge", "known", "label", "large", "laser",
    "later", "laugh", "layer", "learn", "lease",
    "least", "leave", "legal", "level", "lewis",
    "light", "limit", "links", "lives", "local",
    "logic", "loose", "lower", "lucky", "lunch",
    "lying", "magic", "major", "maker", "march",
    "maria", "match", "maybe", "mayor", "meant",
    "media", "metal", "might", "minor", "minus",
    "mixed", "model", "money", "month", "moral",
    "motor", "mount", "mouse", "mouth", "movie",
    "music", "needs", "never", "newly", "night",
    "noise", "north", "noted", "novel", "nurse",
    "occur", "ocean", "offer", "often", "order",
    "other", "ought", "paint", "panel", "paper",
    "party", "peace", "peter", "phase", "phone",
    "photo", "piece", "pilot", "pitch", "place",
    "plain", "plane", "plant", "plate", "point",
    "pound", "power", "press", "price", "pride",
    "prime", "print", "prior", "prize", "proof",
    "proud", "prove", "queen", "quick", "quiet",
    "quite", "radio", "raise", "range", "rapid",
    "ratio", "reach", "ready", "refer", "right",
    "rival", "river", "robin", "roger", "roman",
    "rough", "round", "route", "royal", "rural",
    "scale", "scene", "scope", "score", "sense",
    "serve", "seven", "shall", "shape", "share",
    "sharp", "sheet", "shelf", "shell", "shift",
    "shirt", "shock", "shoot", "short", "shown",
    "sight", "since", "sixth", "sixty", "sized",
    "skill", "sleep", "slice", "slide", "small",
    "smart", "smile", "smith", "smoke", "solid",
    "solve", "sorry", "sound", "south", "space",
    "spare", "speak", "speed", "spend", "spent",
    "split", "spoke", "sport", "staff", "stage",
    "stake", "stand", "start", "state", "steam",
    "steel", "stick", "still", "stock", "stone",
    "stood", "store", "storm", "story", "strip",
    "stuck", "study", "stuff", "style",
    "sugar", "suite", "super", "sweet", "table",
    "taken", "taste", "taxes", "teach", "teeth",
    "terry", "texas", "thank", "theft", "their",
    "theme", "there", "these", "thick", "thing",
    "think", "third", "those", "three", "threw",
    "throw", "tight", "times", "tired", "title",
    "today", "topic", "total", "touch", "tough",
    "tower", "track", "trade", "train", "treat",
    "trend", "trial", "tried", "tries", "truck",
    "truly", "trust", "truth", "twice", "under",
    "undue", "union", "unity", "until", "upper",
    "upset", "urban", "usage", "usual", "valid",
    "value", "video", "virus", "visit", "vital",
    "voice", "waste", "watch", "water", "wheel",
    "where", "which", "while", "white", "whole",
    "whose", "woman", "women", "world", "worry",
    "worse", "worst", "worth", "would", "wound",
    "write", "wrong", "wrote", "yield", "young",
    "youth", "eerie", "llama",
]

# Validate word list at import time (cheap, CPU-only).
for _w in WORDLE_WORDS:
    assert len(_w) == WORDLE_LENGTH, f"word {_w!r} is not {WORDLE_LENGTH} letters"
    assert _w.isalpha(), f"word {_w!r} contains non-alpha characters"

GUESS_TOOL = {
    "type": "function",
    "function": {
        "name": "guess_word",
        "description": "Guess the hidden 5-letter word and receive Wordle feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "pattern": "^[a-zA-Z]{5}$",
                    "description": "Exactly five English letters (case-insensitive).",
                }
            },
            "required": ["word"],
            "additionalProperties": False,
        },
    },
}


def normalize_guess(value: object, *, word_length: int = WORDLE_LENGTH) -> str:
    """Return a validated lowercased guess."""

    guess = str(value).lower().strip()
    if len(guess) != word_length:
        raise ValueError(f"guess must be exactly {word_length} letters, got {len(guess)}")
    if not guess.isalpha():
        raise ValueError(f"guess must contain only letters, got {guess!r}")
    return guess


def score_guess(secret: str, guess: object) -> dict[str, Any]:
    """Score one guess with standard Wordle feedback including repeat-letter rules."""

    secret_lower = secret.lower()
    try:
        normalized_guess = normalize_guess(guess, word_length=len(secret_lower))
    except ValueError as exc:
        return {"valid": False, "error": str(exc), "guess": str(guess)}

    length = len(secret_lower)
    feedback: list[str] = ["absent"] * length
    secret_chars: list[str | None] = list(secret_lower)

    # Phase 1: mark exact matches and consume those secret letters.
    for i in range(length):
        if normalized_guess[i] == secret_chars[i]:
            feedback[i] = "exact"
            secret_chars[i] = None

    # Phase 2: mark present matches with quota counting for repeats.
    for i in range(length):
        if feedback[i] == "exact":
            continue
        char = normalized_guess[i]
        if char in secret_chars:
            feedback[i] = "present"
            secret_chars[secret_chars.index(char)] = None

    return {
        "valid": True,
        "guess": normalized_guess,
        "feedback": feedback,
        "solved": all(f == "exact" for f in feedback),
    }


def make_prompt(record: dict[str, Any]) -> str:
    """Build one prompt without leaking the secret word."""

    max_guesses = int(record.get("max_guesses", DEFAULT_MAX_GUESSES))
    return (
        "Guess the hidden 5-letter English word. "
        f"You have at most {max_guesses} guesses. After each guess, you receive "
        "feedback for each position: 'exact' = correct letter in correct position, "
        "'present' = letter exists in the word but in a different position, "
        "'absent' = letter not in the word. "
        "Call guess_word once per turn and use the feedback to narrow down the word."
    )


def score_episode(
    secret: str,
    guesses: list[object],
    *,
    max_guesses: int = DEFAULT_MAX_GUESSES,
) -> float:
    """Reward solving efficiently, partial progress, and penalize invalid guesses."""

    if not guesses:
        return -0.3
    valid_results = [score_guess(secret, guess) for guess in guesses[:max_guesses]]
    if any(not result["valid"] for result in valid_results):
        return -1.0
    for index, result in enumerate(valid_results, start=1):
        if result["solved"]:
            efficiency = (max_guesses - index) / max(max_guesses - 1, 1)
            return 0.8 + 0.2 * efficiency
    best_information = max(
        result["feedback"].count("exact") + result["feedback"].count("present")
        for result in valid_results
    )
    return 0.1 * best_information / len(secret)


def evaluate_wordle(
    secret: str,
    guesses: list[object],
    *,
    max_guesses: int = DEFAULT_MAX_GUESSES,
) -> dict[str, Any]:
    """Deterministic evaluation reporting solve status and guesses-to-solve."""

    valid_guesses = []
    for guess in guesses[:max_guesses]:
        result = score_guess(secret, guess)
        if not result["valid"]:
            break
        valid_guesses.append(result)
        if result["solved"]:
            return {
                "solved": True,
                "guesses_to_solve": len(valid_guesses),
                "word_length": len(secret),
            }
    return {
        "solved": False,
        "guesses_to_solve": None,
        "word_length": len(secret),
        "valid_guesses": len(valid_guesses),
    }

