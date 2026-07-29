"""Small Wordle helpers for agentic examples."""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import Enum
from functools import cache
from typing import Any

# Word list - MIT licensed wordlist subset
# This is a small subset of common 5-letter English words for demonstration
WORD_LIST = frozenset([
    "about", "above", "abuse", "actor", "acute", "admit", "adopt", "adult",
    "after", "again", "agent", "agree", "ahead", "alarm", "album", "alert",
    "alien", "align", "alike", "alive", "allow", "alone", "along", "alter",
    "among", "anger", "angle", "angry", "apart", "apple", "apply", "arena",
    "argue", "arise", "armor", "array", "arrow", "asset", "avoid", "award",
    "aware", "awful", "basic", "beach", "began", "begin", "being", "below",
    "bench", "birth", "black", "blade", "blame", "blank", "blast", "blend",
    "bless", "blind", "block", "blood", "bloom", "blown", "board", "boost",
    "booth", "bound", "brain", "brand", "brass", "brave", "bread", "break",
    "breed", "brick", "bride", "brief", "bring", "broad", "broke", "brown",
    "brush", "build", "built", "bunch", "burst", "buyer", "cabin", "cable",
    "calif", "carry", "catch", "cause", "chain", "chair", "chart", "chase",
    "cheap", "check", "chest", "chief", "child", "china", "chose", "chunk",
    "civic", "civil", "claim", "class", "clean", "clear", "clerk", "click",
    "cliff", "climb", "clock", "close", "cloth", "cloud", "coast", "color",
    "couch", "could", "count", "court", "cover", "crack", "craft", "crash",
    "crazy", "cream", "crime", "crisp", "cross", "crowd", "crown", "crude",
    "crush", "curve", "cycle", "daily", "dance", "dated", "dealt", "death",
    "debut", "decay", "delay", "depth", "doing", "doubt", "draft", "drain",
    "drama", "drank", "dream", "dress", "drift", "drill", "drink", "drive",
    "drown", "early", "earth", "eight", "elbow", "elder", "elect", "elite",
    "empty", "enemy", "enjoy", "enter", "entry", "equal", "error", "essay",
    "event", "every", "exact", "exist", "extra", "faint", "faith", "false",
    "fancy", "fault", "feast", "fence", "ferry", "fetch", "fever", "fiber",
    "field", "fifth", "fifty", "fight", "final", "first", "fixed", "flame",
    "flash", "fleet", "flesh", "float", "flock", "flood", "floor", "flour",
    "fluid", "flush", "focal", "focus", "force", "forge", "forth", "forty",
    "forum", "found", "frame", "frank", "fraud", "fresh", "front", "frost",
    "fruit", "fully", "funny", "giant", "given", "glass", "globe", "glory",
    "glove", "grace", "grade", "grain", "grand", "grant", "grape", "grasp",
    "grass", "grave", "great", "green", "greet", "grief", "grill", "gross",
    "group", "grove", "grown", "guard", "guess", "guest", "guide", "guild",
    "guilt", "habit", "happy", "harsh", "haste", "haunt", "haven", "heart",
    "heavy", "hedge", "hello", "hence", "hobby", "honey", "honor", "horse",
    "hotel", "house", "human", "humor", "ideal", "image", "imply", "index",
    "inner", "input", "issue", "ivory", "japan", "jimmy", "joint", "jones",
    "judge", "juice", "juicy", "kenny", "knife", "knock", "known", "label",
    "labor", "lance", "large", "laser", "later", "laugh", "layer", "learn",
    "lease", "least", "leave", "legal", "lemon", "level", "lewis", "light",
    "limit", "liner", "links", "liver", "lobby", "local", "logic", "loose",
    "lorry", "loser", "lower", "loyal", "lucky", "lunch", "lyric", "magic",
    "major", "maker", "mango", "manor", "maple", "march", "marry", "match",
    "mayor", "meant", "medal", "media", "melon", "mercy", "merge", "merit",
    "merry", "metal", "meter", "midst", "might", "minor", "minus", "misty",
    "mixed", "model", "modem", "money", "month", "moral", "motor", "motto",
    "mount", "mouse", "mouth", "movie", "music", "naive", "naked", "nasty",
    "naval", "nerve", "never", "newly", "night", "noble", "noise", "north",
    "notch", "noted", "novel", "nurse", "occur", "ocean", "offer", "often",
    "olive", "onion", "opera", "orbit", "order", "organ", "other", "ought",
    "outer", "owner", "oxide", "ozone", "paint", "panel", "panic", "paper",
    "party", "pasta", "patch", "pause", "peace", "peach", "pearl", "penny",
    "phase", "phone", "photo", "piano", "piece", "pilot", "pinch", "pitch",
    "pizza", "place", "plain", "plane", "plant", "plate", "plaza", "plead",
    "pluck", "plumb", "plume", "plump", "plunk", "point", "polar", "poppy",
    "porch", "pouch", "pound", "power", "prank", "press", "price", "pride",
    "prime", "print", "prior", "prize", "probe", "promo", "prone", "proof",
    "prose", "proud", "prove", "proxy", "prune", "pulse", "punch", "pupil",
    "puppy", "purse", "queen", "query", "quest", "queue", "quick", "quiet",
    "quilt", "quirk", "quota", "quote", "radar", "radio", "raise", "rally",
    "ranch", "range", "rapid", "ratio", "reach", "react", "ready", "realm",
    "rebel", "refer", "reign", "relax", "relay", "remit", "renal", "renew",
    "repay", "reply", "reset", "resin", "retro", "rider", "ridge", "rifle",
    "right", "rigid", "risky", "rival", "river", "roast", "robot", "rocky",
    "roman", "rough", "round", "route", "royal", "rugby", "ruler", "rumor",
    "rural", "rusty", "sadly", "saint", "salad", "sales", "salon", "sandy",
    "santa", "sauce", "saved", "scale", "scare", "scarf", "scene", "scent",
    "scope", "score", "scout", "scrap", "seize", "sense", "serve", "setup",
    "seven", "shade", "shady", "shaft", "shake", "shall", "shame", "shape",
    "share", "shark", "sharp", "sheep", "sheer", "sheet", "shelf", "shell",
    "shift", "shine", "shiny", "shirt", "shock", "shoot", "shore", "short",
    "shout", "shown", "siege", "sight", "sigma", "silly", "since", "sixth",
    "sixty", "sized", "skill", "skull", "slave", "sleek", "sleep", "slice",
    "slide", "slope", "small", "smart", "smell", "smile", "smith", "smoke",
    "snake", "solar", "solid", "solve", "songs", "sonic", "sorry", "sorts",
    "souls", "sound", "south", "space", "spare", "spark", "speak", "speed",
    "spell", "spend", "spent", "spice", "spicy", "spill", "spine", "splat",
    "split", "spoke", "spoon", "sport", "spray", "squad", "stack", "staff",
    "stage", "stain", "stake", "stamp", "stand", "stark", "start", "state",
    "stays", "steak", "steal", "steam", "steel", "steep", "steer", "stems",
    "steps", "stick", "stiff", "still", "stock", "stomp", "stone", "stood",
    "stool", "store", "storm", "story", "stove", "strap", "straw", "stray",
    "strip", "stuck", "study", "stuff", "style", "sugar", "suite", "sunny",
    "super", "surge", "swamp", "swear", "sweat", "sweep", "sweet", "swept",
    "swift", "swing", "swiss", "sword", "swore", "sworn", "syrup", "table",
    "taste", "taxes", "teach", "teeth", "tempo", "tense", "tenth", "terms",
    "terry", "texas", "thank", "theft", "theme", "there", "these", "thick",
    "thief", "thigh", "thing", "think", "third", "those", "three", "threw",
    "throw", "thumb", "tiger", "tight", "timer", "tired", "title", "toast",
    "today", "token", "topic", "total", "touch", "tough", "towel", "tower",
    "toxic", "trace", "track", "trade", "trail", "train", "trait", "trash",
    "treat", "trend", "trial", "tribe", "trick", "tried", "tries", "truck",
    "truly", "trunk", "trust", "truth", "tulip", "tumor", "tuned", "twice",
    "twist", "tyler", "ultra", "uncle", "under", "undue", "unfit", "union",
    "unite", "unity", "until", "upper", "upset", "urban", "usage", "usual",
    "valid", "value", "video", "vinyl", "virus", "visit", "vital", "vivid",
    "vocal", "vodka", "vogue", "voice", "voter", "wagon", "waist", "waltz",
    "waste", "watch", "water", "waved", "weary", "weave", "wedge", "weigh",
    "weird", "whale", "wheat", "wheel", "where", "which", "while", "whisk",
    "white", "whole", "whose", "widen", "widow", "width", "wired", "witch",
    "woman", "women", "woods", "world", "worry", "worse", "worst", "worth",
    "would", "wound", "woven", "wrath", "wreck", "wrist", "write", "wrong",
    "wrote", "yacht", "yearn", "yeast", "yield", "young", "yours", "youth",
    "zebra", "zesty", "zonal", "zones",
])

# 5-letter words subset for target words (more common words)
TARGET_WORD_LIST = frozenset([
    "about", "above", "abuse", "actor", "admit", "adopt", "adult", "after",
    "again", "agent", "agree", "ahead", "alarm", "album", "alert", "alien",
    "alike", "alive", "allow", "alone", "along", "alter", "among", "anger",
    "angle", "angry", "apple", "apply", "arena", "argue", "arise", "array",
    "arrow", "avoid", "award", "aware", "basic", "beach", "began", "begin",
    "being", "below", "bench", "birth", "black", "blade", "blame", "blank",
    "blast", "blend", "bless", "blind", "block", "blood", "bloom", "board",
    "boost", "bound", "brain", "brand", "brave", "bread", "break", "breed",
    "brick", "bride", "brief", "bring", "broad", "brown", "brush", "build",
    "bunch", "burst", "buyer", "cabin", "cable", "carry", "catch", "cause",
    "chain", "chair", "chart", "chase", "cheap", "check", "chest", "chief",
    "child", "china", "chose", "civic", "civil", "claim", "class", "clean",
    "clear", "click", "cliff", "climb", "clock", "close", "cloth", "cloud",
    "coast", "color", "could", "count", "court", "cover", "crack", "craft",
    "crash", "crazy", "cream", "crime", "cross", "crowd", "crown", "crude",
    "curve", "cycle", "daily", "dance", "death", "debut", "delay", "depth",
    "doubt", "draft", "drain", "drama", "dream", "dress", "drift", "drill",
    "drink", "drive", "early", "earth", "eight", "elbow", "elder", "elect",
    "empty", "enemy", "enjoy", "enter", "entry", "equal", "error", "essay",
    "event", "every", "exact", "exist", "extra", "faith", "false", "fancy",
    "fault", "feast", "fence", "fetch", "fever", "field", "fifth", "fifty",
    "fight", "final", "first", "flame", "flash", "fleet", "flesh", "float",
    "floor", "fluid", "focus", "force", "forth", "forum", "found", "frame",
    "fresh", "front", "frost", "fruit", "fully", "funny", "giant", "given",
    "glass", "globe", "glory", "glove", "grace", "grade", "grain", "grand",
    "grant", "grape", "grasp", "grass", "grave", "great", "green", "greet",
    "grief", "gross", "group", "grown", "guard", "guess", "guest", "guide",
    "guilt", "happy", "harsh", "heart", "heavy", "hello", "hence", "honey",
    "honor", "horse", "hotel", "house", "human", "humor", "ideal", "image",
    "index", "inner", "input", "issue", "judge", "juice", "knife", "knock",
    "known", "label", "labor", "large", "laser", "later", "laugh", "layer",
    "learn", "least", "leave", "legal", "lemon", "level", "light", "limit",
    "local", "logic", "loose", "lower", "lucky", "lunch", "magic", "major",
    "maker", "manor", "maple", "march", "match", "mayor", "meant", "medal",
    "media", "mercy", "merge", "merit", "merry", "metal", "meter", "might",
    "minor", "model", "money", "month", "moral", "motor", "mount", "mouse",
    "mouth", "movie", "music", "naive", "nasty", "nerve", "never", "night",
    "noble", "noise", "north", "novel", "nurse", "occur", "ocean", "offer",
    "often", "olive", "onion", "opera", "orbit", "order", "organ", "other",
    "outer", "owner", "paint", "panel", "panic", "paper", "party", "pasta",
    "patch", "peace", "peach", "pearl", "penny", "phase", "phone", "photo",
    "piano", "piece", "pilot", "pitch", "pizza", "place", "plain", "plane",
    "plant", "plate", "point", "polar", "poppy", "porch", "pound", "power",
    "press", "price", "pride", "prime", "print", "prior", "prize", "proof",
    "prose", "proud", "prove", "pulse", "punch", "pupil", "puppy", "purse",
    "queen", "query", "quest", "queue", "quick", "quiet", "quilt", "quota",
    "quote", "radar", "radio", "raise", "rally", "ranch", "range", "rapid",
    "ratio", "reach", "react", "ready", "rebel", "refer", "reign", "relax",
    "relay", "remit", "renew", "repay", "reply", "reset", "rider", "ridge",
    "rifle", "right", "rigid", "risky", "rival", "river", "roast", "robot",
    "rough", "round", "route", "royal", "rugby", "ruler", "rumor", "rural",
    "sadly", "saint", "salad", "sales", "salon", "sandy", "sauce", "saved",
    "scale", "scare", "scene", "scent", "scope", "score", "scout", "seize",
    "sense", "serve", "setup", "seven", "shade", "shake", "shall", "shame",
    "shape", "share", "shark", "sharp", "sheep", "sheet", "shelf", "shell",
    "shift", "shine", "shirt", "shock", "shoot", "shore", "short", "shout",
    "siege", "sight", "silly", "since", "sixty", "skill", "slave", "sleep",
    "slice", "slide", "slope", "small", "smart", "smell", "smile", "smoke",
    "snake", "solid", "solve", "sound", "south", "space", "spare", "spark",
    "speak", "speed", "spell", "spend", "spent", "spice", "spill", "spine",
    "split", "spoke", "spoon", "sport", "spray", "squad", "stack", "staff",
    "stage", "stain", "stake", "stamp", "stand", "start", "state", "steak",
    "steal", "steam", "steel", "steep", "stick", "still", "stock", "stone",
    "stood", "store", "storm", "story", "stove", "strap", "strip", "stuck",
    "study", "stuff", "style", "sugar", "suite", "sunny", "super", "surge",
    "swamp", "swear", "sweat", "sweep", "sweet", "swift", "swing", "sword",
    "table", "taste", "teach", "thank", "theft", "theme", "there", "these",
    "thick", "thief", "thing", "think", "third", "those", "three", "throw",
    "tiger", "tight", "timer", "tired", "title", "toast", "today", "token",
    "topic", "total", "touch", "tough", "towel", "tower", "toxic", "trace",
    "track", "trade", "trail", "train", "trash", "treat", "trend", "trial",
    "tribe", "trick", "tried", "truck", "truly", "trunk", "trust", "truth",
    "tulip", "tuned", "twice", "twist", "ultra", "uncle", "under", "union",
    "unite", "unity", "until", "upper", "upset", "urban", "usage", "usual",
    "valid", "value", "video", "vinyl", "virus", "visit", "vital", "vivid",
    "vocal", "vodka", "vogue", "voice", "voter", "wagon", "waist", "waste",
    "watch", "water", "weary", "weave", "wedge", "weigh", "weird", "whale",
    "wheat", "wheel", "where", "which", "while", "whisk", "white", "whole",
    "whose", "widen", "widow", "width", "wired", "witch", "woman", "women",
    "woods", "world", "worry", "worse", "worst", "worth", "would", "wound",
    "woven", "wrath", "wreck", "wrist", "write", "wrong", "wrote", "yacht",
    "yearn", "yeast", "yield", "young", "youth", "zebra", "zesty", "zones",
])

MAX_GUESSES = 6
WORD_LENGTH = 5


class LetterStatus(Enum):
    """Wordle feedback status for each letter."""

    ABSENT = "absent"   # Gray: letter not in word
    PRESENT = "present"  # Yellow: letter in word but wrong position
    EXACT = "exact"    # Green: letter correct and in correct position


class GameState(Enum):
    """Wordle game state."""

    IN_PROGRESS = "in_progress"
    WON = "won"
    LOST = "lost"


# Type aliases
GuessResult = list[LetterStatus]  # Length matches word length
HistoryEntry = tuple[str, GuessResult]  # (guess, result)
WordleGame = dict[str, Any]  # Main game state dictionary


def normalize_word(word: str) -> str:
    """Return a validated lowercase 5-letter word."""
    word = word.lower().strip()
    if len(word) != WORD_LENGTH:
        raise ValueError(f"Word must be {WORD_LENGTH} letters, got: {word}")
    if not word.isalpha():
        raise ValueError(f"Word must contain only letters, got: {word}")
    return word


def is_valid_word(word: str) -> bool:
    """Check if a word is in the valid word list."""
    try:
        normalized = normalize_word(word)
        return normalized in WORD_LIST
    except ValueError:
        return False


def check_guess(guess: str, target: str) -> GuessResult:
    """
    Check a guess against the target word and return feedback.

    Handles repeated letters correctly per Wordle rules:
    - If target has one 'E' and guess has two 'E's:
      - First 'E' matches: mark as EXACT or PRESENT
      - Second 'E' is marked ABSENT if no more targets available

    Returns a list of LetterStatus for each position.
    """
    guess = normalize_word(guess)
    target = normalize_word(target)

    result: list[LetterStatus] = [LetterStatus.ABSENT] * WORD_LENGTH

    # Track which target letters have been matched
    target_letters: dict[str, int] = {}
    for letter in target:
        target_letters[letter] = target_letters.get(letter, 0) + 1

    # First pass: mark exact matches
    for i in range(WORD_LENGTH):
        if guess[i] == target[i]:
            result[i] = LetterStatus.EXACT
            target_letters[guess[i]] -= 1

    # Second pass: mark present (but not already matched)
    for i in range(WORD_LENGTH):
        if result[i] == LetterStatus.EXACT:
            continue
        if guess[i] in target_letters and target_letters[guess[i]] > 0:
            result[i] = LetterStatus.PRESENT
            target_letters[guess[i]] -= 1

    return result


def format_feedback(feedback: GuessResult) -> str:
    """Format feedback as colored symbols for display."""
    symbols = []
    for status in feedback:
        if status == LetterStatus.EXACT:
            symbols.append("[G]")  # Green
        elif status == LetterStatus.PRESENT:
            symbols.append("[Y]")  # Yellow
        else:
            symbols.append("[?]")  # Gray
    return " ".join(symbols)


def result_to_text(feedback: GuessResult) -> str:
    """Convert feedback to text description."""
    parts = []
    for i, status in enumerate(feedback):
        letter_pos = f"position {i + 1}"
        if status == LetterStatus.EXACT:
            parts.append(f"Position {i + 1}: EXACT (correct letter in correct place)")
        elif status == LetterStatus.PRESENT:
            parts.append(f"Position {i + 1}: PRESENT (correct letter, wrong place)")
        else:
            parts.append(f"Position {i + 1}: ABSENT (letter not in word)")
    return "\n".join(parts)


def legal_guesses() -> list[str]:
    """Return list of all valid 5-letter words for guesses."""
    return sorted(WORD_LIST)


def create_new_game(target: str | None = None, *, seed: int | None = None) -> WordleGame:
    """Create a new Wordle game with an optional target word."""
    import random
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    if target is None:
        target = rng.choice(list(TARGET_WORD_LIST))
    else:
        target = normalize_word(target)
        if target not in TARGET_WORD_LIST:
            raise ValueError(f"Target word must be in target word list: {target}")

    return {
        "target": target,
        "guesses": [],
        "feedbacks": [],
        "state": GameState.IN_PROGRESS,
    }


def apply_guess(game: WordleGame, guess: str) -> WordleGame:
    """
    Apply a guess to the game and return updated state.

    Returns a new game state (immutable update).
    """
    guess = normalize_word(guess)
    if not is_valid_word(guess):
        raise ValueError(f"Invalid word: {guess}. Not in word list.")

    target = game["target"]

    # Check for win
    if guess == target:
        return {
            **game,
            "guesses": game["guesses"] + [guess],
            "feedbacks": game["feedbacks"] + [check_guess(guess, target)],
            "state": GameState.WON,
        }

    # Check for loss (max guesses)
    if len(game["guesses"]) >= MAX_GUESSES - 1:
        return {
            **game,
            "guesses": game["guesses"] + [guess],
            "feedbacks": game["feedbacks"] + [check_guess(guess, target)],
            "state": GameState.LOST,
        }

    # In progress
    return {
        **game,
        "guesses": game["guesses"] + [guess],
        "feedbacks": game["feedbacks"] + [check_guess(guess, target)],
        "state": GameState.IN_PROGRESS,
    }


def is_terminal(game: WordleGame) -> bool:
    """Check if the game is over."""
    return game["state"] in (GameState.WON, GameState.LOST)


def game_result(game: WordleGame) -> bool | None:
    """
    Return True if won, False if lost, None if in progress.
    """
    if game["state"] == GameState.WON:
        return True
    elif game["state"] == GameState.LOST:
        return False
    return None


def num_guesses(game: WordleGame) -> int:
    """Return number of guesses made."""
    return len(game["guesses"])


def format_prompt(game: WordleGame) -> str:
    """
    Build the prompt for the Wordle agent.
    Keep it concise so small models (Qwen3-0.6B) can follow.
    """
    lines = [
        f"Wordle: Guess the {WORD_LENGTH}-letter word.",
        f"You have {MAX_GUESSES} attempts.",
        "",
        "Feedback: [G]=correct position, [Y]=wrong position, [?]=not in word.",
    ]

    if game["guesses"]:
        lines.append("")
        lines.append("Previous guesses:")
        for guess, feedback in zip(game["guesses"], game["feedbacks"]):
            feedback_str = format_feedback(feedback)
            lines.append(f"  {guess.upper()} -> {feedback_str}")

    remaining = MAX_GUESSES - num_guesses(game)
    lines.append("")
    lines.append(f"Attempts left: {remaining}")

    if game["state"] == GameState.IN_PROGRESS:
        lines.append("Call guess_word with a valid 5-letter word.")

    return "\n".join(lines)


def format_xml_prompt(game: WordleGame) -> str:
    """
    Build the prompt for the Wordle agent without tools.
    """
    lines = [
        f"Wordle: Guess the {WORD_LENGTH}-letter word.",
        f"You have {MAX_GUESSES} attempts.",
        "",
        "Feedback: [G]=correct position, [Y]=wrong position, [?]=not in word.",
    ]

    if game["guesses"]:
        lines.append("")
        lines.append("Previous guesses:")
        for guess, feedback in zip(game["guesses"], game["feedbacks"]):
            lines.append(f"  {guess.upper()} -> {feedback}")

    remaining = MAX_GUESSES - num_guesses(game)
    lines.append("")
    lines.append(f"Attempts left: {remaining}")
    lines.append('Answer with <guess>WORD</guess>.')

    return "\n".join(lines)


def parse_xml_guess(text: str) -> str | None:
    """Extract the guess word from XML-formatted model response."""
    # Remove think tags
    text = _strip_think_tags(text)
    text = _strip_chat_special_tokens(text).strip()

    # Look for <guess> tag
    match = re.search(r"<guess>\s*([a-zA-Z]{5})\s*</guess>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).lower()

    return None


def _strip_think_tags(text: str) -> str:
    """Remove reasoning spans before parsing the policy action."""
    return re.sub(r"<think\b[^>]*>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)


def _strip_chat_special_tokens(text: str) -> str:
    """Remove chat-template sentinels that may trail generated text."""
    return re.sub(r"<\|[^>]+?\|>|</?s>", " ", text, re.IGNORECASE)


# For backwards compatibility with dataset_loader expectations
def normalize_game(game_data: dict) -> WordleGame:
    """Normalize game data into standard WordleGame format."""
    if "target" in game_data:
        return game_data
    # Handle legacy format
    target = game_data.get("word", game_data.get("target_word"))
    if target:
        return create_new_game(target)
    raise ValueError("Invalid game data format")


def score_game(game: WordleGame, guess: str | None = None) -> float:
    """
    Score a game outcome for RL reward.

    - +1.0: Won the game
    - 0.0: Lost the game (exhausted all guesses)
    - -1.0: Invalid guess or timeout
    """
    if game["state"] == GameState.WON:
        # Bonus for fewer guesses (more efficient)
        num = num_guesses(game)
        efficiency_bonus = (MAX_GUESSES - num) / MAX_GUESSES * 0.5
        return 1.0 + efficiency_bonus
    elif game["state"] == GameState.LOST:
        return 0.0
    return -1.0  # In progress or invalid


# =============================================================================
# Statistics functions for Issue #189 acceptance criteria:
# "report solve rate and guesses-to-solve by word length"
# =============================================================================

def compute_stats(results: list[dict]) -> dict:
    """
    Compute solve rate and average guesses by word length.

    Args:
        results: List of game results, each containing:
            - target: The target word
            - guesses: Number of guesses made (None if not solved)
            - solved: True if solved, False otherwise
            - word_length: Length of the target word (optional, auto-computed)

    Returns:
        Dictionary with:
            - overall_solve_rate: float (0.0 to 1.0)
            - overall_avg_guesses: float (average guesses for solved games)
            - by_word_length: dict mapping word_length to:
                - solve_rate: float
                - avg_guesses: float
                - total_games: int
                - solved_games: int
    """
    if not results:
        return {
            "overall_solve_rate": 0.0,
            "overall_avg_guesses": 0.0,
            "by_word_length": {},
        }

    # Group by word length
    by_length: dict[int, dict] = {}
    total_solved = 0
    total_guesses = 0

    for r in results:
        target = r.get("target", "")
        word_len = len(target)
        solved = r.get("solved", False)
        num_guesses = r.get("guesses")

        if word_len not in by_length:
            by_length[word_len] = {
                "solved": 0,
                "total": 0,
                "guesses_sum": 0,
            }

        by_length[word_len]["total"] += 1
        if solved and num_guesses is not None:
            by_length[word_len]["solved"] += 1
            by_length[word_len]["guesses_sum"] += num_guesses
            total_solved += 1
            total_guesses += num_guesses

    # Compute statistics
    by_word_length = {}
    for length, data in by_length.items():
        total = data["total"]
        solved = data["solved"]
        guesses_sum = data["guesses_sum"]

        solve_rate = solved / total if total > 0 else 0.0
        avg_guesses = guesses_sum / solved if solved > 0 else 0.0

        by_word_length[length] = {
            "solve_rate": round(solve_rate, 4),
            "avg_guesses": round(avg_guesses, 2),
            "total_games": total,
            "solved_games": solved,
        }

    overall_solve_rate = total_solved / len(results) if results else 0.0
    overall_avg_guesses = total_guesses / total_solved if total_solved > 0 else 0.0

    return {
        "overall_solve_rate": round(overall_solve_rate, 4),
        "overall_avg_guesses": round(overall_avg_guesses, 2),
        "by_word_length": by_word_length,
    }


def format_stats(stats: dict, human_readable: bool = True) -> str:
    """
    Format statistics for display.

    Args:
        stats: Output from compute_stats()
        human_readable: If True, use human-readable format with descriptions.
                       If False, output structured format.

    Returns:
        Formatted string representation of statistics.
    """
    if human_readable:
        lines = [
            "Wordle Statistics",
            "=" * 40,
            f"Overall Solve Rate: {stats['overall_solve_rate'] * 100:.1f}%",
            f"Overall Avg Guesses (when solved): {stats['overall_avg_guesses']:.2f}",
            "",
            "By Word Length:",
        ]
        for length in sorted(stats["by_word_length"].keys()):
            data = stats["by_word_length"][length]
            lines.append(
                f"  {length}-letter words: "
                f"{data['solve_rate'] * 100:.1f}% solved, "
                f"avg {data['avg_guesses']:.2f} guesses "
                f"({data['solved_games']}/{data['total_games']})"
            )
        return "\n".join(lines)
    else:
        # Structured output (JSON-like)
        import json
        return json.dumps(stats, indent=2)


def validate_dataset_path(path: str) -> tuple[bool, str | None]:
    """
    Validate dataset path before expensive initialization.

    Args:
        path: Path to dataset file or directory

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if path is valid for processing
        - error_message: None if valid, otherwise descriptive error
    """
    from pathlib import Path

    p = Path(path).expanduser()

    # Check if path exists
    if not p.exists():
        return False, f"Path does not exist: {path}"

    # If directory, check for games.jsonl
    if p.is_dir():
        jsonl_file = p / "games.jsonl"
        if not jsonl_file.exists():
            return False, f"No games.jsonl found in directory: {path}"
        p = jsonl_file

    # Check if it's a file
    if not p.is_file():
        return False, f"Path is not a file: {path}"

    # Check file extension
    if p.suffix not in (".jsonl", ".json"):
        return False, f"Expected .jsonl or .json file, got: {p.suffix}"

    # Check if file is readable and has content
    try:
        if p.stat().st_size == 0:
            return False, f"File is empty: {path}"
    except OSError as e:
        return False, f"Cannot read file: {e}"

    return True, None