"""CPU-only tests for the Wordle agentic RL example — no GPU or tokeniser needed."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "wordle"


def _load_module(name: str):
    """Load a Wordle example module in isolation, mirroring the codebreaker test pattern."""

    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    previous_agentic = sys.modules.get("areno.api.agentic")
    if name == "run_agent":
        sys.modules["areno.api.agentic"] = SimpleNamespace(
            AgentTrajectory=type("AgentTrajectory", (), {}),
            AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
        )
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_wordle_{name}_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game
        if name == "run_agent":
            sys.modules.pop("areno.api.agentic", None)
            if previous_agentic is not None:
                sys.modules["areno.api.agentic"] = previous_agentic


# ── score_guess: success and basic feedback ──────────────────────────────────


def test_correct_guess_all_exact():
    """A perfect guess yields all-exact feedback and solved=True."""

    game = _load_module("game")
    result = game.score_guess("eerie", "eerie")
    assert result["valid"] is True
    assert result["feedback"] == ["exact", "exact", "exact", "exact", "exact"]
    assert result["solved"] is True


def test_all_absent():
    """No matching letters produce all-absent feedback."""

    game = _load_module("game")
    result = game.score_guess("crane", "blimp")
    assert result["feedback"] == ["absent", "absent", "absent", "absent", "absent"]
    assert result["solved"] is False


# ── score_guess: invalid inputs ─────────────────────────────────────────────


def test_invalid_length():
    """A guess shorter than the secret is rejected."""

    game = _load_module("game")
    result = game.score_guess("about", "abcd")
    assert result["valid"] is False
    assert "4" in result["error"]


def test_invalid_non_alpha():
    """A guess containing digits is rejected."""

    game = _load_module("game")
    result = game.score_guess("about", "ab1cd")
    assert result["valid"] is False


# ── score_guess: repeat-letter quota counting ────────────────────────────────


def test_repeat_letters_exact():
    """Correct guess with triple-repeat letters produces all exact."""

    game = _load_module("game")
    result = game.score_guess("eerie", "eerie")
    assert result["feedback"] == ["exact"] * 5
    assert result["solved"] is True


def test_repeat_letters_present_quota():
    """Guess 'eerie' vs 'speed': only 2 of the 3 guess-e's get present (quota)."""

    game = _load_module("game")
    # secret="speed" has 2 e's; guess="eerie" has 3 e's → only 2 present, 1 absent
    result = game.score_guess("speed", "eerie")
    assert result["feedback"] == ["present", "present", "absent", "absent", "absent"]
    assert result["solved"] is False


def test_repeat_letters_mixed():
    """secret='llama', guess='allay' → exact+present combination with repeats."""

    game = _load_module("game")
    # Phase 1: a≠l, l=l(exact), l≠a, a≠m, y≠a → [_,exact,_,_,_]
    #   secret_chars = ['l', None, 'a', 'm', 'a']
    # Phase 2: guess[0]='a' → present (consume 'a' at idx 2)
    #          guess[2]='l' → present (consume 'l' at idx 0)
    #          guess[3]='a' → present (consume 'a' at idx 4)
    #          guess[4]='y' → absent
    result = game.score_guess("llama", "allay")
    assert result["feedback"] == ["present", "exact", "present", "present", "absent"]


# ── normalize_guess ─────────────────────────────────────────────────────────


def test_normalize_guess_case_insensitive():
    """Uppercase input is lowercased."""

    game = _load_module("game")
    assert game.normalize_guess("EERIE") == "eerie"


def test_normalize_guess_with_whitespace():
    """Leading/trailing whitespace is stripped."""

    game = _load_module("game")
    assert game.normalize_guess(" eerie ") == "eerie"


# ── score_episode ───────────────────────────────────────────────────────────


def test_score_episode_solved():
    """Solving at guess 1 of 6 returns 1.0 (0.8 + 0.2 * (6-1)/(6-1))."""

    game = _load_module("game")
    assert game.score_episode("eerie", ["eerie"]) == 1.0


def test_score_episode_solved_later():
    """Solving at guess 2 of 6 returns 0.8 + 0.2 * (6-2)/(6-1) = 0.8 + 0.16."""

    game = _load_module("game")
    reward = game.score_episode("eerie", ["wrong", "eerie"])
    assert abs(reward - (0.8 + 0.2 * 4 / 5)) < 1e-9


def test_score_episode_partial():
    """No solve but with partial matches returns a small positive reward."""

    game = _load_module("game")
    # "wrong" vs "about" → w,r,o,n,g vs a,b,o,u,t → 'o' at index 2 is exact (1)
    # best_information = 1, reward = 0.1 * 1 / 5 = 0.02
    reward = game.score_episode("about", ["wrong"])
    assert abs(reward - 0.02) < 1e-9


def test_score_episode_invalid():
    """Any invalid guess in the list returns -1.0."""

    game = _load_module("game")
    assert game.score_episode("about", ["ab1cd"]) == -1.0


def test_score_episode_no_guesses():
    """An empty guess list returns -1.0."""

    game = _load_module("game")
    assert game.score_episode("about", []) == -1.0


# ── evaluate_wordle ─────────────────────────────────────────────────────────


def test_evaluate_wordle_solved():
    """Solving reports the correct number of guesses."""

    game = _load_module("game")
    result = game.evaluate_wordle("eerie", ["wrong", "eerie"])
    assert result["solved"] is True
    assert result["guesses_to_solve"] == 2
    assert result["word_length"] == 5


def test_evaluate_wordle_not_solved():
    """Failing to solve within max guesses reports solved=False."""

    game = _load_module("game")
    guesses = ["wrong"] * 6
    result = game.evaluate_wordle("eerie", guesses)
    assert result["solved"] is False
    assert result["guesses_to_solve"] is None


def test_guess_exhausted():
    """Six incorrect guesses with max_guesses=6 yields solved=False."""

    game = _load_module("game")
    guesses = ["crane", "slate", "pride", "ghost", "blame", "minor"]
    result = game.evaluate_wordle("eerie", guesses, max_guesses=6)
    assert result["solved"] is False
    assert result["valid_guesses"] == 6


# ── word list integrity ─────────────────────────────────────────────────────


def test_word_list_all_length_5():
    """Every bundled word is exactly 5 letters."""

    game = _load_module("game")
    for word in game.WORDLE_WORDS:
        assert len(word) == game.WORDLE_LENGTH, f"{word!r} is not {game.WORDLE_LENGTH} letters"


def test_word_list_contains_repeat_letter_words():
    """The word list includes words with repeated letters for quota testing."""

    game = _load_module("game")
    has_repeats = any(len(set(w)) < len(w) for w in game.WORDLE_WORDS)
    assert has_repeats, "word list should contain repeat-letter words"


def test_tool_schema_is_closed():
    """GUESS_TOOL has a closed schema with required=['word']."""

    game = _load_module("game")
    params = game.GUESS_TOOL["function"]["parameters"]
    assert params["additionalProperties"] is False
    assert params["required"] == ["word"]
    word_prop = params["properties"]["word"]
    assert "enum" in word_prop or "pattern" in word_prop, "word must have enum or pattern constraint"


# ── generator + loader ──────────────────────────────────────────────────────


def test_generator_is_reproducible_and_loader_hides_secret():
    """Generated records are deterministic and prompts do not leak secrets."""

    generator = _load_module("dataset_generator")
    loader = _load_module("dataset_loader")
    rows = generator.generate_records(8, seed=4)

    assert rows == generator.generate_records(8, seed=4)
    records = loader.load_training_dataset("unused", default_loader=lambda _: rows)
    assert all(record["secret"] not in record["prompt"] for record in records)
    assert all("at most 6 guesses" in record["prompt"] for record in records)


# ── run_agent helpers ───────────────────────────────────────────────────────


def test_agent_executes_guess_and_rejects_bad_calls():
    """_execute_guess accepts valid calls and rejects malformed ones."""

    run_agent = _load_module("run_agent")
    record = {"secret": "eerie", "max_guesses": 6}
    valid = {
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "guess_word", "arguments": json.dumps({"word": "eerie"})},
            }
        ]
    }

    result = run_agent._execute_guess(valid, record)
    assert result["solved"] is True
    assert run_agent._execute_guess({"tool_calls": []}, record) is None
    assert run_agent._execute_guess(
        {"tool_calls": [valid["tool_calls"][0], valid["tool_calls"][0]]}, record
    ) is None
    assert (
        run_agent._execute_guess(
            {"tool_calls": [{"function": {"name": "guess_word", "arguments": "not-json"}}]}, record
        )
        is None
    )


def test_episode_stops_on_success():
    """_run_episode stops after the secret is guessed and appends a summary turn."""

    run_agent = _load_module("run_agent")

    class FakeCompletions:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.messages = []

        async def create(self, **kwargs):
            self.messages.append(kwargs["messages"])
            return next(self.responses)

    def response(word=None):
        calls = []
        if word is not None:
            calls = [
                SimpleNamespace(
                    id=f"call-{word}",
                    type="function",
                    function=SimpleNamespace(name="guess_word", arguments=json.dumps({"word": word})),
                )
            ]
        message = SimpleNamespace(content=None if calls else "I cannot guess", tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    item = SimpleNamespace(
        prompt="guess it",
        record={"secret": "eerie", "max_guesses": 6},
    )
    completions = FakeCompletions([response("crane"), response("eerie"), response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    turns = asyncio.run(run_agent._run_episode(item, client))

    # 2 guess turns + 1 summary turn = 3 total
    assert len(turns) == 3
    # Second call's messages should include the tool result from the first guess
    second_messages = completions.messages[1]
    assert second_messages[-3]["role"] == "assistant"
    assert second_messages[-2]["role"] == "tool"
    assert second_messages[-2]["tool_call_id"] == "call-crane"


def test_episode_stops_on_parser_failure():
    """_run_episode breaks immediately when the model returns no tool call."""

    run_agent = _load_module("run_agent")

    class FakeCompletions:
        def __init__(self, responses):
            self.responses = iter(responses)

        async def create(self, **kwargs):
            return next(self.responses)

    def empty_response():
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="giving up", tool_calls=[]))]
        )

    item = SimpleNamespace(
        prompt="guess it",
        record={"secret": "eerie", "max_guesses": 6},
    )
    failed = FakeCompletions([empty_response()])
    failed_turns = asyncio.run(
        run_agent._run_episode(item, SimpleNamespace(chat=SimpleNamespace(completions=failed)))
    )
    assert len(failed_turns) == 1


# ── reward ──────────────────────────────────────────────────────────────────


def test_reward_separates_paths():
    """reward_fn distinguishes empty, invalid, repeated, partial, and optimal."""

    reward = _load_module("reward")
    source = {"secret": "eerie", "max_guesses": 6}

    def score(guesses):
        calls = [
            {"name": "guess_word", "arguments": json.dumps({"word": guess})}
            for guess in guesses
        ]
        return reward.reward_fn(SimpleNamespace(source_record=source, tool_calls=calls))

    assert score([]) == -1.0
    assert score(["ab1cd"]) == -1.0
    assert score(["eerie"]) == 1.0
    assert score(["eerie", "eerie"]) == -0.5
    assert score(["crane"]) > 0.0

