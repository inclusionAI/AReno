from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "othello"


@contextmanager
def _stubbed_agentic():
    """Load ``run_agent`` without the torch-backed ``areno.api.agentic``.

    ``run_agent.py`` imports ``AgentTrajectory`` / ``AgentTrajectoryTurn`` at
    module top, which pulls in the full areno stack (and torch). In CPU tests we
    stub those names plus ``openai`` / ``httpx`` so the module loads without the
    heavy backend. Restores prior ``sys.modules`` entries on exit.
    """

    import sys as _sys

    saved = {key: _sys.modules.get(key) for key in ("areno.api.agentic", "openai", "httpx")}
    _sys.modules["areno.api.agentic"] = SimpleNamespace(
        AgentTrajectory=lambda turns: SimpleNamespace(turns=turns),
        AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                _sys.modules[key] = value
            else:
                _sys.modules.pop(key, None)


def _load_module(name: str):
    """Load an Othello example module by file path with an isolated ``game`` import.

    Saves/restores ``sys.modules['game']`` so the per-example ``game`` module
    does not leak across tests (mirrors the shopping/codebreaker test pattern).
    """

    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_othello_{name}_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


# --------------------------------------------------------------------------- #
# Environment rules
# --------------------------------------------------------------------------- #


def test_new_board_is_standard_opening():
    game = _load_module("game")

    board = game.new_board()

    assert len(board) == 6 and all(len(row) == 6 for row in board)
    counts = game.count_disks(board)
    assert counts["B"] == 2 and counts["W"] == 2 and counts["."] == 32
    mid = 3
    assert board[mid - 1][mid - 1] == "W"
    assert board[mid - 1][mid] == "B"
    assert board[mid][mid - 1] == "B"
    assert board[mid][mid] == "W"


def test_opening_legal_moves_are_symmetric_four():
    game = _load_module("game")

    moves = game.legal_moves(game.new_board(), "B")

    assert moves == [(1, 2), (2, 1), (3, 4), (4, 3)]


def _blank_board(game, fill_with="."):
    """A 6x6 board filled with the given cell, then renormalized."""

    return game.normalize_board([[fill_with for _ in range(6)] for _ in range(6)])


def _directioned_board(game, mover, opp, target, direction):
    """Build a board where ``mover`` playing ``target`` flips exactly one disc
    sitting one step along ``direction`` from ``target``, bracketed by another
    mover disc two steps along ``direction``.

    ``direction`` is one of the 8 ``(dr, dc)`` unit vectors. Returns the board.
    """

    board = [[game.EMPTY for _ in range(6)] for _ in range(6)]
    tr, tc = target
    dr, dc = direction
    mid = (tr + dr, tc + dc)
    far = (tr + 2 * dr, tc + 2 * dc)
    board[mid[0]][mid[1]] = opp
    board[far[0]][far[1]] = mover
    return game.normalize_board(board)


def test_flips_for_each_of_eight_directions():
    game = _load_module("game")

    target = (2, 2)
    expected_flips = []
    for direction in game.DIRECTIONS:
        dr, dc = direction
        board = _directioned_board(game, "B", "W", target, direction)
        flipped = game.flips_for(board, target[0], target[1], "B")
        mid = (target[0] + dr, target[1] + dc)
        assert flipped == [mid], f"direction {direction} expected flip at {mid}, got {flipped}"
        applied = game.apply_move(board, target[0], target[1], "B")
        assert applied[mid[0]][mid[1]] == "B"
        expected_flips.append(mid)
    assert len(expected_flips) == 8


def test_flips_for_non_empty_or_unbracketed_returns_empty():
    game = _load_module("game")

    board = game.new_board()
    # (2,2) currently W -> non-empty, no flips.
    assert game.flips_for(board, 2, 2, "B") == []
    # Out of bounds.
    assert game.flips_for(board, -1, 0, "B") == []
    # An empty corner with no bracket (a lone opponent disc unbracketed by a mover
    # disc) flips nothing.
    blank = game.normalize_board([["." if (r, c) != (1, 1) else "W" for c in range(6)] for r in range(6)])
    assert game.flips_for(blank, 0, 0, "B") == []


def test_legal_moves_and_apply_move_from_opening():
    game = _load_module("game")

    board = game.new_board()
    assert (1, 2) in game.legal_moves(board, "B")
    next_board = game.apply_move(board, 1, 2, "B")
    # The played cell and the flipped W disc are both B now.
    assert next_board[1][2] == "B"
    assert next_board[2][2] == "B"
    counts = game.count_disks(next_board)
    assert counts["B"] == 4 and counts["W"] == 1


def test_apply_move_rejects_illegal_cells():
    game = _load_module("game")

    board = game.new_board()
    for bad in [(0, 0), (2, 2), (-1, 0), (6, 0)]:
        try:
            game.apply_move(board, bad[0], bad[1], "B")
            raise AssertionError(f"expected ValueError for illegal move {bad}")
        except ValueError:
            pass


def test_force_pass_when_player_has_no_legal_move():
    game = _load_module("game")

    # Evolve a reachable position where Black has no legal move but White does.
    # We search a small prefix of random-but-seeded legal play for such a board.
    import random

    rng = random.Random(123)
    found_board = None
    for _ in range(2000):
        board = game.new_board()
        player = "B"
        for _step in range(rng.randint(6, 30)):
            moves = game.legal_moves(board, player)
            if not moves:
                player = game.opponent(player)
                moves = game.legal_moves(board, player)
                if not moves:
                    break
            move = rng.choice(moves)
            board = game.apply_move(board, move[0], move[1], player)
            player = game.opponent(player)
            if (
                game.legal_moves(board, "B") == []
                and game.has_legal_move(board, "W")
                and not game.is_terminal(board)
            ):
                found_board = board
                break
        if found_board is not None:
            break
    assert found_board is not None, "could not evolve a forced-pass position in the seed budget"
    assert game.legal_moves(found_board, "B") == []
    assert game.has_legal_move(found_board, "W")
    assert not game.is_terminal(found_board)


def test_two_consecutive_passes_is_terminal():
    game = _load_module("game")

    # A full board (no empty cell) means neither side can move -> terminal by
    # the two-consecutive-pass rule.
    full = game.normalize_board(
        [["B" if (r + c) % 2 == 0 else "W" for c in range(6)] for r in range(6)]
    )
    assert not game.has_legal_move(full, "B")
    assert not game.has_legal_move(full, "W")
    assert game.is_terminal(full)


def test_terminal_scoring_black_white_draw():
    game = _load_module("game")

    full = game.normalize_board(
        [["B" if (r + c) % 2 == 0 else "W" for c in range(6)] for r in range(6)]
    )
    result = game.score_board(full)
    assert result["black"] == 18 and result["white"] == 18 and result["winner"] == "draw"

    black_wins = game.normalize_board([["B" for _ in range(6)] for _ in range(6)])
    result = game.score_board(black_wins)
    assert result["winner"] == "B" and result["black"] == 36


def test_score_move_reward_kernel_is_tiered():
    game = _load_module("game")

    board = game.new_board()
    # Non-terminal legal move -> +0.4 (legal-action shaping credit).
    assert game.score_move(board, (1, 2), "B") == 0.4
    # No move / unparseable tool call -> -1.0 (worst tier: no actionable call).
    assert game.score_move(board, None, "B") == -1.0
    # Out-of-bounds coordinate -> -0.5 (bad coordinate, but still emitted a call).
    assert game.score_move(board, (-1, 2), "B") == -0.5
    assert game.score_move(board, (6, 0), "B") == -0.5
    # In-range but illegal cell (occupied or no flank) -> -0.3 (targeted a real
    # cell, better than out-of-range).
    assert game.score_move(board, (0, 0), "B") == -0.3  # empty corner, no flank
    assert game.score_move(board, (2, 2), "B") == -0.3  # occupied cell
    # Every illegal tier must differ from both the no-call tier (-1.0) and each
    # other, so GSPO group-relative advantage is non-zero when outcomes differ.
    tiers = {-1.0, -0.5, -0.3, 0.4}
    assert len(tiers) == 4

    # A move that fills the last empty cell and ends the game with the mover
    # ahead -> 1.0.
    near_full = game.normalize_board(
        [["B" if not (r == 0 and c == 0) else "." for c in range(6)] for r in range(6)]
    )
    # Find a mover+target whose sole legal move on a board that becomes terminal
    # leaves them ahead. Construct: place W so B at (0,0) flips it and wins.
    # Simpler: directly assert on a hand-built terminal-winning board state.
    board_w = game.normalize_board(
        [
            [".", "W", "B", "B", "B", "B"],
            ["B", "B", "B", "B", "B", "B"],
            ["B", "B", "B", "B", "B", "B"],
            ["B", "B", "B", "B", "B", "B"],
            ["B", "B", "B", "B", "B", "B"],
            ["B", "B", "B", "B", "B", "B"],
        ]
    )
    # Black playing (0,0) brackets the W at (0,1) against the B at (0,2).
    assert (0, 0) in game.legal_moves(board_w, "B")
    after = game.apply_move(board_w, 0, 0, "B")
    assert game.is_terminal(after)
    assert game.score_board(after)["winner"] == "B"
    assert game.score_move(board_w, (0, 0), "B") == 1.0


# --------------------------------------------------------------------------- #
# Parsing and reward
# --------------------------------------------------------------------------- #


def test_parse_move_and_parse_tool_move_handle_malformed():
    game = _load_module("game")

    assert game.parse_move("play <move>2,3</move> now") == (2, 3)
    # Last tag wins.
    assert game.parse_move("x <move>5,5</move> y <move>0,1</move>") == (0, 1)
    # Out-of-range coordinates -> None.
    assert game.parse_move("<move>9,9</move>") is None
    # No move tag -> None (never raises).
    assert game.parse_move("I have no idea") is None

    assert game.parse_tool_move([{"name": "choose_move", "arguments": {"row": 1, "col": 2}}]) == (1, 2)
    # JSON-string arguments.
    assert game.parse_tool_move(
        [{"name": "choose_move", "arguments": json.dumps({"row": 4, "col": 0})}]
    ) == (4, 0)
    # Wrong tool, missing fields, out of range, malformed JSON -> None.
    assert game.parse_tool_move([{"name": "other", "arguments": {"row": 1, "col": 2}}]) is None
    assert game.parse_tool_move([{"name": "choose_move", "arguments": {"row": 1}}]) is None
    assert game.parse_tool_move([{"name": "choose_move", "arguments": {"row": 9, "col": 9}}]) is None
    assert game.parse_tool_move([{"name": "choose_move", "arguments": "{bad json"}]) is None
    assert game.parse_tool_move([]) is None
    assert game.parse_tool_move(None) is None


def test_reward_fn_scores_tool_move_only():
    game = _load_module("game")
    reward = _load_module("reward")
    board = game.new_board()
    record = SimpleNamespace(
        source_record={"board": board, "player": "B"},
        completion="<move>1,2</move>",
        tool_calls=[{"name": "choose_move", "arguments": {"row": 0, "col": 0}}],
    )
    # (0,0) is in range but illegal (no flank) -> -0.3.
    assert reward.reward_fn(record) == -0.3

    # Out-of-range coordinate via the tool call -> -0.5 (still emitted a call).
    record.tool_calls = [{"name": "choose_move", "arguments": {"row": 9, "col": 9}}]
    assert reward.reward_fn(record) == -0.5

    record.tool_calls = [{"name": "choose_move", "arguments": {"row": 1, "col": 2}}]
    # (1,2) is a legal non-terminal opening move -> +0.4.
    assert (1, 2) in game.legal_moves(board, "B")
    assert reward.reward_fn(record) == 0.4

    # Malformed / missing tool call -> no parseable coordinates -> -1.0.
    record.tool_calls = [{"name": "choose_move", "arguments": {"row": 1}}]
    assert reward.reward_fn(record) == -1.0
    record.tool_calls = []
    assert reward.reward_fn(record) == -1.0


# --------------------------------------------------------------------------- #
# Dataset generator / loader
# --------------------------------------------------------------------------- #


def test_generator_is_reproducible_and_produces_reachable_openings():
    generator = _load_module("dataset_generator")
    game = _load_module("game")

    rows = generator.generate_records(16, seed=7)

    assert rows == generator.generate_records(16, seed=7)
    assert len(rows) == 16
    for record in rows:
        board = game.normalize_board(record["board"])
        assert game.next_player(board) == "B"
        assert not game.is_terminal(board)
        assert game.legal_moves(board, "B")


def test_generator_rejects_invalid_args():
    generator = _load_module("dataset_generator")

    for bad in [dict(count=0, seed=1), dict(count=5, seed=-1), dict(count=5, seed=1, max_plies=-1)]:
        try:
            generator.generate_records(**bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


def test_loader_formats_records_from_jsonl(tmp_path):
    loader = _load_module("dataset_loader")
    generator = _load_module("dataset_generator")
    game = _load_module("game")

    rows = generator.generate_records(8, seed=3)
    path = tmp_path / "boards.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    records = loader.load_training_dataset(str(path))

    assert len(records) == len(rows) == 8
    for raw, record in zip(rows, records):
        board = game.normalize_board(record["board"])
        assert record["id"] == raw["id"]
        assert record["player"] == "B"
        assert record["valid_moves"] == game.legal_moves(board, "B")
        assert record["prompt"].startswith("You are playing 6x6 Othello")
        assert record["prompt"].endswith("Move:")


def test_loader_falls_back_to_generator_when_missing(tmp_path):
    loader = _load_module("dataset_loader")

    records = loader.load_training_dataset(str(tmp_path / "does-not-exist.jsonl"))

    assert records
    assert all("prompt" in record and "board" in record for record in records)


# --------------------------------------------------------------------------- #
# Agent fn contract + seeded-opponent harness
# --------------------------------------------------------------------------- #


def test_run_agent_signature_accepts_two_positional_args():
    with _stubbed_agentic():
        run_agent = _load_module("run_agent")

    sig = inspect.signature(run_agent.run_agent)
    positionals = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert len(positionals) == 2


def test_choose_move_tool_schema_is_closed_and_bounded():
    with _stubbed_agentic():
        run_agent = _load_module("run_agent")

    tool = run_agent.CHOOSE_MOVE_TOOL["function"]
    params = tool["parameters"]
    assert tool["name"] == "choose_move"
    assert params["additionalProperties"] is False
    assert params["required"] == ["row", "col"]
    assert params["properties"]["row"]["minimum"] == 0
    assert params["properties"]["row"]["maximum"] == 5
    assert params["properties"]["col"]["minimum"] == 0
    assert params["properties"]["col"]["maximum"] == 5


def test_run_agent_builds_trajectories_with_fake_client():
    import sys as _sys

    saved_agentic = _sys.modules.get("areno.api.agentic")
    saved_openai = _sys.modules.get("openai")
    saved_httpx = _sys.modules.get("httpx")
    try:
        _sys.modules["areno.api.agentic"] = SimpleNamespace(
            AgentTrajectory=lambda turns: SimpleNamespace(turns=turns),
            AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
        )

        class FakeCompletions:
            def __init__(self):
                self.requests = []

            async def create(self, **kwargs):
                self.requests.append(kwargs)
                message = SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-0",
                            type="function",
                            function=SimpleNamespace(
                                name="choose_move", arguments=json.dumps({"row": 1, "col": 2})
                            ),
                        )
                    ],
                )
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        class FakeChat:
            def __init__(self):
                self.completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = FakeChat()

            async def close(self):
                pass

        class FakeLimits:
            def __init__(self, **kwargs):
                pass

        class FakeTimeout:
            def __init__(self, *args, **kwargs):
                pass

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        _sys.modules["openai"] = SimpleNamespace(AsyncOpenAI=FakeOpenAI)
        _sys.modules["httpx"] = SimpleNamespace(
            AsyncClient=FakeAsyncClient, Limits=FakeLimits, Timeout=FakeTimeout
        )

        run_agent = _load_module("run_agent")

        class FakeCtx:
            max_running_prompts = 8
            api_key = "k"

            def get_base_url(self):
                return "http://localhost:8000/v1"

        class FakeBatch:
            def __init__(self, prompts):
                self._items = [SimpleNamespace(prompt=p, record={"board": None}) for p in prompts]

            def iter_samples(self):
                return iter(self._items)

        trajectory = asyncio.run(run_agent.run_agent(FakeCtx(), FakeBatch(["p0", "p1"])))

        turns = trajectory.turns
        assert len(turns) == 2
        assert turns[0].item.prompt == "p0"
        assert turns[1].item.prompt == "p1"
        # Both turns forced the choose_move tool choice.
        assert all(turn.tool_choice == {"type": "function", "function": {"name": "choose_move"}} for turn in turns)
    finally:
        for name, saved in (("areno.api.agentic", saved_agentic), ("openai", saved_openai), ("httpx", saved_httpx)):
            if saved is not None:
                _sys.modules[name] = saved
            else:
                _sys.modules.pop(name, None)


def test_evaluate_reports_win_rate_and_invalid_move_rate():
    opponent = _load_module("opponent")
    game = _load_module("game")

    def random_policy(board, player, rng):
        return opponent.random_opponent_pick(board, player, rng)

    report = opponent.evaluate(random_policy, n_games=20, seed=2026, max_steps=60)

    summary = report["summary"]
    assert summary["n_games"] == 20
    assert 0.0 <= summary["win_rate"] <= 1.0
    assert 0.0 <= summary["invalid_move_rate"] <= 1.0
    assert summary["policy_side"] == "B"
    assert len(report["games"]) == 20
    for match in report["games"]:
        assert "winner" in match
        assert match["winner"] in ("B", "W", "draw")
        assert {"black", "white", "steps", "passes", "invalid"}.issubset(match.keys())


def test_play_match_rejects_invalid_args():
    opponent = _load_module("opponent")

    def stub(board, player, rng):
        return None

    for bad in [dict(max_steps=0), dict(max_steps=5, seed=-1)]:
        try:
            opponent.play_match(stub, **bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


def test_help_listed_examples_module_smoke():
    # A backward-compat sanity check: the example directory exposes the three
    # required entry symbols. ``run_agent`` imports ``areno.api.agentic`` at
    # module top, so load it under the same stub used by the integration test.
    for name, symbol in (("dataset_loader", "load_training_dataset"), ("reward", "reward_fn")):
        module = _load_module(name)
        assert callable(getattr(module, symbol, None)), f"{name}.py missing {symbol}"
    with _stubbed_agentic():
        run_agent = _load_module("run_agent")
        assert callable(getattr(run_agent, "run_agent", None))