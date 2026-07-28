from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "2048"

# Sibling module names these example files import as a side effect of loading.
# We snapshot and restore them so a 2048 `import game` cannot leak into other
# example tests (tictactoe/shopping also `import game`) run later in the session.
_SIBLING_MODULES = ("game", "dataset_generator", "dataset_loader")


def _load_module(name: str):
    """Load a 2048 example module by absolute file path (mirrors runtime loading)."""

    saved_path = list(sys.path)
    saved_modules = {key: sys.modules.get(key) for key in _SIBLING_MODULES if key in sys.modules}
    for key in saved_modules:
        sys.modules.pop(key, None)
    try:
        path = EXAMPLE_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"agentic_2048_{name}_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_sibling_globals(saved_path, saved_modules)


def _load_module_without_sys_path(name: str):
    """Load a module the way ``--reward-fn-path`` / ``--agent-fn`` do, without the
    example dir pre-populated on ``sys.path`` — the file must self-bootstrap."""

    saved_path = list(sys.path)
    saved_modules = {key: sys.modules.get(key) for key in _SIBLING_MODULES if key in sys.modules}
    for key in saved_modules:
        sys.modules.pop(key, None)
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != EXAMPLE_DIR.resolve()]
    try:
        path = EXAMPLE_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"agentic_2048_{name}_nosyspath", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_sibling_globals(saved_path, saved_modules)


def _restore_sibling_globals(saved_path: list[str], saved_modules: dict[str, object]) -> None:
    sys.path[:] = saved_path
    for key in _SIBLING_MODULES:
        if key in saved_modules:
            sys.modules[key] = saved_modules[key]
        else:
            sys.modules.pop(key, None)


# --- success path -------------------------------------------------------------


def test_generator_produces_valid_starting_boards():
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    records = generator.generate_records(16, seed=7)

    assert len(records) == 16
    seen = set()
    for record in records:
        board = game.normalize_board(record["board"])
        key = tuple(cell for row in board for cell in row)
        assert key not in seen
        seen.add(key)
        assert not game.is_terminal(board)
        assert game.legal_moves(board)
        assert isinstance(record["seed"], int)
        assert record["random_baseline"]["score"] >= 0
        assert record["id"].startswith("generated-")


# --- merge edge cases ---------------------------------------------------------


def test_slide_row_left_merge_edge_cases():
    game = _load_module("game")

    def left_row(cells):
        board = [list(cells)] + [[0, 0, 0, 0] for _ in range(3)]
        new_board, score, changed = game.slide(board, "left")
        return new_board[0], score, changed

    row, score, changed = left_row([2, 2, 2, 2])
    assert row == [4, 4, 0, 0] and score == 8 and changed

    row, score, changed = left_row([2, 2, 2, 0])
    assert row == [4, 2, 0, 0] and score == 4 and changed  # one merge per move per line

    row, score, changed = left_row([4, 4, 2, 2])
    assert row == [8, 4, 0, 0] and score == 12 and changed

    row, score, changed = left_row([2, 0, 2, 4])
    assert row == [4, 4, 0, 0] and score == 4 and changed

    row, score, changed = left_row([2, 4, 0, 0])
    assert row == [2, 4, 0, 0] and score == 0 and not changed  # no-op


def test_slide_all_four_directions():
    game = _load_module("game")

    # right merges toward the right edge
    board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    new_board, score, changed = game.slide(board, "right")
    assert new_board[0] == [0, 0, 0, 4] and score == 4 and changed

    # up merges a column toward the top
    board = [[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    new_board, score, changed = game.slide(board, "up")
    assert new_board[0][0] == 4 and new_board[1][0] == 0 and score == 4 and changed

    # down merges a column toward the bottom
    board = [[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    new_board, score, changed = game.slide(board, "down")
    assert new_board[3][0] == 4 and new_board[0][0] == 0 and score == 4 and changed


# --- seeded replay determinism ------------------------------------------------


def test_play_episode_is_deterministic_under_seed():
    game = _load_module("game")
    board = [[2, 0, 0, 0], [0, 0, 0, 2], [4, 0, 2, 0], [0, 8, 0, 0]]
    moves = ["left", "up", "right", "down", "left", "up"]

    r1 = game.play_episode(board, moves, seed=42, cap=32)
    r2 = game.play_episode(board, moves, seed=42, cap=32)
    assert (r1.board, r1.score, r1.max_tile, r1.invalid_moves) == (
        r2.board,
        r2.score,
        r2.max_tile,
        r2.invalid_moves,
    )


def test_different_seeds_can_diverge():
    game = _load_module("game")
    board = [[2, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]]
    moves = ["left", "up", "right", "down", "left"]
    outcomes = {
        tuple(cell for row in game.play_episode(board, moves, seed=seed, cap=32).board for cell in row)
        for seed in range(1, 7)
    }
    assert len(outcomes) > 1  # stochastic spawns produce divergent boards


# --- episode cap --------------------------------------------------------------


def test_episode_cap_truncates_long_sequences():
    game = _load_module("game")
    board = [[2, 0, 0, 0], [0, 0, 0, 2], [4, 0, 2, 0], [0, 8, 0, 0]]

    result = game.play_episode(board, ["left"] * 100, seed=1, cap=5)
    assert result.total_moves <= 5
    assert result.truncated

    result_zero = game.play_episode(board, ["left", "right"], seed=1, cap=0)
    assert result_zero.total_moves == 0


# --- random baseline ----------------------------------------------------------


def test_random_baseline_returns_sane_metrics():
    game = _load_module("game")
    board = [[2, 0, 0, 0], [0, 0, 0, 2], [4, 0, 2, 0], [0, 8, 0, 0]]

    summary = game.random_episode(board, seed=2026, cap=32, trials=4)

    assert summary["score"] >= 0
    assert 0 <= summary["invalid_rate"] <= 1
    assert summary["max_tile"] >= 2
    assert summary["trials"] == 4


# --- reward success / invalid / boundary --------------------------------------


def _record(moves=None, completion="", tool_name="choose_moves", arguments=None):
    if arguments is None:
        arguments = {"moves": moves or []}
    tool_calls = [{"name": tool_name, "arguments": arguments}] if tool_name else []
    source = {
        "board": [[2, 0, 0, 0], [0, 0, 0, 2], [4, 0, 2, 0], [0, 8, 0, 0]],
        "seed": 42,
        "random_baseline": {"score": 50.0},
        "id": "test-board-1",
    }
    return SimpleNamespace(source_record=source, completion=completion, tool_calls=tool_calls)


def test_reward_success_returns_float_and_logs_metrics(caplog):
    reward = _load_module("reward")
    game = _load_module("game")

    # score_moves logs on the game module logger; capture at root so the log is
    # caught regardless of which `import game` instance reward.py resolved to.
    with caplog.at_level(logging.INFO):
        value = reward.reward_fn(_record(moves=["left", "up", "right", "down"]))

    assert isinstance(value, float)
    log_text = "\n".join(record.message for record in caplog.records)
    assert "score=" in log_text
    assert "max_tile=" in log_text
    assert "invalid_rate=" in log_text
    assert "improvement=" in log_text
    assert game.INVALID_PENALTY >= 0  # sanity


def test_reward_empty_moves_are_penalized():
    reward = _load_module("reward")

    value = reward.reward_fn(_record(moves=[]))

    # No episode played: score 0 minus baseline 50, no invalid penalty.
    assert value == -50.0


def test_reward_wrong_tool_falls_back_to_completion():
    reward = _load_module("reward")

    value = reward.reward_fn(_record(tool_name="other_tool", completion="left then right"))
    assert isinstance(value, float)  # parsed and replayed without raising


def test_reward_malformed_json_arguments_does_not_raise():
    reward = _load_module("reward")

    value = reward.reward_fn(_record(arguments="{not valid json"))
    assert value == -50.0  # falls through to empty -> baseline-only penalty


def test_reward_out_of_enum_tokens_filtered():
    reward = _load_module("reward")

    # "diagonal" is dropped; remaining legal moves are replayed.
    value = reward.reward_fn(_record(moves=["left", "diagonal", "up"]))
    assert isinstance(value, float)


def test_reward_all_noop_moves_counted_as_invalid():
    reward = _load_module("reward")
    game = _load_module("game")
    # A board where left does nothing (row [2,4] already compressed) on the top
    # row, but we need ALL moves no-op: use a full, unmergeable board.
    board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 128, 256]]
    source = {"board": board, "seed": 1, "random_baseline": {"score": 0.0}, "id": "full"}
    record = SimpleNamespace(
        source_record=source,
        completion="",
        tool_calls=[{"name": "choose_moves", "arguments": {"moves": ["left", "right", "up", "down"]}}],
    )
    value = reward.reward_fn(record)
    result_moves = 4
    assert value == 0.0 - game.INVALID_PENALTY * result_moves


# --- XML no-tool variant ------------------------------------------------------


def test_reward_no_tool_requires_moves_tag():
    reward_no_tool = _load_module("reward_no_tool")

    # No <moves> tag -> empty move list -> baseline-only penalty (no raising).
    record = _record(tool_name=None, completion="I think up and left")
    assert reward_no_tool.reward_fn(record) == -50.0

    # A real moves tag is parsed and replayed to a float reward.
    record = _record(tool_name=None, completion="...<moves>left,up,right,down</moves>...")
    value = reward_no_tool.reward_fn(record)
    assert isinstance(value, float)


def test_no_tool_loader_emits_xml_prompt(tmp_path):
    loader = _load_module("dataset_loader_no_tool")
    generator = _load_module("dataset_generator")

    records = generator.generate_records(2, seed=2026, cap=8, trials=2)
    path = tmp_path / "boards.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        generator.write_jsonl(records, handle)

    loaded = loader.load_training_dataset(str(path))
    assert len(loaded) == 2
    for record in loaded:
        assert "<moves>" in record["prompt"]
        assert record["legal_moves"]


# --- parsing ------------------------------------------------------------------


def test_parse_moves_handles_dict_list_and_string():
    game = _load_module("game")

    assert game.parse_moves({"moves": ["Up", "left", "diagonal", "right"]}) == ["up", "left", "right"]
    assert game.parse_moves(["up", "down", "sideways"]) == ["up", "down"]
    assert game.parse_moves("go left then right? down") == ["left", "right", "down"]
    assert game.parse_moves(None) == []
    assert game.parse_moves({"foo": 1}) == []


def test_parse_xml_moves_extracts_last_tag_and_filters():
    game = _load_module("game")

    assert game.parse_xml_moves("<moves>up,left,down</moves>") == ["up", "left", "down"]
    assert game.parse_xml_moves("noise <moves>Up, diagonal, right</moves> tail") == ["up", "right"]
    # last tag wins, mirroring tictactoe's parse_xml_move behavior
    assert game.parse_xml_moves("<moves>up</moves> then <moves>left,right</moves>") == ["left", "right"]
    assert game.parse_xml_moves("no tag here") == []
    assert game.parse_xml_moves("<moves>up,left</moves><|im_end|>") == ["up", "left"]


# --- file loading contract (no sys.path pre-set) ------------------------------


def test_reward_file_loads_without_sys_path():
    reward = _load_module_without_sys_path("reward")
    assert callable(reward.reward_fn)
    value = reward.reward_fn(_record(moves=["left"]))
    assert isinstance(value, float)


def test_agent_files_compile_without_sys_path():
    # run_agent[_no_tool] import areno.api.agentic (torch-only at runtime);
    # validate the files parse and expose run_agent without importing heavy deps.
    for name in ("run_agent.py", "run_agent_no_tool.py"):
        path = EXAMPLE_DIR / name
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        assert "async def run_agent" in source


# --- backward compatibility / additivity --------------------------------------


def test_existing_examples_unaffected():
    # The 2048 example is purely additive; a sibling example still loads.
    tictactoe_dir = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "tictactoe" / "game.py"
    spec = importlib.util.spec_from_file_location("tictactoe_game_for_2048_test", tictactoe_dir)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.normalize_board([["X", ".", "."], [".", ".", "."], [".", ".", "."]])[0][0] == "X"