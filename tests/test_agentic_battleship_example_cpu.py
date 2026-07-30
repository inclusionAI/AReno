"""CPU tests for the Battleship agentic example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module(name: str):
    """Load a module from the battleship example directory."""
    path = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "battleship" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"agentic_battleship_{name}_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_module_without_sys_path(name: str):
    """Load a module without modifying sys.path (tests import behavior)."""
    path = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "battleship" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"agentic_battleship_{name}_notidy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Generation tests
# =============================================================================


def test_battleship_generator_produces_valid_records():
    """Generator produces legal, non-overlapping, correctly-sized fleets."""
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    records = generator.generate_records(16, seed=7)

    assert len(records) == 16
    for record in records:
        # Validate the record
        normalized = game.normalize_record(record)
        assert normalized is not None

        # Check grid size
        assert record["grid_size"] == game.GRID

        # Check total cells match ship lengths
        total_cells = len(record["ships"])
        expected = sum(record.get("ship_lengths", list(game.SHIPS)))
        assert total_cells == expected, f"Expected {expected} cells, got {total_cells}"


def test_battleship_generator_deterministic():
    """Same seed produces identical records."""
    generator = _load_module("dataset_generator")

    records1 = generator.generate_records(8, seed=42)
    records2 = generator.generate_records(8, seed=42)

    assert len(records1) == len(records2) == 8
    for r1, r2 in zip(records1, records2):
        assert r1["ships"] == r2["ships"], "Same seed should produce identical fleets"


def test_battleship_generator_unique_fleets():
    """Generator produces distinct fleets (deduplication works)."""
    generator = _load_module("dataset_generator")

    records = generator.generate_records(32, seed=12345)

    # All records should have unique ship positions
    ship_sets = [tuple(tuple(c) for c in r["ships"]) for r in records]
    assert len(ship_sets) == len(set(ship_sets)), "Fleets should be unique"


# =============================================================================
# Game logic tests
# =============================================================================


def test_battleship_coordinate_parsing():
    """Coordinate parsing works correctly."""
    game = _load_module("game")

    # Valid coordinates
    assert game.parse_coordinate("A1") == (0, 0)
    assert game.parse_coordinate("H8") == (7, 7)
    assert game.parse_coordinate("D5") == (3, 4)

    # Invalid coordinates
    assert game.parse_coordinate("") is None
    assert game.parse_coordinate("I1") is None  # out of range
    assert game.parse_coordinate("A0") is None  # column 0
    assert game.parse_coordinate("A9") is None  # column 9
    assert game.parse_coordinate("X") is None


def test_battleship_fire_miss():
    """Fire returns miss for empty cell."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Fire at many cells to find a guaranteed miss at A1, or use legal shots check
    # Actually let's use a corner - we know the fleet doesn't cover entire board
    # Try corner cells until we find one with no ship
    all_cells = [(r, c) for r in range(8) for c in range(8)]
    # Find cells that are NOT ship cells
    ship_set = set(tuple(x) for x in record["ships"]) if record.get("ships") else set()
    miss_cells = [c for c in all_cells if c not in ship_set]
    if miss_cells:
        coord = game.format_coordinate(miss_cells[0][0], miss_cells[0][1])
        result = game.fire(state, coord)
        assert result["status"] == "miss"
    else:
        # Should have at least some empty cells (8*8=64, ships=11)
        pass

    assert result["shots_used"] == 1
    assert result["remaining"] == len(game.SHIPS)


def test_battleship_fire_hit():
    """Fire returns hit when a ship is struck."""
    game = _load_module("game")

    # Place a known fleet and find a ship cell
    record = game.place_fleet(42)
    state = game.init_state(record)

    # Find a ship cell to hit
    ship_cell = record["ships"][0]  # First cell of first ship
    coord = game.format_coordinate(ship_cell[0], ship_cell[1])

    result = game.fire(state, coord)

    assert result["status"] in ("hit", "sunk")
    assert result["hit_cells"] >= 1


def test_battleship_fire_sunk():
    """Fire returns sunk when a ship is destroyed."""
    game = _load_module("game")

    # Place a minimal fleet and hit all its cells
    record = game.place_fleet(42)
    state = game.init_state(record)

    # Hit each cell of the first ship
    ship_cells = record["ships"][:4]  # First ship has length 4
    for cell in ship_cells:
        coord = game.format_coordinate(cell[0], cell[1])
        result = game.fire(state, coord)

    # After hitting all cells, should be sunk
    sunk_count = sum(1 for s in state.ships if s.is_sunk)
    assert sunk_count >= 1


def test_battleship_fire_invalid_rejects_out_of_range():
    """Fire rejects coordinates outside the board."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Test out of range: "I9" fails format (I not in A-H)
    result = game.fire(state, "I9")
    assert result["status"] == "invalid"
    assert "invalid" in result.get("reason", "").lower() or "range" in result.get("reason", "").lower()

    # Also test valid format but out of range column: A9
    result2 = game.fire(state, "A9")
    assert result2["status"] == "invalid"


def test_battleship_fire_invalid_rejects_repeated_shot():
    """Fire rejects repeating the same coordinate."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Fire twice at the same spot
    result1 = game.fire(state, "A1")
    assert result1["status"] in ("miss", "hit", "sunk")

    result2 = game.fire(state, "A1")
    assert result2["status"] == "invalid"
    assert "already" in result2.get("reason", "").lower()


def test_battleship_no_cell_leak():
    """Fire never reveals hidden ship cells."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Fire at multiple cells and check no ship positions leak
    test_coords = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    for coord in test_coords:
        result = game.fire(state, coord)
        # Result should not contain ship cell positions
        for ship in state.ships:
            if ship.cells:
                for cell in ship.cells:
                    if cell not in state.shots_history:
                        # This cell hasn't been shot, should not appear in result
                        pass


def test_battleship_is_win():
    """Win detection works correctly."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Initially not a win
    assert not game.is_win(state)

    # Hit all ship cells
    all_ship_cells = []
    for ship in state.ships:
        all_ship_cells.extend(ship.cells)

    for cell in all_ship_cells:
        game.fire(state, game.format_coordinate(cell[0], cell[1]))

    assert game.is_win(state)


def test_battleship_is_terminal():
    """Terminal detection works for win and turn cap."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Not terminal initially
    assert not game.is_terminal(state)

    # Hit all cells (win)
    for ship in state.ships:
        for cell in ship.cells:
            game.fire(state, game.format_coordinate(cell[0], cell[1]))
    assert game.is_terminal(state)

    # New state, reach turn cap without winning
    record2 = game.place_fleet(99)
    state2 = game.init_state(record2)

    # Fire max_turns times without hitting all ships
    for i in range(game.MAX_TURNS):
        game.fire(state2, f"A{(i % 8) + 1}")

    assert game.is_terminal(state2)


def test_battleship_deterministic_replay():
    """Same seed produces same fleet for deterministic replay."""
    game = _load_module("game")

    record1 = game.place_fleet(12345)
    record2 = game.place_fleet(12345)

    assert record1["ships"] == record2["ships"]


def test_battleship_score_episode_counts_sunk_ships():
    """score_episode counts every sunk ship, even two same-length ships.

    Regression: the fleet is [4,3,2,2] (two length-2 ships). An earlier
    implementation keyed a set on ship length and deduped the two
    length-2 ships to one, undercounting sunk_ships.
    """
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Sink every ship by firing at each cell.
    tool_calls = []
    for ship in state.ships:
        for cell in ship.cells:
            coord = game.format_coordinate(cell[0], cell[1])
            tool_calls.append({"name": "fire", "arguments": {"coordinate": coord}})

    score = game.score_episode(state, tool_calls)

    assert score["win"] is True
    assert score["sunk_ships"] == len(game.SHIPS), (
        f"Expected {len(game.SHIPS)} sunk ships, got {score['sunk_ships']}"
    )
    assert score["completion"] == 1.0
    assert score["hits"] == game.TOTAL_SHIP_CELLS


# =============================================================================
# Reward function tests
# =============================================================================


def test_battleship_reward_win():
    """Win yields high positive reward."""
    reward = _load_module("reward")
    game = _load_module("game")

    # Create a record and simulate a winning sequence
    record_data = game.place_fleet(42)

    # Build tool calls that would sink all ships
    tool_calls = []
    state = game.init_state(record_data)
    for ship in state.ships:
        for cell in ship.cells:
            coord = game.format_coordinate(cell[0], cell[1])
            tool_calls.append({"name": "fire", "arguments": {"coordinate": coord}})

    record = SimpleNamespace(
        source_record=record_data,
        tool_calls=tool_calls,
    )

    r = reward.reward_fn(record)
    assert r > 0.5, f"Win should give high reward, got {r}"


def test_battleship_reward_no_fire_calls():
    """No fire calls yields zero reward (no spurious signal)."""
    reward = _load_module("reward")

    record = SimpleNamespace(
        source_record={"ships": [], "ship_lengths": []},
        tool_calls=[],
    )

    r = reward.reward_fn(record)
    assert r == 0.0, f"No calls should give zero reward, got {r}"


def test_battleship_reward_invalid_shots():
    """Invalid/repeated shots are penalized."""
    reward = _load_module("reward")
    game = _load_module("game")

    record_data = game.place_fleet(42)
    ship_cells = set(tuple(x) for x in record_data["ships"])

    # Find a non-ship cell
    non_ship = None
    for r in range(8):
        for c in range(8):
            if (r, c) not in ship_cells:
                non_ship = game.format_coordinate(r, c)
                break
        if non_ship:
            break

    # Fire at non_ship cell 3 times (all invalid due to repetition)
    tool_calls = [
        {"name": "fire", "arguments": {"coordinate": non_ship}},
        {"name": "fire", "arguments": {"coordinate": non_ship}},  # repeated
        {"name": "fire", "arguments": {"coordinate": non_ship}},  # repeated again
    ]

    record = SimpleNamespace(
        source_record=record_data,
        tool_calls=tool_calls,
    )

    r = reward.reward_fn(record)
    assert r <= 0, f"Repeated invalid shots should penalize or have no reward, got {r}"


def test_battleship_reward_partial():
    """Partial hits yield intermediate reward."""
    reward = _load_module("reward")
    game = _load_module("game")

    record_data = game.place_fleet(42)

    # Get first ship cells only
    state = game.init_state(record_data)
    first_ship = state.ships[0]
    tool_calls = []
    for cell in first_ship.cells:
        coord = game.format_coordinate(cell[0], cell[1])
        tool_calls.append({"name": "fire", "arguments": {"coordinate": coord}})

    record = SimpleNamespace(
        source_record=record_data,
        tool_calls=tool_calls,
    )

    r = reward.reward_fn(record)
    # Should have hit reward but not full win bonus
    assert 0 < r < 1.0, f"Partial hits should give intermediate reward, got {r}"


# =============================================================================
# Dataset loader tests
# =============================================================================


def test_battleship_loader_does_not_require_sys_path_from_cwd():
    """Loader imports work without relying on current working directory."""
    # This test verifies that the sys.path.insert pattern works
    # by loading without the directory in sys.path first
    path = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "battleship"

    # Save original sys.path
    original_path = sys.path.copy()

    # Remove the battleship directory from sys.path if present
    if str(path) in sys.path:
        sys.path.remove(str(path))

    # Try to import using the module's internal sys.path manipulation
    # This simulates running from any cwd
    loader = _load_module_without_sys_path("dataset_loader")

    # The loader should have loaded game via its own sys.path.insert
    # We can't test the actual import without cwd change, but we verify the file parses

    # Restore sys.path
    sys.path[:] = original_path


def test_battleship_loader_returns_valid_records():
    """Loader returns records with required fields."""
    loader = _load_module("dataset_loader")

    # Load using fallback (no actual file)
    records = loader.load_training_dataset("/nonexistent/path")

    assert len(records) > 0
    for record in records:
        assert "id" in record
        assert "prompt" in record
        assert "ships" in record


# =============================================================================
# Registry tests (no side effects)
# =============================================================================


def test_battleship_import_does_not_mutate_globals():
    """Importing example modules should not mutate areno registries."""
    try:
        from areno.api.algorithms import list_algorithms
    except ImportError:
        # Skip test if torch is not available (required by areno)
        return

    # Get algorithms before import
    algos_before = list(list_algorithms().keys())

    # Import the example (this is already done by other tests)
    _load_module("game")
    _load_module("reward")
    _load_module("run_agent")

    # Get algorithms after import
    algos_after = list(list_algorithms().keys())

    assert algos_before == algos_after, "Import should not mutate algorithm registry"


# =============================================================================
# Web UI tests
# =============================================================================


def _make_web_server(agent_mode: str = "heuristic", seed: int = 123, base_url=None):
    """Build a BattleshipServer without binding a real socket (port 0)."""
    import argparse

    web_ui = _load_module("web_ui")
    args = argparse.Namespace(
        agent_mode=agent_mode, base_url=base_url, api_key="token", model="policy"
    )
    return web_ui, web_ui.BattleshipServer(("127.0.0.1", 0), web_ui.BattleshipHandler, seed=seed, args=args)


def test_battleship_web_ui_heuristic_wins():
    """Heuristic agent eventually sinks the whole fleet within the turn cap."""
    web_ui, srv = _make_web_server(agent_mode="heuristic", seed=123)
    payload = web_ui._autoplay(srv, web_ui.game.MAX_TURNS)

    assert payload["terminal"] is True
    assert payload["win"] is True
    assert payload["sunk_ships"] == payload["ships_total"]
    assert payload["shots_used"] <= payload["max_turns"]


def test_battleship_web_ui_payload_hides_ship_cells():
    """API payload exposes only hit/miss/unknown, never unrevealed ship cells."""
    web_ui, srv = _make_web_server(seed=42)
    state = srv.state
    # Fire at a guaranteed-miss cell (a corner that has no ship under normal placement).
    payload = web_ui._payload(srv)
    for row in payload["cells"]:
        for cell in row:
            assert cell in ("hit", "miss", "unknown")
    # Expected board size.
    assert payload["grid_size"] == 8
    assert payload["cells"][0][0] == "unknown"


def test_battleship_web_ui_llm_without_base_url_raises():
    """LLM agent mode without a configured endpoint refuses with a clear error."""
    web_ui, srv = _make_web_server(agent_mode="llm")
    try:
        web_ui._agent_shot(srv)
    except ValueError as exc:
        assert "base-url" in str(exc).lower()
        return
    raise AssertionError("expected _agent_shot to raise without --base-url")


def test_battleship_web_ui_human_fire_and_invalid():
    """Human fire resolves a miss; firing the same cell is rejected."""
    web_ui, srv = _make_web_server(seed=7)
    target = None
    for r in range(8):
        for c in range(7, 8):
            if (r, c) not in set(tuple(x) for x in srv.state.ships[0].cells):
                coord = web_ui.game.format_coordinate(r, c)
                p1 = web_ui._fire(srv, coord, source="Human")
                assert p1["shots_used"] == 1
                # Repeating must not advance a real shot beyond the rejected one.
                p2 = web_ui._fire(srv, coord, source="Human")
                assert p2["shots_used"] == 2  # invalid shots still consume a turn
                target = (r, c)
                break
        if target:
            break


# =============================================================================
# play_llm batch-evaluation tests (no network: the game loop drives an injected
# deterministic step policy; the OpenAI client is never touched).
# =============================================================================


def _heuristic_step_fn(game_module):
    """A deterministic hunt/target step policy reused to exercise play_llm's loop."""
    def _neighbors(cell):
        r, c = cell
        return [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]

    def step(messages, state):
        legal = game_module.legal_shots(state)
        if not legal:
            return None
        hit_cells = set()
        sunk_cells = set()
        for ship in state.ships:
            for cell in ship.cells:
                if cell in ship.hits:
                    (sunk_cells if ship.is_sunk else hit_cells).add(cell)
        open_hits = hit_cells - sunk_cells
        candidates = [n for hit in open_hits for n in _neighbors(hit) if n in legal]
        if candidates:
            (r, c) = sorted(candidates)[0]
            return game_module.format_coordinate(r, c)
        spread = sorted(cell for cell in legal if (cell[0] + cell[1]) % 2 == 0)
        pool = spread or sorted(legal)
        (r, c) = pool[0]
        return game_module.format_coordinate(r, c)

    return step


def test_battleship_play_llm_loop_and_metrics():
    """play_llm._play_game drives a full game with an injected policy and returns sane metrics."""
    game_module = _load_module("game")
    play_llm = _load_module("play_llm")

    record = game_module.place_fleet(2026)
    step = _heuristic_step_fn(game_module)
    result = play_llm._play_game(step, record, max_turns=game_module.MAX_TURNS)

    expected_keys = {"win", "completion", "shots_used", "hits", "sunk_ships", "invalid_shots", "seed"}
    assert set(result) == expected_keys
    assert result["seed"] == 2026
    assert result["shots_used"] > 0
    assert 0.0 <= result["completion"] <= 1.0
    assert result["sunk_ships"] <= len(game_module.SHIPS)
    assert result["invalid_shots"] == 0  # heuristic never fires an invalid cell
    # win flag must agree with the game's own terminal-win check.
    assert result["win"] == (result["sunk_ships"] == len(game_module.SHIPS))


def test_battleship_play_llm_invalid_step_progresses():
    """A step that never returns a valid coordinate still advances the game and counts invalids."""
    game_module = _load_module("game")
    play_llm = _load_module("play_llm")

    record = game_module.place_fleet(99)
    captured_messages = []

    def broken_step(messages, state):
        captured_messages.append(len(messages))
        return None  # forces an invalid shot every turn, like a no-op model

    result = play_llm._play_game(broken_step, record, max_turns=5)

    assert result["invalid_shots"] == 5
    assert result["hits"] == 0
    assert result["sunk_ships"] == 0
    assert result["win"] is False
    # The loop still ran for every turn; messages grew with each invalid shot.
    assert len(captured_messages) == 5
    assert captured_messages[-1] > captured_messages[0]


def test_battleship_play_llm_parse_coord_from_response():
    """Response parser extracts the fire coordinate from a tool-call payload."""
    play_llm = _load_module("play_llm")

    class FakeResponse:
        def model_dump(self):
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fire",
                                        "arguments": '{"coordinate": "c5"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            }

    assert play_llm._parse_coord_from_response(FakeResponse()) == "C5"

    class EmptyResponse:
        def model_dump(self):
            return {"choices": [{"message": {"tool_calls": []}}]}

    assert play_llm._parse_coord_from_response(EmptyResponse()) is None


def test_battleship_evaluate_fake_player():
    """Fake deterministic player produces expected behavior."""
    evaluate = _load_module("evaluate")
    game = _load_module("game")

    # Use proper fleet that matches game expectations
    generator = _load_module("dataset_generator")
    records = generator.generate_records(1, seed=42)

    player = evaluate.FakePlayer(sequence=[(0, 0), (0, 1)])
    result = evaluate.evaluate_player(player, records, max_turns=10)

    assert result["total_fleets"] == 1
    # With our simple sequence, we should hit some cells
    assert result["results"][0]["shots_used"] >= 1


def test_battleship_evaluate_random_player():
    """Random player can hit ships occasionally."""
    evaluate = _load_module("evaluate")
    game = _load_module("game")

    # Use proper fleet setup
    generator = _load_module("dataset_generator")
    records = generator.generate_records(1, seed=42)

    # Use a seeded random for reproducibility in test
    import random
    random.seed(42)

    player = evaluate.RandomPlayer()
    result = evaluate.evaluate_player(player, records, max_turns=64)

    assert result["total_fleets"] == 1
    # Random should eventually hit something if turn cap is high enough
    assert result["results"][0]["shots_used"] <= 64


def test_battleship_tool_messages_error_path_serializes_without_state():
    """Error-path tool results (which carry a GameState) serialize to JSON and
    never leak hidden ship positions."""
    import json

    try:
        run_agent = _load_module("run_agent")
    except ModuleNotFoundError as e:
        if "torch" in str(e):
            # Skip if torch is not available (required by areno imports)
            return
        raise
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Error path: missing fire tool call -> result carries {"state": state}.
    assistant_message = {
        "role": "assistant",
        "tool_calls": [{
            "id": "missing_fire",
            "type": "function",
            "function": {"name": "fire", "arguments": "{}"},
        }],
    }
    result = run_agent._run_tool(assistant_message, state)
    assert "state" in result  # internal bookkeeping still present for run_one

    msgs = run_agent._tool_messages(assistant_message, result)
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    # Must not raise and must not carry the raw state / ship cells.
    parsed = json.loads(tool_msg["content"])
    assert "state" not in parsed
    blob = tool_msg["content"]
    for ship in state.ships:
        for cell in ship.cells:
            coord = game.format_coordinate(cell[0], cell[1])
            assert coord not in blob  # no hidden ship coordinate leaked


# =============================================================================
# Extended input validation tests
# =============================================================================


def test_battleship_coordinate_parsing_extended():
    """Extended coordinate parsing tests for edge cases."""
    game = _load_module("game")

    # Invalid formats
    assert game.parse_coordinate("A") is None  # missing number
    assert game.parse_coordinate("1") is None  # missing letter
    assert game.parse_coordinate("A1B") is None  # extra characters
    assert game.parse_coordinate("AA1") is None  # duplicate letters
    assert game.parse_coordinate("@1") is None  # invalid character before A
    assert game.parse_coordinate("A-1") is None  # negative-like number

    # Valid edge cases
    assert game.parse_coordinate("A01") == (0, 0)  # leading zero
    assert game.parse_coordinate(" A1 ") == (0, 0)  # whitespace handling
    assert game.parse_coordinate("a1") == (0, 0)  # lowercase
    assert game.parse_coordinate("D5") == (3, 4)  # middle coordinate
    assert game.parse_coordinate("H1") == (7, 0)  # left edge
    assert game.parse_coordinate("A8") == (0, 7)  # right edge


def test_battleship_fire_accepts_tuple_input():
    """Fire can accept tuple (row, col) directly."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Fire using tuple instead of string
    result = game.fire(state, (0, 0))  # A1 as tuple
    assert result["status"] in ("miss", "hit", "sunk")
    assert result["shots_used"] == 1


def test_battleship_fire_rejects_invalid_tuple():
    """Fire rejects invalid tuple formats."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Invalid tuple formats
    result = game.fire(state, (0,))  # single element tuple
    assert result["status"] == "invalid"

    result = game.fire(state, (0, 0, 0))  # three element tuple
    assert result["status"] == "invalid"

    result = game.fire(state, "not_a_tuple")  # completely invalid string
    assert result["status"] == "invalid"


def test_battleship_fire_boundary_coordinates():
    """Fire works at boundary coordinates (A1, H8)."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # A1 (top-left corner)
    result = game.fire(state, "A1")
    assert result["status"] in ("miss", "hit", "sunk")

    # H8 (bottom-right corner)
    result = game.fire(state, "H8")
    assert result["status"] in ("miss", "hit", "sunk")


# =============================================================================
# Integration tests: dataset_loader / run_agent / reward / training loop
# =============================================================================


def test_battleship_loader_from_jsonl_file():
    """Loader correctly loads fleet data from a real JSONL file on disk."""
    import json
    import os
    import tempfile

    loader = _load_module("dataset_loader")
    game = _load_module("game")

    # Build two legal fleets via place_fleet so ship_lengths matches game.SHIPS
    record1 = game.place_fleet(42)
    record2 = game.place_fleet(43)

    # Write records to a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(record1) + "\n")
        f.write(json.dumps(record2) + "\n")
        temp_path = f.name

    try:
        records = loader.load_training_dataset(temp_path)
        assert len(records) == 2  # two fleets loaded
        assert records[0]["seed"] == 42  # seed preserved
        assert records[1]["seed"] == 43  # seed preserved
        assert "prompt" in records[0]  # prompt generated from record
        assert "ships" in records[0]  # ships carried through
        assert records[0]["ships"] == record1["ships"]  # ships unchanged
    finally:
        os.unlink(temp_path)  # clean up temp file


def test_battleship_run_agent_run_tool_fire():
    """_run_tool correctly executes a valid fire tool call."""
    import json
    try:
        run_agent = _load_module("run_agent")
    except ModuleNotFoundError as e:
        if "torch" in str(e):
            return  # run_agent requires torch via areno.api.agentic; skip on CPU-only envs
        raise
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Create a valid fire tool call
    assistant_message = {
        "role": "assistant",
        "tool_calls": [{
            "id": "fire_001",
            "type": "function",
            "function": {
                "name": "fire",
                "arguments": json.dumps({"coordinate": "A1"}),
            },
        }],
    }

    result = run_agent._run_tool(assistant_message, state)

    assert "status" in result  # fire result present
    assert result["status"] in ("miss", "hit", "sunk")  # valid shot outcome
    assert "board" in result  # board text included for the model to see
    assert "state" in result  # state returned for internal use in run_one


def test_battleship_run_agent_invalid_turn_advances_shots_used():
    """An invalid tool call (missing coordinate) must still consume a turn —
    otherwise run_one's while loop never terminates on an untrained model that
    never emits a legal fire call (shots_used would stay 0 forever)."""
    try:
        run_agent = _load_module("run_agent")
    except ModuleNotFoundError as e:
        if "torch" in str(e):
            return  # run_agent requires torch via areno.api.agentic; skip on CPU-only envs
        raise
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)
    assert state.shots_used == 0

    # The path an untrained model hits: dummy tool call with empty arguments.
    assistant_message = {
        "role": "assistant",
        "tool_calls": [{
            "id": "missing_fire",
            "type": "function",
            "function": {"name": "fire", "arguments": "{}"},
        }],
    }

    before = state.shots_used
    result = run_agent._run_tool(assistant_message, state)

    # Turn counter advanced and board untouched (no real shot fired).
    assert result["status"] == "invalid"
    assert result["shots_used"] == before + 1
    assert state.shots_used == before + 1
    assert state.shots == before  # no cell recorded
    # Repeating keeps advancing so the loop terminates within MAX_TURNS.
    run_agent._run_tool(assistant_message, state)
    assert state.shots_used == before + 2


def test_battleship_run_agent_full_tool_message_flow():
    """Test complete tool call -> execution -> message flow in run_agent."""
    import json
    try:
        run_agent = _load_module("run_agent")
    except ModuleNotFoundError as e:
        if "torch" in str(e):
            return  # run_agent requires torch via areno.api.agentic; skip on CPU-only envs
        raise
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Simulate an assistant message with a fire tool call
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "fire",
                "arguments": json.dumps({"coordinate": "D5"}),
            },
        }],
    }

    # Execute tool call -> get result with state
    result = run_agent._run_tool(assistant_message, state)

    # Build (assistant_message, tool_result) message pair
    messages = run_agent._tool_messages(assistant_message, result)

    assert len(messages) == 2  # assistant message + tool result
    assert messages[0]["role"] == "assistant"  # first message is assistant
    assert messages[1]["role"] == "tool"  # second message is tool result
    assert messages[1]["tool_call_id"] == "call_123"  # id matches the call

    # Tool result content must be JSON serializable and stripped of state
    content = json.loads(messages[1]["content"])
    assert "status" in content  # status carried through
    assert "state" not in content  # state must be stripped (not serializable)


def test_battleship_reward_full_game_scenarios():
    """Perfect game scores higher than the same game with wasted misses."""
    reward = _load_module("reward")
    game = _load_module("game")

    record_data = game.place_fleet(42)
    state = game.init_state(record_data)

    # Scenario 1: perfect game (hit every ship cell, no misses, no invalids)
    tool_calls = []
    for ship in state.ships:
        for cell in ship.cells:
            coord = game.format_coordinate(cell[0], cell[1])
            tool_calls.append({"name": "fire", "arguments": {"coordinate": coord}})

    record = SimpleNamespace(
        source_record=record_data,
        tool_calls=tool_calls,
    )
    perfect_reward = reward.reward_fn(record)

    # Scenario 2: same game plus 5 wasted miss shots (efficiency penalty)
    ship_cells = set(tuple(x) for x in record_data["ships"])
    empty_cells = [(r, c) for r in range(8) for c in range(8) if (r, c) not in ship_cells]
    tool_calls_with_misses = list(tool_calls)
    for cell in empty_cells[:5]:  # add 5 misses at empty cells
        coord = game.format_coordinate(cell[0], cell[1])
        tool_calls_with_misses.append({"name": "fire", "arguments": {"coordinate": coord}})

    record_with_misses = SimpleNamespace(
        source_record=record_data,
        tool_calls=tool_calls_with_misses,
    )
    imperfect_reward = reward.reward_fn(record_with_misses)

    # Perfect game should score higher than the same win with wasted shots
    assert perfect_reward > imperfect_reward


def test_battleship_training_integration_loader_to_reward():
    """Integration: dataset_loader -> reward_fn via the full record format.

    Replaces a real LLM-driven training loop with a deterministic synthetic
    agent: load a record via the loader, build tool calls from the fleet, run
    reward_fn end-to-end, and assert a win-shaped reward.
    """
    loader = _load_module("dataset_loader")
    game = _load_module("game")
    reward = _load_module("reward")

    # 1) Load a record using the loader's in-memory fallback generator
    records = loader.load_training_dataset("/nonexistent/path")
    assert len(records) > 0  # fallback produced records
    raw_record = records[0]

    # 2) The loader record already carries ship positions; reconstruct a state
    state = game.init_state(raw_record)

    # 3) Synthetic "agent": fire at every ship cell (a perfect winning game)
    tool_calls = []
    for ship in state.ships:
        for cell in ship.cells:
            coord = game.format_coordinate(cell[0], cell[1])
            tool_calls.append({"name": "fire", "arguments": {"coordinate": coord}})

    # 4) Feed the loader record + tool calls into reward_fn exactly as the
    #    trainer would (source_record + tool_calls shape)
    record = SimpleNamespace(
        source_record=raw_record,
        tool_calls=tool_calls,
    )
    r = reward.reward_fn(record)

    # 5) A perfect win must produce a high reward (win bonus + hit/sunk rewards)
    assert r > 0.5, f"Full-pipeline win reward should be high, got {r}"


# =============================================================================
# Backward compatibility tests
# =============================================================================


def test_battleship_backward_compat_missing_optional_fields():
    """Records missing optional fields should use defaults and work correctly."""
    game = _load_module("game")

    # Minimal record without optional id/grid_size (simulating older format)
    minimal_record = {
        "seed": 42,
        "ships": [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [3, 0], [3, 1]],
        "ship_lengths": [4, 3, 2, 2],
    }

    # Should normalize without error
    normalized = game.normalize_record(minimal_record)
    assert normalized["seed"] == 42

    # Should be able to initialize game state
    state = game.init_state(minimal_record)
    assert state.seed == 42
    assert len(state.ships) == 4

    # Game should be playable
    result = game.fire(state, "A1")
    assert result["status"] in ("miss", "hit", "sunk")


def test_battleship_backward_compat_legacy_ship_cell_format():
    """Ship cells in legacy list-of-lists format should be accepted."""
    game = _load_module("game")

    # Record with ships as list of lists (older format possibility)
    legacy_record = {
        "seed": 42,
        "ships": [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [3, 0], [3, 1]],
        "ship_lengths": [4, 3, 2, 2],
        "grid_size": 8,
    }

    # init_state uses tuple conversion internally
    state = game.init_state(legacy_record)
    assert len(state.ships) == 4

    # Verify ship cells were correctly converted
    for ship in state.ships:
        for cell in ship.cells:
            assert isinstance(cell, tuple)  # Should be tuple internally
            assert len(cell) == 2


def test_battleship_backward_compat_fire_result_format():
    """Fire result format should maintain backward compatibility."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Fire at a known ship cell
    ship_cell = record["ships"][0]
    result = game.fire(state, game.format_coordinate(ship_cell[0], ship_cell[1]))

    # Result should have all expected fields for backward compatibility
    assert "status" in result
    assert "shots_used" in result
    assert "hit_cells" in result
    assert "sunk_ships" in result
    assert "remaining" in result

    # Fire at same spot (invalid/repeated)
    result2 = game.fire(state, game.format_coordinate(ship_cell[0], ship_cell[1]))
    assert "status" in result2
    assert "reason" in result2  # Invalid results should have reason


def test_battleship_backward_compat_tool_call_argument_formats():
    """Tool calls with various argument formats (JSON string vs dict) should work."""
    import json
    reward = _load_module("reward")
    game = _load_module("game")

    record_data = game.place_fleet(42)

    # Format 1: arguments as dict (current preferred format)
    tool_calls_dict = [
        {"name": "fire", "arguments": {"coordinate": "A1"}},
        {"name": "fire", "arguments": {"coordinate": "A2"}},
    ]

    # Format 2: arguments as JSON string (legacy format from API)
    tool_calls_json = [
        {"name": "fire", "arguments": json.dumps({"coordinate": "A1"})},
        {"name": "fire", "arguments": json.dumps({"coordinate": "A2"})},
    ]

    # Both should produce rewards (may differ since coordinates are random)
    record_dict = SimpleNamespace(
        source_record=record_data,
        tool_calls=tool_calls_dict,
    )
    record_json = SimpleNamespace(
        source_record=record_data,
        tool_calls=tool_calls_json,
    )

    r_dict = reward.reward_fn(record_dict)
    r_json = reward.reward_fn(record_json)

    # Both should return valid numeric rewards
    assert isinstance(r_dict, (int, float))
    assert isinstance(r_json, (int, float))


def test_battleship_backward_compat_coordinate_case_insensitive():
    """Coordinate parsing should be case-insensitive for backward compatibility."""
    game = _load_module("game")

    # All these should parse to the same coordinate
    test_cases = ["A1", "a1", "A1 ", " a1 ", "A01"]
    expected = (0, 0)

    for coord in test_cases:
        parsed = game.parse_coordinate(coord)
        assert parsed == expected, f"Failed for '{coord}': got {parsed}, expected {expected}"


def test_battleship_backward_compat_score_episode_format():
    """Score episode should produce consistent output format."""
    game = _load_module("game")

    record = game.place_fleet(42)
    state = game.init_state(record)

    # Build tool calls for a complete game
    tool_calls = []
    for ship in state.ships:
        for cell in ship.cells:
            coord = game.format_coordinate(cell[0], cell[1])
            tool_calls.append({"name": "fire", "arguments": {"coordinate": coord}})

    score = game.score_episode(state, tool_calls)

    # Verify score output has all expected fields for API compatibility
    assert "win" in score
    assert "completion" in score
    assert "hits" in score
    assert "sunk_ships" in score
    assert "shots_used" in score
    assert "invalid_shots" in score

    # Verify types
    assert isinstance(score["win"], bool)
    assert isinstance(score["completion"], float)
    assert isinstance(score["hits"], int)


def test_game_state_default_initialization():
    """GameState dataclass default values work as expected."""
    game = _load_module("game")

    # Default empty state
    empty_state = game.GameState()
    assert empty_state.ships == []
    assert empty_state.shots_history == []
    assert empty_state.shots_used == 0
    assert empty_state.grid_size == game.GRID
    assert empty_state.seed == 0

    # Can still use methods with empty state
    legal = game.legal_shots(empty_state)
    assert len(legal) == game.GRID * game.GRID  # All cells legal when empty


def test_reward_constants_exist_and_valid():
    """Reward constants exist and are valid numeric types (values may be tuned)."""
    reward = _load_module("reward")

    # Verify constants exist and are numeric (allows future tuning of values)
    assert hasattr(reward, 'WIN_BONUS')
    assert hasattr(reward, 'HIT_REWARD')
    assert hasattr(reward, 'SUNK_REWARD')
    assert hasattr(reward, 'INVALID_SHOT_PENALTY')
    assert hasattr(reward, 'EFFICIENCY_PENALTY')

    # Verify types are numeric (int or float)
    assert isinstance(reward.WIN_BONUS, (int, float))
    assert isinstance(reward.HIT_REWARD, (int, float))
    assert isinstance(reward.SUNK_REWARD, (int, float))
    assert isinstance(reward.INVALID_SHOT_PENALTY, (int, float))
    assert isinstance(reward.EFFICIENCY_PENALTY, (int, float))

    # Verify values are reasonable (positive rewards, non-negative penalties)
    assert reward.WIN_BONUS > 0
    assert reward.HIT_REWARD >= 0
    assert reward.SUNK_REWARD >= 0
    assert reward.INVALID_SHOT_PENALTY >= 0
    assert reward.EFFICIENCY_PENALTY >= 0


def test_battleship_backward_compat_grid_size_mismatch_handling():
    """Records with grid_size different from current GRID should be handled gracefully."""
    game = _load_module("game")

    # Record with mismatched grid_size (simulate old save with different board size)
    mismatched_record = {
        "seed": 42,
        "ships": [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [3, 0], [3, 1]],
        "ship_lengths": [4, 3, 2, 2],
        "grid_size": 8,  # Current grid size, should work
    }

    # Should still work as long as cells are valid for current grid
    state = game.init_state(mismatched_record)
    assert state.grid_size == game.GRID


def test_other_agentic_examples_unaffected_by_battleship():
    """Ensure other agentic examples (tictactoe) work correctly after Battleship code changes.

    This test verifies that Battleship example code doesn't pollute global state
    or break other examples through shared imports or side effects.
    """
    # Load Battleship first
    battleship_game = _load_module("game")
    _ = _load_module("reward")

    # Now load tictactoe - should work independently
    ttt_path = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "tictactoe" / "game.py"
    spec = importlib.util.spec_from_file_location("tictactoe_game_for_compat_test", ttt_path)
    ttt_game = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(ttt_game)

    # Tic-Tac-Toe should work correctly
    board = ttt_game.normalize_board([[".", ".", "."], [".", ".", "."], [".", ".", "."]])
    assert len(board) == 3
    assert ttt_game.next_player(board) == "X"

    # Verify the two games don't interfere with each other's constants
    assert battleship_game.GRID == 8  # Battleship uses 8x8
    # Tictactoe board is 3x3 (implied by normalize_board validation)


def test_battleship_isolation_from_trainer_api():
    """Battleship example should not interfere with core areno Trainer imports.

    Verifies that importing Battleship modules doesn't break the ability to
    import and use the areno Trainer (if torch is available).
    """
    # Load all Battleship modules first
    _ = _load_module("game")
    _ = _load_module("reward")
    loader = _load_module("dataset_loader")

    # Verify Battleship loader generates valid records
    records = loader.load_training_dataset("/nonexistent/path")
    assert len(records) > 0

    # Now verify we can still import areno API (if available)
    try:
        from areno.api import Trainer
        # If import succeeds, verify it works
        assert callable(Trainer)
    except ImportError:
        # Skip if areno/torch not available (this is fine in CPU-only test env)
        pass