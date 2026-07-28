"""Focused CPU tests for the water-jug agentic example.

Covers: solvable/unsolvable inputs, invalid jug identifiers, no-op pours,
seeded generation, action exhaustion, boundary values, and reward logic.
Run with: python -m pytest examples/agentic/water_jug/test_water_jug.py -v
Or standalone:  python examples/agentic/water_jug/test_water_jug.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure we can import game, dataset_loader, reward, dataset_generator
sys.path.insert(0, str(Path(__file__).resolve().parent))

import game
import dataset_generator
import dataset_loader
import reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeRecord:
    """Minimal stand-in for AReno's RewardRecord used by reward_fn."""

    def __init__(self, tool_calls, source_record):
        self.tool_calls = tool_calls
        self.source_record = source_record


def make_tool_call(action: str) -> dict:
    return {
        "name": "water_jug_action",
        "arguments": json.dumps({"action": action}),
    }


def solve_actions(caps, target):
    """Return the list of actions to solve a puzzle."""
    return game.shortest_path(caps, (0,) * len(caps), target)


# ---------------------------------------------------------------------------
# 1. Game logic: solvable puzzles
# ---------------------------------------------------------------------------

def test_classic_3_5_target4():
    """Classic 3L/5L jug puzzle, target 4L."""
    path = game.shortest_path((3, 5), (0, 0), 4)
    assert path is not None
    assert len(path) == 6
    # Verify the path actually reaches the target
    state = (0, 0)
    for action in path:
        state = game.apply_action((3, 5), state, action)
    assert game.is_goal(state, 4)


def test_fill_directly():
    """Target equals a jug capacity — 1-step solution."""
    path = game.shortest_path((5, 3), (0, 0), 5)
    assert path == ["fill(0)"]


def test_three_jugs():
    """Three-jug puzzle."""
    path = game.shortest_path((6, 10, 4), (0, 0, 0), 8)
    assert path is not None
    state = (0, 0, 0)
    for a in path:
        state = game.apply_action((6, 10, 4), state, a)
    assert game.is_goal(state, 8)


# ---------------------------------------------------------------------------
# 2. Unsolvable inputs
# ---------------------------------------------------------------------------

def test_unsolvable_target():
    """Target 4 with jugs (3, 6) is unsolvable (gcd=3, 4 not divisible)."""
    path = game.shortest_path((3, 6), (0, 0), 4)
    assert path is None


def test_unsolvable_bfs_distance():
    """bfs_distance returns None for unsolvable targets."""
    dist = game.bfs_distance((3, 6), (0, 0), 4)
    assert dist is None


def test_unsolvable_reward_is_zero():
    """Reward for unsolvable puzzle should be 0.0."""
    image = {"capacities": [3, 6], "initial_state": [0, 0], "target": 4, "oracle_steps": 0}
    record = FakeRecord([make_tool_call("fill(0)")], {"image": image})
    r = reward.reward_fn(record)
    assert r == 0.0


# ---------------------------------------------------------------------------
# 3. Invalid jug identifiers
# ---------------------------------------------------------------------------

def test_invalid_jug_index_high():
    """Jug index out of range raises ValueError."""
    try:
        game.apply_action((3, 5), (0, 0), "fill(5)")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_invalid_jug_index_negative():
    """Negative jug index raises ValueError."""
    try:
        game.apply_action((3, 5), (0, 0), "fill(-1)")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_pour_same_jug():
    """Pouring from a jug into itself raises ValueError."""
    try:
        game.apply_action((3, 5), (0, 0), "pour(0,0)")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_unknown_action():
    """Unknown action string raises ValueError."""
    try:
        game.apply_action((3, 5), (0, 0), "shake(0)")
        assert False, "Should have raised"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 4. No-op pours (pour when source is empty or dest is full)
# ---------------------------------------------------------------------------

def test_noop_pour_empty_source():
    """Pouring from an empty jug is a no-op (state unchanged)."""
    state = game.apply_action((3, 5), (0, 0), "pour(0,1)")
    assert state == (0, 0)  # nothing happens


def test_noop_pour_full_dest():
    """Pouring into a full jug is a no-op."""
    state = game.apply_action((3, 5), (0, 0), "fill(1)")  # (0, 5)
    state = game.apply_action((3, 5), state, "pour(0,1)")  # pour from empty 0 into full 1
    assert state == (0, 5)


# ---------------------------------------------------------------------------
# 5. Seeded generation (deterministic)
# ---------------------------------------------------------------------------

def test_seed_deterministic(tmp_path):
    """Same seed produces identical puzzles."""
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    os.system(f"{sys.executable} {Path(__file__).parent / 'dataset_generator.py'} -o {p1} -n 20 --seed 42")
    os.system(f"{sys.executable} {Path(__file__).parent / 'dataset_generator.py'} -o {p2} -n 20 --seed 42")
    assert p1.read_text() == p2.read_text()


def test_different_seed_differs(tmp_path):
    """Different seeds produce different puzzles."""
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    os.system(f"{sys.executable} {Path(__file__).parent / 'dataset_generator.py'} -o {p1} -n 20 --seed 42")
    os.system(f"{sys.executable} {Path(__file__).parent / 'dataset_generator.py'} -o {p2} -n 20 --seed 999")
    assert p1.read_text() != p2.read_text()


def test_all_generated_solvable(tmp_path):
    """Every generated puzzle is solvable."""
    p = tmp_path / "p.jsonl"
    os.system(f"{sys.executable} {Path(__file__).parent / 'dataset_generator.py'} -o {p} -n 50 --seed 2026")
    with p.open() as f:
        for line in f:
            d = json.loads(line.strip())
            dist = game.bfs_distance(d["capacities"], d["initial_state"], d["target"])
            assert dist is not None, f"Unsolvable puzzle: {d}"


# ---------------------------------------------------------------------------
# 6. Action exhaustion (max turns without solving)
# ---------------------------------------------------------------------------

def test_action_exhaustion_reward():
    """Many actions that don't solve should give partial/zero reward."""
    image = {"capacities": [7, 10], "initial_state": [0, 0], "target": 9, "oracle_steps": 10}
    # 5 random actions that don't solve
    calls = [make_tool_call(a) for a in ["fill(0)", "empty(0)", "fill(1)", "empty(1)", "fill(0)"]]
    record = FakeRecord(calls, {"image": image})
    r = reward.reward_fn(record)
    assert 0.0 <= r < 1.0  # not solved, partial reward at most


# ---------------------------------------------------------------------------
# 7. Reward function: solved and efficiency
# ---------------------------------------------------------------------------

def test_reward_solved_optimal():
    """Solving in optimal steps gives >1.0 reward (efficiency bonus)."""
    caps = (3, 5)
    target = 4
    image = {"capacities": list(caps), "initial_state": [0, 0], "target": target, "oracle_steps": 6}
    actions = solve_actions(caps, target)
    calls = [make_tool_call(a) for a in actions]
    record = FakeRecord(calls, {"image": image})
    r = reward.reward_fn(record)
    assert r >= 1.0  # solved with bonus


def test_reward_solved_excess_actions():
    """Solving with more steps than oracle still gives positive reward but less."""
    caps = (3, 5)
    target = 4
    image = {"capacities": list(caps), "initial_state": [0, 0], "target": target, "oracle_steps": 6}
    # Optimal is 6 steps, add 4 wasted steps
    actions = solve_actions(caps, target) + ["empty(0)", "fill(0)", "empty(0)", "fill(0)"]
    calls = [make_tool_call(a) for a in actions]
    record = FakeRecord(calls, {"image": image})
    r = reward.reward_fn(record)
    assert 0.1 <= r < 1.0  # solved but with penalty


def test_reward_no_actions():
    """No tool calls at all gives 0 reward."""
    image = {"capacities": [3, 5], "initial_state": [0, 0], "target": 4, "oracle_steps": 6}
    record = FakeRecord([], {"image": image})
    r = reward.reward_fn(record)
    assert r == 0.0


# ---------------------------------------------------------------------------
# 8. Dataset loader
# ---------------------------------------------------------------------------

def test_dataset_loader(tmp_path):
    """Dataset loader produces correct prompt and image fields."""
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps({
        "id": "test-0", "capacities": [3, 5],
        "initial_state": [0, 0], "target": 4, "oracle_steps": 6
    }) + "\n")
    items = dataset_loader.load_training_dataset(str(p))
    assert len(items) == 1
    assert "prompt" in items[0]
    assert "image" in items[0]
    assert items[0]["image"]["target"] == 4
    assert items[0]["image"]["capacities"] == [3, 5]


def test_dataset_loader_missing_file():
    """Missing dataset file raises FileNotFoundError."""
    try:
        dataset_loader.load_training_dataset("/nonexistent/path.jsonl")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# 9. Board formatting
# ---------------------------------------------------------------------------

def test_format_board():
    """format_board produces readable output with target and jug info."""
    board = game.format_board((3, 5), (1, 4), 4)
    assert "Target" in board
    assert "Jug 0" in board
    assert "Jug 1" in board
    assert "1/3" in board
    assert "4/5" in board


def test_build_user_prompt():
    """build_user_prompt includes target and available actions."""
    prompt = game.build_user_prompt((3, 5), 4)
    assert "4" in prompt
    assert "fill" in prompt
    assert "empty" in prompt
    assert "pour" in prompt


# ---------------------------------------------------------------------------
# 10. Solve rate and excess actions metric (acceptance criterion)
# ---------------------------------------------------------------------------

def test_solve_rate_and_excess_actions(tmp_path):
    """Generate puzzles, solve them, and report solve rate + excess actions."""
    p = tmp_path / "puzzles.jsonl"
    os.system(f"{sys.executable} {Path(__file__).parent / 'dataset_generator.py'} -o {p} -n 30 --seed 2026")

    solved = 0
    total = 0
    excess_actions = []

    with p.open() as f:
        for line in f:
            d = json.loads(line.strip())
            total += 1
            path = game.shortest_path(d["capacities"], d["initial_state"], d["target"])
            if path is not None:
                solved += 1
                excess = len(path) - d["oracle_steps"]
                excess_actions.append(excess)

    solve_rate = solved / total
    avg_excess = sum(excess_actions) / len(excess_actions) if excess_actions else 0

    print(f"\nSolve rate: {solved}/{total} = {solve_rate:.1%}")
    print(f"Average excess actions over oracle: {avg_excess:.2f}")

    assert solve_rate == 1.0  # all generated puzzles should be solvable
    assert avg_excess == 0.0  # BFS gives optimal, so excess should be 0


# ---------------------------------------------------------------------------
# Standalone runner (no pytest needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    state = {"passed": 0, "failed": 0, "failures": []}

    def run_test(name, fn):
        try:
            fn()
            print(f"  PASS  {name}")
            state["passed"] += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            state["failed"] += 1
            state["failures"].append((name, str(e)))

    # Use a temp dir for tests that need tmp_path
    tmp = Path(tempfile.mkdtemp())

    # 1. Solvable
    run_test("test_classic_3_5_target4", test_classic_3_5_target4)
    run_test("test_fill_directly", test_fill_directly)
    run_test("test_three_jugs", test_three_jugs)

    # 2. Unsolvable
    run_test("test_unsolvable_target", test_unsolvable_target)
    run_test("test_unsolvable_bfs_distance", test_unsolvable_bfs_distance)
    run_test("test_unsolvable_reward_is_zero", test_unsolvable_reward_is_zero)

    # 3. Invalid jug IDs
    run_test("test_invalid_jug_index_high", test_invalid_jug_index_high)
    run_test("test_invalid_jug_index_negative", test_invalid_jug_index_negative)
    run_test("test_pour_same_jug", test_pour_same_jug)
    run_test("test_unknown_action", test_unknown_action)

    # 4. No-op pours
    run_test("test_noop_pour_empty_source", test_noop_pour_empty_source)
    run_test("test_noop_pour_full_dest", test_noop_pour_full_dest)

    # 5. Seeded generation (need tmp_path workaround)
    run_test("test_seed_deterministic", lambda: test_seed_deterministic(tmp))
    run_test("test_different_seed_differs", lambda: test_different_seed_differs(tmp))
    run_test("test_all_generated_solvable", lambda: test_all_generated_solvable(tmp))

    # 6. Action exhaustion
    run_test("test_action_exhaustion_reward", test_action_exhaustion_reward)

    # 7. Reward
    run_test("test_reward_solved_optimal", test_reward_solved_optimal)
    run_test("test_reward_solved_excess_actions", test_reward_solved_excess_actions)
    run_test("test_reward_no_actions", test_reward_no_actions)

    # 8. Dataset loader
    run_test("test_dataset_loader", lambda: test_dataset_loader(tmp))
    run_test("test_dataset_loader_missing_file", test_dataset_loader_missing_file)

    # 9. Formatting
    run_test("test_format_board", test_format_board)
    run_test("test_build_user_prompt", test_build_user_prompt)

    # 10. Solve rate
    run_test("test_solve_rate_and_excess_actions", lambda: test_solve_rate_and_excess_actions(tmp))

    print(f"\n{'='*50}")
    print(f"Results: {state['passed']} passed, {state['failed']} failed, {state['passed'] + state['failed']} total")
    if state["failures"]:
        print("\nFailures:")
        for name, err in state["failures"]:
            print(f"  - {name}: {err}")
    print(f"{'='*50}")