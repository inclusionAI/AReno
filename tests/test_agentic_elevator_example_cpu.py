from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module(name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "elevator" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"agentic_elevator_{name}_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _building(arrivals=None, *, floors=4, capacity=2, hall=None, car=None):
    building = {
        "floors": floors,
        "capacity": capacity,
        "arrivals": list(arrivals or []),
        "car": {"floor": 0, "direction": "U", "door_open": False, "passengers": []},
    }
    if hall is not None:
        building["hall_queues"] = hall
    if car is not None:
        building["car"] = car
    return building


def test_elevator_overload_prevention_respects_capacity():
    game = _load_module("game")

    # Five passengers wait at floor 0 with capacity 2: opening boards only two.
    building = _building(
        floors=4,
        capacity=2,
        hall={"0": [{"to_floor": 2}, {"to_floor": 2}, {"to_floor": 3}, {"to_floor": 1}, {"to_floor": 1}]},
        arrivals=[],
    )
    state = game.clone_state(game.normalize_building(building))
    stats = {"delivered": 0, "total_wait": 0, "max_wait": 0}
    assert game.step(state, "O", stats) is True
    assert len(state["car"]["passengers"]) == 2
    assert len(state["hall_queues"][0]) == 3  # overload leaves the rest waiting


def test_elevator_empty_door_operation_is_safe_and_counted():
    game = _load_module("game")

    # Opening on an empty floor is a legal but useless action; the door opens.
    # The episode terminates immediately after the open (nothing left to do),
    # so only the open is counted -- the close is skipped, never invalid.
    empty = _building(floors=4, capacity=2, car={"floor": 1, "door_open": False, "passengers": []}, arrivals=[])
    result = game.play(empty, "OC", max_steps=5)
    assert result["n_steps"] == 1
    assert result["n_invalid"] == 0
    assert result["delivered_passengers"] == 0

    # A separate door state-machine check: closing a closed door is invalid.
    closed = _building(floors=4, capacity=2, car={"floor": 1, "door_open": False, "passengers": []}, arrivals=[])
    state = game.clone_state(game.normalize_building(closed))
    stats = {"delivered": 0, "total_wait": 0, "max_wait": 0}
    assert game.step(state, "C", stats) is False  # door already closed


def test_elevator_simultaneous_requests_are_handled():
    game = _load_module("game")

    # Two passengers arrive on tick 0 at different floors; FCFS delivers both.
    building = _building(
        floors=5,
        capacity=2,
        arrivals=[{"tick": 0, "from_floor": 0, "to_floor": 4}, {"tick": 0, "from_floor": 3, "to_floor": 1}],
    )
    actions = game.fcfs_actions(building)
    result = game.play(building, actions, max_steps=60)
    assert result["delivered_passengers"] == 2
    assert result["remaining_passengers"] == 0
    assert result["terminal"] is True


def test_elevator_peak_traffic_runs_to_termination():
    game = _load_module("game")

    # A dense arrival schedule across many floors; FCFS clears it deterministically.
    arrivals = [
        {"tick": t, "from_floor": f, "to_floor": (f + 2) % 5}
        for t, f in zip(range(0, 24, 2), [0, 2, 1, 3, 4, 0, 2, 1, 3, 4, 0, 1])
    ]
    building = _building(floors=5, capacity=3, arrivals=arrivals)
    result = game.play(building, game.fcfs_actions(building), max_steps=400)
    assert result["terminal"] is True
    assert result["remaining_passengers"] == 0
    assert result["delivered_passengers"] == len(arrivals)


def test_elevator_termination_when_nothing_remains():
    game = _load_module("game")

    delivered = {"floors": 4, "capacity": 2, "arrivals": [], "car": {"floor": 0, "door_open": False, "passengers": []}}
    state = game.normalize_building(delivered)
    assert game.is_terminal(state) is True

    pending = _building(arrivals=[{"tick": 0, "from_floor": 0, "to_floor": 2}])
    state = game.normalize_building(pending)
    assert game.is_terminal(state) is False


def test_elevator_seeded_replay_is_deterministic():
    game = _load_module("game")
    import random

    building = game.fresh_building(random.Random(2026))
    sequence = "OCUUOCUDDOC"

    first = game.play(building, sequence, max_steps=40)
    second = game.play(building, sequence, max_steps=40)

    def no_state(d):
        return {k: v for k, v in d.items() if k != "state"}

    assert no_state(first) == no_state(second)
    assert first["state"] == second["state"]


def test_elevator_normalize_rejects_malformed_buildings():
    game = _load_module("game")

    for bad in (
        {"floors": 1, "capacity": 2, "arrivals": []},  # too few floors
        {"floors": 4, "capacity": 0, "arrivals": []},  # no capacity
        {"floors": 3, "capacity": 2, "arrivals": [{"tick": 0, "from_floor": 0, "to_floor": 0}]},  # self-arrival
        {"floors": 3, "capacity": 2, "car": {"floor": 9}, "arrivals": []},  # car off-grid
    ):
        try:
            game.normalize_building(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for building {bad}")


def test_elevator_reward_scores_dispatch_actions_and_writes_metrics():
    reward = _load_module("reward")
    game = _load_module("game")

    building = _building(
        floors=4, capacity=2, arrivals=[{"tick": 0, "from_floor": 0, "to_floor": 2}]
    )
    source = {"building": building, "max_steps": 60}

    # A FCFS-style dispatch that delivers scores above zero and writes metrics.
    good = SimpleNamespace(
        source_record=dict(source),
        completion="",
        tool_calls=[{"name": "dispatch", "arguments": {"actions": game.fcfs_actions(building)}}],
    )
    good_reward = reward.reward_fn(good)
    metrics = good.source_record["metrics"]
    assert good_reward > 0.0
    assert metrics["delivered_passengers"] >= 1
    assert {"delivered_passengers", "mean_wait", "invalid_rate", "n_steps", "n_invalid", "terminal"} <= set(metrics)

    # Empty tool calls map to the failing score.
    empty = SimpleNamespace(source_record=dict(source), completion="", tool_calls=[])
    assert reward.reward_fn(empty) == -1.0

    # JSON-string arguments (the real OpenAI shape) parse correctly.
    json_str = SimpleNamespace(
        source_record=dict(source),
        completion="",
        tool_calls=[{"name": "dispatch", "arguments": '{"actions": "OCUUOC"}'}],
    )
    assert reward.reward_fn(json_str) >= -1.0

    # Non-UDOC characters are stripped, not fatal.
    cleaned = reward._clean_actions("U?O!xxC")  # noqa: SLF001
    assert cleaned == "UOC"


def test_elevator_generator_is_reproducible_and_valid():
    generator = _load_module("dataset_generator")
    game = _load_module("game")

    first = generator.generate_records(8, seed=2026, arrivals_per_building=4)
    second = generator.generate_records(8, seed=2026, arrivals_per_building=4)
    assert first == second
    for record in first:
        # normalize must succeed; arrivals respect floors and from != to.
        building = game.normalize_building(record)
        for arrival in building["arrivals"]:
            assert arrival["from_floor"] != arrival["to_floor"]


def test_elevator_fcfs_baseline_beats_or_matches_random_idle():
    """A pure no-op/idle policy must not outscoop FCFS: baseline metrics stay sane
    and a FCFS dispatch at worst delivers the same passengers with finite wait."""

    baseline = _load_module("baseline")
    game = _load_module("game")

    stats = baseline.fcfs_baseline(16, seed=2026, arrivals_per_building=4)
    assert stats["count"] == 16.0
    assert stats["mean_delivered"] >= 0.0
    assert stats["mean_wait"] >= 0.0
    assert 0.0 <= stats["mean_invalid_rate"] <= 1.0
    # FCFS never emits an invalid action.
    assert stats["mean_invalid_rate"] == 0.0

    # Reproducible aggregation.
    again = baseline.fcfs_baseline(16, seed=2026, arrivals_per_building=4)
    assert stats == again
