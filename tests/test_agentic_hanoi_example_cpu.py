"""Focused CPU tests for the Towers of Hanoi agentic example.

Loaded standalone via importlib (mirroring ``test_agentic_tictactoe_example_cpu.py``)
so the suite runs on CPU without importing the AReno package or triggering the
CUDA build. Each example module is loaded under a unique name to avoid
collisions with other examples' top-level ``game`` / ``dataset_generator``.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "hanoi" / f"{name}.py"
    mod_name = f"hanoi_{name}_for_tests"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # 3.9 dataclass + annotations workaround
    spec.loader.exec_module(module)
    return module


game = _load("game")
gen = _load("dataset_generator")
loader = _load("dataset_loader")
reward = _load("reward")


# --- Oracle -----------------------------------------------------------------


@pytest.mark.parametrize("n,expected", [(3, 7), (4, 15), (5, 31), (6, 63)])
def test_optimal_steps_closed_form(n, expected):
    assert game.optimal_steps(n) == expected


def test_optimal_steps_rejects_out_of_range():
    for bad in (0, 1, 2, 7, 8, -1):
        with pytest.raises(ValueError):
            game.optimal_steps(bad)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_optimal_solution_length_and_validity(n):
    moves = game.optimal_solution(n)
    assert len(moves) == game.optimal_steps(n)
    assert game.validate_solution(moves, n)


def test_optimal_solution_respects_source_target():
    moves = game.optimal_solution(3, source=0, target=2)
    assert moves[0][0] == 0
    assert moves[-1][1] == 2


# --- Environment core -------------------------------------------------------


def test_make_state_canonical_start():
    state = game.make_state(3)
    assert state.pegs == ((3, 2, 1), (), ())
    assert state.n == 3
    assert not game.is_terminal(state)


def test_make_state_rejects_out_of_range():
    for bad in (2, 7):
        with pytest.raises(ValueError):
            game.make_state(bad)


def test_legal_move_advances_state():
    state = game.make_state(3)
    next_state, _reward, _done, info = game.step(state, (0, 2))
    assert next_state.pegs == ((3, 2), (), (1,))
    assert info["move"] == (0, 2)
    assert info["completed"] is False


def test_optimal_solution_drives_to_completion():
    for n in (3, 4, 5):
        state = game.make_state(n)
        done = False
        for mv in game.optimal_solution(n):
            state, _reward, done, info = game.step(state, mv)
            assert not info["illegal"]
        assert done and info["completed"]


def test_completion_reward_decreases_with_excess_steps():
    n = 3
    state = game.make_state(n)
    reward_opt = 0.0
    for mv in game.optimal_solution(n):
        state, reward, done, _ = game.step(state, mv)
        if done:
            reward_opt = reward
    assert reward_opt == pytest.approx(game.COMPLETION_REWARD)  # excess == 0

    # build a deliberately longer-but-legal path then re-score terminal
    state = game.make_state(n)
    s, _, _, _ = game.step(state, (0, 1))
    s, _, _, _ = game.step(s, (1, 0))
    done = False
    reward_long = 0.0
    for mv in game.optimal_solution(n):
        s, reward, done, _ = game.step(s, mv)
        if done:
            reward_long = reward
    assert reward_long < reward_opt


# --- Illegal actions --------------------------------------------------------


def test_empty_source_rejected():
    state = game.make_state(3)
    next_state, reward, done, info = game.step(state, (1, 2))
    assert info["illegal"] and info["reason"] == "empty_source"
    assert next_state == state  # unchanged
    assert reward == game.ILLEGAL_PENALTY
    assert not done  # penalize does not terminate


def test_larger_on_smaller_rejected():
    state = game.make_state(3)
    # move disk 1 (smallest) to peg 2, then try to put disk 2 on top of it
    s, _, _, _ = game.step(state, (0, 2))
    next_state, reward, _done, info = game.step(s, (0, 2))
    assert info["illegal"] and info["reason"] == "larger_on_smaller"
    assert next_state == s
    assert reward == game.ILLEGAL_PENALTY


def test_out_of_range_and_noop_rejected():
    state = game.make_state(3)
    for action in [(0, 3), (0, -1), (0, 0)]:
        _, _reward, _done, info = game.step(state, action)
        assert info["illegal"]


def test_malformed_action_rejected():
    state = game.make_state(3)
    for action in [None, "foo", {"wrong": 1}, (0,), 5]:
        _, _reward, _done, info = game.step(state, action)
        assert info["illegal"] and info["reason"] in {
            "malformed",
            "out_of_range",
            "no_op",
            "empty_source",
        }


def test_illegal_terminate_policy_ends_episode():
    state = game.make_state(3)
    _ns, _r, done, info = game.step(state, (1, 2), illegal_policy="terminate")
    assert info["illegal"]
    assert done is True


def test_dict_and_string_actions_acceptable():
    state = game.make_state(3)
    s1, _, _, info1 = game.step(state, {"source": 0, "target": 2})
    assert not info1["illegal"]
    _s2, _, _, info2 = game.step(s1, "0 -> 1")
    assert not info2["illegal"]


# --- Trace replay -----------------------------------------------------------


def test_serialize_and_parse_roundtrip():
    moves = [(0, 2), (0, 1), (2, 1)]
    trace = game.serialize_trace(moves)
    assert trace == "0->2,0->1,2->1"
    assert game.parse_trace(trace) == moves


def test_replay_optimal_trace_completes_with_zero_excess():
    for n in (3, 4, 5, 6):
        trace = game.serialize_trace(game.optimal_solution(n))
        result = game.replay(trace, n)
        assert result.completed
        assert result.illegal_count == 0
        assert result.excess_moves == 0
        assert result.legal_count == game.optimal_steps(n)


def test_replay_counts_illegal_without_terminating():
    # insert a no-op illegal move in the middle (before completion) so the
    # replay actually processes it; default penalize policy keeps going.
    optimal = game.optimal_solution(3)
    moves = [optimal[0], (1, 1)] + list(optimal[1:])
    trace = game.serialize_trace(moves)
    result = game.replay(trace, 3)
    assert result.completed
    assert result.illegal_count == 1
    assert result.legal_count == game.optimal_steps(3)


def test_replay_failure_trace_does_not_complete():
    moves = [(0, 1), (1, 0), (0, 1), (1, 0)]
    result = game.replay(game.serialize_trace(moves), 3)
    assert not result.completed


def test_replay_as_text_is_human_readable():
    result = game.replay(game.serialize_trace(game.optimal_solution(3)), 3)
    text = result.as_text()
    assert "completed" in text
    assert "excess_moves_over_optimum=0" in text


# --- Evaluation -------------------------------------------------------------


def test_evaluate_completion_rate_and_excess():
    n = 3
    traces = [
        game.serialize_trace(game.optimal_solution(n)),  # completes, 0 excess
        game.serialize_trace([(0, 1), (1, 0)] * 4),  # fails
    ]
    report = game.evaluate(traces, n)
    assert report["sample_count"] == 2
    assert report["completion_rate"] == 0.5
    assert report["oracle_steps"] == 7
    assert report["avg_excess_moves"] == 0.0


# --- Determinism & backward-compat -----------------------------------------


def test_deterministic_output_same_inputs():
    n = 4
    trace = game.serialize_trace(game.optimal_solution(n))
    r1 = game.replay(trace, n).as_dict()
    r2 = game.replay(trace, n).as_dict()
    assert r1 == r2


def test_default_illegal_policy_is_penalize():
    # default behavior: illegal move does not terminate
    state = game.make_state(3)
    _, _r, done, _info = game.step(state, (1, 2))  # empty source
    assert done is False


def test_prompt_contains_legal_moves():
    state = game.make_state(3)
    prompt = game.format_prompt(state)
    assert "move_disk" in prompt
    assert "0" in prompt and "2" in prompt


# --- Generator fixtures -----------------------------------------------------


def test_default_fixture_count_is_all_scenarios_times_disks():
    records = gen.generate_records()
    assert len(records) == (game.MAX_DISKS - game.MIN_DISKS + 1) * len(gen.SCENARIOS)
    assert len(records) == 16


def test_count_truncates_deterministically():
    a = gen.generate_records(5)
    b = gen.generate_records(5)
    assert a == b
    assert len(a) == 5


def test_every_record_has_required_fields_and_consistent_oracle():
    for record in gen.generate_records():
        for key in (
            "id",
            "n",
            "scenario",
            "pegs",
            "legal_moves",
            "oracle_steps",
            "optimal_moves",
            "trace",
            "expected",
        ):
            assert key in record, f"{record['id']} missing {key}"
        n = record["n"]
        assert record["oracle_steps"] == 2**n - 1
        assert len(record["optimal_moves"]) == 2**n - 1


def test_optimal_moves_are_valid_solutions():
    for record in gen.generate_records():
        moves = [tuple(mv) for mv in record["optimal_moves"]]
        assert game.validate_solution(moves, record["n"])


def test_record_to_state_roundtrips_canonical_start():
    for record in gen.generate_records():
        state = gen.record_to_state(record)
        assert state.n == record["n"]
        assert tuple(state.pegs[0]) == tuple(range(record["n"], 0, -1))  # all disks on peg 0
        assert state.pegs[1] == () and state.pegs[2] == ()


def test_record_to_trace_parses_pairs():
    rec = {"trace": [[0, 2], [0, 1], [2, 1]]}
    assert gen.record_to_trace(rec) == [(0, 2), (0, 1), (2, 1)]


@pytest.mark.parametrize("scenario", gen.SCENARIOS)
def test_expected_outcomes_match_actual_replay(scenario):
    # For each scenario on n=3, the stored expected outcome equals a fresh replay.
    record = next(r for r in gen.generate_records() if r["n"] == 3 and r["scenario"] == scenario)
    expected = gen.expected_outcome(record)
    actual = game.replay(gen.record_to_trace(record), 3).as_dict()
    assert actual["completed"] == expected["completed"]
    assert actual["legal_count"] == expected["legal_count"]
    assert actual["illegal_count"] == expected["illegal_count"]
    assert actual["excess_moves"] == expected["excess_moves"]


def test_optimal_scenario_zero_illegal_zero_excess():
    for n in (3, 4, 5, 6):
        record = next(r for r in gen.generate_records() if r["n"] == n and r["scenario"] == "optimal")
        exp = gen.expected_outcome(record)
        assert exp["completed"] is True
        assert exp["illegal_count"] == 0
        assert exp["excess_moves"] == 0


def test_contains_illegal_scenario_has_one_illegal_and_completes():
    for n in (3, 4, 5, 6):
        record = next(r for r in gen.generate_records() if r["n"] == n and r["scenario"] == "contains_illegal")
        exp = gen.expected_outcome(record)
        assert exp["completed"] is True
        assert exp["illegal_count"] == 1


def test_boundary_scenario_has_one_illegal_and_completes():
    for n in (3, 4, 5, 6):
        record = next(r for r in gen.generate_records() if r["n"] == n and r["scenario"] == "boundary")
        exp = gen.expected_outcome(record)
        assert exp["completed"] is True
        assert exp["illegal_count"] == 1  # the empty-source attempt


def test_failure_scenario_does_not_complete():
    for n in (3, 4, 5, 6):
        record = next(r for r in gen.generate_records() if r["n"] == n and r["scenario"] == "failure")
        exp = gen.expected_outcome(record)
        assert exp["completed"] is False


def test_write_jsonl_round_trips_through_parse():
    import json as _json

    records = gen.generate_records(3)
    buf = io.StringIO()
    gen.write_jsonl(records, buf)
    buf.seek(0)
    parsed = [_json.loads(line) for line in buf if line.strip()]
    assert len(parsed) == 3
    assert parsed[0]["id"] == records[0]["id"]


def test_deterministic_across_calls():
    import json as _json

    r1 = gen.generate_records()
    r2 = gen.generate_records()
    assert [_json.dumps(r, sort_keys=True) for r in r1] == [_json.dumps(r, sort_keys=True) for r in r2]


# --- Loader -----------------------------------------------------------------


def _write_fixtures(tmp_path: Path, count: int | None = None) -> Path:
    out = tmp_path / loader.DEFAULT_FILENAME
    with out.open("w", encoding="utf-8") as handle:
        gen.write_jsonl(gen.generate_records(count), handle)
    return out


def test_load_training_dataset_returns_prompt_records(tmp_path):
    _write_fixtures(tmp_path)
    records = loader.load_training_dataset(str(tmp_path))

    assert len(records) == 16
    for rec in records:
        for key in (
            "id",
            "prompt",
            "state",
            "n",
            "oracle_steps",
            "best_action",
            "best_actions",
            "legal_actions",
            "trace",
            "expected",
        ):
            assert key in rec, f"{rec['id']} missing {key}"
        assert "move_disk" in rec["prompt"]  # prompt mentions the tool
        assert "Current pegs" in rec["prompt"]  # and the board


def test_best_action_is_first_optimal_move(tmp_path):
    _write_fixtures(tmp_path)
    records = loader.load_training_dataset(str(tmp_path))
    for rec in records:
        assert rec["best_action"] == rec["best_actions"][0]
        assert rec["best_actions"] == rec["state"]["optimal_moves"]


def test_prompt_matches_game_format_prompt(tmp_path):
    _write_fixtures(tmp_path)
    records = loader.load_training_dataset(str(tmp_path))
    for rec in records:
        state = gen.record_to_state(rec["state"])
        assert rec["prompt"] == game.format_prompt(state)


def test_load_from_explicit_file_path(tmp_path):
    file_path = _write_fixtures(tmp_path, count=4)
    records = loader.load_training_dataset(str(file_path))
    assert len(records) == 4
    assert records[0]["id"] == "hanoi-n3-optimal"


def test_missing_dataset_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        loader.load_training_dataset(str(tmp_path / "nope.jsonl"))
    assert "dataset_generator.py" in str(exc.value)


def test_default_loader_arg_accepted_and_ignored(tmp_path):
    # AReno may pass default_loader; the loader accepts and ignores it.
    _write_fixtures(tmp_path)
    records = loader.load_training_dataset(str(tmp_path), default_loader=None, extra="x")
    assert len(records) == 16


def test_roundtrip_generator_to_loader_in_memory():
    import json as _json

    buf = io.StringIO()
    gen.write_jsonl(gen.generate_records(3), buf)
    buf.seek(0)
    raw = [_json.loads(line) for line in buf if line.strip()]
    formatted = [loader._format_record(raw=r, index=i) for i, r in enumerate(raw, start=1)]
    assert len(formatted) == 3
    assert formatted[0]["best_action"] == [0, 2]  # first optimal move for n=3


def test_loader_records_are_json_serializable(tmp_path):
    import json as _json

    _write_fixtures(tmp_path)
    records = loader.load_training_dataset(str(tmp_path))
    for rec in records:  # prompt records must be plain dicts
        _json.dumps(rec)


# --- Reward -----------------------------------------------------------------


def _record(n, scenario, *, tool_moves=None, completion=None):
    fixture = next(r for r in gen.generate_records() if r["n"] == n and r["scenario"] == scenario)
    tool_calls = []
    if tool_moves is not None:
        tool_calls = [{"name": "move_disk", "arguments": {"moves": [list(m) for m in tool_moves]}}]
    return SimpleNamespace(
        source_record={"state": fixture},
        completion=completion,
        tool_calls=tool_calls,
    )


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_optimal_solution_rewards_full_score(n):
    optimal = [tuple(m) for m in game.optimal_solution(n)]
    rec = _record(n, "optimal", tool_moves=optimal)
    assert reward.reward_fn(rec) == pytest.approx(game.COMPLETION_REWARD)


def test_longer_legal_solution_scores_lower():
    n = 3
    optimal = [tuple(m) for m in game.optimal_solution(n)]
    # insert a no-op illegal mid-run (penalize keeps going) -> still completes, +1 step
    moves = [optimal[0], (1, 1)] + list(optimal[1:])
    rec = _record(n, "optimal", tool_moves=moves)
    score = reward.reward_fn(rec)
    assert score < game.COMPLETION_REWARD
    assert score == pytest.approx(game.COMPLETION_REWARD - game.EXCESS_STEP_PENALTY * 1)


def test_failure_sequence_scores_zero_progress():
    # Unsolved traces earn partial credit only for PROGRESS toward peg 2, not
    # for legal steps. The failure fixture oscillates the smallest disk without
    # ever building the target stack, so peg 2 ends empty and it scores 0 —
    # and so does a model that tries to "steal" reward with legal-but-stagnant
    # moves (see test_collapse_sequence_no_partial_credit).
    n = 3
    moves = [(0, 2), (2, 0)] * 4  # never solves, peg 2 ends empty
    rec = _record(n, "failure", tool_moves=moves)
    assert not game.replay(moves, n).completed
    assert reward.reward_fn(rec) == pytest.approx(0.0)
    assert reward.reward_fn(rec) < game.COMPLETION_REWARD


def test_empty_moves_scores_zero():
    rec = _record(3, "optimal", tool_moves=[])
    assert reward.reward_fn(rec) == 0.0


def test_wrong_tool_name_ignored_falls_back_to_completion():
    # If the model calls a wrong tool name, reward falls back to completion text.
    rec = _record(
        3,
        "optimal",
        tool_moves=None,
        completion='{"moves": [[0,2],[0,1],[2,1],[0,2],[1,0],[1,2],[0,2]]}',
    )
    assert reward.reward_fn(rec) == pytest.approx(game.COMPLETION_REWARD)


def test_string_arguments_are_parsed():
    import json as _json

    n = 3
    optimal = [list(m) for m in game.optimal_solution(n)]
    rec = SimpleNamespace(
        source_record={"state": next(r for r in gen.generate_records() if r["n"] == n and r["scenario"] == "optimal")},
        completion=None,
        tool_calls=[{"name": "move_disk", "arguments": _json.dumps({"moves": optimal})}],
    )
    assert reward.reward_fn(rec) == pytest.approx(game.COMPLETION_REWARD)


def test_source_target_args_extracted_without_crash():
    # Regression for the cold-start extraction stall: a weak base model emits
    # the prose form {"source": s, "target": t} instead of the schema's
    # {"moves": [[s,t]...]}. _tool_moves must still extract it (not crash / not
    # silently drop) — before the fix, tool_calls=8/8 yet reward_mean=0.0. The
    # extracted single move (0,2) puts disk 1 on peg 2, but disk 1 is not the
    # correct bottom disk (3 for n=3), so under progress-based partial credit
    # it scores 0; this asserts the extraction path, not the score.
    rec = _record(3, "optimal", tool_moves=None)
    rec.tool_calls = [{"name": "move_disk", "arguments": {"source": 0, "target": 2}}]
    assert reward.reward_fn(rec) == 0.0


def test_single_move_dict_wrapped_extracted_without_crash():
    # Also tolerate {"moves": {"source": 0, "target": 2}} — the wrapped-dict
    # extraction path must not crash; the single move scores 0 under
    # progress-based credit (disk 1 is not the correct bottom disk for n=3).
    import json as _json

    rec = SimpleNamespace(
        source_record={"state": next(r for r in gen.generate_records() if r["n"] == 3 and r["scenario"] == "optimal")},
        completion=None,
        tool_calls=[{"name": "move_disk", "arguments": _json.dumps({"moves": {"source": 0, "target": 2}})}],
    )
    assert reward.reward_fn(rec) == 0.0


def test_moves_field_as_json_string_parsed():
    # Regression (2026-07-29 rollout): the Qwen tool-call path serializes the
    # moves list as a JSON *string* — arguments {"moves": "[[0,2],..."}. The
    # extractor must deserialize the string, not reject it as non-list, which
    # made tool_calls=8/8 yet reward_mean=0.0 until _coerce_moves learned to
    # json.loads a str-shaped moves field.
    import json as _json

    optimal = [list(m) for m in game.optimal_solution(3)]
    rec = SimpleNamespace(
        source_record={"state": next(r for r in gen.generate_records() if r["n"] == 3 and r["scenario"] == "optimal")},
        completion=None,
        tool_calls=[{"name": "move_disk", "arguments": _json.dumps({"moves": _json.dumps(optimal)})}],
    )
    assert reward.reward_fn(rec) == pytest.approx(game.COMPLETION_REWARD)


def test_progress_toward_peg2_rewarded():
    # Partial credit is PROGRESS-based: disks correctly stacked on peg 2 from
    # the bottom earn a little. For n=3, the first 4 optimal moves park disk 3
    # on peg 2 (progress=1) without completing, so reward = 0.02 (not 0, not 1.0).
    n = 3
    moves = [tuple(m) for m in game.optimal_solution(n)[:4]]  # disk 3 -> peg 2, not yet solved
    rec = _record(n, "optimal", tool_moves=moves)
    result = game.replay(moves, n)
    assert not result.completed
    assert reward.reward_fn(rec) == pytest.approx(reward.PROGRESS_BONUS * 1)


def test_collapse_sequence_no_partial_credit():
    # Regression (2026-07-29 rollout mode collapse): the model locked reward at
    # 0.04 by replaying [[0,2],[0,1]] — two legal moves that don't push any disk
    # onto peg 2 in the correct bottom-up order. Progress-based partial credit
    # scores this 0, removing the incentive to freeze on this shortcut.
    rec = _record(3, "optimal", tool_moves=[(0, 2), (0, 1)])
    assert reward.reward_fn(rec) == 0.0


def test_malformed_arguments_score_zero_not_crash():
    rec = _record(3, "optimal", tool_moves=None, completion="not json at all")
    assert reward.reward_fn(rec) == 0.0


def test_reward_matches_replay_outcome_for_every_fixture():
    # For each fixture's own trace, reward is consistent with game.replay.
    for fixture in gen.generate_records():
        n = fixture["n"]
        moves = [tuple(m) for m in fixture["trace"]]
        rec = _record(n, fixture["scenario"], tool_moves=moves)
        score = reward.reward_fn(rec)
        result = game.replay(moves, n)
        if result.completed:
            assert score > 0.0
            assert score <= game.COMPLETION_REWARD
        else:
            # unsolved: partial credit is PROGRESS-based — disks correctly
            # stacked on peg 2 from the bottom — and stays below completion.
            expected = min(reward.PROGRESS_BONUS * reward._progress_count(result), reward.PARTIAL_CREDIT_CAP)
            assert score == pytest.approx(expected)
            assert score < game.COMPLETION_REWARD
