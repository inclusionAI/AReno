"""CPU tests for the Battleship autoplay-mode comparison script."""

from __future__ import annotations

import argparse
import json

from tests.test_agentic_battleship_example_cpu import _load_module

compare = _load_module("compare_modes")


# =============================================================================
# Schema / fairness tests (offline, no LLM)
# =============================================================================


def _run_offline(modes: str = "heuristic,random", games: int = 5):
    """Invoke compare.run over a few seeds without an LLM endpoint."""
    args = argparse.Namespace(
        games=games,
        seed=2026,
        max_turns=None,
        modes=modes,
        heuristic_seed=42,
        base_url=None,
        model="policy",
        api_key="token",
        output=None,
        show_boards=False,
    )
    return compare.run(args)


def test_compare_offline_schema_and_seed_alignment():
    """Offline run returns the expected per-game schema and uses the same seeds per mode."""
    report = _run_offline()

    assert set(report) == {"config", "per_mode"}
    assert report["config"]["modes"] == ["heuristic", "random"]

    expected_keys = {"win", "completion", "shots_used", "hits", "sunk_ships", "invalid_shots", "seed"}
    for mode in ("heuristic", "random"):
        per_mode = report["per_mode"][mode]
        assert {"summary", "results"} == set(per_mode)
        for r in per_mode["results"]:
            assert expected_keys <= set(r)
            assert 0.0 <= r["completion"] <= 1.0
            assert isinstance(r["win"], bool)
        # fairness: every mode saw the identical seed sequence
        assert [r["seed"] for r in per_mode["results"]] == report["config"]["seeds"]

    summary = report["per_mode"]["heuristic"]["summary"]
    assert 0.0 <= summary["win_rate"] <= 1.0
    assert summary["mode"] == "heuristic"


def test_compare_llm_drops_without_base_url(capsys):
    """When --base-url is omitted, 'llm' is dropped and only offline modes run."""
    report = _run_offline(modes="heuristic,random,llm")
    assert "llm" not in report["per_mode"]
    assert "llm" not in report["config"]["modes"]
    captured = capsys.readouterr()
    assert "dropping 'llm'" in captured.err


# =============================================================================
# Aggregate / std unit tests
# =============================================================================


def test_compare_std_matches_sample_std():
    """_std uses sample standard deviation (n-1) and 0.0 for <2 points."""
    assert compare._std([1.0, 2.0, 3.0, 4.0]) == 1.2909944487358056
    assert compare._std([1.0]) == 0.0
    assert compare._std([]) == 0.0


def test_compare_summary_llm_extras():
    """_summarize adds latency/token means for the llm mode only."""
    results = [
        {
            "win": True,
            "completion": 1.0,
            "shots_used": 5,
            "hits": 11,
            "sunk_ships": 4,
            "invalid_shots": 0,
            "latency_ms": 100.0,
            "prompt_tokens": 50,
            "completion_tokens": 10,
        },
        {
            "win": False,
            "completion": 0.5,
            "shots_used": 40,
            "hits": 5,
            "sunk_ships": 1,
            "invalid_shots": 2,
            "latency_ms": 300.0,
            "prompt_tokens": 150,
            "completion_tokens": 30,
        },
    ]
    s = compare._summarize("llm", results, max_turns=40)
    assert s["win_rate"] == 0.5
    assert s["latency_ms_mean"] == 200.0
    assert s["prompt_tokens_mean"] == 100.0
    assert "latency_ms_mean" not in compare._summarize("random", results, max_turns=40)


# =============================================================================
# LLM path with a stubbed endpoint (no network, no `openai` install required)
# =============================================================================


class _FakeUsage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c


class _FakeResponse:
    """Minimal stand-in for an OpenAI chat completion response.

    Always fires at A1; repeated shots become invalid but the loop still drives
    shots_used up to max_turns and accumulates latency + token stats.
    """

    def __init__(self) -> None:
        self.usage = _FakeUsage(10, 2)

    def model_dump(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"function": {"name": "fire", "arguments": json.dumps({"coordinate": "A1"})}}]
                    }
                }
            ]
        }


class _FakeCompletions:
    def create(self, **kwargs) -> _FakeResponse:
        return _FakeResponse()


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def test_compare_llm_game_attaches_latency_and_tokens():
    """_play_llm_game drives the loop via a stub client and reports latency/tokens."""
    game = _load_module("game")
    record = game.place_fleet(2026)
    result = compare._play_llm_game(_FakeClient(), "policy", record, max_turns=5)

    expected_keys = {
        "win",
        "completion",
        "shots_used",
        "hits",
        "sunk_ships",
        "invalid_shots",
        "seed",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
    }
    assert expected_keys <= set(result)
    assert result["latency_ms"] >= 0.0
    assert result["prompt_tokens"] > 0
    assert result["completion_tokens"] > 0
