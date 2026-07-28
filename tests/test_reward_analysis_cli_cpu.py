"""CPU tests for the ``areno reward-analysis`` CLI subcommand.

Asserts emitted fields and error messages, not only exit status.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from areno.cli.reward_analysis import reward_analysis_command


def _fixture(dirpath: Path) -> Path:
    path = dirpath / "reward_components.0.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step": 0, "name": "a", "value": 1.0}),
                json.dumps({"step": 0, "name": "b", "value": 0.0}),
                json.dumps({"step": 1, "name": "a", "value": 0.0}),
                json.dumps({"step": 1, "name": "b", "value": 1.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_success_json_emits_component_fields():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        _fixture(Path(tmp))
        result = runner.invoke(reward_analysis_command, ["--metrics-dir", tmp, "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    names = {c["name"] for c in payload["snapshot"]["components"]}
    assert names == {"a", "b"}
    comp = next(c for c in payload["snapshot"]["components"] if c["name"] == "a")
    for field in (
        "current",
        "mean",
        "std",
        "zero_fraction",
        "outlier_fraction",
        "non_finite_fraction",
        "missing_count",
        "weighted_contribution",
        "contribution_fraction",
        "history",
        "distribution",
    ):
        assert field in comp, field
    assert len(payload["snapshot"]["steps"]) == 2
    assert payload["errors"] == []


def test_success_human_table_contains_headers():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        _fixture(Path(tmp))
        result = runner.invoke(reward_analysis_command, ["--metrics-dir", tmp])

    assert result.exit_code == 0, result.output
    assert "Reward components in" in result.output
    assert "component" in result.output
    assert "contrib%" in result.output


def test_missing_path_raises_usage_error_with_stage_and_input():
    runner = CliRunner()
    result = runner.invoke(reward_analysis_command, ["--metrics-dir", "/tmp/areno-nope-xyz"])
    assert result.exit_code != 0
    assert "artifact resolution" in result.output
    assert "/tmp/areno-nope-xyz" in result.output


def test_empty_dir_reports_no_data_cleanly():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        result = runner.invoke(reward_analysis_command, ["--metrics-dir", tmp])
    assert result.exit_code == 0, result.output
    assert "No reward component data" in result.output


def test_malformed_row_reported_without_sample_text():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "reward_components.0.jsonl"
        path.write_text(
            json.dumps({"step": 0, "name": "a", "value": 1.0}) + "\n{broken with secret-prompt-text}\n",
            encoding="utf-8",
        )
        result = runner.invoke(reward_analysis_command, ["--metrics-dir", tmp, "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["errors"], "expected at least one artifact error"
    rendered = json.dumps(payload["errors"])
    assert "artifact parse" in rendered
    assert "secret-prompt-text" not in rendered
