from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_repository_agent_skills_are_valid():
    """Project skills should retain valid metadata, links, and script entrypoints."""

    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, str(root / ".agents/scripts/validate_skills.py")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if process.returncode != 0:
        raise AssertionError(
            f"validate_skills.py failed with exit code {process.returncode}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    result = json.loads(process.stdout)
    assert result["skill_count"] == 10
    assert result["script_count"] >= 15


def test_transcript_validator_accepts_normalized_argument_objects(tmp_path):
    root = Path(__file__).resolve().parents[1]
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "guess_code", "arguments": {"code": "0123"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": '{"solved":true}'},
                ]
            }
        ),
        encoding="utf-8",
    )

    process = subprocess.run(
        [
            sys.executable,
            str(root / ".agents/skills/areno-build-agentic-workflow/scripts/validate_transcript.py"),
            str(transcript),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout)["ok"] is True
