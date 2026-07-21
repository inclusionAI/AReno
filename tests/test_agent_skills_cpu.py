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
    result = json.loads(process.stdout)

    assert process.returncode == 0, result
    assert result["skill_count"] == 10
    assert result["script_count"] >= 15
