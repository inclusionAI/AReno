from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODING_DIR = ROOT / "examples" / "agentic" / "coding"
if str(CODING_DIR) not in sys.path:
    sys.path.insert(0, str(CODING_DIR))

from coding_tools import CodingWorkspace, run_tool  # noqa: E402


def test_background_command_output_range(tmp_path):
    workspace = CodingWorkspace(task={}, root=tmp_path, cleanup_on_close=False)
    command = (
        f"{sys.executable} -c \"import time; "
        "print('alpha', flush=True); time.sleep(0.2); print('omega', flush=True)\""
    )

    started = run_tool(workspace, "run_command", {"command": command, "background": True})
    assert started["running"] is True
    assert started["task_id"] == "bg-1"

    time.sleep(0.5)
    output = run_tool(workspace, "read_background_output", {"task_id": "bg-1", "start": 0, "end": 200})

    assert output["running"] is False
    assert output["returncode"] == 0
    assert "alpha" in output["output"]
    assert "omega" in output["output"]
    assert output["end"] <= output["output_chars"]


def test_background_command_can_read_later_range(tmp_path):
    workspace = CodingWorkspace(task={}, root=tmp_path, cleanup_on_close=False)
    command = f"{sys.executable} -c \"print('0123456789')\""
    started = workspace.run_command(command, background=True)

    time.sleep(0.2)
    output = workspace.read_background_output(started["task_id"], start=3, end=7)

    assert output["output"] == "3456"
    assert output["start"] == 3
    assert output["end"] == 7
