from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_uses_version_file(tmp_path: Path) -> None:
    for filename in ("pyproject.toml", "setup.py", "README.md", "LICENSE"):
        shutil.copy(ROOT / filename, tmp_path / filename)
    expected_version = "9.8.7.dev6"
    (tmp_path / "VERSION").write_text(f"{expected_version}\n")

    env = os.environ.copy()
    env["ARENO_BUILD_EXT"] = "0"

    result = subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == expected_version
