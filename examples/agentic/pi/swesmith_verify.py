"""Grade a sqlparse workspace against immutable SWE-smith regression tests."""

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def verify(workspace: Path, protected_files: dict[str, str], test_ids: list[str]) -> bool:
    # Grade only submitted package code in a fresh directory. Changes to tests,
    # conftest, pytest configuration or scripts cannot weaken the trusted suite.
    with tempfile.TemporaryDirectory(prefix="areno-pi-grade-") as directory:
        root = Path(directory)
        package = workspace / "sqlparse"
        if not package.is_dir() or package.is_symlink():
            raise ValueError("missing sqlparse package")
        for source in package.rglob("*"):
            if source.is_symlink():
                raise ValueError("symlinks are not supported in submitted package code")
        shutil.copytree(package, root / "sqlparse", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name, encoded in protected_files.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "tests":
                raise ValueError("invalid protected test path")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(encoded, validate=True))
        runner = """import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import pytest
class Outcomes:
    def __init__(self):
        self.items = {}
    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.items[report.nodeid] = report.outcome
        elif report.failed or report.skipped:
            self.items[report.nodeid] = 'error' if report.failed else 'skipped'
plugin = Outcomes()
code = pytest.main(['-q', '-c', '/dev/null', '--rootdir=.', *json.loads(sys.argv[2])], plugins=[plugin])
Path('outcomes.json').write_text(json.dumps({'exit_code': int(code), 'tests': plugin.items}))
raise SystemExit(code)
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", runner, str(root), json.dumps(test_ids)],
            cwd=root,
            env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH="", PYTHONHASHSEED="0"),
            check=False,
        )
        outcome_path = root / "outcomes.json"
        outcomes = (
            json.loads(outcome_path.read_text())
            if outcome_path.exists()
            else {"tests": {}, "exit_code": result.returncode}
        )
        outcomes["expected"] = len(test_ids)
        outcomes["complete"] = set(outcomes["tests"]) == set(test_ids)
        print("ARENO_SWE_TEST_RESULT=" + json.dumps(outcomes), flush=True)
        return (
            result.returncode == 0
            and outcomes["complete"]
            and bool(test_ids)
            and all(value == "passed" for value in outcomes["tests"].values())
        )
