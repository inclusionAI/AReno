"""Run immutable upstream regression tests against submitted package code."""

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def unpack(payload, root):
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz") as archive:
        for member in archive:
            path = Path(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] == ".pi"
            ):
                raise ValueError("invalid snapshot entry")
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())


def verify(workspace, snapshot, package, test_ids, protected):
    with tempfile.TemporaryDirectory(prefix="areno-pi-grade-") as directory:
        root = Path(directory)
        unpack(snapshot, root)
        source = workspace / package
        if not source.is_dir() or source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("invalid submitted package")
        if (root / package).exists():
            shutil.rmtree(root / package)
        shutil.copytree(source, root / package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # Restore tests nested inside the editable package (e.g. pydicom).
        for name in protected:
            target = root / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        unpack(snapshot, root)
        # snapshot excludes editable code, so restoring it cannot undo the repair.
        runner = r"""import json, sys
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / 'src')]
import pytest
class Outcomes:
    def __init__(self): self.items = {}
    def pytest_runtest_logreport(self, report):
        if report.when == 'call': self.items[report.nodeid] = report.outcome
        elif report.failed or report.skipped: self.items[report.nodeid] = 'error' if report.failed else 'skipped'
plugin = Outcomes()
code = pytest.main(['-q', '-c', '/dev/null', '--rootdir=.', *json.loads(sys.argv[1])], plugins=[plugin])
Path('outcomes.json').write_text(json.dumps({'exit_code': int(code), 'tests': plugin.items}))
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", runner, json.dumps(test_ids)],
            cwd=root,
            env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH="", PYTHONHASHSEED="0"),
            check=False,
        )
        path = root / "outcomes.json"
        outcomes = json.loads(path.read_text()) if path.exists() else {"tests": {}, "exit_code": result.returncode}
        outcomes["complete"] = set(outcomes["tests"]) == set(test_ids)
        print("ARENO_SWE_TEST_RESULT=" + json.dumps(outcomes), flush=True)
        return (
            outcomes["exit_code"] == 0
            and outcomes["complete"]
            and bool(test_ids)
            and all(v == "passed" for v in outcomes["tests"].values())
        )
