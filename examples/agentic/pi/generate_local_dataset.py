"""Convert ModelScope BigCodeBench utility tasks for pi in the existing container."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET = "bigcode/bigcodebench"
REVISION = "a4da68573cf2ead10e049a580ba0016d9eb5f281"
SPLIT = "v0.1.4"
# File processing, archives, serialization, checksums and SQLite; no services or task images.
TASK_IDS = tuple(f"BigCodeBench/{number}" for number in (7, 19, 24, 25, 118, 127, 539, 992, 1130, 1134))


def verifier_source(test: str, entry_point: str) -> str:
    # Use the same namespace for implementation and tests: upstream tests can
    # refer to module constants and patch __main__ names as well as the function.
    return (
        "from pathlib import Path as _pi_Path\n"
        "exec(compile(_pi_Path('solution.py').read_text(), 'solution.py', 'exec'), globals())\n"
        f"if not callable(globals().get({entry_point!r})): raise RuntimeError('Missing task entry point')\n"
        f"exec(compile({test!r}, '<trusted-tests>', 'exec'), globals())\n"
        "import unittest as _pi_unittest\n"
        "_pi_suite = _pi_unittest.defaultTestLoader.loadTestsFromTestCase(TestCases)\n"
        "if _pi_suite.countTestCases() == 0: raise RuntimeError('No grading tests found')\n"
        "_pi_result = _pi_unittest.TextTestRunner(verbosity=2).run(_pi_suite)\n"
        "raise SystemExit(0 if _pi_result.wasSuccessful() and not _pi_result.skipped else 1)\n"
    )


def convert_row(raw: dict, *, source_sha256: str) -> dict:
    for key in ("task_id", "complete_prompt", "instruct_prompt", "canonical_solution", "test", "entry_point"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"BigCodeBench row requires {key}")
    if not raw["entry_point"].isidentifier():
        raise ValueError("invalid entry_point")
    tree = ast.parse(raw["complete_prompt"] + raw["canonical_solution"] + "\n" + raw["test"])
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    libs = ast.literal_eval(raw["libs"]) if isinstance(raw.get("libs"), str) else raw.get("libs", [])
    if not isinstance(libs, list) or not all(isinstance(name, str) for name in libs):
        raise ValueError("libs must be a list of module names")
    imports.update(name.split(".")[0] for name in libs)
    extra = imports - sys.stdlib_module_names
    if extra:
        raise ValueError(f"{raw['task_id']} requires non-stdlib modules: {', '.join(sorted(extra))}")
    tests = ast.parse(raw["test"])
    if not any(isinstance(node, ast.ClassDef) and node.name == "TestCases" for node in tests.body):
        raise ValueError("BigCodeBench task requires its upstream TestCases class")
    stub = raw["complete_prompt"].rstrip() + "\n    raise NotImplementedError('Implement this feature')\n"
    ast.parse(stub)
    prompt = (
        "Implement the requested software utility in solution.py. Read its existing API and docstring; "
        "preserve the function signature and module constants. Use the Python standard library. "
        "Inspect the workspace, edit the code, and write/run your own local tests. "
        "The controller will run separate upstream tests after you finish.\n\n" + raw["instruct_prompt"]
    )
    return {
        "prompt": prompt,
        "files": {"solution.py": stub, "README.md": prompt + "\n"},
        "verify": verifier_source(raw["test"], raw["entry_point"]),
        "source": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": SPLIT,
            "task_id": raw["task_id"],
            "sha256": source_sha256,
        },
        "max_turns": 32,
        "timeout": 600,
        "verify_timeout": 30,
    }


def check_reference(row: dict, reference: str) -> None:
    """Check the upstream solution passes and the unfinished implementation fails."""
    for label, source, expected in (("reference", reference, 0), ("stub", row["files"]["solution.py"], 1)):
        with tempfile.TemporaryDirectory(prefix="areno-pi-check-") as directory:
            (Path(directory) / "solution.py").write_text(source)
            result = subprocess.run(
                [sys.executable, "-I", "-c", row["verify"]],
                cwd=directory,
                env=dict(os.environ, TMPDIR=directory, TMP=directory, TEMP=directory),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
            if result.returncode != expected:
                raise ValueError(
                    f"{row['source']['task_id']} {label} check failed (exit {result.returncode}):\n{result.stdout[-4000:]}"
                )


def generate(snapshot: Path, output: Path, *, task_ids=None, limit=None, validate=False) -> int:
    import pyarrow.parquet as pq

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected = list(TASK_IDS if task_ids is None else task_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("select at least one task, without duplicate task IDs")
    if limit is not None:
        selected = selected[:limit]
    found = {}
    for path in sorted((snapshot / "data").glob(f"{SPLIT}-*.parquet")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for raw in pq.read_table(path).to_pylist():
            if raw["task_id"] not in selected:
                continue
            if raw["task_id"] in found:
                raise ValueError(f"duplicate source task: {raw['task_id']}")
            row = convert_row(raw, source_sha256=digest)
            if validate:
                check_reference(row, raw["complete_prompt"] + raw["canonical_solution"])
            found[raw["task_id"]] = row
    missing = set(selected) - found.keys()
    if missing:
        raise ValueError(f"tasks missing from {SPLIT}: {', '.join(sorted(missing))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(found[key], ensure_ascii=False) + "\n" for key in selected), encoding="utf-8")
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="take the first N of the ten selected utility tasks")
    parser.add_argument("--task-id", action="append", help="select other upstream task IDs; repeat for multiple tasks")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--check-reference", action="store_true", help="execute upstream tests on reference and stub code"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    from modelscope import snapshot_download

    snapshot = snapshot_download(
        DATASET,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[f"data/{SPLIT}-*.parquet", "README.md"],
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    try:
        count = generate(
            Path(snapshot), args.output, task_ids=args.task_id, limit=args.limit, validate=args.check_reference
        )
    except (ValueError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {count} BigCodeBench utility tasks to {args.output}")


if __name__ == "__main__":
    main()
