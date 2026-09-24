"""Convert ModelScope BigCodeBench utility tasks for pi in the existing container."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PureWindowsPath

DATASET = "bigcode/bigcodebench"
REVISION = "a4da68573cf2ead10e049a580ba0016d9eb5f281"
SPLIT = "v0.1.4"
# File processing, archives, serialization, checksums and SQLite; no services or task images.
TASK_IDS = tuple(f"BigCodeBench/{number}" for number in (7, 19, 24, 25, 118, 127, 539, 992, 1130, 1134))
# Installed once in the existing environment; no per-task installation.
SHARED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "mpl_toolkits": "matplotlib",
    "seaborn": "seaborn",
    "PIL": "Pillow",
    "faker": "Faker",
    "dateutil": "python-dateutil",
    "pytz": "pytz",
}
EXTERNAL_MODULES = {
    "subprocess",
    "socket",
    "http",
    "urllib",
    "ftplib",
    "smtplib",
    "telnetlib",
    "ctypes",
    "multiprocessing",
    "signal",
    "webbrowser",
    "turtle",
    "tkinter",
}


def check_portability(tree: ast.AST, imports: set[str]) -> None:
    """Conservative screening, not a sandbox or proof of runtime requirements."""
    external = imports & EXTERNAL_MODULES
    if external:
        raise ValueError(f"external process/network/GUI modules: {', '.join(sorted(external))}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if "\n" not in value and (value.startswith(("/", "~/", "../")) or PureWindowsPath(value).is_absolute()):
                raise ValueError("absolute/shared or parent-relative path literal")
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "fork", "execv", "spawnv"}:
            raise ValueError(f"external process operation: {node.attr}")


def verifier_source(test: str, entry_point: str) -> str:
    # Use the same namespace for implementation and tests: upstream tests can
    # refer to module constants and patch __main__ names as well as the function.
    return (
        "import os as _pi_os\n"
        "_pi_os.environ['MPLBACKEND'] = 'Agg'\n"
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


def convert_row(raw: dict, *, source_sha256: str, profile: str = "stdlib") -> dict:
    if profile not in {"smoke", "stdlib", "extended"}:
        raise ValueError(f"unknown profile: {profile}")
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
    unsupported = extra - (SHARED_PACKAGES.keys() if profile == "extended" else set())
    if unsupported:
        raise ValueError(f"{raw['task_id']} requires unsupported non-stdlib modules: {', '.join(sorted(unsupported))}")
    # These ten pinned tasks were inspected individually, including their
    # harmless nonexistent-path test literals. Screen all other tasks.
    if raw["task_id"] not in TASK_IDS:
        check_portability(tree, imports)
    tests = ast.parse(raw["test"])
    if not any(isinstance(node, ast.ClassDef) and node.name == "TestCases" for node in tests.body):
        raise ValueError("BigCodeBench task requires its upstream TestCases class")
    stub = raw["complete_prompt"].rstrip() + "\n    raise NotImplementedError('Implement this feature')\n"
    ast.parse(stub)
    libraries = "Use the Python standard library. "
    if extra:
        libraries = f"Available third-party modules: {', '.join(sorted(extra))}. Do not install packages. "
    prompt = (
        "Implement the requested software utility in solution.py. Read its existing API and docstring; "
        "preserve the function signature and module constants. "
        + libraries
        + "Inspect the workspace, edit the code, and write/run your own local tests. "
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
            "required_packages": sorted({SHARED_PACKAGES[name] for name in extra}),
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
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", row["verify"]],
                cwd=directory,
                env=dict(
                    os.environ,
                    TMPDIR=directory,
                    TMP=directory,
                    TEMP=directory,
                    MPLBACKEND="Agg",
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                output, _ = process.communicate(timeout=row["verify_timeout"])
            finally:
                # Reap reference/test child processes even after a timeout.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
            if process.returncode != expected:
                raise ValueError(
                    f"{row['source']['task_id']} {label} check failed (exit {process.returncode}):\n{output[-4000:]}"
                )


def generate(
    snapshot: Path,
    output: Path,
    *,
    task_ids=None,
    limit=None,
    validate=False,
    profile="stdlib",
    report_path=None,
    progress=False,
) -> int:
    import pyarrow.parquet as pq

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if profile not in {"smoke", "stdlib", "extended"}:
        raise ValueError(f"unknown profile: {profile}")
    report_path = Path(report_path) if report_path else output.with_suffix(output.suffix + ".report.json")
    if report_path.resolve() == output.resolve():
        raise ValueError("report path must differ from dataset output")
    selected = list(task_ids) if task_ids is not None else (list(TASK_IDS) if profile == "smoke" else None)
    if selected is not None and (not selected or len(selected) != len(set(selected))):
        raise ValueError("select at least one task, without duplicate task IDs")
    if selected is not None and limit is not None:
        selected = selected[:limit]
    source = {}
    for path in sorted((snapshot / "data").glob(f"{SPLIT}-*.parquet")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for raw in pq.read_table(path).to_pylist():
            if raw["task_id"] in source:
                raise ValueError(f"duplicate source task: {raw['task_id']}")
            source[raw["task_id"]] = (raw, digest)
    missing = set(selected or []) - source.keys()
    if missing:
        raise ValueError(f"tasks missing from {SPLIT}: {', '.join(sorted(missing))}")
    candidates = selected if selected is not None else sorted(source, key=lambda key: int(key.rsplit("/", 1)[-1]))
    rows, excluded = [], []
    for index, key in enumerate(candidates):
        raw, digest = source[key]
        stage = "selection"
        try:
            row = convert_row(raw, source_sha256=digest, profile=profile)
            if validate:
                stage = "reference_check"
                if progress:
                    print(f"[{index + 1}/{len(candidates)}] Checking {key}; kept {len(rows)}", flush=True)
                check_reference(row, raw["complete_prompt"] + raw["canonical_solution"])
            rows.append(row)
        except (ValueError, SyntaxError, subprocess.TimeoutExpired) as exc:
            reason = (
                f"reference/stub exceeded {exc.timeout}s" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
            )
            excluded.append({"task_id": key, "stage": stage, "reason": reason})
        if limit is not None and len(rows) >= limit:
            break
    packages = sorted({package for row in rows for package in row["source"]["required_packages"]})
    installed = {}
    for package in packages:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    report = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": SPLIT,
        "profile": profile,
        "python": sys.version,
        "reference_checked": validate,
        "source_rows": len(source),
        "selected_task_ids": [row["source"]["task_id"] for row in rows],
        "accepted": len(rows),
        "excluded": excluded,
        "not_considered": len(candidates) - len(rows) - len(excluded),
        "required_packages": packages,
        "installed_versions": installed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Explicit selections and the curated smoke set must not silently shrink.
    if selected is not None and excluded:
        raise ValueError(f"requested tasks failed; see {report_path}: {excluded[0]['reason']}")
    if not rows:
        raise ValueError(f"no eligible tasks; see {report_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["stdlib", "extended", "smoke"], default="stdlib")
    parser.add_argument("--limit", type=int, help="stop after N accepted tasks (default: all eligible tasks)")
    parser.add_argument("--task-id", action="append", help="select other upstream task IDs; repeat for multiple tasks")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--report", type=Path, help="selection report (default: OUTPUT.report.json)")
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
            Path(snapshot),
            args.output,
            task_ids=args.task_id,
            limit=args.limit,
            validate=args.check_reference,
            profile=args.profile,
            report_path=args.report,
            progress=True,
        )
    except (ValueError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {count} BigCodeBench utility tasks to {args.output}")
    print(f"Selection report: {args.report or str(args.output) + '.report.json'}")


if __name__ == "__main__":
    main()
