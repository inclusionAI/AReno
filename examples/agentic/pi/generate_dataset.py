"""Generate repository-level sqlparse repair tasks from ModelScope SWE-smith."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

DATASET = "SWE-bench/SWE-smith"
REVISION = "552363dc45a3bdce7ff4f3d7cdaf6945117f66ca"
REPO = "swesmith/andialbrecht__sqlparse.e57923b3"
UPSTREAM = "https://github.com/andialbrecht/sqlparse.git"
COMMIT = "e57923b3aa823c524c807953cecc48cf6eec2cb2"
# The pinned source locates this repository in this one shard.
SHARD = "data/train-00003-of-00011.parquet"


def repository_files(cache: Path) -> dict[str, bytes]:
    if not (cache / ".git").exists():
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM, str(cache)], check=True)
    archive = subprocess.check_output(["git", "-C", str(cache), "archive", COMMIT])
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.isdir():
                continue
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError(f"unsupported repository entry: {member.name}")
            files[member.name] = tar.extractfile(member).read()
    return files


def write_snapshot(root: Path, files: dict[str, bytes]):
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def check_task(root: Path, verifier: str, python: str) -> dict:
    verifier_path = root / ".areno-verify.py"
    verifier_path.write_text(verifier)
    process = subprocess.Popen(
        [python, "-I", str(verifier_path)],
        cwd=root,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=120)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
    summaries = [line.split("=", 1)[1] for line in output.splitlines() if line.startswith("ARENO_SWE_TEST_RESULT=")]
    if not summaries:
        raise ValueError(f"verifier failed without outcomes (exit {process.returncode}): {output[-2000:]}")
    summary = json.loads(summaries[-1])
    summary["returncode"] = process.returncode
    return summary


def convert_row(raw: dict, pristine: dict[str, bytes], *, python: str, digest: str) -> tuple[dict, dict]:
    if raw.get("repo") != REPO or not raw.get("problem_statement"):
        raise ValueError("task must have an issue statement for the pinned repository")
    paths = re.findall(r"^\+\+\+ b/(.+)$", raw["patch"], re.MULTILINE)
    if not paths or any(
        not name.startswith("sqlparse/") or not name.endswith(".py") or ".." in Path(name).parts for name in paths
    ):
        raise ValueError("select package-only repairs")
    if len(set(paths)) != 1:
        raise ValueError("task requires a single-file package repair")
    f2p, p2p = raw["FAIL_TO_PASS"], raw["PASS_TO_PASS"]
    if not f2p or not p2p or not all(isinstance(t, str) and t.startswith("tests/") for t in f2p + p2p):
        raise ValueError("task requires upstream FAIL_TO_PASS and PASS_TO_PASS test IDs")
    test_ids = sorted(set(f2p + p2p))
    protected = {
        name: base64.b64encode(content).decode() for name, content in pristine.items() if name.startswith("tests/")
    }
    verifier = Path(__file__).with_name("swesmith_verify.py").read_text()
    verifier += f"\nraise SystemExit(0 if verify(Path.cwd(), {protected!r}, {test_ids!r}) else 1)\n"
    with tempfile.TemporaryDirectory(prefix="areno-pi-reference-") as directory:
        root = Path(directory)
        write_snapshot(root, pristine)
        gold = check_task(root, verifier, python)
        if gold["returncode"] != 0 or not gold["complete"]:
            raise ValueError("pristine reference fails the fixed shared environment")
        # SWE-smith patch introduces the bug; it is NOT a gold repair patch.
        subprocess.run(["git", "apply", "--check", "-"], cwd=root, input=raw["patch"], text=True, check=True)
        subprocess.run(["git", "apply", "-"], cwd=root, input=raw["patch"], text=True, check=True)
        broken = check_task(root, verifier, python)
        if (
            broken["returncode"] != 1
            or not broken["complete"]
            or any(broken["tests"].get(t) != "failed" for t in f2p)
            or any(broken["tests"].get(t) != "passed" for t in p2p)
        ):
            raise ValueError("injected bug does not reproduce the upstream F2P/P2P outcomes")
        files, binary = {}, {}
        for name in pristine:
            content = (root / name).read_bytes()
            try:
                files[name] = content.decode("utf-8")
            except UnicodeDecodeError:
                binary[name] = base64.b64encode(content).decode()
    files.setdefault(
        "AGENTS.md",
        "# Repository workspace\n\n"
        "The current working directory is this task's sqlparse repository root. "
        "Use relative paths here; do not search other workspaces or /testbed.\n\n"
        "- sqlparse/sql.py: Token and TokenList classes.\n"
        "- sqlparse/engine/grouping.py: SQL token grouping.\n"
        "- sqlparse/keywords.py and sqlparse/tokens.py: keyword and token definitions.\n"
        "- tests/: existing pytest regression tests.\n\n"
        "Python and pytest are already installed. Do not install packages or fetch source. "
        "Reproduce the reported failing tests, edit package code, and rerun the regression suite.\n",
    )
    prompt = (
        "Fix the following bug report in this sqlparse repository. Inspect the implementation, reproduce "
        "the failures, edit the package, and run relevant pytest tests. Preserve existing public behavior. "
        "Your current working directory is the repository root. Use relative paths within it; "
        "do not inspect other workspaces. "
        "Only changes under sqlparse/ are graded; the controller runs immutable upstream regression tests "
        "in a fresh directory. pytest is already installed in the shared Python environment. "
        "Do not install packages, fetch external source, or replace the repository with an upstream copy.\n\n"
        + raw["problem_statement"]
        + "\n\nStart by reproducing these failing regression tests:\n"
        + shlex.join(["python", "-m", "pytest", "-q", "-c", "/dev/null", *f2p])
        + "\nRead the relevant implementation sections and apply a minimal repair. "
        "Use read/edit tools or a Python heredoc for multiline code. Then rerun the failing tests "
        "and regression suite before finishing."
    )
    row = {
        "prompt": prompt,
        "files": files,
        "binary_files": binary,
        "verify": verifier,
        "source": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "train",
            "task_id": raw["instance_id"],
            "repo": UPSTREAM,
            "base_commit": COMMIT,
            "sha256": digest,
            "fail_to_pass": f2p,
            "pass_to_pass": p2p,
            "difficulty": "focused",
        },
        "max_turns": 32,
        "timeout": 1800,
        "verify_timeout": 120,
    }
    evidence = {"task_id": raw["instance_id"], "changed_files": paths, "reference": gold, "broken": broken}
    return row, evidence


def select_candidates(raw_rows: list[dict], seed: int) -> list[dict]:
    candidates = [
        row
        for row in raw_rows
        if row["problem_statement"]
        and (".func_basic__" in row["instance_id"] or ".lm_rewrite__" in row["instance_id"])
        and row["patch"].count("diff --git") == 1
        and 1 <= len(row["FAIL_TO_PASS"]) <= 8
    ]
    candidates.sort(key=lambda row: row["instance_id"])
    random.Random(seed).shuffle(candidates)
    return candidates


def generate(
    snapshot: Path,
    output: Path,
    *,
    repo_cache: Path,
    python: str,
    limit: int = 4,
    seed: int = 0,
) -> dict:
    import pyarrow.parquet as pq

    if limit <= 0:
        raise ValueError("limit must be positive")
    path = snapshot / SHARD
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    raw_rows = pq.read_table(path, filters=[("repo", "==", REPO)]).to_pylist()
    candidates = select_candidates(raw_rows, seed)
    pristine = repository_files(repo_cache)
    rows, evidence, excluded = [], [], []
    for raw in candidates:
        print(f"Checking {raw['instance_id']}", flush=True)
        try:
            row, check = convert_row(raw, pristine, python=python, digest=digest)
        except (ValueError, subprocess.SubprocessError) as exc:
            excluded.append({"task_id": raw["instance_id"], "reason": str(exc)})
            print(f"Excluded: {exc}", flush=True)
            continue
        rows.append(row)
        evidence.append(check)
        if len(rows) >= limit:
            break
    environment = subprocess.check_output(
        [
            python,
            "-I",
            "-c",
            "import json,sys,pytest; print(json.dumps({'python':sys.version,'pytest':pytest.__version__}))",
        ],
        text=True,
    )
    report = {
        "dataset": DATASET,
        "revision": REVISION,
        "repo": UPSTREAM,
        "base_commit": COMMIT,
        "source_sha256": digest,
        "seed": seed,
        "difficulty": "focused",
        "candidate_count": len(candidates),
        "accepted": len(rows),
        "selected_task_ids": [r["source"]["task_id"] for r in rows],
        "reference_checks": evidence,
        "excluded": excluded,
        "environment": json.loads(environment),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(output.suffix + ".report.json").write_text(json.dumps(report, indent=2) + "\n")
    if len(rows) < limit:
        raise ValueError("not enough reproducible tasks; see selection report")
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("pi-swesmith.jsonl"))
    parser.add_argument("--repo-cache", type=Path, default=Path(".cache/pi-sqlparse-source"))
    parser.add_argument("--python", default=sys.executable, help="shared task environment Python executable")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("limit must be positive")
    from modelscope import snapshot_download

    snapshot = snapshot_download(DATASET, repo_type="dataset", revision=REVISION, allow_patterns=[SHARD])
    report = generate(
        Path(snapshot),
        args.output,
        repo_cache=args.repo_cache,
        python=args.python,
        limit=args.limit,
        seed=args.seed,
    )
    print(f"Wrote {report['accepted']} verified repository repair tasks to {args.output}")


if __name__ == "__main__":
    main()
