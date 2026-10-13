"""Prepare real upstream issue repairs from a pinned ModelScope SWE-bench release."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

from generate_dataset import check_task, write_snapshot

DATASET = "AI-ModelScope/SWE-bench"
REVISION = "8fe7c991907c740e8400dd51d8b08af51271acf8"
PACKAGES = {
    "sphinx-doc/sphinx": "sphinx",
    "pydicom/pydicom": "pydicom",
    "pylint-dev/astroid": "astroid",
    "marshmallow-code/marshmallow": "src/marshmallow",
    "psf/requests": "requests",
    "pallets/flask": "src/flask",
}


def archive_files(repository, commit):
    data = subprocess.check_output(["git", "-C", str(repository), "archive", commit])
    result = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for entry in archive:
            if entry.isdir():
                continue
            if not entry.isfile() or Path(entry.name).is_absolute() or ".." in Path(entry.name).parts:
                raise ValueError(f"unsupported archive entry: {entry.name}")
            result[entry.name] = archive.extractfile(entry).read()
    return result


def compress(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    return base64.b64encode(buffer.getvalue()).decode()


def apply(root, patch):
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"], cwd=root, input=patch, text=True, check=True, capture_output=True
    )


def prepare(raw, cache, python):
    package = PACKAGES[raw["repo"]]
    paths = re.findall(r"^\+\+\+ b/(.+)$", raw["patch"], re.MULTILINE)
    if not paths or any(not p.startswith(package + "/") or "/tests/" in p for p in paths):
        raise ValueError("repair changes files outside the editable package")
    selection = raw.get("selection", {})
    f2p = selection.get("fail_to_pass", json.loads(raw["FAIL_TO_PASS"]))
    all_p2p = json.loads(raw["PASS_TO_PASS"])
    # Keep every failing regression plus a deterministic sample of passing tests.
    p2p = sorted(t for t in set(all_p2p) - set(f2p) if ("[" not in t or t.endswith("]")) and "/n/fs/" not in t)[:20]
    p2p = selection.get("pass_to_pass", p2p)
    tests = sorted(set(f2p + p2p))
    if not f2p or not p2p:
        raise ValueError("missing regression outcomes")
    pristine = archive_files(cache / raw["repo"].split("/")[-1], raw["base_commit"])
    with tempfile.TemporaryDirectory(prefix="areno-pi-real-") as directory:
        root = Path(directory)
        write_snapshot(root, pristine)
        apply(root, raw["test_patch"])
        snapshot = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        protected = {
            name: content
            for name, content in snapshot.items()
            if not name.startswith(package + "/") or "/tests/" in name or "/data/" in name
        }
        verifier = Path(__file__).with_name("swebench_verify.py").read_text()
        verifier += f"\nraise SystemExit(0 if verify(Path.cwd(), {compress(protected)!r}, {package!r}, {tests!r}, {[package + '/tests', package + '/data']!r}) else 1)\n"
        broken = check_task(root, verifier, python)
        if (
            not broken["complete"]
            or any(broken["tests"].get(t) != "failed" for t in f2p)
            or any(broken["tests"].get(t) != "passed" for t in p2p)
        ):
            raise ValueError("base regression mismatch: " + broken["failure_output"][-2500:])
        apply(root, raw["patch"])
        gold = check_task(root, verifier, python)
        if gold["returncode"] or not gold["complete"]:
            raise ValueError("reference repair fails: " + gold["failure_output"][-2500:])
    files = {}
    files["REPRODUCTION.md"] = "# Verified regression failures\n\n```text\n" + broken["failure_output"] + "\n```\n"
    files["AGENTS.md"] = (
        "# Repair workspace\n\nThe current directory is the issue's repository at its historical base commit. "
        f"Edit {package}/. Read REPRODUCTION.md and relevant tests, use rg to locate implementation, "
        "then apply a focused repair. Python dependencies are already installed. Do not install packages "
        "or fetch upstream source. Tests and configuration are immutable during grading. "
        "Run tests with python -m pytest -q -c /dev/null.\n"
    )
    source = {k: raw[k] for k in ["repo", "instance_id", "base_commit", "split"]}
    source.update(
        dataset=DATASET,
        revision=REVISION,
        synthetic=False,
        pull_request_url=f"https://github.com/{raw['repo']}/pull/{raw['instance_id'].rsplit('-', 1)[1]}",
        fail_to_pass=f2p,
        pass_to_pass=p2p,
        available_pass_to_pass=len(all_p2p),
        test_patch_sha256=hashlib.sha256(raw["test_patch"].encode()).hexdigest(),
    )
    source["task_id"] = raw["instance_id"]
    for key in ["issue_url", "issue_title", "problem_statement_sha256"]:
        if key in selection:
            source[key] = selection[key]
    return {
        "prompt": f"Resolve this real upstream issue in {raw['repo']}. Your current directory is the repository root. "
        f"Only changes under {package}/ are graded against immutable upstream tests. "
        "Use the installed Python environment; do not install dependencies or fetch upstream fixes. "
        "Read REPRODUCTION.md, inspect the affected implementation, repair it, and run regression tests.\n\n"
        + raw["problem_statement"],
        "files": files,
        "workspace_archive": compress(snapshot),
        "verify": verifier,
        "source": source,
        "max_turns": 32,
        "timeout": 3600,
        "verify_timeout": 120,
    }, {"task_id": raw["instance_id"], "broken": broken, "reference": gold}


def source_rows(snapshot):
    import pyarrow.parquet as pq

    checksums = {
        "test": "db4f70ef735b3162c74801ddcdf8d7bae8d704193788c6d844f898c20b571cbb",
        "dev": "d758d54540aa4140d0274ed0cc93b8288aa6f323c3c603e2557e7666a47fc41b",
    }
    rows = {}
    for split, expected in checksums.items():
        path = snapshot / f"data/{split}-00000-of-00001.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{split} source does not match the pinned ModelScope release")
        rows.update({row["instance_id"]: dict(row, split=split) for row in pq.read_table(path).to_pylist()})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Previously downloaded pinned ModelScope snapshot")
    parser.add_argument("--repo-cache", type=Path, default=Path.home() / ".cache/areno/pi-repositories")
    parser.add_argument("--python", required=True, help="Shared task environment Python")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("swebench_100.json"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    selection = json.loads(args.manifest.read_text())[: args.limit]
    if args.limit < 1 or args.workers < 1 or len(selection) != args.limit:
        parser.error("limit must fit the manifest; workers must be positive")
    if len({r["issue_url"] for r in selection}) != len(selection):
        raise ValueError("manifest must contain distinct upstream issues")
    if args.snapshot is None:
        from modelscope.hub.snapshot_download import snapshot_download

        args.snapshot = Path(
            snapshot_download(
                DATASET,
                repo_type="dataset",
                revision=REVISION,
                allow_patterns=[f"data/{split}-00000-of-00001.parquet" for split in ("test", "dev")],
            )
        )
    source = source_rows(args.snapshot)
    rows = []
    for item in selection:
        row = source[item["task_id"]]
        if (
            row["repo"] != item["repo"]
            or row["base_commit"] != item["base_commit"]
            or hashlib.sha256(row["problem_statement"].encode()).hexdigest() != item["problem_statement_sha256"]
        ):
            raise ValueError(f"manifest/source mismatch: {item['task_id']}")
        rows.append(dict(row, selection=item))
    args.repo_cache.mkdir(parents=True, exist_ok=True)
    for repo in sorted({row["repo"] for row in rows}):
        path = args.repo_cache / repo.split("/")[-1]
        if not (path / ".git").exists():
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--no-checkout", f"https://github.com/{repo}.git", str(path)],
                check=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tasks = args.output.with_suffix(".tasks")
    tasks.mkdir(exist_ok=True)
    environment = subprocess.check_output(
        [
            args.python,
            "-I",
            "-c",
            "import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))",
        ],
        text=True,
    )
    report = {"environment": json.loads(environment), "accepted": [], "rejected": []}

    def work(row):
        try:
            task, evidence = prepare(row, args.repo_cache, args.python)
            (tasks / (row["instance_id"] + ".json")).write_text(json.dumps(task))
            return row["instance_id"], evidence, None
        except Exception as exc:
            return row["instance_id"], None, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for task_id, evidence, error in executor.map(work, rows):
            if error:
                report["rejected"].append({"task_id": task_id, "reason": error})
                print("REJECT", task_id, error[-250:].replace("\n", " "), flush=True)
            else:
                report["accepted"].append(evidence)
                print("ACCEPT", len(report["accepted"]), task_id, flush=True)
            args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    if len(report["accepted"]) != args.limit:
        raise SystemExit(f"Only {len(report['accepted'])}/{args.limit} passed; training dataset not published")
    pending = args.output.with_suffix(".pending.jsonl")
    with pending.open("w") as stream:
        for evidence in report["accepted"]:
            stream.write((tasks / (evidence["task_id"] + ".json")).read_text() + "\n")
    pending.replace(args.output)
    print(f"Wrote {args.limit} validated real-issue tasks to {args.output}")


if __name__ == "__main__":
    main()
