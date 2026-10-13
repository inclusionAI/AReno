"""Normalize self-contained pi coding tasks for AReno training."""

import json
from pathlib import Path


def _rows(dataset_path, default_loader):
    path = Path(dataset_path)
    if path.is_file() and path.suffix == ".jsonl":
        # Archives can exceed an Arrow JSON reader's block size. Stream local
        # records directly, without making a second full-file string copy.
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from default_loader(dataset_path)


def load_training_dataset(dataset_path, *, default_loader, **kwargs):
    records = []
    for row in _rows(dataset_path, default_loader):
        record = dict(row)
        prompt = record.get("prompt") or record.get("problem_statement")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("pi task requires prompt or problem_statement")
        if not isinstance(record.get("files"), dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in record["files"].items()
        ):
            raise ValueError("pi task files must map relative paths to text")
        if not isinstance(record.get("verify"), str) or not record["verify"].strip():
            raise ValueError("pi task requires trusted Python verify code")
        binary = record.get("binary_files", {})
        if "workspace_archive" in record and not isinstance(record["workspace_archive"], str):
            raise ValueError("workspace_archive must be a base64 tar.gz string")
        if not isinstance(binary, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in binary.items()
        ):
            raise ValueError("binary_files must map relative paths to base64 strings")
        record["prompt"] = prompt
        records.append(record)
    return records
