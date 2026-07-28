"""CLI command for near-duplicate training example detection (issue #218).

``areno dedup`` scans a JSONL/JSON dataset and reports duplicate groups
without modifying the source data.  Two modes are available:

* ``--mode exact`` (default): normalised-text fingerprinting; catches
  exact duplicates and formatting-only variations.
* ``--mode near``: character n-gram shingling with Jaccard similarity;
  catches paraphrases and minor edits.

Output can be human-readable (default) or JSON (``--json``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# Supported file suffixes for dataset loading.
_SUPPORTED_SUFFIXES = {".jsonl", ".json", ".ndjson"}


def _load_records(path: str) -> list[dict]:
    """Load records from a JSONL or JSON file.

    JSONL (.jsonl, .ndjson): one JSON object per line.
    JSON (.json): a single JSON array of objects.
    """

    p = Path(path)
    if not p.exists():
        raise click.ClickException(f"File not found: {path}")
    if p.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise click.ClickException(
            f"Unsupported file type: {p.suffix}. Supported: {', '.join(sorted(_SUPPORTED_SUFFIXES))}"
        )

    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        records: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise click.ClickException(f"Invalid JSON on line {line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise click.ClickException(f"Line {line_no} is not a JSON object")
                records.append(obj)
        return records

    # .json — expect an array of objects.
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise click.ClickException("JSON file must contain an array of objects")
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise click.ClickException(f"Element at index {i} is not a JSON object")
    return data


@click.command(
    name="dedup",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--data-path", required=True, help="Path to JSONL or JSON dataset file.")
@click.option(
    "--mode",
    type=click.Choice(["exact", "near"]),
    default="exact",
    show_default=True,
    help="Detection mode: exact (normalised fingerprinting) or near (n-gram similarity).",
)
@click.option(
    "--scope",
    type=click.Choice(["prompt", "full"]),
    default="prompt",
    show_default=True,
    help="Comparison scope: prompt (primary text field) or full (all string fields).",
)
@click.option(
    "--threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Jaccard similarity threshold for near mode (0.0-1.0).",
)
@click.option(
    "--ngram-size",
    type=int,
    default=5,
    show_default=True,
    help="Character n-gram size for near mode shingling.",
)
@click.option(
    "--text-keys",
    default=None,
    help="Comma-separated field names for text extraction (default: prompt,question,text,content,input,query).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON output.")
def dedup_command(
    data_path: str,
    mode: str,
    scope: str,
    threshold: float,
    ngram_size: int,
    text_keys: str | None,
    as_json: bool,
) -> None:
    """Find near-duplicate training examples without deleting them."""

    from areno.dedup import find_duplicates, format_duplicate_report

    records = _load_records(data_path)
    if not records:
        click.echo("No records found in dataset.", err=True)
        sys.exit(1)

    keys = text_keys.split(",") if text_keys else None

    report = find_duplicates(
        records,
        mode=mode,
        scope=scope,
        text_keys=keys,
        threshold=threshold,
        ngram_size=ngram_size,
    )

    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(format_duplicate_report(report))