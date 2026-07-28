"""CLI command for conversation role normalization and tool-message pairing.

Reads a JSON file containing a list of conversations (each a list of message
dicts) and outputs a human-readable or structured JSON normalization report.

Usage::

    areno normalize-conversation data.json
    areno normalize-conversation data.json --json
    areno normalize-conversation data.json --human
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.command(name="normalize-conversation", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit a machine-readable JSON report instead of human-readable text.",
)
@click.option(
    "--human",
    "as_human",
    is_flag=True,
    help="Emit a human-readable report (default when --json is not set).",
)
def normalize_conversation_command(input_file: Path, as_json: bool, as_human: bool) -> None:
    """Normalize conversation roles and validate tool-message pairing.

    INPUT_FILE is a JSON file containing a list of conversations.
    Each conversation is a list of message dicts with ``role`` and ``content`` keys.
    """

    # Import lazily so the CLI doesn't pull torch on help.
    from areno.engine.data.conversation_normalizer import normalize_dataset

    try:
        raw = json.loads(input_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        click.echo(f"Error: failed to parse JSON: {exc}", err=True)
        sys.exit(1)

    if not isinstance(raw, list):
        click.echo("Error: input JSON must be a list of conversations.", err=True)
        sys.exit(1)

    report = normalize_dataset(raw)

    if as_json:
        click.echo(report.to_json())
    else:
        click.echo(report.to_human_string())

    if report.failed > 0:
        sys.exit(1)