"""CLI command for streaming JSONL quality scanning.

This module registers the ``areno scan`` sub-command, which wraps the
core :func:`areno.api.quality_scanner.scan_jsonl` function and exposes
it through the CLI with options for schema validation, output format,
and error preview size.

The command supports two input modes:
  - **File path**: ``areno scan data.jsonl``
  - **stdin pipe**: ``cat data.jsonl | areno scan``

Two output formats are available:
  - ``table`` (default): human-readable summary with error preview
  - ``json``: machine-readable JSON for CI/CD integration
"""

from __future__ import annotations

import sys

import click

from areno.api.quality_scanner import (
    render_json,
    render_table,
    scan_jsonl,
)


@click.command(name="scan", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("source", required=False, default="-",
                type=click.Path(exists=False),
                help="JSONL file path, or '-' for stdin (default).")
@click.option("--required-fields", "required_fields", default=None,
              help="Comma-separated list of fields that every record must contain "
                   "with non-empty values. Example: --required-fields prompt,response")
@click.option("--max-errors", "max_errors", default=100, show_default=True,
              type=int, help="Maximum number of error entries to retain in the "
                             "report. Additional errors are counted but not stored.")
@click.option("--format", "output_format",
              type=click.Choice(["table", "json"]), default="table", show_default=True,
              help="Output format: 'table' for human-readable, 'json' for "
                   "machine-readable (CI/CD friendly).")
def scan_command(source: str, required_fields: str | None, max_errors: int,
                 output_format: str) -> None:
    """Scan a JSONL file (or stdin) and report data quality issues.

    Detects blank lines, JSON parse errors, non-object records, and
    schema violations (missing/empty fields).  Uses bounded memory —
    suitable for million-line files.

    \b
    Examples:
      areno scan data.jsonl
      areno scan data.jsonl --required-fields prompt,response
      cat data.jsonl | areno scan --format json
    """

    # Parse comma-separated required fields into a list, if provided.
    fields = None
    if required_fields:
        fields = [f.strip() for f in required_fields.split(",") if f.strip()]

    # '-' means read from stdin; otherwise treat as a file path.
    if source == "-":
        result = scan_jsonl(sys.stdin, required_fields=fields, max_errors=max_errors)
    else:
        result = scan_jsonl(source, required_fields=fields, max_errors=max_errors)

    # Render the result in the requested format.
    if output_format == "json":
        click.echo(render_json(result))
    else:
        click.echo(render_table(result))
