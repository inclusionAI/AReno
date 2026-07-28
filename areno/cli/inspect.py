"""CLI inspection commands that run without loading model weights.

These commands provide pre-flight diagnostics for model adaptation.  The
``chat-template`` sub-command checks whether a tokenizer's chat template can
correctly render all message types used by AReno's training pipeline.
"""

from __future__ import annotations

import json
import sys

import click

from areno.api.chat_template_inspector import ChatTemplateInspector
from areno.cli.model_refs import resolve_model_ref
from areno.engine.data.tokenizer import load_tokenizer


@click.group(name="inspect", context_settings={"help_option_names": ["-h", "--help"]})
def inspect_command() -> None:
    """Inspect model artifacts without loading weights."""


@click.command(
    name="chat-template",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--model",
    required=True,
    help="Model name or local checkpoint path (tokenizer only, weights not loaded).",
)
@click.option(
    "--output-format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format: human-readable table (text) or machine-readable JSON.",
)
def inspect_chat_template_command(model: str, output_format: str) -> None:
    """Check chat template compatibility without loading model weights."""

    local_path = resolve_model_ref(model)
    tokenizer = load_tokenizer(local_path)

    report = ChatTemplateInspector.inspect(model_name=model, tokenizer=tokenizer)

    if output_format == "json":
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(report.to_human_readable())

    if report.overall_status == "fail":
        sys.exit(1)


inspect_command.add_command(inspect_chat_template_command)