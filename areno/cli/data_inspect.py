"""CLI for inspecting and validating datasets against AReno data contracts.

``areno data inspect`` loads a dataset through the same loader path used by
``areno train``, then validates the post-loader records against the per-mode
contract before any model or worker initialization.  Output is available in
human-readable terminal form or structured JSON.
"""

from __future__ import annotations

import json

import click

from areno.api.data_contract import (
    ContractError,
    ContractReport,
    list_contract_modes,
    validate_contract,
)


@click.group(name="data", context_settings={"help_option_names": ["-h", "--help"]})
def data_command() -> None:
    """Inspect or validate datasets against AReno data contracts."""


@data_command.command(name="inspect", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset-path", required=True, help="Dataset path, directory, or remote ref.")
@click.option(
    "--dataset-loader-fn",
    default=None,
    help="Optional Python dataset loader function as file.py or file.py:function.",
)
@click.option(
    "--model-hub",
    type=click.Choice(["hf", "modelscope"], case_sensitive=False),
    default="modelscope",
    show_default=True,
    help="Remote hub for non-local model and dataset refs.",
)
@click.option(
    "--contract",
    "run_contract",
    is_flag=True,
    help="Validate the loaded dataset against the mode-specific data contract.",
)
@click.option(
    "--mode",
    type=click.Choice(list_contract_modes(), case_sensitive=False),
    default=None,
    help="Training mode to validate against (required with --contract).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON report.")
@click.option(
    "--max-samples",
    type=int,
    default=100,
    show_default=True,
    help="Maximum dataset records to scan.",
)
@click.option(
    "--max-errors",
    type=int,
    default=20,
    show_default=True,
    help="Maximum errors to collect before truncating.",
)
def inspect_command(
    dataset_path: str,
    dataset_loader_fn: str | None,
    model_hub: str,
    run_contract: bool,
    mode: str | None,
    as_json: bool,
    max_samples: int,
    max_errors: int,
) -> None:
    """Load a dataset and optionally validate it against a data contract."""

    if run_contract and mode is None:
        raise click.UsageError("--mode is required when --contract is used")

    dataset = _load_dataset(dataset_path, dataset_loader_fn=dataset_loader_fn, model_hub=model_hub)
    records = list(dataset)
    total = len(records)

    if not run_contract:
        if as_json:
            click.echo(json.dumps({"total_records": total, "contract": None}, indent=2))
        else:
            click.echo(f"Dataset: {dataset_path}")
            click.echo(f"  records: {total}")
            if dataset_loader_fn:
                click.echo(f"  loader:  {dataset_loader_fn}")
            click.echo()
            if total > 0:
                click.echo("First record keys:")
                first = records[0]
                if isinstance(first, dict):
                    for key in sorted(first.keys()):
                        click.echo(f"  {key}: {type(first[key]).__name__}")
        return

    report = validate_contract(records, mode=mode, max_samples=max_samples, max_errors=max_errors)

    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        _print_contract_report(report, total)

    if not report.ok:
        raise click.exceptions.Exit(1)


def _load_dataset(
    dataset_path: str,
    *,
    dataset_loader_fn: str | None,
    model_hub: str,
):
    """Load a dataset using the same path as ``areno train``."""

    from areno.cli.train import _load_dataset_for_training

    from datasets import load_dataset, load_from_disk

    return _load_dataset_for_training(
        dataset_path,
        dataset_loader_fn=dataset_loader_fn,
        model_hub=model_hub,
        load_dataset=load_dataset,
        load_from_disk=load_from_disk,
    )


def _print_contract_report(report: ContractReport, total_records: int) -> None:
    """Print a human-readable contract validation report."""

    status = "passed" if report.ok else "failed"
    click.echo(
        f"Contract validation: mode={report.mode}  scanned={report.total_scanned}"
        f"  total={total_records}  errors={len(report.errors)}  warnings={len(report.warnings)}"
        f"  status={status}"
    )
    click.echo()

    for err in report.errors:
        _print_error("ERROR", err)
    for warn in report.warnings:
        _print_error("WARN", warn)

    if report.ok:
        click.echo("Passed: no contract violations found.")
    else:
        click.echo(
            f"Failed: {len(report.errors)} error(s) found."
            "  Fix the listed fields or adjust the dataset loader."
        )


def _print_error(label: str, err: ContractError) -> None:
    """Print one contract error or warning line."""

    idx = err.sample_index if err.sample_index >= 0 else "-"
    click.echo(
        f"{label:<5} sample={idx}  field='{err.field_path}'"
        f"  expected={err.expected}  got={err.actual}"
    )
    click.echo(f"      hint: {err.hint}")
