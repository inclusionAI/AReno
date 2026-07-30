"""CLI command for explaining SFT packer loss-mask output.

Produces a human-readable or JSON report that maps token-level training masks
back to roles (prompt / response) and reports per-span token counts and loss
flags.  Runs entirely on CPU — no model weights or GPU workers are loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from areno.api.data import LossMaskExplanation
from areno.api.data_utils import prompt_response_to_tokens_and_mask, spans_from_prompt_mask
from areno.api.loss_mask_explainer import explain_loss_mask


@click.command(name="explain-mask", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--ckpt", required=True, help="Tokenizer / model checkpoint path (loaded on CPU only).")
@click.option("--dataset-path", required=True, help="Dataset path (same format as `areno train`).")
@click.option(
    "--dataset-loader-fn",
    required=True,
    help="Python file with a loader function (same format as `areno train --dataset-loader-fn`).",
)
@click.option("--max-rows", default=5, help="Number of dataset rows to process (default 5).")
@click.option("--show-text", is_flag=True, help="Decode and show span text (default: hidden).")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON output.")
def explain_mask_command(
    ckpt: str,
    dataset_path: str,
    dataset_loader_fn: str,
    max_rows: int,
    show_text: bool,
    as_json: bool,
) -> None:
    """Explain SFT packer loss-mask: map tokens back to roles and counts."""

    # 1. Validate inputs before loading anything expensive.
    if max_rows <= 0:
        raise click.UsageError("--max-rows must be positive")
    loader_file = Path(dataset_loader_fn.split(":")[0] if ":" in dataset_loader_fn else dataset_loader_fn)
    if not loader_file.exists():
        raise click.UsageError(f"--dataset-loader-fn file does not exist: {loader_file}")

    # 2. Load tokenizer (CPU-only).
    from areno.api.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(ckpt)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # 3. Load dataset and apply loader function (same flow as `areno train`).
    from areno.cli.train import _load_dataset_for_training

    try:
        from datasets import load_dataset as hf_load_dataset
        from datasets import load_from_disk as hf_load_from_disk
    except ImportError:
        raise click.UsageError(
            "The `datasets` package is required to load datasets. Install it with `pip install datasets`."
        )

    dataset = _load_dataset_for_training(
        dataset_path,
        dataset_loader_fn=dataset_loader_fn,
        load_dataset=hf_load_dataset,
        load_from_disk=hf_load_from_disk,
    )

    # 4. Process rows.
    explanations: list[tuple[int, LossMaskExplanation]] = []
    row_idx = 0
    for record in dataset:
        if row_idx >= max_rows:
            break
        record = dict(record) if not isinstance(record, dict) else record
        if "prompt" not in record or "response" not in record:
            continue
        prompt = str(record["prompt"]) if record["prompt"] is not None else ""
        response = str(record["response"]) if record["response"] is not None else ""
        if not response:
            continue
        tokens, prompt_mask = prompt_response_to_tokens_and_mask(prompt, response, tokenizer, eos_token_id)
        if len(tokens) < 2:
            continue
        spans = spans_from_prompt_mask(prompt_mask)
        loss_mask = [not m for m in prompt_mask]
        explanation = explain_loss_mask(
            tokens, loss_mask, spans, tokenizer=tokenizer if show_text else None, show_text=show_text
        )
        explanations.append((row_idx, explanation))
        row_idx += 1

    if not explanations:
        click.echo("No valid rows found in the dataset.", err=True)
        raise click.exceptions.Exit(1)

    # 5. Output.
    if as_json:
        _emit_json(explanations)
    else:
        _emit_terminal(explanations, show_text)


def _emit_json(explanations: list[tuple[int, LossMaskExplanation]]) -> None:
    rows = []
    for row_idx, exp in explanations:
        rows.append(
            {
                "row": row_idx,
                "total_tokens": exp.total_tokens,
                "loss_tokens": exp.loss_tokens,
                "spans": [
                    {
                        "role": s.role,
                        "start": s.start,
                        "end": s.end,
                        "loss": s.loss,
                        "turn": s.turn,
                        "token_count": s.end - s.start,
                    }
                    for s in exp.spans
                ],
                "summary": exp.summary,
                "text_preview": exp.text_preview,
            }
        )
    click.echo(json.dumps({"rows": rows}, indent=2))


def _emit_terminal(explanations: list[tuple[int, LossMaskExplanation]], show_text: bool) -> None:
    for row_idx, exp in explanations:
        click.echo(f"\nRow {row_idx}: total={exp.total_tokens} tokens, loss={exp.loss_tokens} tokens")
        click.echo("")
        header = "  Span  Role      Start  End  Tokens  Loss"
        if show_text:
            header += "  Text"
        click.echo(header)
        click.echo(f"  {'─' * (len(header) - 2)}")
        for i, span in enumerate(exp.spans):
            line = (
                f"  {i:<5}  {span.role:<9} {span.start:<5}  {span.end:<3}  "
                f"{span.end - span.start:<6}  {'Yes' if span.loss else 'No'}"
            )
            if show_text and exp.text_preview and i in exp.text_preview:
                text = exp.text_preview[i].replace("\n", " ")[:40]
                line += f"  {text}"
            click.echo(line)
        click.echo("")
        click.echo("  Summary:")
        for entry in exp.summary:
            click.echo(f"    {entry['role']}: {entry['token_count']} tokens, {entry['loss_tokens']} loss tokens")
