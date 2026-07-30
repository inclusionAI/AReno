"""CLI inspection commands that run without loading model weights.

These commands provide pre-flight diagnostics for model adaptation.  The
``chat-template`` sub-command checks whether a tokenizer's chat template can
correctly render all message types used by AReno's training pipeline.  The
``loss-mask`` sub-command explains which tokens contribute to the training
loss by mapping packer output back to conversational structure.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from areno.api.chat_template_inspector import ChatTemplateInspector
from areno.api.loss_mask_explainer import LossMaskExplainer
from areno.api.openai_chat import normalize_messages
from areno.api.tokenizer import apply_chat_template_with_options
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
    """Check chat template compatibility without loading model weights.

    Resolves the model reference to a local path, loads only the tokenizer,
    and runs the :class:`ChatTemplateInspector` across five canonical
    message scenarios.  Exits with code 1 if any check fails, making the
    command suitable for CI pipelines.
    """

    # Resolve to a local checkpoint directory — downloads from ModelScope
    # or Hugging Face only if the model is not already cached locally.
    local_path = resolve_model_ref(model)

    # Load only the tokenizer; model weights are never read.
    tokenizer = load_tokenizer(local_path)

    # Run all diagnostic checks and produce the aggregated report.
    report = ChatTemplateInspector.inspect(model_name=model, tokenizer=tokenizer)

    # Emit output in the requested format.
    if output_format == "json":
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(report.to_human_readable())

    # Non-zero exit code on failure so CI tools can detect the problem.
    if report.overall_status == "fail":
        sys.exit(1)


inspect_command.add_command(inspect_chat_template_command)


@click.command(
    name="loss-mask",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--model",
    required=True,
    help="Model name or local checkpoint path (tokenizer only, weights not loaded).",
)
@click.option(
    "--messages",
    required=True,
    type=click.Path(exists=True),
    help="Path to a JSON file containing the OpenAI-style message list.",
)
@click.option(
    "--show-full-text",
    is_flag=True,
    default=False,
    help="Show full decoded text per span (default: truncate to 50 characters).",
)
@click.option(
    "--output-format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format: human-readable table (text) or machine-readable JSON.",
)
def inspect_loss_mask_command(
    model: str, messages: str, show_full_text: bool, output_format: str
) -> None:
    """Explain which tokens contribute to the training loss.

    Loads only the tokenizer, renders the provided messages through the
    chat template to obtain token ids and a loss mask, then maps the
    mask back to conversational spans (roles, turns, text previews).
    """

    local_path = resolve_model_ref(model)
    tokenizer = load_tokenizer(local_path)

    with open(messages, "r", encoding="utf-8") as f:
        raw_messages = json.load(f)
    msg_list = normalize_messages(raw_messages)

    # Render the full conversation to obtain token ids.  The loss mask is
    # derived from role boundaries: system/user/tool tokens are masked out
    # (not trained), assistant tokens are masked in (trained).
    token_ids = apply_chat_template_with_options(
        tokenizer, msg_list, tokenize=True
    )
    # Normalise to a plain list of ints.
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    if hasattr(token_ids, "ids"):
        token_ids = token_ids.ids
    if not isinstance(token_ids, (list, tuple)):
        token_ids = list(token_ids)

    # Build a loss mask from role boundaries: only assistant spans are
    # trainable.  This mirrors the SFT packer's behaviour where prompt
    # tokens (system/user/tool) are masked out.
    loss_mask = _build_loss_mask(tokenizer, msg_list, token_ids)

    report = LossMaskExplainer.explain(
        model_name=model,
        tokenizer=tokenizer,
        token_ids=list(token_ids),
        loss_mask=loss_mask,
        messages=msg_list,
        show_full_text=show_full_text,
    )

    if output_format == "json":
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(report.to_human_readable())


def _build_loss_mask(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    token_ids: list[int],
) -> list[bool]:
    """Build a loss mask where only assistant tokens are trainable.

    This mirrors the SFT packer: system/user/tool tokens are masked out,
    assistant tokens are masked in.  For agentic trajectories the caller
    would pass the packer's actual ``loss_mask`` instead.
    """

    total = len(token_ids)
    mask = [False] * total
    prev_len = 0

    for i, msg in enumerate(messages):
        partial = messages[: i + 1]
        try:
            rendered = apply_chat_template_with_options(
                tokenizer, partial, tokenize=True
            )
        except Exception:
            continue
        if hasattr(rendered, "input_ids"):
            rendered = rendered.input_ids
        if hasattr(rendered, "ids"):
            rendered = rendered.ids
        if not isinstance(rendered, (list, tuple)):
            rendered = list(rendered)

        cur_len = min(len(rendered), total)
        if msg.get("role") == "assistant":
            for i in range(prev_len, cur_len):
                mask[i] = True
        prev_len = cur_len

    return mask


inspect_command.add_command(inspect_loss_mask_command)