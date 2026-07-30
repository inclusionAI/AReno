"""``areno inspect-tokenizer``:tokenizer 对齐检查 CLI。

聚焦、只读的诊断命令:对 plain prompt / chat messages / tool calls 渲染
token ids、pieces、special-token 标记、EOS 位置、roles、loss-mask spans,
并把 tokenizer 词表大小与模型 ``config.json`` 的 ``vocab_size`` 对齐比较。

重依赖(transformers / modelscope)在命令函数内才真正触发:模块级只 import
轻量的函数引用(load_tokenizer / resolve_model_ref 内部延迟 import,
inspect_* 为纯函数),因此 ``import areno.cli.inspect_tokenizer`` 不会拉起
torch/engine,与 ``areno.cli.diagnostics`` 的轻量精神一致。
"""

from __future__ import annotations

import json
from typing import Any

import click

from areno.api.tokenizer_inspect import (
    InspectionReport,
    inspect_messages,
    inspect_prompt,
    inspect_tool_call,
)
from areno.cli.model_refs import resolve_model_ref
from areno.engine.data.tokenizer import load_tokenizer


@click.command(name="inspect-tokenizer", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--model", "model", required=True, metavar="PATH", help="Tokenizer/model path (local dir or hub id).")
@click.option("--prompt", default=None, help="Plain prompt text to inspect.")
@click.option(
    "--messages",
    default=None,
    help='JSON array of chat messages, e.g. [{"role":"user","content":"hi"}].',
)
@click.option(
    "--tool-call",
    "tool_call",
    default=None,
    help="JSON array of messages that include assistant tool_calls and tool-role replies.",
)
@click.option(
    "--max-length",
    type=int,
    default=None,
    help="If set, truncate the inspected token sequence to this length and report truncation.",
)
@click.option(
    "--enable-thinking/--disable-thinking",
    default=None,
    help="Force the chat template enable_thinking switch on/off (default: tokenizer default).",
)
@click.option("--no-vocab-align", is_flag=True, default=False, help="Skip tokenizer vs model config vocab_size comparison.")
@click.option(
    "--add-generation-prompt/--no-add-generation-prompt",
    default=True,
    help="Append the assistant generation prompt for --messages (default on).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON report.")
def inspect_tokenizer_command(  # noqa: PLR0913 - 选项数量与 train/serve 风格一致
    model: str,
    prompt: str | None,
    messages: str | None,
    tool_call: str | None,
    max_length: int | None,
    enable_thinking: bool | None,
    no_vocab_align: bool,
    add_generation_prompt: bool,
    as_json: bool,
) -> None:
    """Inspect tokenizer alignment for prompts, chat messages, or tool calls."""

    # 1. 输入校验(在加载模型前):恰好一种输入,JSON 必须可解析且形状正确。
    input_kind, parsed = _parse_inputs(prompt, messages, tool_call)

    # 2. 解析 model ref 并加载 tokenizer(只读,不初始化 engine/worker)。
    try:
        path = resolve_model_ref(model)
        tokenizer = load_tokenizer(path)
    except Exception as exc:  # noqa: BLE001 - 任何加载失败都转为可读错误,不暴露堆栈
        click.echo(f"failed to load tokenizer from {model!r}: {exc}", err=True)
        raise click.exceptions.Exit(1)

    # 3. 生成检查报告(--no-vocab-align 时传 model_path=None 跳过词表对齐)。
    vocab_path: str | None = None if no_vocab_align else path
    try:
        if input_kind == "prompt":
            report = inspect_prompt(tokenizer, parsed, model_path=vocab_path, max_length=max_length)
        elif input_kind == "messages":
            report = inspect_messages(
                tokenizer,
                parsed,
                model_path=vocab_path,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                max_length=max_length,
            )
        else:  # tool_call
            report = inspect_tool_call(
                tokenizer,
                parsed,
                model_path=vocab_path,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                max_length=max_length,
            )
    except Exception as exc:  # noqa: BLE001
        click.echo(f"inspection failed: {exc}", err=True)
        raise click.exceptions.Exit(1)

    # 4. 输出 + 退出码。
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_report(report)
    # 词表不一致视为需要关注的失败;invalid input 已在校验阶段以 UsageError 退出(exit 2)。
    if report.vocab_alignment.status == "FAIL":
        raise click.exceptions.Exit(1)


def _parse_inputs(
    prompt: str | None,
    messages: str | None,
    tool_call: str | None,
) -> tuple[str, Any]:
    """校验并解析互斥输入,返回 (kind, value)。失败抛 click.UsageError。"""

    provided = [
        (name, val)
        for name, val in (("prompt", prompt), ("messages", messages), ("tool_call", tool_call))
        if val is not None
    ]
    if len(provided) != 1:
        raise click.UsageError("provide exactly one of --prompt / --messages / --tool-call.")
    name, raw = provided[0]
    if name == "prompt":
        return "prompt", raw
    # messages / tool_call 走 JSON 解析。
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        flag = "--" + name.replace("_", "-")
        raise click.UsageError(f"{flag} is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, list):
        flag = "--" + name.replace("_", "-")
        raise click.UsageError(f"{flag} must be a JSON array of message objects.")
    for msg in parsed:
        if not isinstance(msg, dict) or "role" not in msg:
            raise click.UsageError(f"each message must be an object with a 'role' field; got {msg!r}")
    return ("messages" if name == "messages" else "tool_call"), parsed


def _print_report(report: InspectionReport) -> None:
    """人类可读输出:概要 + 逐 token 表 + EOS/警告。"""

    click.echo(f"kind: {report.kind}")
    rt = report.round_trip
    click.echo(f"round-trip: {'OK' if rt.ok else 'DIFF'}" + (f"  {rt.diff_note}" if rt.diff_note else ""))
    va = report.vocab_alignment
    click.echo(f"vocab: {va.status}  {va.note}")
    click.echo(f"truncated: {report.truncated}  has_unknown: {report.has_unknown}")
    click.echo()
    click.echo(f"{'idx':>3}  {'id':>6}  {'piece':<16}  {'S':>1}  {'E':>1}  {'role':<10}  {'loss':>3}")
    for idx, seg in enumerate(report.segments):
        piece = seg.text
        if len(piece) > 16:
            piece = piece[:15] + "…"
        click.echo(
            f"{idx:>3}  {seg.token_ids[0]:>6}  {piece:<16}  "
            f"{'S' if seg.is_special else '.':>1}  {'E' if seg.is_eos else '.':>1}  "
            f"{str(seg.role):<10}  {'Y' if seg.loss_mask else 'N':>3}"
        )
    if report.eos_positions:
        click.echo(f"eos_positions: {report.eos_positions}")
    for warning in report.warnings:
        click.echo(f"WARNING: {warning}")
