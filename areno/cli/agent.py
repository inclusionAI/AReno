"""Local coding-agent CLI for AReno train/serve operations."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

DEFAULT_KNOWLEDGE_FILE = Path(__file__).resolve().parents[1] / "agentic" / "coding" / "ops_knowledge.md"
DEFAULT_KNOWLEDGE = DEFAULT_KNOWLEDGE_FILE.read_text(encoding="utf-8")
CONFIG_FILE = Path.home() / ".areno" / "agent_config.json"
DEFAULT_AGENT_TURN_LIMIT = 1_000_000
JUDGE_CONTEXT_CHARS = 24000
CONSOLE = Console(highlight=False)

SYSTEM_TEMPLATE = """You are an AReno operations coding agent.

You can inspect and modify the current checkout and run shell commands through
tools. Your task is to complete the user's train or serve request in this
environment. Work iteratively: inspect the environment, choose a conservative
command, run it, read failures, and retry with adjusted parameters when possible.
Use exactly one tool call per assistant turn. Call submit with status=solved only
after a train/serve command has completed successfully or is clearly running and
verified. Call submit with status=blocked only after a non-recoverable blocker.

Train command policy:

- For rollout/RL algorithms, use `--n-samples 8` by default unless the user
  explicitly asks for another value.
- Add `--drop-rollout-state` to train and train-smoke commands by default unless
  the user explicitly asks to keep rollout state.
- A tiny smoke command only proves startup. After it succeeds, do not submit yet
  if the goal is a train run. Run additional smoke attempts with larger
  `--max-running-prompts`, `--batch-size`, and/or `--mini-bs` to use available
  GPU memory, then run the real train command with the largest stable settings.
- If a smoke command underuses GPU memory, increase rollout concurrency first.
- Before using remote model or dataset refs, check whether Hugging Face is
  reachable. If Hugging Face is reachable and the requested ref is an HF ref,
  use `--model-hub hf`. If Hugging Face is unreachable or times out, use
  ModelScope with `--model-hub modelscope` instead.

Background knowledge:

{knowledge}
"""


@click.command("agent", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--set", "set_config", is_flag=True, help="Store agent connection config under ~/.areno and exit.")
@click.option("--base-url", default=None, help="OpenAI-compatible base URL to store with --set.")
@click.option("--model", default=None, help="Model name to store with --set.")
@click.option("--api-key", default=None, help="API key to store with --set.")
@click.option("--command-timeout-s", default=1800.0, show_default=True, help="Maximum timeout for run_command tools.")
@click.option(
    "--knowledge-file",
    default=str(DEFAULT_KNOWLEDGE_FILE),
    show_default=True,
    help="File storing AReno train/serve background knowledge.",
)
@click.option(
    "--refresh-knowledge",
    is_flag=True,
    help="Refresh the knowledge file with the configured LLM and exit.",
)
@click.argument("job", nargs=-1)
def agent_command(
    *,
    set_config: bool,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    command_timeout_s: float,
    knowledge_file: str,
    refresh_knowledge: bool,
    job: tuple[str, ...],
) -> None:
    """Ask an OpenAI-compatible coding agent to run an AReno train/serve job."""

    if set_config:
        if refresh_knowledge or job:
            raise click.UsageError("--set cannot be combined with --refresh-knowledge or a job")
        _write_agent_config(base_url=base_url, model=model, api_key=api_key)
        click.echo(f"stored agent config: {CONFIG_FILE}")
        return
    if base_url or model or api_key:
        raise click.UsageError("--base-url, --model, and --api-key are only used with --set")

    config = _load_agent_config()
    resolved_base_url = config.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
    resolved_model = config.get("model") or os.environ.get("OPENAI_MODEL") or "policy"
    resolved_api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    repo = "."
    if refresh_knowledge:
        refresh_args = argparse.Namespace(
            base_url=resolved_base_url,
            model=resolved_model,
            api_key=resolved_api_key,
            repo=repo,
            knowledge_file=knowledge_file,
        )
        raise SystemExit(asyncio.run(_refresh_knowledge_async(refresh_args)))

    instruction = " ".join(job).strip()
    if not instruction:
        raise click.UsageError("provide a natural-language train/serve job, or use --refresh-knowledge")

    args = argparse.Namespace(
        base_url=resolved_base_url,
        model=resolved_model,
        api_key=resolved_api_key,
        repo=repo,
        max_turns=DEFAULT_AGENT_TURN_LIMIT,
        command_timeout_s=command_timeout_s,
        knowledge_file=knowledge_file,
        instruction=instruction,
    )
    raise SystemExit(asyncio.run(_main_async(args)))


async def _refresh_knowledge_async(args: argparse.Namespace) -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("The agent CLI requires `openai`. Install it with `pip install openai`.") from exc

    repo = Path(args.repo).expanduser().resolve()
    knowledge_path = Path(args.knowledge_file).expanduser()
    context = _collect_refresh_context(repo)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    try:
        from areno.agentic.coding.agent_loop import create_chat_completion_with_retry

        response = await create_chat_completion_with_retry(
            client,
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You maintain a concise operations knowledge file for an AReno coding agent. "
                        "Return markdown only. Do not wrap the answer in code fences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Refresh this AReno train/serve operations knowledge file. Keep it practical and compact, "
                        "but include command usage, GPU inspection, memory tuning rules, retry strategy, checkpoint "
                        "save/load testing, drop-rollout-state meaning, and common failure fixes.\n\n"
                        f"Current built-in knowledge:\n{DEFAULT_KNOWLEDGE}\n\n"
                        f"Fresh local context:\n{context}"
                    ),
                },
            ],
            stream=False,
        )
    finally:
        await client.close()
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise click.ClickException("LLM returned empty knowledge content")
    knowledge_path.parent.mkdir(parents=True, exist_ok=True)
    knowledge_path.write_text(content + "\n", encoding="utf-8")
    click.echo(f"refreshed knowledge file with LLM: {knowledge_path}")
    return 0


def _write_agent_config(*, base_url: str | None, model: str | None, api_key: str | None) -> None:
    missing = [
        name
        for name, value in [
            ("--base-url", base_url),
            ("--model", model),
            ("--api-key", api_key),
        ]
        if not value
    ]
    if missing:
        raise click.UsageError("--set requires " + ", ".join(missing))
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_url": _b64_encode(base_url or ""),
        "model": _b64_encode(model or ""),
        "api_key": _b64_encode(api_key or ""),
    }
    CONFIG_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(CONFIG_FILE, 0o600)


def _load_agent_config() -> dict[str, str]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"failed to read agent config {CONFIG_FILE}: {exc}") from exc
    config: dict[str, str] = {}
    for key in ("base_url", "model", "api_key"):
        value = raw.get(key)
        if value is not None:
            config[key] = _b64_decode(str(value), key)
    return config


def _b64_encode(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _b64_decode(value: str, key: str) -> str:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise click.ClickException(f"invalid base64 value for {key} in {CONFIG_FILE}") from exc


async def _main_async(args: argparse.Namespace) -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("The agent CLI requires `openai`. Install it with `pip install openai`.") from exc

    from areno.agentic.coding.agent_loop import run_conversation_turns
    from areno.agentic.coding.coding_tools import CodingWorkspace

    knowledge = _load_knowledge(Path(args.knowledge_file).expanduser())
    task = {
        "instance_id": "areno_ops_local",
        "repo": str(Path(args.repo).expanduser().resolve()),
        "base_commit": "current-workspace",
        "problem_statement": args.instruction,
        "test_commands": [],
        "max_turns": int(args.max_turns),
    }
    item = SimpleNamespace(record=task, prompt=args.instruction)
    workspace = CodingWorkspace.from_current_repo(task, args.repo)
    workspace.max_command_timeout_s = float(args.command_timeout_s)
    workspace.command_output_callback = _print_command_output_event
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(knowledge=knowledge)},
        {"role": "user", "content": _job_prompt(args.instruction, workspace.root)},
    ]
    _print_banner(args.instruction, workspace.root, args.model)
    try:
        while True:
            await run_conversation_turns(
                client=client,
                item=item,
                workspace=workspace,
                model=args.model,
                messages=messages,
                max_turns=int(args.max_turns),
                record_trajectory=False,
                on_event=_print_event,
            )
            if workspace.submitted is None:
                click.echo("agent stopped without submit")
                return 2
            judgment = await _judge_goal_done(
                client=client,
                model=args.model,
                instruction=args.instruction,
                submitted=workspace.submitted,
                messages=messages,
                command_history=workspace.command_history,
            )
            _print_judgment(judgment)
            if judgment.get("done") is True:
                _print_panel(
                    "done",
                    _json_view(workspace.submitted),
                    style="cyan",
                )
                return 0 if workspace.submitted.get("status") == "solved" else 1
            feedback = str(judgment.get("feedback") or judgment.get("reason") or "").strip()
            if not feedback:
                feedback = "The goal is not actually complete. Inspect the current state and continue."
            workspace.submitted = None
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "A reviewer checked the previous submit and decided the goal is not actually done.\n\n"
                        f"Reviewer feedback:\n{feedback}\n\n"
                        "Continue from the existing context. Do more inspection or rerun adjusted commands, "
                        "then call submit again only when the original user goal is actually complete."
                    ),
                }
            )
    finally:
        await client.close()
        workspace.close()


async def _judge_goal_done(
    *,
    client: Any,
    model: str,
    instruction: str,
    submitted: dict[str, Any],
    messages: list[dict[str, Any]],
    command_history: list[dict[str, Any]],
) -> dict[str, Any]:
    from areno.agentic.coding.agent_loop import create_chat_completion_with_retry

    response = await create_chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict reviewer for an AReno train/serve operations agent. "
                    "Decide whether the original goal is actually complete. A submit is not enough by itself. "
                    "Look for concrete evidence such as successful command output, a running verified server, "
                    "a completed train step, a saved and reload-tested checkpoint when requested, or a truly "
                    "non-recoverable blocker. For rollout/RL train goals, do not accept a single tiny smoke "
                    "success as complete unless the user only requested smoke. Check that the agent used "
                    "--n-samples 8 by default, included --drop-rollout-state by default, and tried to increase "
                    "GPU utilization after basic smoke success when memory headroom was available. Return JSON "
                    "only with keys: done, reason, feedback."
                ),
            },
            {
                "role": "user",
                "content": _judge_prompt(
                    instruction=instruction,
                    submitted=submitted,
                    messages=messages,
                    command_history=command_history,
                ),
            },
        ],
        stream=False,
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        judgment = json.loads(_extract_json_object(content))
    except json.JSONDecodeError:
        return {
            "done": False,
            "reason": "reviewer returned non-JSON output",
            "feedback": content[:2000] or "Reviewer output was empty; continue and verify the goal explicitly.",
        }
    if not isinstance(judgment, dict):
        return {"done": False, "reason": "reviewer returned non-object JSON", "feedback": "Continue and verify the goal."}
    judgment["done"] = bool(judgment.get("done"))
    judgment["reason"] = str(judgment.get("reason") or "")
    judgment["feedback"] = str(judgment.get("feedback") or "")
    return judgment


def _extract_json_object(content: str) -> str:
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        return content[start : end + 1]
    return content


def _judge_prompt(
    *,
    instruction: str,
    submitted: dict[str, Any],
    messages: list[dict[str, Any]],
    command_history: list[dict[str, Any]],
) -> str:
    payload = {
        "original_goal": instruction,
        "submitted": submitted,
        "recent_messages": _trim_for_judge(_json_dumps(messages[-24:])),
        "recent_command_history": _trim_for_judge(_json_dumps(command_history[-20:])),
    }
    return _json_dumps(payload)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _trim_for_judge(text: str) -> str:
    if len(text) <= JUDGE_CONTEXT_CHARS:
        return text
    return text[-JUDGE_CONTEXT_CHARS:]


def _load_knowledge(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_KNOWLEDGE, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def _collect_refresh_context(repo: Path) -> str:
    commands = [
        ["areno", "--help"],
        ["areno", "train", "--help"],
        ["areno", "serve", "--help"],
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv"],
    ]
    rows = [f"Repository: {repo}"]
    examples_dir = repo / "examples"
    if examples_dir.exists():
        rows.append("Examples tree:\n" + _run_context_command(["find", "examples", "-maxdepth", "3", "-type", "f"], repo))
    for command in commands:
        rows.append("$ " + " ".join(command))
        rows.append(_run_context_command(command, repo))
    return "\n\n".join(rows)


def _run_context_command(command: list[str], cwd: Path) -> str:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<failed: {exc}>"
    output = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    output = output.strip() or f"<exit {proc.returncode}, no output>"
    if len(output) > 12000:
        output = output[:12000] + "\n<truncated>"
    return output


def _job_prompt(instruction: str, root: Path) -> str:
    return (
        f"Workspace: {root}\n"
        f"User job: {instruction}\n\n"
        "Complete this job in the current environment. If the job asks for training, run a real small train "
        "or the requested train command and adjust parameters on recoverable failures. If the job asks for "
        "serving, start the server and verify it with a local request when possible.\n\n"
        "Operational requirements for train jobs:\n"
        "- Use --n-samples 8 for RL/rollout algorithms unless the user provided another value.\n"
        "- Include --drop-rollout-state by default unless the user asks to keep rollout state.\n"
        "- Before remote downloads, check Hugging Face availability. If it is unavailable, use "
        "--model-hub modelscope; if it is available and the requested ref is an HF ref, use --model-hub hf.\n"
        "- Use smoke-infer/smoke-train before long runs, but do not treat one tiny smoke success as completion.\n"
        "- After smoke succeeds, retry with larger max-running-prompts, batch-size, or mini-bs to fill GPU memory "
        "safely before choosing final train settings."
    )


def _print_event(event: str, payload: dict[str, Any]) -> None:
    if event == "assistant":
        content = payload.get("content")
        if content:
            _print_panel("assistant", Text(str(content), style="bright_cyan"), style="cyan")
        calls = payload.get("tool_calls") or []
        if calls:
            call = calls[0]
            tool_name = call["function"]["name"]
            arguments = call["function"].get("arguments", "")
            _print_panel(f"tool call: {tool_name}", _format_tool_arguments(tool_name, arguments), style="magenta")
    elif event == "tool":
        name = str(payload.get("name") or "tool")
        _print_panel(f"tool result: {name}", _format_tool_result(name, str(payload.get("content", ""))), style="blue")


def _print_command_output_event(event: dict[str, Any]) -> None:
    kind = event.get("kind")
    if kind == "start":
        CONSOLE.print(Rule("[bold magenta]command output[/bold magenta]", style="magenta"))
        CONSOLE.print(Text(str(event.get("command") or ""), style="magenta"))
        CONSOLE.print(Text(f"timeout_s: {event.get('timeout_s')}", style="dim"))
    elif kind == "line":
        stream = str(event.get("stream") or "stdout")
        line = str(event.get("line") or "").rstrip()
        prefix_style = "red" if stream == "stderr" else "cyan"
        line_style = "bright_red" if stream == "stderr" else "white"
        text = Text()
        text.append(f"{stream}> ", style=prefix_style)
        text.append(line, style=line_style)
        CONSOLE.print(text, soft_wrap=True)
    elif kind == "end":
        skipped = int(event.get("skipped_stream_lines") or 0)
        returncode = event.get("returncode")
        timed_out = bool(event.get("timed_out"))
        style = "red" if timed_out or returncode not in (0, None) else "cyan"
        summary = f"returncode={returncode}"
        if timed_out:
            summary += " timed_out=true"
        if skipped:
            summary += f" streamed_screened={skipped} skipped_lines"
        CONSOLE.print(Rule(f"[bold {style}]{summary}[/bold {style}]", style=style))


def _print_banner(instruction: str, root: Path, model: str) -> None:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("model", model)
    table.add_row("workspace", str(root))
    table.add_row("goal", instruction)
    _print_panel("areno agent", table, style="cyan")


def _print_judgment(judgment: dict[str, Any]) -> None:
    done = bool(judgment.get("done"))
    title = "judge: done" if done else "judge: continue"
    _print_panel(title, _json_view(judgment), style="cyan" if done else "yellow")


def _format_tool_arguments(tool_name: str, raw: str) -> Any:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return Text(raw, style="white")
    if tool_name == "run_command":
        command = str(parsed.get("command", ""))
        timeout = parsed.get("timeout_s")
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold magenta", no_wrap=True)
        table.add_column()
        table.add_row("command", Syntax(command, "bash", theme="ansi_dark", word_wrap=True))
        if timeout is not None:
            table.add_row("timeout_s", str(timeout))
        return table
    return _json_view(parsed)


def _format_tool_result(tool_name: str, raw: str) -> Any:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return Text(raw, style="white")
    if tool_name == "run_command":
        return _format_run_command_result(parsed)
    return _json_view(parsed)


def _format_run_command_result(parsed: dict[str, Any]) -> Any:
    returncode = parsed.get("returncode")
    timed_out = bool(parsed.get("timed_out"))
    screened = bool(parsed.get("screened"))
    status_style = "red" if timed_out or returncode not in (0, None) else "cyan"
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold blue", no_wrap=True)
    table.add_column()
    table.add_row("command", Syntax(str(parsed.get("command") or ""), "bash", theme="ansi_dark", word_wrap=True))
    table.add_row("returncode", Text(str(returncode), style=status_style))
    table.add_row("screened", "yes" if screened else "no")
    table.add_row("streamed", str(parsed.get("streamed_lines", 0)))
    skipped = int(parsed.get("skipped_stream_lines") or 0)
    if skipped:
        table.add_row("live_skipped", str(skipped))
    if timed_out:
        table.add_row("timed_out", Text("yes", style="red"))
    output = str(parsed.get("output") or parsed.get("stdout") or parsed.get("stderr") or "")
    if not output:
        return table
    output_panel = Panel(
        Text(output, style="white"),
        title="screened output" if screened else "output",
        border_style="bright_black",
        padding=(0, 1),
    )
    return Group(table, output_panel)


def _json_view(value: Any) -> Any:
    return Syntax(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        "json",
        theme="ansi_dark",
        word_wrap=True,
    )


def _print_panel(title: str, body: Any, *, style: str) -> None:
    CONSOLE.print()
    CONSOLE.print(Panel(body, title=f"[bold {style}]{title}[/bold {style}]", border_style="bright_black", padding=(0, 1)))


if __name__ == "__main__":
    agent_command()
