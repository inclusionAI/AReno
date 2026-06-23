"""Interactive Codex-style CLI for the agentic coding example."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_loop import initial_messages, run_conversation_turns  # noqa: E402
from coding_tools import CodingWorkspace  # noqa: E402


async def _main_async(args: argparse.Namespace) -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("The coding CLI requires `openai`. Install it with `pip install openai`.") from exc

    task = _task_from_args(args)
    item = SimpleNamespace(record=task, prompt=task["problem_statement"])
    workspace = _workspace_from_args(task, args)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    # Keep coloring local to the CLI; training traces stay plain JSON/messages.
    colors = _Colors(enabled=_color_enabled(args.color))
    try:
        print(colors.header("AReno coding agent"))
        print(f"{colors.label('workspace')} {workspace.root}")
        if args.verbose:
            print(f"{colors.label('task')} {task['problem_statement']}")
            if task["test_commands"]:
                print(f"{colors.label('suggested tests')} {', '.join(task['test_commands'])}")
        await _run_interactive_session(
            client=client, item=item, workspace=workspace, args=args, task=task, colors=colors
        )
        return 0
    finally:
        await client.close()
        workspace.close()


def _task_from_args(args: argparse.Namespace) -> dict:
    prompt = args.prompt or _prompt_multiline("Describe the coding task. Finish with an empty line:")
    if not prompt.strip():
        raise SystemExit("task prompt must be non-empty")
    commands = list(args.test_command or [])
    return {
        "instance_id": "interactive__local-001",
        "repo": str(Path(args.repo).expanduser().resolve()),
        "base_commit": "interactive-workspace",
        "problem_statement": prompt.strip(),
        "FAIL_TO_PASS": [],
        "PASS_TO_PASS": [],
        "test_commands": commands,
        "max_turns": args.max_turns or 12,
    }


def _workspace_from_args(task: dict, args: argparse.Namespace) -> CodingWorkspace:
    # Standalone CLI mirrors Codex behavior and edits the selected checkout
    # directly. The training entrypoint still uses temp workspaces.
    return CodingWorkspace.from_current_repo(task, args.repo)


async def _run_interactive_session(
    *,
    client,
    item,
    workspace: CodingWorkspace,
    args: argparse.Namespace,
    task: dict,
    colors: _Colors,
) -> None:
    messages = initial_messages(task)
    next_input: str | None = None
    turn_limit = int(args.max_turns or task.get("max_turns") or 12)
    while True:
        if next_input is not None:
            messages.append({"role": "user", "content": next_input})
        messages = _compact_messages(
            messages,
            max_chars=int(args.compact_chars),
            keep_recent=int(args.compact_keep_messages),
            colors=colors,
        )
        print(colors.dim("running coding agent...\n"))
        await run_conversation_turns(
            client=client,
            item=item,
            workspace=workspace,
            model=args.model,
            messages=messages,
            max_turns=turn_limit,
            record_trajectory=False,
            on_event=lambda _event, message: _print_message(message, verbose=args.verbose, colors=colors),
        )
        if workspace.submitted is None:
            print()
            print(colors.dim("no final submission yet; continuing agent loop...\n"))
            continue
        print()
        print(colors.header("last submission"))
        print(_format_json(workspace.submitted, colors=colors))
        next_input = _prompt_multiline("Next instruction (empty line to exit):")
        if not next_input.strip():
            break


def _prompt_multiline(header: str) -> str:
    print(header)
    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


def _print_message(message: dict, *, verbose: bool, colors: _Colors) -> None:
    # Called from the agent loop after every model/tool event, so users see
    # progress immediately instead of waiting for the whole trajectory.
    role = message["role"]
    if role == "tool":
        content = message.get("content", "")
        print(colors.section(f"tool:{message.get('name')}"), flush=True)
        print(_format_json_text(content, colors=colors), flush=True)
    elif role == "assistant":
        calls = message.get("tool_calls") or []
        if message.get("content"):
            print(colors.section("assistant"), flush=True)
            print(message["content"], flush=True)
        if calls:
            call = calls[0]
            print(colors.section(f"assistant -> {call['function']['name']}"), flush=True)
            print(_format_json_text(call["function"]["arguments"], colors=colors), flush=True)
    elif verbose:
        print(colors.section(role), flush=True)
        print(message.get("content", ""), flush=True)


def _compact_messages(
    messages: list[dict],
    *,
    max_chars: int,
    keep_recent: int,
    colors: _Colors | None = None,
) -> list[dict]:
    if max_chars <= 0 or _messages_chars(messages) <= max_chars:
        return messages
    if len(messages) <= 3:
        return messages
    head = messages[:2]
    recent = messages[2:][-max(keep_recent, 1) :]
    while recent and recent[0].get("role") == "tool":
        recent = recent[1:]
    compacted = messages[2 : len(messages) - len(recent)]
    note = {
        "role": "user",
        "content": "Compacted prior conversation:\n" + _summarize_messages(compacted),
    }
    if colors is not None:
        print(colors.dim(f"compacted conversation history from {_messages_chars(messages)} chars\n"), flush=True)
    messages[:] = [*head, note, *recent]
    return messages


def _messages_chars(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, sort_keys=True))


def _summarize_messages(messages: list[dict]) -> str:
    rows = []
    for message in messages:
        role = message.get("role", "unknown")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                call = calls[0]
                rows.append(
                    f"- assistant called {call['function']['name']}({call['function'].get('arguments', '')[:240]})"
                )
            elif message.get("content"):
                rows.append(f"- assistant: {_one_line(message['content'])}")
        elif role == "tool":
            rows.append(f"- tool {message.get('name')}: {_one_line(message.get('content', ''))}")
        elif role == "user" and message.get("content"):
            rows.append(f"- user: {_one_line(message['content'])}")
    return "\n".join(rows[-80:]) or "- prior messages omitted"


def _one_line(text: str, limit: int = 300) -> str:
    line = " ".join(str(text).split())
    if len(line) <= limit:
        return line
    return line[:limit] + "..."


def _format_json_text(text: str, *, colors: _Colors) -> str:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return str(text)
    return _format_json(value, colors=colors)


def _format_json(value: object, *, colors: _Colors) -> str:
    if not colors.enabled:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return _pretty_json(value, colors=colors)


def _pretty_json(value: object, *, colors: _Colors, indent: int = 0) -> str:
    pad = " " * indent
    next_pad = " " * (indent + 2)
    if isinstance(value, dict):
        if not value:
            return "{}"
        rows = ["{"]
        items = sorted(value.items(), key=lambda item: str(item[0]))
        for idx, (key, item) in enumerate(items):
            comma = "," if idx < len(items) - 1 else ""
            rows.append(
                f"{next_pad}{colors.blue(json.dumps(str(key), ensure_ascii=False))}: {_pretty_json(item, colors=colors, indent=indent + 2)}{comma}"
            )
        rows.append(f"{pad}}}")
        return "\n".join(rows)
    if isinstance(value, list):
        if not value:
            return "[]"
        rows = ["["]
        for idx, item in enumerate(value):
            comma = "," if idx < len(value) - 1 else ""
            rows.append(f"{next_pad}{_pretty_json(item, colors=colors, indent=indent + 2)}{comma}")
        rows.append(f"{pad}]")
        return "\n".join(rows)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return colors.magenta(json.dumps(value, ensure_ascii=False))
    return json.dumps(value, ensure_ascii=False)


def _color_enabled(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class _Colors:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    def header(self, text: str) -> str:
        return self.cyan_bold(text)

    def section(self, text: str) -> str:
        return f"\n{self.cyan('▶')} {self.bold(text)}"

    def label(self, text: str) -> str:
        return self.dim(f"{text}:")

    def bold(self, text: str) -> str:
        return self._wrap(text, "1")

    def dim(self, text: str) -> str:
        return self._wrap(text, "2")

    def cyan(self, text: str) -> str:
        return self._wrap(text, "36")

    def cyan_bold(self, text: str) -> str:
        return self._wrap(text, "1;36")

    def blue(self, text: str) -> str:
        return self._wrap(text, "34")

    def green(self, text: str) -> str:
        return self._wrap(text, "32")

    def magenta(self, text: str) -> str:
        return self._wrap(text, "35")

    def _wrap(self, text: str, code: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an interactive coding agent against an OpenAI-compatible server.")
    parser.add_argument("--repo", default=".", help="Local repository directory to edit in place.")
    parser.add_argument("--prompt", help="Coding task prompt. If omitted, read interactively.")
    parser.add_argument(
        "--test-command",
        action="append",
        help="Allowed test command. Repeat to allow multiple commands. If omitted, prompt interactively.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible /v1 base URL.")
    parser.add_argument("--api-key", default="areno-agentic", help="API key passed to the OpenAI client.")
    parser.add_argument("--model", default="policy", help="Model name passed to chat.completions.")
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum tool-use turns.")
    parser.add_argument("--compact-chars", type=int, default=24000, help="Auto-compact history above this size.")
    parser.add_argument(
        "--compact-keep-messages",
        type=int,
        default=16,
        help="Number of recent raw messages to keep after compaction.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print system/user messages as well as tool turns.")
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Colorize CLI output.",
    )
    return asyncio.run(_main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
