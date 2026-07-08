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

DEFAULT_KNOWLEDGE_FILE = Path(__file__).resolve().parents[1] / "agentic" / "coding" / "ops_knowledge.md"
DEFAULT_KNOWLEDGE = DEFAULT_KNOWLEDGE_FILE.read_text(encoding="utf-8")
CONFIG_FILE = Path.home() / ".areno" / "agent_config.json"
DEFAULT_AGENT_TURN_LIMIT = 1_000_000

SYSTEM_TEMPLATE = """You are an AReno operations coding agent.

You can inspect and modify the current checkout and run shell commands through
tools. Your task is to complete the user's train or serve request in this
environment. Work iteratively: inspect the environment, choose a conservative
command, run it, read failures, and retry with adjusted parameters when possible.
Use exactly one tool call per assistant turn. Call submit with status=solved only
after a train/serve command has completed successfully or is clearly running and
verified. Call submit with status=blocked only after a non-recoverable blocker.

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
        response = await client.chat.completions.create(
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
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(knowledge=knowledge)},
        {"role": "user", "content": _job_prompt(args.instruction, workspace.root)},
    ]
    try:
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
        click.echo(json.dumps(workspace.submitted, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if workspace.submitted.get("status") == "solved" else 1
    finally:
        await client.close()
        workspace.close()


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
        "serving, start the server and verify it with a local request when possible."
    )


def _print_event(event: str, payload: dict[str, Any]) -> None:
    if event == "assistant":
        content = payload.get("content")
        if content:
            click.echo(f"\nassistant:\n{content}")
        calls = payload.get("tool_calls") or []
        if calls:
            call = calls[0]
            click.echo(f"\nassistant -> {call['function']['name']}: {call['function'].get('arguments', '')}")
    elif event == "tool":
        click.echo(f"\ntool:{payload.get('name')}")
        click.echo(payload.get("content", ""))


if __name__ == "__main__":
    agent_command()
