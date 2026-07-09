"""Local coding-agent CLI for AReno train/serve operations."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

DEFAULT_KNOWLEDGE_FILE = Path(__file__).resolve().parents[1] / "agentic" / "coding" / "ops_knowledge.md"
DEFAULT_KNOWLEDGE = DEFAULT_KNOWLEDGE_FILE.read_text(encoding="utf-8")
CONFIG_FILE = Path.home() / ".areno" / "agent_config.json"
DEFAULT_AGENT_TURN_LIMIT = 1_000_000
JUDGE_CONTEXT_CHARS = 24000
PROMPT_PAUSE = "__areno_agent_pause__"
RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
CYAN = "\x1b[38;2;137;221;255m"
BLUE = "\x1b[38;2;130;170;255m"
MAGENTA = "\x1b[38;2;199;146;234m"
YELLOW = "\x1b[38;2;255;203;107m"
RED = "\x1b[38;2;240;113;120m"
MUTED = "\x1b[38;2;103;110;149m"
WHITE = "\x1b[38;2;166;172;205m"
GREEN = "\x1b[38;2;195;232;141m"
AGENT_CONSOLE = Console(color_system=None, force_terminal=False, no_color=True)

SYSTEM_TEMPLATE = """You are an AReno operations coding agent.

You can inspect and modify the current checkout and run shell commands through
tools. Your task is to complete the user's train or serve request in this
environment. Work iteratively: inspect the environment, choose the largest
plausible smoke target, run it, read failures, and retry with adjusted
parameters when possible.
Use exactly one tool call per assistant turn. Call submit with status=solved only
after a train/serve command has completed successfully or is clearly running and
verified. Call submit with status=blocked only after a non-recoverable blocker.

Train command policy:

- For rollout/RL algorithms, use `--n-samples 8` by default unless the user
  explicitly asks for another value.
- Add `--drop-rollout-state` to train and train-smoke commands by default unless
  the user explicitly asks to keep rollout state.
- For rollout/RL jobs, keep `batch_size * n_samples` aligned with
  `--max-running-prompts`: the normal target is
  `max_running_prompts >= batch_size * n_samples`, and if you raise
  `max-running-prompts` for utilization you should also consider raising
  `batch-size` so the batch can actually feed that concurrency.
- Do not start smoke tuning from tiny settings unless the user only requested a
  startup check. Estimate the largest plausible `--max-running-prompts` and
  `--mini-bs`, try those first, and binary search down on recoverable OOM.
- Do not tune `--max-new-tokens` to make smoke or train fit. Treat generation
  length as a task quality target unless the user explicitly changes it.
- For agentic train or serve tasks, if the user did not provide generation
  length or context capacity, ask for `--max-new-tokens` and
  `--max-context-len` before running commands. Do not silently assume defaults
  for these two agentic limits.
- If a large smoke command succeeds with lots of free memory, raise the upper
  bound briefly, but do not run excessive smoke attempts. One large attempt plus
  two or three capacity retries is usually enough before the real train command.
- Keep smoke and final settings within `mem_frac <= 0.9`; leave GPU memory
  headroom instead of choosing a configuration that uses nearly all memory.
- For agentic train tasks, always set `--max-context-len` explicitly after the
  user has confirmed the context cap.
- Never use Hugging Face model hub. For remote model or dataset refs, always use
  `--model-hub modelscope` unless the user explicitly provides a local path.

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
    instruction = asyncio.run(
        _enrich_instruction_with_user_answers_async(
            instruction,
            base_url=resolved_base_url,
            model=resolved_model,
            api_key=resolved_api_key,
            knowledge_file=knowledge_file,
        )
    )

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
    raise SystemExit(_run_agent_console(args))


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


async def _enrich_instruction_with_user_answers_async(
    instruction: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    knowledge_file: str,
) -> str:
    questions = await _llm_questions_for_instruction(
        instruction,
        base_url=base_url,
        model=model,
        api_key=api_key,
        knowledge_file=knowledge_file,
    )
    if not questions:
        return instruction
    click.echo("A few run parameters are missing. Press Enter to accept the recommended value.")
    answers: list[str] = []
    for question in questions[:6]:
        key = str(question.get("key") or "preference").strip() or "preference"
        prompt = str(question.get("question") or key).strip()
        default = str(question.get("default") or "").strip()
        if not prompt or not default:
            continue
        value = await _prompt_value_async(prompt, default=default)
        value = str(value).strip()
        if value:
            answers.append(f"- {key}: {value}")
    if not answers:
        return instruction
    return instruction + "\n\nUser-provided run preferences:\n" + "\n".join(answers)


async def _llm_questions_for_instruction(
    instruction: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    knowledge_file: str,
) -> list[dict[str, str]]:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return []
    from areno.agentic.coding.agent_loop import create_chat_completion_with_retry

    knowledge = _load_knowledge(Path(knowledge_file).expanduser())
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    try:
        response = await create_chat_completion_with_retry(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You prepare a short preflight questionnaire for an AReno train/serve operations agent. "
                        "Read the user goal and ask only for parameters that are genuinely missing and materially "
                        "affect running the command. Examples include checkpoint/model, dataset, algorithm, "
                        "max_new_tokens, max_context_len for agentic training, GPU/TP preset, serve port, or save/load "
                        "requirements. For agentic train or serve goals, you must ask for max_new_tokens and "
                        "max_context_len if either is not already provided. Do not ask questions already answered by "
                        "the user. Do not ask more than six "
                        "questions. Return JSON only: {\"questions\":[{\"key\":\"...\",\"question\":\"...\",\"default\":\"...\"}]}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"User goal:\n{instruction}\n\nRelevant AReno operations knowledge:\n{knowledge[:12000]}",
                },
            ],
            stream=False,
        )
    except Exception as exc:  # noqa: BLE001 - preflight questions are helpful but non-critical.
        click.echo(f"skipping LLM preflight questions: {exc}", err=True)
        return []
    finally:
        await client.close()
    content = (response.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(_extract_json_object(content))
    except json.JSONDecodeError:
        return []
    questions = parsed.get("questions") if isinstance(parsed, dict) else None
    if not isinstance(questions, list):
        return []
    return [question for question in questions if isinstance(question, dict)]


async def _prompt_value_async(question: str, *, default: str = "") -> str:
    if not sys.stdin.isatty():
        return default
    _load_prompt_toolkit()
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.patch_stdout import patch_stdout

    suffix = f" [{default}]" if default else ""
    AGENT_CONSOLE.print(Text(f"{question}{suffix}", style="dim"))
    session = _create_prompt_session()
    with patch_stdout():
        value = await session.prompt_async(HTML('<style fg="#89ddff">❯</style> '))
    if value == PROMPT_PAUSE:
        return default
    return str(value or default)


def _load_prompt_toolkit() -> None:
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError as exc:
        raise click.ClickException("the agent CLI requires prompt-toolkit; install project dependencies first") from exc


def _create_prompt_session(*, model: str | None = None, cwd: Path | None = None) -> Any:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    bindings = KeyBindings()

    @bindings.add("escape")
    def _pause(event: Any) -> None:
        event.app.exit(result=PROMPT_PAUSE)

    @bindings.add("c-c")
    def _ctrl_c(event: Any) -> None:
        event.app.exit(result=PROMPT_PAUSE)

    return PromptSession(
        key_bindings=bindings,
        bottom_toolbar=lambda: _toolbar_html(model=model, cwd=cwd),
        erase_when_done=True,
        style=Style.from_dict(
            {
                "bottom-toolbar": "bg:#11131a #a6accd",
                "": "#a6accd",
                "prompt": "#89ddff bold",
            }
        ),
    )


def _toolbar_html(*, model: str | None, cwd: Path | None) -> Any:
    from html import escape

    from prompt_toolkit.formatted_text import HTML

    display_model = escape(model or "agent")
    display_cwd = escape(_short_path(cwd or Path.cwd()))
    return HTML(
        '<style fg="#89ddff"> agent </style>'
        '<style fg="#676e95"> TVD ▲ </style>'
        f'<style fg="#c792ea">⣶ {display_model}</style>'
        '<style fg="#676e95"> | </style>'
        '<style fg="#a6accd">0.00%</style>'
        '<style fg="#676e95"> | </style>'
        '<style fg="#a6accd"> NRML </style>'
        '<style fg="#676e95"> | </style>'
        f'<style fg="#a6accd">{display_cwd}</style>'
        '<style fg="#676e95"> | </style>'
        '<style fg="#a6accd">areno agent</style>'
    )


def _run_agent_console(args: argparse.Namespace) -> int:
    ui = AgentConsoleUI(args)
    return ui.run()


class AgentConsoleUI:
    """Pretty terminal output for the local AReno operations agent."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.console = AGENT_CONSOLE

    def run(self) -> int:
        _load_prompt_toolkit()
        from prompt_toolkit.patch_stdout import patch_stdout

        with patch_stdout():
            self.startup()
            try:
                return asyncio.run(_main_async(self.args, ui=self))
            except BaseException as exc:  # noqa: BLE001 - surface uncaught agent failures before exiting.
                self.write_panel("error", str(exc))
                return 1

    def startup(self) -> None:
        root = Path(self.args.repo).resolve()
        self.console.print(_banner_renderable(self.args.instruction, root, self.args.model))
        if sys.stdout.isatty():
            self.console.print(_startup_help(), soft_wrap=True)

    def agent_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "assistant":
            content = payload.get("content")
            if content:
                self.write_panel("agent", Markdown(str(content)), direction="in", right=self.args.model)
            calls = payload.get("tool_calls") or []
            if calls:
                call = calls[0]
                tool_name = call["function"]["name"]
                arguments = call["function"].get("arguments", "")
                if tool_name != "run_command":
                    summary = _tool_call_summary(tool_name, arguments)
                    if tool_name in {"read_file", "inspect_tree", "rg", "search"}:
                        self._print_header(tool_name, summary, direction="out")
                    else:
                        self.write_panel(
                            tool_name,
                            _format_tool_arguments(tool_name, arguments),
                            direction="out",
                            right=summary,
                        )
        elif event == "tool":
            name = str(payload.get("name") or "tool")
            content = str(payload.get("content", ""))
            if name == "run_command" and _run_command_was_streamed(content):
                return
            self.write_panel(
                name,
                _format_tool_result(name, content),
                direction="in",
                right=_tool_result_summary(name, content),
            )

    def command_output_event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind == "start":
            command = str(event.get("command") or "")
            self._print_header("shell", f"timeout: {event.get('timeout_s')}", direction="out")
            text = Text()
            text.append("$ ", style="dim")
            text.append(command, style="#a6accd")
            self.console.print(text, soft_wrap=True)
        elif kind == "line":
            stream = str(event.get("stream") or "stdout")
            line = str(event.get("line") or "").rstrip()
            body_style = "#f07178" if stream == "stderr" else "#a6accd"
            text = Text()
            text.append(line, style=body_style)
            self.console.print(text, soft_wrap=True)
        elif kind == "end":
            skipped = int(event.get("skipped_stream_lines") or 0)
            returncode = event.get("returncode")
            timed_out = bool(event.get("timed_out"))
            interrupted = bool(event.get("interrupted"))
            summary = f"returncode={returncode}"
            if interrupted:
                summary += " interrupted=true"
            if timed_out:
                summary += " timed_out=true"
            if skipped:
                summary += f" streamed_screened={skipped} skipped_lines"
            style = "#f07178" if timed_out or returncode not in (0, None) else "#89ddff"
            self.console.print(_shell_exit_line(summary, style=style))

    def judgment(self, judgment: dict[str, Any]) -> None:
        done = bool(judgment.get("done"))
        title = "judge: done" if done else "judge: continue"
        self.write_panel(title, _pretty_value(judgment))

    def done(self, submitted: dict[str, Any]) -> None:
        self.write_panel("done", _pretty_value(submitted))

    def write_panel(
        self,
        title: str,
        body: str | RenderableType,
        *,
        direction: str = "in",
        right: str = "",
    ) -> None:
        self._print_header(title, right, direction=direction)
        self.console.print(_renderable_from_body(body), soft_wrap=True)

    def write(self, text: str) -> None:
        self.console.print(Text.from_ansi(text), end="")

    def _print_header(self, left: str, right: str = "", *, direction: str = "in") -> None:
        self.console.print(_header_line(left, right, self.console.size.width, direction=direction))


class InteractiveAgentInput:
    """Read Esc pauses and line hints without blocking the asyncio loop."""

    def __init__(self, ui: AgentConsoleUI, workspace: Any) -> None:
        self.ui = ui
        self.workspace = workspace
        self._hints: asyncio.Queue[str] = asyncio.Queue()
        self._stop = False
        self._paused = asyncio.Event()
        self._resume_after_pause = False
        self._prompt_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        _load_prompt_toolkit()
        self.ui.console.print(
            Text(
                "interactive: type a hint then Enter to send it on the next model turn; Esc pauses execution.",
                style="dim",
            )
        )
        self._prompt_task = asyncio.create_task(self._prompt_loop())

    def stop(self) -> None:
        self._stop = True
        if self._prompt_task is not None:
            self._prompt_task.cancel()

    async def apply_pending(self, messages: list[dict[str, Any]], phase: str = "before_turn") -> bool:
        if phase == "assistant_no_tool":
            return await self._wait_for_user_answer(messages)
        was_paused = await self._wait_if_paused()
        self._apply_queued_hints(messages)
        should_skip_current_tool = was_paused and phase == "after_assistant"
        if should_skip_current_tool:
            self._resume_after_pause = True
        return not should_skip_current_tool

    def consume_resume_after_pause(self) -> bool:
        should_resume = self._resume_after_pause
        self._resume_after_pause = False
        return should_resume

    async def _wait_for_user_answer(self, messages: list[dict[str, Any]]) -> bool:
        if not sys.stdin.isatty():
            return False
        self.workspace.interrupt_requested = False
        self._paused.set()
        self.ui.console.print(Text("waiting for input: enter a value or hint to continue.", style="dim"))
        await self._wait_if_paused()
        self._apply_queued_hints(messages)
        return True

    async def _wait_if_paused(self) -> bool:
        was_paused = False
        while self._paused.is_set() and not self._stop:
            was_paused = True
            try:
                hint = await asyncio.wait_for(self._hints.get(), timeout=0.1)
            except TimeoutError:
                continue
            await self._hints.put(hint)
            self._paused.clear()
            self.workspace.interrupt_requested = False
            break
        return was_paused

    def _apply_queued_hints(self, messages: list[dict[str, Any]]) -> None:
        while True:
            try:
                hint = self._hints.get_nowait()
            except asyncio.QueueEmpty:
                break
            messages.append({"role": "user", "content": f"User runtime hint:\n{hint}"})
            self.ui.write_panel("user hint", hint)

    async def _prompt_loop(self) -> None:
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.patch_stdout import patch_stdout

        session = _create_prompt_session(model=self.ui.args.model, cwd=Path(self.ui.args.repo).resolve())
        while not self._stop:
            try:
                with patch_stdout():
                    hint = await session.prompt_async(HTML('<style fg="#89ddff">❯</style> '))
            except EOFError:
                self.workspace.interrupt_requested = True
                self._paused.set()
                break
            except asyncio.CancelledError:
                break
            if hint == PROMPT_PAUSE:
                self.workspace.interrupt_requested = True
                self._paused.set()
                self.ui.console.print(
                    Text("paused: current command will stop; enter a hint to continue.", style="#ffcb6b")
                )
                continue
            hint = str(hint).strip()
            if hint:
                await self._hints.put(hint)


async def _main_async(args: argparse.Namespace, *, ui: AgentConsoleUI) -> int:
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
    workspace.command_output_callback = ui.command_output_event
    interaction = InteractiveAgentInput(ui, workspace)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(knowledge=knowledge)},
        {"role": "user", "content": _job_prompt(args.instruction, workspace.root)},
    ]
    try:
        interaction.start()
        while True:
            await run_conversation_turns(
                client=client,
                item=item,
                workspace=workspace,
                model=args.model,
                messages=messages,
                max_turns=int(args.max_turns),
                record_trajectory=False,
                on_event=ui.agent_event,
                interaction_hook=interaction.apply_pending,
            )
            if workspace.submitted is None:
                if interaction.consume_resume_after_pause():
                    continue
                ui.write_panel("stopped", "agent stopped without submit")
                return 2
            judgment = await _judge_goal_done(
                client=client,
                model=args.model,
                instruction=args.instruction,
                submitted=workspace.submitted,
                messages=messages,
                command_history=workspace.command_history,
            )
            ui.judgment(judgment)
            if judgment.get("done") is True:
                ui.done(workspace.submitted)
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
        interaction.stop()
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
                    "--n-samples 8 by default, included --drop-rollout-state by default, kept batch_size * "
                    "n_samples aligned with --max-running-prompts, and searched from a large plausible smoke "
                    "target with a bounded number of binary-search style retries on recoverable capacity "
                    "failures without tuning --max-new-tokens or exceeding mem_frac 0.9. For agentic train "
                    "or serve tasks, check that missing max-new-tokens and max-context-len were asked before "
                    "running and that --max-context-len was set explicitly for train. Return JSON "
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
        "- Keep batch_size * n_samples aligned with max-running-prompts. Usually set "
        "max-running-prompts >= batch_size * n_samples; if increasing max-running-prompts for throughput, "
        "increase batch-size too when the dataset and memory allow it.\n"
        "- Never use Hugging Face model hub. For remote model or dataset refs, always use --model-hub modelscope "
        "unless the user explicitly provides a local path.\n"
        "- Use smoke-infer/smoke-train before long runs. Start from the largest plausible settings you can infer, "
        "then binary search down on recoverable OOM. Do not treat one tiny smoke success as completion.\n"
        "- Do not tune max-new-tokens to make smoke or train fit unless the user explicitly asks for shorter "
        "generation length.\n"
        "- For agentic train or serve tasks, if the user did not provide max-new-tokens or max-context-len, ask "
        "for those values before running commands. Do not silently assume these two agentic limits.\n"
        "- If large smoke succeeds with lots of free memory, raise the upper bound briefly, but avoid too many "
        "smoke runs. One large attempt plus two or three capacity retries is usually enough before choosing "
        "final train settings.\n"
        "- Keep smoke and final settings within mem_frac <= 0.9 so CUDA graphs, allocator fragmentation, and "
        "transient buffers have headroom.\n"
        "- For agentic train tasks, always set --max-context-len explicitly after the user confirms it."
    )


def _banner_text(instruction: str, root: Path, model: str) -> str:
    return f"model: {model}\nworkspace: {root}\ngoal: {instruction}"


def _banner_renderable(instruction: str, root: Path, model: str) -> RenderableType:
    text = Text()
    text.append("AReno operations agent\n", style="#89ddff bold")
    text.append("model: ", style="dim")
    text.append(model, style="#c792ea")
    text.append("  workspace: ", style="dim")
    text.append(_short_path(root), style="#a6accd")
    text.append("\n")
    text.append("goal: ", style="dim")
    text.append(instruction, style="#a6accd")
    return text


def _startup_help() -> Text:
    text = Text()
    text.append("Use ", style="dim")
    text.append("Enter", style="#89ddff")
    text.append(" to queue guidance, ", style="dim")
    text.append("Esc/Ctrl-C", style="#c792ea")
    text.append(" to pause the current run.\n", style="dim")
    text.append("Commands stream as they run. Tool results are summarized; full context stays in the session.\n", style="dim")
    text.append("Agent shortcuts:\n", style="dim")
    text.append("  ", style="dim")
    text.append("smoke-infer / smoke-train", style="#89ddff")
    text.append(" for memory checks, ", style="dim")
    text.append("drop-rollout-state", style="#89ddff")
    text.append(" for RL memory pressure, ", style="dim")
    text.append("modelscope", style="#89ddff")
    text.append(" for remote model refs.\n", style="dim")
    return text


def _short_path(path: Path, *, max_len: int = 34) -> str:
    text = str(path.expanduser())
    home = str(Path.home())
    if text.startswith(home):
        text = "~" + text[len(home) :]
    if len(text) <= max_len:
        return text
    return "…" + text[-(max_len - 1) :]


def _format_tool_arguments(tool_name: str, raw: str) -> RenderableType:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
    if tool_name == "run_command":
        command = str(parsed.get("command", ""))
        timeout = parsed.get("timeout_s")
        command_text = Text()
        command_text.append("$ ", style="dim")
        command_text.append(command, style="#a6accd")
        rows: list[RenderableType] = [command_text]
        if timeout is not None:
            timeout_text = Text("timeout_s: ", style="dim")
            timeout_text.append(str(timeout), style="#89ddff")
            rows.append(timeout_text)
        return Group(*rows)
    return _format_mapping(parsed)


def _format_tool_result(tool_name: str, raw: str) -> RenderableType:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
    if tool_name == "run_command":
        return _format_run_command_result(parsed)
    if tool_name == "read_file":
        return _format_read_file_result(parsed)
    if tool_name == "inspect_tree":
        return _format_tree_result(parsed)
    if tool_name in {"rg", "search"}:
        return _format_search_result(parsed)
    return _format_mapping(parsed)


def _tool_call_summary(tool_name: str, raw: str) -> str:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return ""
    if tool_name == "read_file":
        return str(parsed.get("path") or "")
    if tool_name in {"rg", "search"}:
        parts = [str(parsed.get("pattern") or parsed.get("query") or "").strip()]
        path = str(parsed.get("path") or "").strip()
        if path and path != ".":
            parts.append(path)
        return " · ".join(part for part in parts if part)
    if tool_name == "inspect_tree":
        return str(parsed.get("path") or ".")
    return ""


def _tool_result_summary(tool_name: str, raw: str) -> str:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return ""
    if tool_name == "read_file":
        path = str(parsed.get("path") or "")
        start = parsed.get("start_line")
        end = parsed.get("end_line")
        if start is not None and end is not None:
            return f"{path} · lines {start}-{end}"
        return path
    if tool_name == "inspect_tree":
        tree = parsed.get("tree")
        count = len(tree) if isinstance(tree, list) else 0
        return f"{count} entries"
    if tool_name in {"rg", "search"}:
        matches = parsed.get("matches")
        count = len(matches) if isinstance(matches, list) else 0
        return f"{count} matches"
    return ""


def _run_command_was_streamed(raw: str) -> bool:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return False
    return int(parsed.get("streamed_lines") or 0) > 0


def _format_read_file_result(parsed: dict[str, Any]) -> RenderableType:
    path = str(parsed.get("path") or "")
    content = str(parsed.get("content") or "")
    language = _language_for_path(path)
    return Syntax(content, language, theme="material", line_numbers=False, word_wrap=True)


def _format_tree_result(parsed: dict[str, Any]) -> RenderableType:
    tree = parsed.get("tree")
    if not isinstance(tree, list):
        return _format_mapping(parsed)
    text = "\n".join(str(item) for item in tree)
    rows: list[RenderableType] = [Syntax(text or "<empty>", "text", theme="material", word_wrap=True)]
    if parsed.get("truncated"):
        rows.append(Text("truncated", style="dim"))
    return Group(*rows)


def _format_search_result(parsed: dict[str, Any]) -> RenderableType:
    matches = parsed.get("matches")
    if not isinstance(matches, list):
        return _format_mapping(parsed)
    lines: list[str] = []
    for match in matches[:80]:
        if isinstance(match, dict):
            path = match.get("path", "")
            line = match.get("line", "")
            text = match.get("text", "")
            lines.append(f"{path}:{line}: {text}")
        else:
            lines.append(str(match))
    if parsed.get("truncated"):
        lines.append("<truncated>")
    return Syntax("\n".join(lines) or "<no matches>", "text", theme="material", word_wrap=True)


def _format_run_command_result(parsed: dict[str, Any]) -> RenderableType:
    returncode = parsed.get("returncode")
    timed_out = bool(parsed.get("timed_out"))
    interrupted = bool(parsed.get("interrupted"))
    screened = bool(parsed.get("screened"))
    streamed = int(parsed.get("streamed_lines") or 0)
    if streamed > 0:
        parts = [f"returncode={returncode}", f"streamed={streamed}"]
        if interrupted:
            parts.append("interrupted=true")
        if timed_out:
            parts.append("timed_out=true")
        skipped = int(parsed.get("skipped_stream_lines") or 0)
        if skipped:
            parts.append(f"screened={skipped}")
        return Text(" ".join(parts), style="dim")
    rows: list[RenderableType] = [
        Text("command", style="dim"),
        Syntax(str(parsed.get("command") or ""), "bash", theme="material", word_wrap=True),
        _kv_text("returncode", returncode),
        _kv_text("screened", "yes" if screened else "no"),
        _kv_text("streamed", streamed),
    ]
    skipped = int(parsed.get("skipped_stream_lines") or 0)
    if skipped:
        rows.append(_kv_text("live_skipped", skipped))
    if interrupted:
        rows.append(_kv_text("interrupted", "yes", value_style="#f07178"))
    if timed_out:
        rows.append(_kv_text("timed_out", "yes", value_style="#f07178"))
    output = str(parsed.get("output") or parsed.get("stdout") or parsed.get("stderr") or "")
    if output:
        title = "screened output" if screened else "output"
        rows.append(Text(title, style="dim"))
        rows.append(Syntax(output, "text", theme="material", word_wrap=True))
    return Group(*rows)


def _pretty_value(value: Any, *, indent: int = 0) -> RenderableType:
    if indent == 0:
        return _format_mapping(value) if isinstance(value, dict) else Text(str(value), style="#a6accd")
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return f"{prefix}{MUTED}<empty>{RESET}"
        rows: list[str] = []
        for key, item in value.items():
            label = f"{prefix}{BLUE}{key}{RESET}"
            if isinstance(item, dict | list):
                rows.append(f"{label}:")
                rows.append(_pretty_value(item, indent=indent + 2))
            else:
                rows.append(f"{label}: {_colored_scalar(item)}")
        return "\n".join(rows)
    if isinstance(value, list):
        if not value:
            return f"{prefix}{MUTED}<empty>{RESET}"
        rows = []
        for idx, item in enumerate(value):
            label = f"{prefix}{MUTED}[{idx}]{RESET}"
            if isinstance(item, dict | list):
                rows.append(f"{label}:")
                rows.append(_pretty_value(item, indent=indent + 2))
            else:
                rows.append(f"{label}: {_colored_scalar(item)}")
        return "\n".join(rows)
    return f"{prefix}{_colored_scalar(value)}"


def _format_mapping(value: dict[str, Any]) -> RenderableType:
    rows: list[RenderableType] = []
    for key, item in value.items():
        if key in {"content", "output", "stdout", "stderr"} and isinstance(item, str) and "\n" in item:
            rows.append(Text(str(key), style="dim"))
            rows.append(Syntax(item, "text", theme="material", word_wrap=True))
            continue
        if isinstance(item, dict):
            rows.append(_kv_text(str(key), "<object>", value_style="dim"))
            rows.append(_format_mapping(item))
            continue
        if isinstance(item, list):
            rows.append(_kv_text(str(key), _compact_list(item)))
            continue
        rows.append(_kv_text(str(key), item))
    return Group(*(rows or [Text("<empty>", style="dim")]))


def _compact_list(value: list[Any], *, limit: int = 8) -> str:
    items = [str(item) for item in value[:limit]]
    suffix = f", … +{len(value) - limit}" if len(value) > limit else ""
    return "[" + ", ".join(items) + suffix + "]"


def _language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".json": "json",
        ".jsonl": "json",
        ".md": "markdown",
        ".rst": "rst",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".css": "css",
        ".html": "html",
    }.get(suffix, "text")


def _colored_scalar(value: Any) -> str:
    if value is None:
        return f"{MUTED}null{RESET}"
    if isinstance(value, bool):
        return f"{GREEN if value else RED}{str(value).lower()}{RESET}"
    if isinstance(value, int | float):
        return f"{CYAN}{value}{RESET}"
    text = str(value)
    if "\n" in text:
        return f"{WHITE}{text}{RESET}"
    return f"{WHITE}{text}{RESET}"


def _kv_text(key: str, value: Any, *, value_style: str = "#89ddff") -> Text:
    text = Text()
    text.append(f"{key}: ", style="dim")
    text.append(str(value), style=value_style)
    return text


def _renderable_from_body(body: str | RenderableType) -> RenderableType:
    if isinstance(body, str):
        if "\x1b[" in body:
            return Text.from_ansi(body)
        return Text(body, style="#a6accd")
    return body


def _header_line(left: str, right: str, width: int, *, direction: str = "in") -> Text:
    arrow = "▶" if direction == "out" else "◀"
    left_text = Text()
    left_text.append("▎", style="dim")
    left_text.append(arrow, style="#89ddff")
    left_text.append(" ", style="dim")
    left_text.append(left, style="#89ddff bold")
    right_text = Text(right.strip(), style="dim") if right.strip() else Text()
    line = Text()
    line.append_text(left_text)
    if right_text:
        remaining = width - left_text.cell_len - right_text.cell_len
        if remaining >= 3:
            line.append(" ", style="default")
            line.append("─" * (remaining - 2), style="dim")
            line.append(" ", style="default")
            line.append_text(right_text)
            return line
        line.append(" ", style="default")
        line.append_text(right_text)
        return line
    remaining = width - left_text.cell_len
    if remaining >= 2:
        line.append(" ", style="default")
        line.append("─" * (remaining - 1), style="dim")
    return line


def _shell_exit_line(summary: str, *, style: str) -> Text:
    text = Text()
    text.append("▎", style="dim")
    text.append(f" {summary} ", style=f"{style} reverse")
    return text


def _text_panel(title: str, body: str) -> str:
    clean = body.rstrip()
    width = _panel_width(title)
    inner_width = width - 4
    top = "╭" + "─" * (width - 2) + "╮"
    bottom = "╰" + "─" * (width - 2) + "╯"
    title_line = _panel_line(f"{CYAN}{BOLD}{title}{RESET}", inner_width)
    body_lines = _wrap_panel_body(clean, inner_width)
    middle = "\n".join(_panel_line(line, inner_width) for line in body_lines)
    return (
        f"\n{MUTED}{top}{RESET}\n"
        f"{title_line}\n"
        f"{MUTED}├{'─' * (width - 2)}┤{RESET}\n"
        f"{middle}\n"
        f"{MUTED}{bottom}{RESET}\n"
    )


def _panel_width(title: str) -> int:
    term_width = shutil.get_terminal_size(fallback=(100, 24)).columns
    max_width = max(44, min(term_width - 2, 120))
    return max(min(max_width, term_width), min(max(len(_strip_ansi(title)) + 8, max_width), max_width))


def _wrap_panel_body(body: str, width: int) -> list[str]:
    if not body:
        return [""]
    rows: list[str] = []
    for raw_line in body.splitlines():
        rows.extend(_wrap_ansi_line(raw_line, width) or [""])
    return rows


def _wrap_ansi_line(line: str, width: int) -> list[str]:
    if width <= 1 or _visible_len(line) <= width:
        return [line]

    rows: list[str] = []
    current: list[str] = []
    visible = 0
    index = 0
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    while index < len(line):
        match = ansi_pattern.match(line, index)
        if match:
            current.append(match.group(0))
            index = match.end()
            continue
        char = line[index]
        if visible >= width:
            rows.append("".join(current).rstrip())
            current = []
            visible = 0
            if char == " ":
                index += 1
                continue
        current.append(char)
        visible += 1
        index += 1
    rows.append("".join(current).rstrip())
    return rows


def _panel_line(text: str, width: int) -> str:
    visible = _visible_len(text)
    padding = " " * max(width - visible, 0)
    return f"{MUTED}│{RESET} {text}{RESET}{padding} {MUTED}│{RESET}"


def _visible_len(text: str) -> int:
    return len(_strip_ansi(text))


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


if __name__ == "__main__":
    agent_command()
