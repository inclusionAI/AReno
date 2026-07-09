"""Local coding-agent CLI for AReno train/serve operations."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import os
import re
import subprocess
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Box, Frame, Label

DEFAULT_KNOWLEDGE_FILE = Path(__file__).resolve().parents[1] / "agentic" / "coding" / "ops_knowledge.md"
DEFAULT_KNOWLEDGE = DEFAULT_KNOWLEDGE_FILE.read_text(encoding="utf-8")
CONFIG_FILE = Path.home() / ".areno" / "agent_config.json"
DEFAULT_AGENT_TURN_LIMIT = 1_000_000
JUDGE_CONTEXT_CHARS = 24000
RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
CYAN = "\x1b[38;2;34;211;238m"
BLUE = "\x1b[38;2;96;165;250m"
MAGENTA = "\x1b[38;2;217;70;239m"
YELLOW = "\x1b[38;2;251;191;36m"
RED = "\x1b[38;2;248;113;113m"
MUTED = "\x1b[38;2;148;163;184m"
WHITE = "\x1b[38;2;226;232;240m"
GREEN = "\x1b[38;2;45;212;191m"

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
- If a large smoke command succeeds with lots of free memory, raise the upper
  bound briefly, but do not run excessive smoke attempts. One large attempt plus
  two or three capacity retries is usually enough before the real train command.
- Keep smoke and final settings within `mem_frac <= 0.9`; leave GPU memory
  headroom instead of choosing a configuration that uses nearly all memory.
- For agentic train tasks, always set `--max-context-len` explicitly. If the
  user does not provide one, use a practical cap such as 32768.
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
    raise SystemExit(_run_agent_tui(args))


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
        value = click.prompt(prompt, default=default, show_default=True)
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
                        "requirements. Do not ask questions already answered by the user. Do not ask more than six "
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


def _run_agent_tui(args: argparse.Namespace) -> int:
    app = AgentTuiApp(args)
    result = app.run()
    return int(result or 0)


class AgentTuiApp:
    """prompt_toolkit interface for the local AReno operations agent."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.exit_code = 1
        self._log_chunks: deque[str] = deque()
        self._log_chars = 0
        self._log_text = ""
        self._lock = threading.Lock()
        self._agent_thread: threading.Thread | None = None
        self.output_control = FormattedTextControl(lambda: ANSI(self._log_text), focusable=True)
        self.output_window = Window(
            content=self.output_control,
            wrap_lines=True,
            right_margins=[ScrollbarMargin(display_arrows=True)],
            always_hide_cursor=True,
        )
        self.application = self._build_application()

    def run(self) -> int:
        self._agent_thread = threading.Thread(target=self._run_agent_thread, name="areno-agent", daemon=True)
        self._agent_thread.start()
        result = self.application.run()
        return int(result or self.exit_code or 0)

    def _build_application(self) -> Application[int]:
        bindings = KeyBindings()

        @bindings.add("c-c")
        @bindings.add("q")
        def _quit(_event: Any) -> None:
            self.application.exit(result=self.exit_code)

        @bindings.add("pagedown")
        def _page_down(_event: Any) -> None:
            self.output_window.vertical_scroll += 20
            self.application.invalidate()

        @bindings.add("pageup")
        def _page_up(_event: Any) -> None:
            self.output_window.vertical_scroll = max(0, self.output_window.vertical_scroll - 20)
            self.application.invalidate()

        @bindings.add("end")
        def _end(_event: Any) -> None:
            self._scroll_to_end()
            self.application.invalidate()

        sidebar = HSplit(
            [
                Label(text="AReno Agent"),
                Label(text=""),
                Label(text=f"Model\n{self.args.model}"),
                Label(text=""),
                Label(text=f"Workspace\n{Path(self.args.repo).resolve()}"),
                Label(text=""),
                Label(text=f"Goal\n{self.args.instruction}"),
            ]
        )
        root = HSplit(
            [
                Label(text=self._header_text()),
                VSplit(
                    [
                        Box(Frame(sidebar, title="Run"), width=34, padding=1),
                        Frame(self.output_window, title="Live Activity"),
                    ]
                ),
                Label(text=" q / Ctrl-C: quit   |   output streams live   |   scroll: mouse / PageUp / PageDown "),
            ]
        )
        style = Style.from_dict(
            {
                "frame.border": "#3b4758",
                "label": "#c8d3df bg:#111820",
                "text-area": "#d6dee8 bg:#0b0f14",
                "window": "bg:#0b0f14",
            }
        )
        return Application(
            layout=Layout(root, focused_element=self.output_window),
            key_bindings=bindings,
            full_screen=True,
            mouse_support=True,
            refresh_interval=0.1,
            style=style,
        )

    def _header_text(self) -> str:
        return " AReno Operations Agent  |  model: " + self.args.model

    def _run_agent_thread(self) -> None:
        self.write_panel("areno agent", _banner_text(self.args.instruction, Path(self.args.repo).resolve(), self.args.model))
        try:
            self.exit_code = asyncio.run(_main_async(self.args, ui=self))
        except BaseException as exc:  # noqa: BLE001 - surface uncaught agent failures in the TUI before exiting.
            self.exit_code = 1
            self.write_panel("error", str(exc))
        finally:
            self._exit_from_thread()

    def _exit_from_thread(self) -> None:
        loop = getattr(self.application, "loop", None)
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(lambda: self.application.exit(result=self.exit_code))
        else:
            self.application.exit(result=self.exit_code)

    def agent_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "assistant":
            content = payload.get("content")
            if content:
                self.write_panel("assistant", str(content))
            calls = payload.get("tool_calls") or []
            if calls:
                call = calls[0]
                tool_name = call["function"]["name"]
                arguments = call["function"].get("arguments", "")
                self.write_panel(f"tool call: {tool_name}", _format_tool_arguments(tool_name, arguments))
        elif event == "tool":
            name = str(payload.get("name") or "tool")
            self.write_panel(f"tool result: {name}", _format_tool_result(name, str(payload.get("content", ""))))

    def command_output_event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind == "start":
            self.write(f"\n{MAGENTA}{BOLD}━━ command output ━━{RESET}\n")
            self.write(f"{MAGENTA}$ {event.get('command') or ''}{RESET}\n")
            self.write(f"{MUTED}timeout_s: {event.get('timeout_s')}{RESET}\n")
        elif kind == "line":
            stream = str(event.get("stream") or "stdout")
            line = str(event.get("line") or "").rstrip()
            prefix = RED if stream == "stderr" else CYAN
            body = RED if stream == "stderr" else WHITE
            self.write(f"{prefix}{stream}>{RESET} {body}{line}{RESET}\n")
        elif kind == "end":
            skipped = int(event.get("skipped_stream_lines") or 0)
            returncode = event.get("returncode")
            timed_out = bool(event.get("timed_out"))
            summary = f"returncode={returncode}"
            if timed_out:
                summary += " timed_out=true"
            if skipped:
                summary += f" streamed_screened={skipped} skipped_lines"
            color = RED if timed_out or returncode not in (0, None) else GREEN
            self.write(f"{color}{BOLD}━━ {summary} ━━{RESET}\n")

    def judgment(self, judgment: dict[str, Any]) -> None:
        done = bool(judgment.get("done"))
        title = "judge: done" if done else "judge: continue"
        self.write_panel(title, _pretty_value(judgment))

    def done(self, submitted: dict[str, Any]) -> None:
        self.write_panel("done", _pretty_value(submitted))

    def write_panel(self, title: str, body: str) -> None:
        self.write(_text_panel(title, body))

    def write(self, text: str) -> None:
        with self._lock:
            self._log_chunks.append(text)
            self._log_chars += len(text)
            current = "".join(self._log_chunks)
        self._set_output_text(current)

    def _set_output_text(self, text: str) -> None:
        def update() -> None:
            self._log_text = text
            self._scroll_to_end()
            self.application.invalidate()

        loop = getattr(self.application, "loop", None)
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(update)
        else:
            update()

    def _scroll_to_end(self) -> None:
        self.output_window.vertical_scroll = max(0, self._log_text.count("\n"))


async def _main_async(args: argparse.Namespace, *, ui: AgentTuiApp) -> int:
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
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(knowledge=knowledge)},
        {"role": "user", "content": _job_prompt(args.instruction, workspace.root)},
    ]
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
                on_event=ui.agent_event,
            )
            if workspace.submitted is None:
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
                    "tasks, check that --max-context-len was set explicitly. Return JSON "
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
        "- If large smoke succeeds with lots of free memory, raise the upper bound briefly, but avoid too many "
        "smoke runs. One large attempt plus two or three capacity retries is usually enough before choosing "
        "final train settings.\n"
        "- Keep smoke and final settings within mem_frac <= 0.9 so CUDA graphs, allocator fragmentation, and "
        "transient buffers have headroom.\n"
        "- For agentic train tasks, always set --max-context-len explicitly. If the user does not provide one, "
        "use a practical cap such as 32768."
    )


def _banner_text(instruction: str, root: Path, model: str) -> str:
    return f"model: {model}\nworkspace: {root}\ngoal: {instruction}"


def _format_tool_arguments(tool_name: str, raw: str) -> str:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
    if tool_name == "run_command":
        command = str(parsed.get("command", ""))
        timeout = parsed.get("timeout_s")
        rows = [f"{MUTED}command{RESET}:\n{WHITE}{command}{RESET}"]
        if timeout is not None:
            rows.append(f"{MUTED}timeout_s{RESET}: {CYAN}{timeout}{RESET}")
        return "\n".join(rows)
    return _pretty_value(parsed)


def _format_tool_result(tool_name: str, raw: str) -> str:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
    if tool_name == "run_command":
        return _format_run_command_result(parsed)
    return _pretty_value(parsed)


def _format_run_command_result(parsed: dict[str, Any]) -> str:
    returncode = parsed.get("returncode")
    timed_out = bool(parsed.get("timed_out"))
    screened = bool(parsed.get("screened"))
    rows = [
        f"{MUTED}command{RESET}:\n{WHITE}{parsed.get('command') or ''}{RESET}",
        f"{MUTED}returncode{RESET}: {_colored_scalar(returncode)}",
        f"{MUTED}screened{RESET}: {_colored_scalar('yes' if screened else 'no')}",
        f"{MUTED}streamed{RESET}: {_colored_scalar(parsed.get('streamed_lines', 0))}",
    ]
    skipped = int(parsed.get("skipped_stream_lines") or 0)
    if skipped:
        rows.append(f"{MUTED}live_skipped{RESET}: {_colored_scalar(skipped)}")
    if timed_out:
        rows.append(f"{MUTED}timed_out{RESET}: {RED}yes{RESET}")
    output = str(parsed.get("output") or parsed.get("stdout") or parsed.get("stderr") or "")
    if output:
        title = "screened output" if screened else "output"
        rows.append(f"{MUTED}{title}{RESET}:\n{WHITE}{output}{RESET}")
    return "\n".join(rows)


def _pretty_value(value: Any, *, indent: int = 0) -> str:
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


def _text_panel(title: str, body: str) -> str:
    clean = body.rstrip()
    width = min(max(len(_strip_ansi(title)) + 8, 44), 96)
    top = "╭" + "─" * (width - 2) + "╮"
    bottom = "╰" + "─" * (width - 2) + "╯"
    return f"\n{MUTED}{top}{RESET}\n{CYAN}{BOLD}│ {title}{RESET}\n{MUTED}├{'─' * (width - 2)}┤{RESET}\n{clean}\n{MUTED}{bottom}{RESET}\n"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


if __name__ == "__main__":
    agent_command()
