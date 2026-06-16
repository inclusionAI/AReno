"""Reusable multi-turn coding-agent loop for AReno and the standalone CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from coding_tools import CodingWorkspace, run_tool

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = """You are a coding agent working in an isolated repository.
Use one tool call per turn. Prefer inspect_tree/read_file/rg to understand the
code, replace_text for simple exact replacements, write_file for creating or
overwriting small files, apply unified diffs for structured edits, run tests, and call submit
when the task is solved or blocked. For repository explanation or architecture
questions, inspect the tree and then read README/docs/key source files before
summarizing. Do not infer architecture from filenames alone. Do not claim
success until tests pass for code-change tasks. Tool calls must use valid JSON
arguments matching the tool schema."""

TOOLS = [
    # Tool schemas intentionally mirror Codex-like actions while keeping each
    # turn to one constrained operation for stable agentic RL trajectories.
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List source files under a workspace path. Prefer relative paths like '.' or 'areno/cli'.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_tree",
            "description": "Inspect a compact directory tree. Prefer relative paths like '.' or 'areno/cli'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer", "minimum": 1, "maximum": 6},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a bounded line range from a workspace file. Prefer relative paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rg",
            "description": "Run a ripgrep-style regex search over workspace files. Prefer relative paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "case_sensitive": {"type": "boolean"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a valid unified diff patch with ---/+++/@@ headers to workspace files.",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": "Replace exact text in one workspace file. Use this for simple renames or wording changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "count": {"type": "integer", "minimum": 0},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite one workspace file with exact content. Set append=true to append.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run one shell/test command with a short timeout. Destructive rm commands are blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "number", "minimum": 0.1, "maximum": 10.0},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit final coding-task status and a compact summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["solved", "blocked"]},
                    "summary": {"type": "string"},
                },
                "required": ["status"],
                "additionalProperties": False,
            },
        },
    },
]


async def run_agentic_coding_loop(ctx, batch) -> AgentTrajectory:
    """Run the coding loop for every expanded prompt/sample item."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("The coding agentic example requires `openai` and `httpx`. Install `openai`.") from exc

    items = list(batch.iter_samples())
    logger.info("Coding agent start tasks=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        # Training tasks are materialized into a disposable repo so model
        # patches/tests cannot affect the source checkout running AReno.
        workspace = CodingWorkspace.from_task(item.record)
        try:
            _, turns = await run_single_task(client=client, item=item, workspace=workspace, model="policy")
            return turns
        finally:
            workspace.close()

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


async def run_single_task(
    *,
    client: Any,
    item: Any,
    workspace: CodingWorkspace,
    model: str,
    max_turns: int | None = None,
    record_trajectory: bool = True,
    on_event: Any | None = None,
) -> tuple[list[dict[str, Any]], list[AgentTrajectoryTurn]]:
    """Run one coding task using a standard OpenAI-compatible client."""

    task = workspace.task
    turn_limit = int(max_turns or task.get("max_turns") or 8)
    messages = initial_messages(task)
    turns = await run_conversation_turns(
        client=client,
        item=item,
        workspace=workspace,
        model=model,
        messages=messages,
        max_turns=turn_limit,
        record_trajectory=record_trajectory,
        on_event=on_event,
        implicit_submit_on_plain_text=True,
    )
    return messages, turns


async def run_conversation_turns(
    *,
    client: Any,
    item: Any,
    workspace: CodingWorkspace,
    model: str,
    messages: list[dict[str, Any]],
    max_turns: int,
    record_trajectory: bool = True,
    on_event: Any | None = None,
    implicit_submit_on_plain_text: bool = True,
) -> list[AgentTrajectoryTurn]:
    """Continue an existing coding-agent conversation for up to ``max_turns`` model calls."""

    turns: list[AgentTrajectoryTurn] = []
    for _ in range(max_turns):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        if record_trajectory:
            # AReno trains from explicit per-turn trajectories; no proxy-side
            # prompt matching is needed to reconstruct the multi-turn sample.
            turns.append(AgentTrajectoryTurn(item=item, messages=list(messages), response=response, tools=TOOLS))
        assistant_message = _assistant_message_from_response(response)
        messages.append(assistant_message)
        # The standalone CLI uses this hook to stream model/tool activity as it happens.
        _emit(on_event, "assistant", assistant_message)
        call = _first_tool_call(assistant_message)
        if call is None:
            if not implicit_submit_on_plain_text:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not include a tool call. Continue the agent loop by "
                            "outputting exactly one tool call in the next assistant response: use an available "
                            "inspect/read/search/edit/test tool, or call submit if the task is solved or blocked."
                        ),
                    }
                )
                continue
            # Training can still accept a plain final answer as an implicit submission.
            result = workspace.submit(status="solved", summary=assistant_message.get("content", ""))
            tool_message = {
                "role": "tool",
                "tool_call_id": "implicit_submit",
                "name": "submit",
                "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
            }
            messages.append(tool_message)
            _emit(on_event, "tool", tool_message)
            break
        result = _execute_tool_call(workspace, call)
        tool_message = {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": call["function"]["name"],
            "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
        }
        messages.append(tool_message)
        # Append tool results as chat messages so the next model turn can recover
        # from failed patches/tests using the same context a real coding agent sees.
        _emit(on_event, "tool", tool_message)
        if call["function"]["name"] == "submit":
            break
    return turns


def initial_messages(task: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _task_prompt(task)},
    ]


def _task_prompt(task: dict[str, Any]) -> str:
    commands = ", ".join(str(command) for command in task.get("test_commands") or [])
    instance_id = task.get("instance_id", task.get("id", "unknown"))
    problem = task.get("problem_statement", task.get("instruction", ""))
    fail_to_pass = ", ".join(str(test) for test in task.get("FAIL_TO_PASS", []))
    pass_to_pass = ", ".join(str(test) for test in task.get("PASS_TO_PASS", []))
    return (
        f"SWE-bench instance: {instance_id}\n"
        f"Repository: {task.get('repo', 'local/example')} @ {task.get('base_commit', 'workspace')}\n"
        f"Problem statement: {problem}\n"
        f"Fail-to-pass tests: {fail_to_pass or 'listed in suggested commands'}\n"
        f"Pass-to-pass tests: {pass_to_pass or 'none listed'}\n"
        f"Suggested test commands: {commands or 'none'}\n"
        "Work inside the provided repository. Use tools to inspect, edit, test, and submit. "
        "If this is an information request rather than a code-change request, read enough files to answer with evidence "
        "and then call submit with a concise summary."
    )


def _assistant_message_from_response(response: Any) -> dict[str, Any]:
    message = response.choices[0].message
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (message.tool_calls or [])[:1]
        ],
    }


def _first_tool_call(message: dict[str, Any]) -> dict[str, Any] | None:
    calls = message.get("tool_calls") or []
    return calls[0] if calls else None


def _execute_tool_call(workspace: CodingWorkspace, call: dict[str, Any]) -> dict[str, Any]:
    try:
        arguments = _parse_tool_arguments(call["function"].get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        return {"error": f"invalid JSON arguments: {exc.msg}"}
    if not isinstance(arguments, dict):
        return {"error": "tool arguments must be a JSON object"}
    return run_tool(workspace, call["function"]["name"], arguments)


def _parse_tool_arguments(raw: Any) -> Any:
    value = raw
    for _ in range(3):
        if not isinstance(value, str):
            break
        if value is not raw and not value.strip().startswith(("{", "[")):
            break
        value = json.loads(value or "{}")
    if isinstance(value, dict):
        return _normalize_tool_arguments(value)
    return value


def _normalize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in arguments.items():
        key = _canonical_tool_argument_key(str(raw_key))
        repair_dangling_quote = key in {"path", "pattern", "query", "command", "status"}
        value = (
            _unescape_tool_argument_string(raw_value, repair_dangling_quote=repair_dangling_quote)
            if isinstance(raw_value, str)
            else raw_value
        )
        if key in {"count", "max_depth", "max_lines", "start_line"}:
            value = _coerce_int_argument(value)
        elif key in {"case_sensitive"}:
            value = _coerce_bool_argument(value)
        normalized[key] = value
    return normalized


def _unescape_tool_argument_string(value: Any, *, repair_dangling_quote: bool = False) -> str:
    text = str(value)
    for _ in range(3):
        stripped = text.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
            try:
                text = json.loads(stripped)
                continue
            except json.JSONDecodeError:
                break
        break
    if repair_dangling_quote and isinstance(text, str) and text.endswith('"') and not text.startswith('"'):
        text = text[:-1]
    return str(text)


def _canonical_tool_argument_key(value: str) -> str:
    return _unescape_tool_argument_string(value).strip().lstrip("{").rstrip("}").strip().strip('"').strip()


def _coerce_int_argument(value: Any) -> int:
    if isinstance(value, int):
        return value
    match = re.match(r"^-?\d+", str(value).strip())
    if match is None:
        return 0
    return int(match.group(0))


def _coerce_bool_argument(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _emit(callback: Any | None, event: str, payload: dict[str, Any]) -> None:
    if callback is not None:
        callback(event, payload)
