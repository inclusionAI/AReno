"""AReno-repo coding-agent entrypoint for target-checkpoint tasks.

Unlike ``run_agent.py``, this runner is for records that point at the real
AReno repository. Each sample copies the local AReno checkout into a disposable
workspace first, then lets the shared coding tools inspect, patch, test, and
submit inside that copied checkout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_loop import run_conversation_turns  # noqa: E402
from coding_tools import CodingWorkspace, ToolError  # noqa: E402

from areno.api.agentic import AgentTrajectory  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_ARENO_SOURCE = "/home/admin/AReno"
DEFAULT_OUTPUT_ROOT = "/tmp/areno-agentic-targets"
TOOL_CUDA_VISIBLE_DEVICES = "4,5,6,7"
_RUN_LOCK = asyncio.Lock()

ARENO_SYSTEM_PROMPT = """You are a coding agent working in a copied AReno repository.
Use one tool call per turn. Prefer inspect_tree/read_file/rg to understand the
code, replace_text for simple exact replacements, write_file for creating or
overwriting small files, apply unified diffs for structured edits, run tests, and call submit
when the task is solved or blocked. Do not clone, download, or create another checkout:
the runner has already copied /home/admin/AReno into your workspace.
Keep edits scoped to the requested agentic example and its directly related docs/tests.
Run AReno training or other long commands with run_command(background=true). Then
use a short command such as sleep 5 to wait, and use read_background_output to
inspect an output range from the background task before deciding the next step.
Write generated datasets and logs under the task output directory shown in the
user prompt; do not write them under the copied source workspace. Do not pass
--save-path to areno train unless the user explicitly asks for checkpoint output."""


async def run_agent(ctx, batch) -> AgentTrajectory:
    """Copy AReno for each sample, then run the shared coding-agent loop."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("The AReno coding agent requires `openai` and `httpx`. Install `openai`.") from exc

    items = list(batch.iter_samples())
    logger.info("AReno coding agent start tasks=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        workspace = await asyncio.to_thread(_copy_workspace, item.record)
        try:
            messages = _initial_messages(item.record)
            return await run_conversation_turns(
                client=client,
                item=item,
                workspace=workspace,
                model="policy",
                messages=messages,
                max_turns=int(item.record.get("max_turns") or 14),
            )
        finally:
            workspace.close()

    try:
        grouped = []
        async with _RUN_LOCK:
            for item in items:
                grouped.append(await run_one(item))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()
        await http_client.aclose()


def _copy_workspace(record: dict[str, Any]) -> CodingWorkspace:
    source = Path(str(record.get("repo_path") or DEFAULT_ARENO_SOURCE)).expanduser().resolve()
    if not source.is_dir():
        raise ToolError(f"AReno source checkout does not exist: {source}")
    workspace = Path(tempfile.mkdtemp(prefix="areno-repo-agent-"))
    try:
        shutil.copytree(
            source,
            workspace,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                "build",
                "dist",
            ),
        )
        return ArenoRepoWorkspace(task=dict(record), root=workspace, cleanup_on_close=True)
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


class ArenoRepoWorkspace(CodingWorkspace):
    """Workspace that pins tool-run subprocesses to the AReno target GPUs."""

    def _command_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = TOOL_CUDA_VISIBLE_DEVICES
        return env

    def _visible_command_env(self) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": TOOL_CUDA_VISIBLE_DEVICES}

    def _background_output_dir(self) -> Path:
        return _output_dir(self.task)


def _initial_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ARENO_SYSTEM_PROMPT},
        {"role": "user", "content": _task_prompt(record)},
    ]


def _task_prompt(record: dict[str, Any]) -> str:
    commands = ", ".join(str(command) for command in record.get("test_commands") or [])
    target = record.get("target_example", "agentic example")
    expected = record.get("expected_artifacts") or []
    expected_text = (
        "\n".join(f"- {item}" for item in expected)
        if expected
        else "- A focused code/docs change for the target."
    )
    return (
        f"AReno repository task: {record.get('instance_id', record.get('id', 'unknown'))}\n"
        f"Repository source: {record.get('repo_path', DEFAULT_ARENO_SOURCE)}\n"
        f"Output directory: {_output_dir(record)}\n"
        f"Target example: {target}\n\n"
        f"Goal:\n{record.get('problem_statement', record.get('instruction', ''))}\n\n"
        f"Expected artifacts:\n{expected_text}\n\n"
        f"Allowed tests: {commands or 'none'}\n"
        "The AReno repository has already been copied into the current workspace. "
        "Keep the copied source workspace as the command working directory, but write generated datasets "
        "and logs under the output directory above. Do not pass --save-path to areno train. "
        "Use the coding tools to inspect files, patch the repository, run the allowed tests when practical, "
        "and submit the result."
    )


def _output_dir(record: dict[str, Any]) -> Path:
    value = record.get("output_dir")
    if value:
        return Path(str(value)).expanduser().resolve()
    instance_id = str(record.get("instance_id", record.get("id", "unknown"))).replace("/", "_")
    return Path(DEFAULT_OUTPUT_ROOT, instance_id).resolve()
