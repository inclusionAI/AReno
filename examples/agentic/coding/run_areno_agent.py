"""AReno-repo coding-agent entrypoint for target-checkpoint tasks.

Unlike ``run_agent.py``, this runner is for records that point at the real
AReno repository. Each sample clones the requested repo/ref into a disposable
workspace first, then lets the shared coding tools inspect, patch, test, and
submit inside that cloned checkout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_loop import run_conversation_turns  # noqa: E402
from coding_tools import (  # noqa: E402
    DEFAULT_TIMEOUT_S,
    MAX_OUTPUT_CHARS,
    CodingWorkspace,
    ToolError,
    _is_dangerous_rm_command,
    _truncate,
)

from areno.api.agentic import AgentTrajectory  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_ARENO_REPO = "https://github.com/inclusionAI/AReno.git"
DEFAULT_ARENO_REF = "main"
TOOL_CUDA_VISIBLE_DEVICES = "4,5,6,7"
_RUN_LOCK = asyncio.Lock()

ARENO_SYSTEM_PROMPT = """You are a coding agent working in a cloned AReno repository.
Use one tool call per turn. Prefer inspect_tree/read_file/rg to understand the
code, replace_text for simple exact replacements, write_file for creating or
overwriting small files, apply unified diffs for structured edits, run tests, and call submit
when the task is solved or blocked. Do not clone, download, or create another checkout:
the runner has already cloned the requested AReno repository into your workspace.
Keep edits scoped to the requested agentic example and its directly related docs/tests."""


async def run_agent(ctx, batch) -> AgentTrajectory:
    """Clone AReno for each sample, then run the shared coding-agent loop."""

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
        workspace = await asyncio.to_thread(_clone_workspace, item.record)
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


def _clone_workspace(record: dict[str, Any]) -> CodingWorkspace:
    repo_url = str(record.get("repo_url") or DEFAULT_ARENO_REPO)
    repo_ref = str(record.get("repo_ref") or DEFAULT_ARENO_REF)
    workspace = Path(tempfile.mkdtemp(prefix="areno-repo-agent-"))
    try:
        clone_cmd = ["git", "clone", "--depth", "1"]
        if repo_ref and not _looks_like_commit(repo_ref):
            clone_cmd += ["--branch", repo_ref]
        clone_cmd += [repo_url, str(workspace)]
        subprocess.run(clone_cmd, check=True, text=True, capture_output=True, timeout=120)
        if repo_ref and _looks_like_commit(repo_ref):
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", repo_ref],
                cwd=workspace,
                check=True,
                text=True,
                capture_output=True,
                timeout=120,
            )
            subprocess.run(
                ["git", "checkout", "--detach", repo_ref],
                cwd=workspace,
                check=True,
                text=True,
                capture_output=True,
                timeout=60,
            )
        return ArenoRepoWorkspace(task=dict(record), root=workspace, cleanup_on_close=True)
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(workspace, ignore_errors=True)
        raise ToolError(f"git clone timed out after {exc.timeout}s") from exc
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(workspace, ignore_errors=True)
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise ToolError(f"git clone failed: {stderr[:500]}") from exc
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


def _looks_like_commit(value: str) -> bool:
    return len(value) >= 7 and all(char in "0123456789abcdefABCDEF" for char in value)


class ArenoRepoWorkspace(CodingWorkspace):
    """Workspace that pins tool-run subprocesses to the AReno target GPUs."""

    def run_command(self, command: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
        if _is_dangerous_rm_command(command):
            raise ToolError(f"dangerous rm command is not allowed: {command}")
        timeout = min(max(float(timeout_s), 0.1), DEFAULT_TIMEOUT_S)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = TOOL_CUDA_VISIBLE_DEVICES
        proc = subprocess.run(
            command,
            cwd=self.root,
            shell=True,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        result = {
            "command": command,
            "returncode": int(proc.returncode),
            "stdout": _truncate(proc.stdout, MAX_OUTPUT_CHARS),
            "stderr": _truncate(proc.stderr, MAX_OUTPUT_CHARS),
            "env": {"CUDA_VISIBLE_DEVICES": TOOL_CUDA_VISIBLE_DEVICES},
        }
        self.command_history.append(result)
        return result


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
        f"Repository: {record.get('repo_url', DEFAULT_ARENO_REPO)} @ {record.get('repo_ref', DEFAULT_ARENO_REF)}\n"
        f"Target example: {target}\n\n"
        f"Goal:\n{record.get('problem_statement', record.get('instruction', ''))}\n\n"
        f"Expected artifacts:\n{expected_text}\n\n"
        f"Allowed tests: {commands or 'none'}\n"
        "The AReno repository has already been cloned into the current workspace. "
        "Use the coding tools to inspect files, patch the repository, run the allowed tests when practical, "
        "and submit the result."
    )
