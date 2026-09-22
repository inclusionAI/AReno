"""Run the real pi CLI against a secondary, trajectory-capturing proxy."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path

from areno.api.agentic import AgentTrajectory

# AReno loads --agent-fn by filename rather than importing an examples package.
_EXAMPLE_DIR = str(Path(__file__).resolve().parent)
if _EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLE_DIR)
from pi_proxy import PiProxy  # noqa: E402

logger = logging.getLogger(__name__)


def write_files(workspace: Path, files: dict) -> None:
    for name, content in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] == ".pi":
            raise ValueError(f"invalid task file path: {name}")
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def configure_pi(config_dir: Path, proxy: PiProxy) -> None:
    config_dir.mkdir()
    provider = {
        "baseUrl": proxy.base_url,
        "api": "openai-completions",
        "apiKey": proxy.api_key,
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False},
        "models": [
            {
                "id": "policy",
                "name": "AReno rollout policy",
                "reasoning": False,
                "input": ["text"],
                "contextWindow": 131072,
                "maxTokens": 4096,
            }
        ],
    }
    (config_dir / "models.json").write_text(json.dumps({"providers": {"areno": provider}}))
    # Keep auxiliary summarization and automatic retries out of training traces.
    (config_dir / "settings.json").write_text(
        json.dumps({"compaction": {"enabled": False}, "retry": {"enabled": False, "provider": {"maxRetries": 0}}})
    )


async def run_process(command, *, cwd, env, timeout, log_path):
    """Bound execution and reap the process group, including tool children."""
    with log_path.open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return await asyncio.wait_for(process.wait(), timeout), False
        except asyncio.TimeoutError:
            return None, True
        finally:
            # Kill remaining tool processes even if pi itself has already exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()


def log_tail(path, limit=8000):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read().decode(errors="replace")


async def run_agent(ctx, batch):
    executable = shutil.which(os.environ.get("ARENO_PI_EXECUTABLE", "pi"))
    if executable is None:
        raise RuntimeError("pi is not installed; install @mariozechner/pi-coding-agent or set ARENO_PI_EXECUTABLE")
    semaphore = asyncio.Semaphore(ctx.max_running_prompts)

    async def run_sample(original):
        async with semaphore:
            # iter_samples shares dataset records across samples. Rewards must not.
            item = copy.deepcopy(original)
            timeout = float(item.record.get("timeout", 300))
            verify_timeout_s = float(item.record.get("verify_timeout", 30))
            if not math.isfinite(verify_timeout_s) or verify_timeout_s <= 0:
                raise ValueError("verify_timeout must be finite and positive")
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be finite and positive")
            with tempfile.TemporaryDirectory(prefix="areno-pi-") as directory:
                root = Path(directory)
                workspace = root / "workspace"
                workspace.mkdir()
                write_files(workspace, item.record["files"])
                async with PiProxy(ctx, max_turns=int(item.record.get("max_turns", 32)), timeout=timeout) as proxy:
                    config_dir = root / "pi-config"
                    configure_pi(config_dir, proxy)
                    env = dict(os.environ, PI_CODING_AGENT_DIR=str(config_dir))
                    command = [
                        executable,
                        "--print",
                        "--no-session",
                        "--no-extensions",
                        "--no-skills",
                        "--no-prompt-templates",
                        "--no-themes",
                        "--provider",
                        "areno",
                        "--model",
                        "policy",
                        "Task:\n" + item.prompt,
                    ]
                    code, timed_out = await run_process(
                        command, cwd=workspace, env=env, timeout=timeout, log_path=root / "pi.log"
                    )
                result = {
                    "returncode": code,
                    "timed_out": timed_out,
                    "log": log_tail(root / "pi.log"),
                    "proxy_errors": proxy.errors,
                    "turn_limit_reached": proxy.limit_reached,
                }
                item.record["pi_result"] = result
                if proxy.errors or not proxy.trace:
                    logger.warning(
                        "pi rollout invalid prompt=%d sample=%d result=%s", item.prompt_index, item.sample_index, result
                    )
                    return AgentTrajectory(invalid_items=[item])
                # The verifier comes from the trusted dataset, outside editable files.
                # -I avoids PYTHONPATH/user site; insert only the task workspace.
                verifier = "import sys; sys.path.insert(0, sys.argv[1]);\n" + item.record["verify"]
                verify_code, verify_timeout = await run_process(
                    [sys.executable, "-I", "-c", verifier, str(workspace)],
                    cwd=workspace,
                    env=dict(os.environ),
                    timeout=verify_timeout_s,
                    log_path=root / "verify.log",
                )
                result.update(
                    verify_returncode=verify_code,
                    verify_timed_out=verify_timeout,
                    verify_log=log_tail(root / "verify.log"),
                )
                # Failed/timed-out attempts with usable traces remain negative examples.
                result["reward"] = float(
                    code == 0 and not timed_out and not proxy.limit_reached and verify_code == 0 and not verify_timeout
                )
                return AgentTrajectory(turns=proxy.turns(item))

    tasks = [asyncio.create_task(run_sample(item)) for item in batch.iter_samples()]
    try:
        trajectories = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return AgentTrajectory(
        turns=[turn for trajectory in trajectories for turn in trajectory.turns],
        invalid_items=[item for trajectory in trajectories for item in trajectory.invalid_items],
    )
