"""Compose Harbor's native Pi agent with the SVG icon example adapter.

Harbor owns Pi execution, session logs, turn limits and ATIF conversion.
ARENO_HARBOR_ENVIRONMENT selects docker (default) or local. Docker uses a
preinstalled Pi image supplied by ARENO_HARBOR_IMAGE; local reuses host tools.
Native environment/agent configuration can also be passed to create_runner().
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from areno.api.harness import HarnessRunner, RecordLoader
from examples.agentic.pi import svg_dataset  # noqa: F401 -- register the example adapter


def create_runner(
    *,
    harness: str = "pi",
    environment: str | None = None,
    dataset: str = "svg_icon",
    harness_options: dict[str, Any] | None = None,
    environment_options: dict[str, Any] | None = None,
    dataset_options: dict[str, Any] | None = None,
    loader_fn: RecordLoader | str | Path | None = None,
    loader_kwargs: dict[str, Any] | None = None,
    require_agent_success: bool = False,
) -> HarnessRunner:
    environment = (environment or os.environ.get("ARENO_HARBOR_ENVIRONMENT", "docker")).strip()
    if environment.lower() in {"docker", "local"}:
        environment = environment.lower()
    options = json.loads(os.environ.get("ARENO_HARBOR_ENVIRONMENT_OPTIONS", "{}"))
    options.update(environment_options or {})
    if environment != "local" and "ARENO_HARBOR_IMAGE" in os.environ:
        options.setdefault("image", os.environ["ARENO_HARBOR_IMAGE"])
    if "ARENO_PI_PYTHON" in os.environ:
        options.setdefault("python", os.environ["ARENO_PI_PYTHON"])
    agent_options = dict(
        harness_options
        if harness_options is not None
        else (
            {"kwargs": {"thinking": "off", "model_api": "openai-completions"}}
            if harness.strip().lower() == "pi"
            else {}
        )
    )
    if harness.strip().lower() == "pi" and environment == "local":
        harness = "examples.agentic.pi.local_pi:LocalPi"
        options.setdefault("inherit_env", ["PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "NODE_PATH"])
    return HarnessRunner(
        harness=harness,
        environment=environment,
        dataset=dataset,
        harness_options=agent_options,
        environment_options=options,
        dataset_options=dict(dataset_options or {}),
        loader_fn=loader_fn,
        loader_kwargs=dict(loader_kwargs or {}),
        result_dir=os.environ.get("ARENO_PI_RESULT_DIR"),
        attempt_dir=os.environ.get("ARENO_HARBOR_ATTEMPT_DIR"),
        require_agent_success=require_agent_success,
    )


async def run_agent(ctx, batch):
    return await create_runner()(ctx, batch)
