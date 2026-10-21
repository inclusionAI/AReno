"""Compose Harbor's native Pi agent with the frontend example adapter.

Harbor owns Pi execution, session logs, turn limits and ATIF conversion. Supply
an existing image with Pi and task tools installed via ARENO_HARBOR_IMAGE, or
pass native Harbor environment/agent configuration to create_runner().
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from runpy import run_path
from typing import Any

from areno.api.harness import HarnessRunner, RecordLoader


def create_runner(
    *,
    harness: str = "pi",
    environment: str = "docker",
    dataset: str = "frontend",
    harness_options: dict[str, Any] | None = None,
    environment_options: dict[str, Any] | None = None,
    dataset_options: dict[str, Any] | None = None,
    loader_fn: RecordLoader | str | Path | None = None,
    loader_kwargs: dict[str, Any] | None = None,
    require_agent_success: bool = False,
) -> HarnessRunner:
    run_path(str(Path(__file__).with_name("frontend_dataset.py")))
    options = json.loads(os.environ.get("ARENO_HARBOR_ENVIRONMENT_OPTIONS", "{}"))
    options.update(environment_options or {})
    if "ARENO_HARBOR_IMAGE" in os.environ:
        options.setdefault("image", os.environ["ARENO_HARBOR_IMAGE"])
    if "ARENO_PI_PYTHON" in os.environ:
        options.setdefault("python", os.environ["ARENO_PI_PYTHON"])
    return HarnessRunner(
        harness=harness,
        environment=environment,
        dataset=dataset,
        harness_options=dict(
            harness_options
            if harness_options is not None
            else (
                {"kwargs": {"thinking": "off", "model_api": "openai-completions"}}
                if harness.strip().lower() == "pi"
                else {}
            )
        ),
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
