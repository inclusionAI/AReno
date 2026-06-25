"""AReno agent entrypoint for the multi-turn coding example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_loop import run_agentic_coding_loop  # noqa: E402


async def run_agent(ctx, batch):
    """Run the shared coding-agent loop and return explicit trajectories."""

    return await run_agentic_coding_loop(ctx, batch)
