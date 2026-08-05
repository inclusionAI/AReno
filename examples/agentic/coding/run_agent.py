"""AReno agent entrypoint for the multi-turn coding example."""

from __future__ import annotations

from areno.agent import run_agentic_coding_loop


async def run_agent(ctx, batch):
    """Run the shared coding-agent loop and return explicit trajectories."""

    return await run_agentic_coding_loop(ctx, batch)
