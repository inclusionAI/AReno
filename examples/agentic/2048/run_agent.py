"""Agent entrypoint for episode-based 2048 tool-call rollouts.

Each prompt is one 2048 starting board. The policy returns a bounded ``moves``
array in a single ``choose_moves`` tool call; the engine (in ``game.py``) replays
the whole episode deterministically at reward time.
"""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are an expert 2048 player. "
    "Choose a sequence of legal directions by calling the choose_moves tool. "
    "The tool name is always choose_moves; never use up, down, left, or right as the tool name. "
    "Order moves to maximize merges and grow toward larger tiles. "
    "Stop once no direction changes the board; do not pad with no-op moves."
)

CHOOSE_MOVES_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_moves",
        "description": "Choose the 2048 direction sequence to play from the current board.",
        "parameters": {
            "type": "object",
            "properties": {
                "moves": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "description": "Legal directions to play, in order.",
                    "items": {"type": "string", "enum": ["up", "down", "left", "right"]},
                }
            },
            "required": ["moves"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call model request for each 2048 board."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 2048 agentic example requires `openai`. Install it with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("2048 agent start requests=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    # Limit concurrent HTTP requests to avoid overwhelming the engine's
    # single-file command queue.  Without this, the asyncio.gather below
    # submits *all* requests at once; the proxy/engine serialises them, and
    # the worker refill logic can lose/defer responses, causing a permanent
    # hang.  8 is a safe default that keeps the GPU fed without saturation.
    max_concurrent = min(ctx.max_running_prompts, 8)
    max_connections = max(max_concurrent, ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(item):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        tool_choice = {"type": "function", "function": {"name": "choose_moves"}}
        async with semaphore:
            response = await client.chat.completions.create(
                model="policy",
                messages=messages,
                tools=[CHOOSE_MOVES_TOOL],
                tool_choice=tool_choice,
                stream=False,
            )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[CHOOSE_MOVES_TOOL],
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()