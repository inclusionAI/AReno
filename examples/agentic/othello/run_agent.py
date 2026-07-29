"""Agent entrypoint for one-step 6x6 Othello tool-call rollouts."""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a careful 6x6 Othello player. You are given the current board and "
    "your color. Choose exactly one legal move by calling the choose_move tool "
    "with a row and col in [0, 5]. Labels like (row,col) on empty cells are "
    "coordinate references, not discs. Prefer moves that flip more discs and "
    "keep stable corners; avoid passing unless you have no legal move."
)

CHOOSE_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_move",
        "description": "Choose the next 6x6 Othello move (row, col) for the player.",
        "parameters": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "description": "The row index to place the disc in (0-5).",
                },
                "col": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "description": "The column index to place the disc in (0-5).",
                },
            },
            "required": ["row", "col"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call model request for each board."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The Othello agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("Othello agent start requests=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        tool_choice = {"type": "function", "function": {"name": "choose_move"}}
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=[CHOOSE_MOVE_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[CHOOSE_MOVE_TOOL],
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()