"""Agent entrypoint for single-turn maze tool-call rollouts.

The model sees the full local view and returns a sequence of actions in one
tool call. This avoids the multi-round latency of step-by-step interaction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are navigating a maze. You can see the full map. "
    "Reach the goal G by moving through open floor, "
    "picking up keys K, and opening doors D with held keys. "
    "Call the act tool with a sequence of actions. /no_think"
)

ACT_TOOL = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "Submit a sequence of maze actions to execute in order.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "description": "Ordered list of actions to execute.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["move", "pickup", "use_key"],
                                "description": "The action to perform.",
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["UP", "DOWN", "LEFT", "RIGHT"],
                                "description": "Direction for move or use_key. Omit for pickup.",
                            },
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["actions"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call model request for each maze."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The maze agentic example requires `openai`. Install it with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("Maze agent start tasks=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        record = item.record.get("state", item.record)
        state = game.make_state_from_record(record)
        radius = record.get("view_radius", 1)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": game.format_prompt(state, radius)},
        ]
        tool_choice = {"type": "function", "function": {"name": "act"}}
        try:
            response = await client.chat.completions.create(
                model="policy",
                messages=messages,
                tools=[ACT_TOOL],
                tool_choice=tool_choice,
                stream=False,
            )
        except Exception as exc:
            logger.warning("Maze agent rollout request failed: %s", exc)
            response = None
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[ACT_TOOL],
            tool_choice=tool_choice,
        )

    try:
        turns = list(await asyncio.gather(*(run_one(item) for item in items)))
        return AgentTrajectory(turns=turns)
    finally:
        await client.close()
