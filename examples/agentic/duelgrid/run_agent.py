"""Agent entrypoint for one-step DuelGrid tool-call rollouts."""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are an expert DuelGrid player controlling A. "
    "Choose one or more legal actions by calling the choose_action tool. "
    "The tool name is always choose_action; never use MOVE, ATTACK, or another action as the tool name. "
    "Copy legal action objects from the prompt into the actions array until energy is spent. "
    "Unspent energy is penalized."
)

CHOOSE_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_action",
        "description": "Choose the next DuelGrid action sequence for player A.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "description": "Legal actions to execute this turn, in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["MOVE", "ATTACK", "RANGED_ATTACK", "SHIELD", "PICKUP", "WAIT"],
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["UP", "DOWN", "LEFT", "RIGHT"],
                            },
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["actions"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call model request for each DuelGrid state."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The DuelGrid agentic example requires `openai`. Install it with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("DuelGrid agent start requests=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
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
        tool_choice = {"type": "function", "function": {"name": "choose_action"}}
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=[CHOOSE_ACTION_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[CHOOSE_ACTION_TOOL],
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()
