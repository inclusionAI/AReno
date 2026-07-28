"""Agent entrypoint for one-step Countdown tool-call rollouts."""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a careful Countdown numbers player. "
    "You are given a set of numbers and a target value. "
    "Pick exactly two numbers and one operation (+, -, *, /) to get as close "
    "to the target as possible. Call the calculate tool with your choice. "
    "Division is only allowed when the result is a whole number."
)

CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Pick two numbers and an operation for the Countdown game.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "The first number to use.",
                },
                "b": {
                    "type": "integer",
                    "description": "The second number to use.",
                },
                "op": {
                    "type": "string",
                    "enum": ["+", "-", "*", "/"],
                    "description": "The arithmetic operation to perform.",
                },
            },
            "required": ["a", "b", "op"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call model request for each Countdown puzzle."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The Countdown agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info(
        "Countdown agent start requests=%d max_running_prompts=%d",
        len(items),
        ctx.max_running_prompts,
    )
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        ),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(
        base_url=ctx.get_base_url(),
        api_key=ctx.api_key,
        http_client=http_client,
        max_retries=0,
    )

    async def run_one(item):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        tool_choice = {"type": "function", "function": {"name": "calculate"}}
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=[CALCULATE_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[CALCULATE_TOOL],
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(
            turns=list(await asyncio.gather(*(run_one(item) for item in items)))
        )
    finally:
        await client.close()