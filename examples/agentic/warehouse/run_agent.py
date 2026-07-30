"""Agent entrypoint for multi-turn warehouse-picking tool-call rollouts.

The Agent receives a warehouse layout and an order, then uses four tools
(query_inventory, move, pick, submit) across multiple turns to navigate
the warehouse, collect items, and complete the order.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    TOOLS,
    SYSTEM_PROMPT,
    make_prompt,
    query_inventory,
    move,
    pick,
    submit,
    get_agent_view,
    generate_small,
    generate_medium,
    generate_hard,
    WarehouseState,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

TOOL_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS}

MAX_TURNS = 30
"""Maximum tool-call turns before the episode is terminated."""


async def run_agent(ctx, batch):
    """Run multi-turn tool-call episodes for each warehouse-picking task."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The warehouse agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info(
        "Warehouse agent start tasks=%d max_running_prompts=%d",
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
        """Run one warehouse-picking episode."""

        source = dict(item.record)
        difficulty = source.get("difficulty", "small")
        seed = source.get("seed", 42)

        # Reconstruct the warehouse state from the record.
        if difficulty == "medium":
            state = generate_medium(seed=seed)
        elif difficulty == "hard":
            state = generate_hard(seed=seed)
        else:
            state = generate_small(seed=seed)

        # Override order if specified in the record.
        if "order" in source:
            from game import Order
            state.order = Order(order_id="order_1", items=source["order"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_prompt(state)},
        ]

        turns: list[AgentTrajectoryTurn] = []

        for turn_idx in range(MAX_TURNS):
            if state.completed:
                break

            response = await client.chat.completions.create(
                model="policy",
                messages=messages,
                tools=TOOLS,
                stream=False,
            )

            assistant_message = response.choices[0].message

            # If no tool calls, the model is done (or stuck).
            if not assistant_message.tool_calls:
                turns.append(
                    AgentTrajectoryTurn(
                        item=item,
                        messages=list(messages),
                        response=response,
                        tools=TOOLS,
                    )
                )
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_message.tool_calls
                    ],
                }
            )

            # Execute each tool call.
            for tc in assistant_message.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                result = _run_tool(tool_name, args, state)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=list(messages),
                    response=response,
                    tools=TOOLS,
                )
            )

        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


def _run_tool(name: str, args: dict, state: WarehouseState) -> dict:
    """Dispatch a tool call to the corresponding game function."""

    if name == "query_inventory":
        return query_inventory(state, args.get("shelf_id", ""))
    if name == "move":
        return move(state, args.get("direction", ""))
    if name == "pick":
        return pick(state, args.get("item", ""), args.get("quantity", 1))
    if name == "submit":
        return submit(state)
    return {"ok": False, "error": f"unknown tool: {name}"}