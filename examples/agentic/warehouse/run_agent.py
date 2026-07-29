"""Agent entrypoint for two-turn warehouse-picking tool-call rollouts."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    WarehouseState,
    build_state,
    pick_from_shelf,
    submit_order,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a warehouse picking robot. Use the pick_from_shelf tool to move to a shelf "
    "and pick a required item, then use submit_order to complete the order. "
    "You can only move to adjacent shelves. Do not answer in plain text."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pick_from_shelf",
            "description": "Move to an adjacent shelf and pick a quantity of a SKU from it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shelf_id": {
                        "type": "string",
                        "description": "The adjacent shelf to move to and pick from, e.g. A1, B2",
                    },
                    "sku": {"type": "string", "description": "The SKU to pick"},
                    "qty": {"type": "integer", "minimum": 1, "description": "Quantity to pick"},
                },
                "required": ["shelf_id", "sku", "qty"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_order",
            "description": "Submit the completed order for validation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS}

TURN_PROMPTS = {
    "pick_from_shelf": "Turn 1: call pick_from_shelf only. Move to a shelf that has a needed SKU and pick it.",
    "submit_order": "Turn 2: call submit_order only. Submit the completed order.",
}

_state_cache: dict[int, WarehouseState] = {}


def reset_state_cache() -> None:
    """Clear the state cache. Called at run_agent entry and test setup."""

    _state_cache.clear()


def _get_or_build_state(record: dict) -> WarehouseState:
    rid = record.get("id", id(record))
    if rid not in _state_cache:
        _state_cache[rid] = build_state(record)
    return _state_cache[rid]


async def run_agent(ctx, batch):
    """Run two tool-call turns for each warehouse picking task."""

    reset_state_cache()

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The warehouse agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("Warehouse agent start tasks=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        turns = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        for tool_name in ["pick_from_shelf", "submit_order"]:
            assistant_msg, turn = await _call_model(item, client, messages, tool_name)
            turns.append(turn)
            messages.extend(_tool_messages(assistant_msg, _run_tool(assistant_msg, item.record)))
        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


async def _call_model(item, client, messages: list[dict], tool_name: str):
    turn_messages = [*messages, {"role": "user", "content": TURN_PROMPTS[tool_name]}]
    tools = [TOOL_BY_NAME[tool_name]]
    tool_choice = {"type": "function", "function": {"name": tool_name}}
    response = await client.chat.completions.create(
        model="policy",
        messages=turn_messages,
        tools=tools,
        tool_choice=tool_choice,
        stream=False,
    )
    message = response.choices[0].message
    tool_calls = [call for call in (message.tool_calls or []) if call.function.name == tool_name][:1]
    assistant_message = {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ],
    }
    if not assistant_message["tool_calls"]:
        assistant_message["tool_calls"] = [
            {
                "id": f"missing_{tool_name}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": "{}",
                },
            }
        ]
    return assistant_message, AgentTrajectoryTurn(
        item=item,
        messages=turn_messages,
        response=response,
        tools=tools,
        tool_choice=tool_choice,
    )


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    messages = [assistant_message]
    for call in assistant_message.get("tool_calls") or []:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )
    return messages


def _run_tool(assistant_message: dict, record: dict) -> dict:
    """Execute the environment logic for a tool call."""

    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return {"error": "missing tool call"}
    call = calls[0]
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid JSON arguments"}

    state = _get_or_build_state(record)

    if name == "pick_from_shelf":
        result = pick_from_shelf(
            state, args.get("shelf_id", ""), args.get("sku", ""), int(args.get("qty", 0))
        )
        return {"success": result.success, "message": result.message, "data": result.data}
    if name == "submit_order":
        result = submit_order(state)
        return {"success": result.success, "message": result.message, "data": result.data}
    return {"error": f"unknown tool: {name}"}