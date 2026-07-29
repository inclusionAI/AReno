"""Agent entrypoint for 2-turn warehouse-picking tool-call rollouts.

Design reference: shopping example (4 forced turns, 1 tool per turn)
Optimized for low memory: max 2 turns, concise prompts, message truncation
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
    WarehouseState,
    build_state,
    pick_from_shelf,
    submit_order,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 简洁的 system prompt
SYSTEM_PROMPT = (
    "You are a warehouse robot. Use pick_from_shelf to pick items, "
    "then submit_order to complete. Only move to adjacent shelves."
)

# 两个工具定义
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pick_from_shelf",
            "description": "Move to adjacent shelf and pick items: shelf_id, sku, qty",
            "parameters": {
                "type": "object",
                "properties": {
                    "shelf_id": {"type": "string", "description": "Shelf ID like A1, B2"},
                    "sku": {"type": "string", "description": "SKU to pick"},
                    "qty": {"type": "integer", "minimum": 1, "description": "Quantity"},
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
            "description": "Submit completed order for validation.",
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

# 简短的 turn prompts (学习 shopping)
TURN_PROMPTS = {
    "pick_from_shelf": "Pick items: call pick_from_shelf to get required items.",
    "submit_order": "Done: call submit_order to complete the order.",
}

# 状态缓存
_state_cache: dict[int, WarehouseState] = {}


def reset_state_cache() -> None:
    """Clear state cache between batches."""
    _state_cache.clear()


def _get_or_build_state(record: dict) -> WarehouseState:
    rid = record.get("id", id(record))
    if rid not in _state_cache:
        _state_cache[rid] = build_state(record)
    return _state_cache[rid]


async def run_agent(ctx, batch):
    """Run 2-turn agentic rollout (pick -> submit).

    Design principles:
    - Max 2 turns to control memory
    - Each turn exposes only 1 tool (forced tool_choice)
    - Concise prompts
    - Truncate history to 2 messages per turn
    """

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
    logger.info("Warehouse agent start tasks=%d", len(items))

    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(300.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        turns = []
        # 初始消息
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]

        # Turn 1: pick_from_shelf only
        pick_msg, pick_turn = await _call_model(item, client, messages, "pick_from_shelf")
        turns.append(pick_turn)

        tool_result = _run_tool(pick_msg, item.record)
        messages = _tool_messages(pick_msg, tool_result)
        # 截断：只保留 system + user + 这一轮的结果
        messages = messages[:4]  # system, user, assistant, tool

        # 检查是否已完成（可能第一次就完成了）
        completed = tool_result.get("data", {}).get("completed", False)

        if not completed:
            # Turn 2: submit_order only
            submit_msg, submit_turn = await _call_model(item, client, messages, "submit_order")
            turns.append(submit_turn)

            tool_result = _run_tool(submit_msg, item.record)
            # 不需要继续了，2 轮是上限

        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


async def _call_model(item, client, messages: list[dict], tool_name: str):
    """Call model with forced tool choice (1 tool per turn, like shopping)."""
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

    # 如果模型没调用工具，添加空调用
    if not assistant_message["tool_calls"]:
        assistant_message["tool_calls"] = [
            {
                "id": f"missing_{tool_name}",
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
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
    """Build tool result message."""
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
    """Execute tool call."""
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return {"success": False, "message": "no tool call", "data": {}}

    call = calls[0]
    name = call["function"]["name"]

    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return {"success": False, "message": "invalid JSON", "data": {}}

    state = _get_or_build_state(record)

    if name == "pick_from_shelf":
        result = pick_from_shelf(
            state,
            args.get("shelf_id", ""),
            args.get("sku", ""),
            int(args.get("qty", 0)),
        )
        return {"success": result.success, "message": result.message, "data": result.data}

    if name == "submit_order":
        result = submit_order(state)
        return {"success": result.success, "message": result.message, "data": result.data}

    return {"success": False, "message": f"unknown tool: {name}", "data": {}}