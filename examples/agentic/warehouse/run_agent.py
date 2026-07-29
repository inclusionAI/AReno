"""Agent entrypoint for multi-turn warehouse-picking tool-call rollouts."""

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
    "You are a warehouse picking robot. Your goal is to pick all items in the order "
    "and submit the completed order. Use the pick_from_shelf tool to move to a shelf "
    "and pick items, then use submit_order to complete the order once finished. "
    "You can only move to adjacent shelves. Do not answer in plain text - always use a tool."
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
            "description": "Submit the completed order for validation when finished picking.",
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

# 动态提示，根据当前购物车状态调整
def get_turn_prompt(state: WarehouseState, is_first_turn: bool) -> str:
    """根据当前状态生成提示"""
    if state.cart:
        cart_items = ", ".join(f"{sku}×{qty}" for sku, qty in state.cart.items())
        return (
            f"You have picked: {cart_items}. "
            "Use pick_from_shelf to pick more items, or submit_order when done."
        )
    return "Use pick_from_shelf to pick the first item for your order."


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
    """Run multi-turn tool-call rollout until order is completed or max turns reached."""

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

    # 限制最大轮次，避免无限循环和 OOM
    MAX_TURNS = 8
    # 保留最近 N 轮对话以控制序列长度
    MAX_KEEP_TURNS = 3

    async def run_one(item):
        turns = []
        state = _get_or_build_state(item.record)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        # 消息截断：保留 system + user 初始消息 + 最近 N 轮
        def truncate_messages(msgs, keep_turns: int = MAX_KEEP_TURNS):
            """保留 system + user + 最近 keep_turns 轮对话"""
            if len(msgs) <= 2 + keep_turns * 2:
                return msgs
            # 保留前两条 (system + user)
            base = msgs[:2]
            # 保留最近 keep_turns 轮 (每轮 = assistant + tool)
            recent = msgs[-(keep_turns * 2):]
            # 还要保留当前购物车状态在最后tool result中
            return base + recent

        for turn_num in range(MAX_TURNS):
            # 获取当前状态的提示
            turn_prompt = get_turn_prompt(state, turn_num == 0)
            assistant_msg, turn = await _call_model(
                item, client, messages, tools=TOOLS, turn_prompt=turn_prompt
            )
            turns.append(turn)

            # 执行工具并获取结果
            tool_result = _run_tool(assistant_msg, item.record)
            messages.extend(_tool_messages(assistant_msg, tool_result))

            # 控制消息长度，只保留最近 N 轮
            messages = truncate_messages(messages)

            # 检查是否完成订单
            if tool_result.get("data", {}).get("completed"):
                # 订单完成，不再继续
                break

        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


async def _call_model(item, client, messages: list[dict], tools: list[dict], turn_prompt: str):
    """Call the model with tools and prompt."""
    turn_messages = [*messages, {"role": "user", "content": turn_prompt}]
    response = await client.chat.completions.create(
        model="policy",
        messages=turn_messages,
        tools=tools,
        # 不强制工具，让模型自己决定
        stream=False,
    )
    message = response.choices[0].message
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
            for call in (message.tool_calls or [])
        ],
    }
    if not assistant_message["tool_calls"]:
        # 如果没有工具调用，记录下来
        assistant_message["tool_calls"] = []
    return assistant_message, AgentTrajectoryTurn(
        item=item,
        messages=turn_messages,
        response=response,
        tools=tools,
        tool_choice=None,  # 不强制工具选择
    )


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    messages = [assistant_message]
    # 为每个工具调用添加结果
    for call in assistant_message.get("tool_calls") or []:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )
    # 如果没有工具调用，添加一个空结果
    if not assistant_message.get("tool_calls"):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "no_tool_call",
                "name": "none",
                "content": json.dumps({"error": "no tool call made", "data": {}}),
            }
        )
    return messages


def _run_tool(assistant_message: dict, record: dict) -> dict:
    """Execute the environment logic for tool calls."""

    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return {"success": False, "message": "no tool call made", "data": {}}

    # 执行第一个工具调用
    call = calls[0]
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return {"success": False, "message": "invalid JSON arguments", "data": {}}

    state = _get_or_build_state(record)

    if name == "pick_from_shelf":
        result = pick_from_shelf(
            state, args.get("shelf_id", ""), args.get("sku", ""), int(args.get("qty", 0))
        )
        return {"success": result.success, "message": result.message, "data": result.data}
    if name == "submit_order":
        result = submit_order(state)
        return {"success": result.success, "message": result.message, "data": result.data}

    # 其他工具未知
    return {"success": False, "message": f"unknown tool: {name}", "data": {}}