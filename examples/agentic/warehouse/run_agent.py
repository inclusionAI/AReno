"""Agent entrypoint for multi-turn warehouse-picking with small tools.

Design:
- Each action (move, check, pick, submit) is a separate tool
- Each tool call gives immediate feedback via tool result
- Model can freely choose actions instead of forced sequence
- Max 6 turns to balance learning vs memory
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
    move,
    pick,
    query_inventory,
    submit_order,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a warehouse robot. Your goal is to collect items and complete the order. "
    "You can: move_to adjacent shelves, check_shelf for inventory, pick_item from current shelf, "
    "and submit_order when done. Plan your actions wisely."
)

# 拆分为小工具：每个 action 独立
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": "Move one step to an adjacent shelf. You can only move to directly adjacent shelves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shelf_id": {
                        "type": "string",
                        "description": "Adjacent shelf ID to move to, e.g., A1, B2",
                    },
                },
                "required": ["shelf_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_shelf",
            "description": "Check what items are on the current shelf.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick_item",
            "description": "Pick items from current shelf. Must be on the shelf to pick.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "SKU to pick"},
                    "qty": {"type": "integer", "minimum": 1, "description": "Quantity to pick"},
                },
                "required": ["sku", "qty"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_order",
            "description": "Submit the order when you believe it's complete.",
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

# 紧凑状态提示 - 不传完整历史
def make_compact_prompt(state: WarehouseState, order: list[dict], turn: int) -> str:
    """生成紧凑的当前状态提示"""
    order_str = ", ".join(f"{item['sku']}×{item['qty']}" for item in order)

    # 购物车状态
    if state.cart:
        cart_str = ", ".join(f"{sku}×{qty}" for sku, qty in state.cart.items())
        picked = set(state.cart.keys())
        remaining = [f"{item['sku']}×{item['qty']}" for item in order if item['sku'] not in picked]
        remaining_str = ", ".join(remaining) if remaining else "DONE!"
    else:
        cart_str = "empty"
        remaining_str = order_str

    # 可达位置
    neighbors = state.adjacency.get(state.agent_pos, [])
    neighbors_str = ", ".join(neighbors) if neighbors else "none"

    return (
        f"[Turn {turn}] Order: {order_str} | Cart: {cart_str} | "
        f"Need: {remaining_str} | At: {state.agent_pos} | Adj: {neighbors_str}"
    )

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
    """Run multi-turn agent with small tools.

    Design principles:
    - 最多 6 轮（平衡学习 vs 内存）
    - 所有工具都可自由选择，不强制顺序
    - 每轮都有即时反馈
    - 只保留最近 2 轮对话以控制 token
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

    MAX_TURNS = 6
    MAX_KEEP_TURNS = 2  # 只保留最近 2 轮

    async def run_one(item):
        turns = []
        state = _get_or_build_state(item.record)
        order = item.record.get("order", [])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]

        for turn_num in range(1, MAX_TURNS + 1):
            # 检查是否已完成
            if state.completed:
                break

            # 生成紧凑的当前状态提示
            turn_prompt = make_compact_prompt(state, order, turn_num)

            # 调用模型（自由选择工具）
            assistant_msg, turn = await _call_model(item, client, messages, turn_prompt)
            turns.append(turn)

            # 执行工具并获取结果
            tool_result = _run_tool(assistant_msg, item.record)
            messages.extend(_tool_messages(assistant_msg, tool_result))

            # 截断消息：只保留 system + user + 最近 2 轮
            if len(messages) > 2 + MAX_KEEP_TURNS * 2:
                messages = messages[:2] + messages[-(MAX_KEEP_TURNS * 2):]

            # 检查是否被提交完成
            if tool_result.get("data", {}).get("completed"):
                state.completed = True
                break

        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


async def _call_model(item, client, messages: list[dict], turn_prompt: str):
    """Call model with all 4 tools available, no forced choice."""
    turn_messages = [*messages, {"role": "user", "content": turn_prompt}]

    response = await client.chat.completions.create(
        model="policy",
        messages=turn_messages,
        tools=TOOLS,
        tool_choice="auto",  # 自由选择，不强制
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

    # 如果没有工具调用，记录为无调用
    if not assistant_message["tool_calls"]:
        assistant_message["tool_calls"] = []

    return assistant_message, AgentTrajectoryTurn(
        item=item,
        messages=turn_messages,
        response=response,
        tools=TOOLS,
        tool_choice=None,
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
    # 如果没有工具调用，添加空结果
    if not assistant_message.get("tool_calls"):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "no_tool",
                "name": "none",
                "content": json.dumps({"message": "no tool called", "data": {}}),
            }
        )
    return messages


def _run_tool(assistant_message: dict, record: dict) -> dict:
    """Execute any tool call."""
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return {"success": False, "message": "no tool called", "data": {}}

    # 执行第一个工具调用
    call = calls[0]
    name = call["function"]["name"]

    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return {"success": False, "message": "invalid JSON", "data": {}}

    state = _get_or_build_state(record)

    if name == "move_to":
        target = args.get("shelf_id", "")
        result = move(state, target)
        return {"success": result.success, "message": result.message, "data": result.data}

    if name == "check_shelf":
        # 查询当前货架
        result = query_inventory(state, state.agent_pos)
        return {"success": result.success, "message": result.message, "data": result.data}

    if name == "pick_item":
        sku = args.get("sku", "")
        qty = int(args.get("qty", 0))
        result = pick(state, sku, qty)
        return {"success": result.success, "message": result.message, "data": result.data}

    if name == "submit_order":
        result = submit_order(state)
        return {"success": result.success, "message": result.message, "data": result.data}

    return {"success": False, "message": f"unknown tool: {name}", "data": {}}