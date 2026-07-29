"""Bounded multi-turn agent loop for warehouse picking."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    ActionResult,
    WarehouseState,
    baseline_action_count,
    baseline_distance,
    build_state,
    execute_action,
    state_metrics,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

TURN_SLACK = 2

SYSTEM_PROMPT = (
    "You are a warehouse robot. Collect exactly the requested items and complete the order. "
    "On each action turn, call exactly one available tool. Use query_inventory to locate requested "
    "SKUs, move_to for one adjacent step, check_shelf to verify stock after arrival, pick_item only "
    "after inspecting the current shelf, and submit_order only when the cart exactly matches the "
    "order. Prefer the shortest route and never invent stock or call multiple tools at once."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_inventory",
            "description": "Find every current shelf location and quantity for one SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "SKU whose warehouse locations should be returned.",
                    }
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": "Move one step to a directly adjacent shelf.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shelf_id": {
                        "type": "string",
                        "description": "Adjacent shelf ID, such as A2 or B1.",
                    }
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
            "description": "Inspect the stock on the current shelf.",
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
            "description": "Pick a positive quantity of one SKU from the current shelf.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "SKU to pick.",
                    },
                    "qty": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Quantity to pick.",
                    },
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
            "description": "Validate and submit the current cart.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


def make_state_prompt(
    state: WarehouseState,
    *,
    turn_number: int,
    turn_limit: int,
) -> str:
    """Build a compact state reminder without discarding message history."""

    order = ", ".join(f"{item['sku']} x{item['qty']}" for item in state.order)
    cart = ", ".join(f"{sku} x{quantity}" for sku, quantity in sorted(state.cart.items())) if state.cart else "empty"
    remaining = []
    for item in state.order:
        quantity = item["qty"] - state.cart.get(item["sku"], 0)
        if quantity > 0:
            remaining.append(f"{item['sku']} x{quantity}")
    needed = ", ".join(remaining) if remaining else "nothing; submit the order"
    neighbors = ", ".join(state.adjacency.get(state.agent_pos, [])) or "none"
    return (
        f"Action turn {turn_number} of {turn_limit}. Order: {order}. Cart: {cart}. "
        f"Still needed: {needed}. Current shelf: {state.agent_pos}. Adjacent shelves: {neighbors}. "
        "Call exactly one tool."
    )


async def run_agent(ctx, batch):
    """Run one isolated bounded warehouse episode per prompt/sample pair."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The warehouse agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    episodes = [(item, build_state(item.record)) for item in items]
    logger.info(
        "Warehouse agent start requests=%d max_running_prompts=%d",
        len(episodes),
        ctx.max_running_prompts,
    )

    max_connections = max(len(episodes), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        ),
        timeout=httpx.Timeout(300.0, connect=30.0),
    )
    client = AsyncOpenAI(
        base_url=ctx.get_base_url(),
        api_key=ctx.api_key,
        http_client=http_client,
        max_retries=0,
    )

    try:
        grouped = await asyncio.gather(*(_run_episode(item, state, client) for item, state in episodes))
        return AgentTrajectory(turns=[turn for episode in grouped for turn in episode])
    finally:
        await client.close()


async def _run_episode(
    item,
    state: WarehouseState,
    client,
) -> list[AgentTrajectoryTurn]:
    """Run one episode while preserving exact assistant/tool ordering."""

    turns: list[AgentTrajectoryTurn] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt},
    ]
    baseline = baseline_distance(state)
    turn_limit = baseline_action_count(state) + TURN_SLACK
    termination_reason = "turn_limit"
    needs_final_turn = False

    for turn_number in range(1, turn_limit + 1):
        turn_messages = [
            *messages,
            {
                "role": "user",
                "content": make_state_prompt(
                    state,
                    turn_number=turn_number,
                    turn_limit=turn_limit,
                ),
            },
        ]
        assistant_message, turn = await _call_model(
            item,
            client,
            turn_messages,
        )
        turns.append(turn)
        messages = [*turn_messages, assistant_message]

        calls = list(assistant_message.get("tool_calls") or [])
        if not calls:
            logger.warning(
                "Warehouse model returned no tool call prompt_index=%s sample_index=%s turn=%d",
                getattr(item, "prompt_index", None),
                getattr(item, "sample_index", None),
                turn_number,
            )
            termination_reason = "missing_tool_call"
            needs_final_turn = False
            break

        if len(calls) != 1:
            state.invalid_actions += 1
            message = f"expected exactly one tool call, received {len(calls)}"
            results = [
                _result_payload(
                    ActionResult(
                        False,
                        message,
                        {
                            "stage": "tool_protocol",
                            "call_count": len(calls),
                        },
                    ),
                    state,
                    baseline,
                )
                for _ in calls
            ]
            messages.extend(_tool_messages(assistant_message, results))
            termination_reason = "invalid_tool_count"
            needs_final_turn = True
            break

        result = _execute_tool_call(calls[0], state)
        payload = _result_payload(result, state, baseline)
        messages.extend(_tool_messages(assistant_message, [payload]))
        metrics = payload["data"]["metrics"]
        logger.info(
            "Warehouse action prompt_index=%s sample_index=%s turn=%d tool=%s "
            "success=%s completed=%d mistakes=%d invalid=%d empty_checks=%d "
            "distance=%d baseline=%d",
            getattr(item, "prompt_index", None),
            getattr(item, "sample_index", None),
            turn_number,
            calls[0]["function"]["name"],
            result.success,
            metrics["complete_orders"],
            metrics["picking_mistakes"],
            metrics["invalid_actions"],
            metrics["empty_shelf_checks"],
            metrics["distance"],
            metrics["baseline_distance"],
        )
        needs_final_turn = True

        if state.completed:
            termination_reason = "completed"
            break

    if needs_final_turn:
        turns.append(
            await _final_turn(
                item,
                client,
                messages,
                state,
                baseline,
                termination_reason,
            )
        )
    return turns


async def _call_model(
    item,
    client,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], AgentTrajectoryTurn]:
    response = await client.chat.completions.create(
        model="policy",
        messages=messages,
        tools=TOOLS,
        tool_choice="required",
        stream=False,
    )
    return _assistant_message(response), AgentTrajectoryTurn(
        item=item,
        messages=messages,
        response=response,
        tools=TOOLS,
        tool_choice="required",
    )


async def _final_turn(
    item,
    client,
    messages: list[dict[str, Any]],
    state: WarehouseState,
    baseline: int,
    termination_reason: str,
) -> AgentTrajectoryTurn:
    metrics = state_metrics(state, baseline=baseline)
    final_messages = [
        *messages,
        {
            "role": "user",
            "content": (
                f"The episode ended with reason {termination_reason}. "
                f"Metrics: {json.dumps(metrics, sort_keys=True)}. "
                "Briefly summarize the outcome without calling a tool."
            ),
        },
    ]
    response = await client.chat.completions.create(
        model="policy",
        messages=final_messages,
        tools=TOOLS,
        tool_choice="none",
        stream=False,
    )
    return AgentTrajectoryTurn(
        item=item,
        messages=final_messages,
        response=response,
        tools=TOOLS,
        tool_choice="none",
    )


def _assistant_message(response) -> dict[str, Any]:
    message = response.choices[0].message
    return {
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


def _execute_tool_call(
    call: dict[str, Any],
    state: WarehouseState,
) -> ActionResult:
    function = call.get("function")
    if not isinstance(function, dict):
        state.invalid_actions += 1
        return ActionResult(
            False,
            "tool call function must be an object",
            {"stage": "tool_protocol", "input": "function"},
        )
    name = function.get("name")
    if not isinstance(name, str) or not name:
        state.invalid_actions += 1
        return ActionResult(
            False,
            "tool call function name must be a non-empty string",
            {"stage": "tool_protocol", "input": "name"},
        )

    raw_arguments = function.get("arguments", "")
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        state.invalid_actions += 1
        return ActionResult(
            False,
            "tool arguments must be valid JSON",
            {"stage": "tool_validation", "input": "arguments"},
        )
    return execute_action(state, name, arguments)


def _result_payload(
    result: ActionResult,
    state: WarehouseState,
    baseline: int,
) -> dict[str, Any]:
    data = dict(result.data)
    data["metrics"] = state_metrics(state, baseline=baseline)
    return {
        "success": result.success,
        "message": result.message,
        "data": data,
    }


def _tool_messages(
    assistant_message: dict[str, Any],
    tool_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls = list(assistant_message.get("tool_calls") or [])
    if len(calls) != len(tool_results):
        raise ValueError(f"tool result count {len(tool_results)} does not match call count {len(calls)}")

    messages: list[dict[str, Any]] = []
    for call, result in zip(calls, tool_results, strict=True):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
    return messages
