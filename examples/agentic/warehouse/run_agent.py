"""Bounded multi-turn agent loop for warehouse navigation."""

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

SYSTEM_PROMPT = (
    "You are a warehouse robot. Navigate to the target shelf and submit the order. "
    "On each turn, call exactly one tool. Use move_to to move one adjacent shelf at a time. "
    "Use submit_order when you are at the target shelf to complete the order."
)

TOOLS = [
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
            "name": "submit_order",
            "description": "Submit the order. Only works when you are at the target shelf.",
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


def make_state_prompt(
    state: WarehouseState,
    *,
    turn_number: int,
    turn_limit: int,
    is_submit_turn: bool,
) -> str:
    """Build a compact state reminder for the current turn."""

    neighbors = ", ".join(state.adjacency.get(state.agent_pos, [])) or "none"
    if is_submit_turn:
        return (
            f"Turn {turn_number} of {turn_limit}. You are at shelf {state.agent_pos}. "
            f"Target shelf: {state.target_shelf}. Call submit_order."
        )
    return (
        f"Turn {turn_number} of {turn_limit}. You are at shelf {state.agent_pos}. "
        f"Target shelf: {state.target_shelf}. Adjacent shelves: {neighbors}. "
        f"Call move_to to navigate toward the target."
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
        timeout=httpx.Timeout(900.0, connect=30.0),
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
    """Run one episode with per-turn forced single tool (shopping pattern)."""

    turns: list[AgentTrajectoryTurn] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt},
    ]
    baseline = baseline_distance(state)
    turn_limit = baseline_action_count(state)

    for turn_number in range(1, turn_limit + 1):
        is_submit_turn = (turn_number == turn_limit)
        tool_name = "submit_order" if is_submit_turn else "move_to"

        turn_messages = [
            *messages,
            {
                "role": "user",
                "content": make_state_prompt(
                    state,
                    turn_number=turn_number,
                    turn_limit=turn_limit,
                    is_submit_turn=is_submit_turn,
                ),
            },
        ]
        assistant_message, turn = await _call_model(item, client, turn_messages, tool_name)
        turns.append(turn)

        result = _execute_tool_call(assistant_message, state)
        payload = _result_payload(result, state, baseline)
        messages = [*turn_messages, *_tool_messages(assistant_message, payload)]
        metrics = payload["data"]["metrics"]
        logger.info(
            "Warehouse action prompt_index=%s sample_index=%s turn=%d tool=%s "
            "success=%s completed=%d invalid=%d distance=%d baseline=%d",
            getattr(item, "prompt_index", None),
            getattr(item, "sample_index", None),
            turn_number,
            tool_name,
            result.success,
            metrics["complete_orders"],
            metrics["invalid_actions"],
            metrics["distance"],
            metrics["baseline_distance"],
        )

        if state.completed:
            break

    return turns


async def _call_model(
    item,
    client,
    messages: list[dict[str, Any]],
    tool_name: str,
) -> tuple[dict[str, Any], AgentTrajectoryTurn]:
    """Call model with only one tool exposed and forced tool_choice (shopping pattern)."""

    tools = [TOOL_BY_NAME[tool_name]]
    tool_choice = {"type": "function", "function": {"name": tool_name}}
    response = await client.chat.completions.create(
        model="policy",
        messages=messages,
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
        messages=messages,
        response=response,
        tools=tools,
        tool_choice=tool_choice,
    )


def _execute_tool_call(
    assistant_message: dict[str, Any],
    state: WarehouseState,
) -> ActionResult:
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        state.invalid_actions += 1
        return ActionResult(
            False,
            "missing tool call",
            {"stage": "tool_protocol"},
        )
    call = calls[0]
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
    tool_result: dict[str, Any],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [assistant_message]
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