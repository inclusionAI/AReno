"""Agent entrypoint for multi-turn maze tool-call rollouts.

The agent receives a local view of the maze and calls ``move`` repeatedly
until it reaches the goal or exhausts its step budget.  Maze state is
maintained locally inside ``run_agent`` — the AReno infrastructure only
sees the message trajectory and the final reward.
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
    "You are navigating a partially observable maze. "
    "You can only see cells around your current position. "
    "Move one step at a time by calling the move tool with a direction. "
    "Find the key to unlock doors, then reach the goal (G). "
    "The local view shows: @ = you, # = wall, . = empty, "
    "k = key, D = locked door, G = goal, ? = unseen."
)

MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "move",
        "description": "Move one step in the maze.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Direction to move.",
                }
            },
            "required": ["direction"],
            "additionalProperties": False,
        },
    },
}

TOOL_CHOICE = {"type": "function", "function": {"name": "move"}}


async def run_agent(ctx, batch):
    """Run multi-turn move requests for each maze task."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The maze agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
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
        turns = []
        state = game.deserialize_maze(item.record)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]

        while not state.steps_taken >= state.max_steps:
            assistant_message, turn = await _call_model(item, client, messages)
            turns.append(turn)
            tool_result = _run_tool(assistant_message, state)
            messages.extend(_tool_messages(assistant_message, tool_result))

            # Advance state.
            direction = _extract_direction(assistant_message)
            if direction is not None:
                result = game.apply_move(state, direction)
                state = result.state
                if result.terminal:
                    break
            else:
                # Invalid / missing tool call — waste a step but keep going.
                state = game.replace(state, steps_taken=state.steps_taken + 1)
                if state.steps_taken >= state.max_steps:
                    break

        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


async def _call_model(item, client, messages: list[dict]):
    """Call the policy model, returning the assistant message dict and a trajectory turn."""

    response = await client.chat.completions.create(
        model="policy",
        messages=messages,
        tools=[MOVE_TOOL],
        tool_choice=TOOL_CHOICE,
        stream=False,
    )
    message = response.choices[0].message
    tool_calls = [call for call in (message.tool_calls or []) if call.function.name == "move"][:1]
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
                "id": "missing_move",
                "type": "function",
                "function": {"name": "move", "arguments": "{}"},
            }
        ]
    return assistant_message, AgentTrajectoryTurn(
        item=item,
        messages=list(messages),
        response=response,
        tools=[MOVE_TOOL],
        tool_choice=TOOL_CHOICE,
    )


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    """Build the assistant and tool result messages to append."""

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


def _run_tool(assistant_message: dict, state: game.MazeState) -> dict:
    """Execute the move and return a result dict for the tool message."""

    direction = _extract_direction(assistant_message)
    if direction is None:
        return {"error": "missing or invalid direction", "observation": game.local_view(state)}

    result = game.apply_move(state, direction)
    return {
        "direction": direction,
        "success": result.success,
        "reason": result.reason,
        "terminal": result.terminal,
        "observation": result.observation,
        "steps_taken": result.state.steps_taken,
        "max_steps": result.state.max_steps,
        "has_key": result.state.has_key,
    }


def _extract_direction(assistant_message: dict) -> str | None:
    """Extract the direction from the assistant's tool call."""

    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return None
    call = calls[0]
    if call["function"]["name"] != "move":
        return None
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return None
    direction = args.get("direction")
    if direction in ("up", "down", "left", "right"):
        return direction
    return None
