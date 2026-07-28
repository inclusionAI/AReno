"""Agent entrypoint for multi-turn partially-observable maze rollouts.

Each step the model sees only a local view of the maze and must call the
``act`` tool with a single action (``move``, ``pickup``, or ``use_key``).
The agent loops until the goal is reached or ``max_steps`` is exhausted.
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
    "You are navigating a partially observable maze. You can only see tiles "
    "immediately around you. Reach the goal G by moving through open floor, "
    "picking up keys K, and opening doors D with held keys. "
    "Each turn call the act tool with exactly one action."
)

ACT_TOOL = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "Perform one maze action: move, pickup, or use_key.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["move", "pickup", "use_key"],
                    "description": "The action to perform.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["UP", "DOWN", "LEFT", "RIGHT"],
                    "description": "Direction for move or use_key. Omit for pickup.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}

DEFAULT_MAX_TURNS = 30


async def run_agent(ctx, batch):
    """Run multi-turn tool-call maze episodes for each item."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The maze agentic example requires `openai`. Install it with `pip install openai`."
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
        record = item.record.get("state", item.record)
        state = game.make_state_from_record(record)
        radius = record.get("view_radius", 1)
        max_turns = min(record.get("max_steps", DEFAULT_MAX_TURNS), DEFAULT_MAX_TURNS)
        turns: list[AgentTrajectoryTurn] = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": game.format_prompt(state, radius)},
        ]

        for _ in range(max_turns):
            if state.done:
                break
            tool_choice = {"type": "function", "function": {"name": "act"}}
            try:
                response = await client.chat.completions.create(
                    model="policy",
                    messages=messages,
                    tools=[ACT_TOOL],
                    tool_choice=tool_choice,
                    stream=False,
                )
            except Exception as exc:
                logger.warning("Maze agent rollout request failed, ending episode early: %s", exc)
                break
            message = response.choices[0].message
            tool_calls = [call for call in (message.tool_calls or []) if call.function.name == "act"][:1]
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
                        "id": "missing_act",
                        "type": "function",
                        "function": {"name": "act", "arguments": "{}"},
                    }
                ]

            turn = AgentTrajectoryTurn(
                item=item,
                messages=list(messages),
                response=response,
                tools=[ACT_TOOL],
                tool_choice=tool_choice,
            )
            turns.append(turn)

            # Execute the tool call locally
            action = _parse_tool_action(tool_calls)
            next_state, _reward, done, info = game.step(state, action)
            feedback = _format_feedback(info)
            tool_result = {"state": "ok", "feedback": feedback, "done": done}
            if done:
                tool_result["result"] = "reached_goal" if state.reached_goal or next_state.reached_goal else "timeout"

            messages.append(assistant_message)
            messages.append({
                "role": "tool",
                "tool_call_id": assistant_message["tool_calls"][0]["id"],
                "name": "act",
                "content": json.dumps(tool_result, ensure_ascii=False),
            })
            if done:
                break
            # Add follow-up user message with new view
            messages.append({
                "role": "user",
                "content": game.format_step_prompt(next_state, radius, feedback),
            })
            state = next_state

        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


def _parse_tool_action(tool_calls: list) -> dict[str, str] | None:
    """Extract an action dict from the first act tool call."""

    if not tool_calls:
        return None
    call = tool_calls[0]
    try:
        args = json.loads(call.function.arguments or "{}")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(args, dict):
        return None
    return {"action": str(args.get("action", "")), "direction": str(args.get("direction", ""))}


def _format_feedback(info: dict) -> str:
    """Turn step info into a short feedback string for the model."""

    if info.get("goal"):
        return "You reached the goal!"
    if info.get("timeout"):
        return "Out of steps."
    if info.get("illegal"):
        return f"Invalid move: {info.get('reason', 'unknown')}"
    if info.get("picked_up"):
        return f"Picked up key {info['picked_up']}."
    if info.get("opened_door"):
        return f"Opened door ({info.get('direction', '')})."
    if info.get("moved"):
        return f"Moved {info.get('direction', '')}."
    return ""
