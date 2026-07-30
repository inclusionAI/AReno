"""Bounded multi-turn agent loop for elevator dispatch rollouts.

Each sample is one elevator episode. The agent calls move/open_door/close_door
each turn; the deterministic environment advances and the new state is fed back
as a tool result. The loop ends on a ``done`` tool, a terminal state, or the
horizon cap. No external services are used: the environment runs in-process.
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
    "You are an elevator dispatcher. On every turn call exactly one tool: "
    "move, open_door, close_door, or done. Move one floor at a time with the door closed. "
    "Open the door at a floor to alight riders whose destination is the current floor and "
    "to board waiting passengers up to capacity. Minimize waiting time, maximize delivered "
    "passengers, and never call a tool when the door is already in the requested state."
)


async def run_agent(ctx, batch):
    """Run bounded concurrent elevator episodes, preserving exact model outputs."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The elevator agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("Elevator agent start episodes=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    try:
        grouped = await asyncio.gather(*(_run_episode(item, client) for item in items))
        return AgentTrajectory(turns=[turn for episode in grouped for turn in episode])
    finally:
        await client.close()


async def _run_episode(item, client) -> list[AgentTrajectoryTurn]:
    """Drive one episode until done, terminal, or horizon exhaustion."""

    state = game.build_state(item.record)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{item.prompt}\n\nCurrent state: {game.format_state(state)}"},
    ]
    turns: list[AgentTrajectoryTurn] = []
    # hard cap slightly beyond horizon so a stuck agent still terminates
    max_steps = state.horizon + 2
    for _ in range(max_steps):
        if game.is_terminal(state):
            break
        turn_messages = [*messages, {"role": "user", "content": "Pick the next action by calling one tool."}]
        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=game.TOOLS,
            tool_choice="auto",
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=turn_messages,
                response=response,
                tools=game.TOOLS,
                tool_choice="auto",
            )
        )
        assistant_message = _assistant_message(response)
        action = game.parse_action(assistant_message)
        if action is None:
            # model produced no usable tool call; record the turn and stop the episode
            logger.warning("Elevator model returned no executable tool call; ending episode")
            break
        result = game.step(state, action)
        messages.extend(_tool_messages(assistant_message, result))
        if action["name"] == "done" or game.is_terminal(state):
            break
    return turns


def _assistant_message(response) -> dict:
    """Normalize the OpenAI response message into a serializable dict."""

    message = response.choices[0].message
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (message.tool_calls or [])
        ],
    }


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    """Append the assistant tool call and its environment result to the transcript."""

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
