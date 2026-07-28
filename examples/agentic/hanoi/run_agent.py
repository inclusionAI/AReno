"""Bounded multi-turn agent loop for the Towers of Hanoi example."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    MOVE_TOOL,
    apply_move,
    default_max_moves,
    illegal_reason,
    initial_state,
    is_legal_move,
    is_terminal,
    state_to_text,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a careful Towers of Hanoi solver. On every turn call the move(source, target) "
    "tool exactly once with peg names A, B, or C. Only move the top disk of a peg and never "
    "place a larger disk on a smaller one. Use peg B as the auxiliary. After the puzzle is "
    "solved or an illegal move ends the episode, summarize the outcome without a tool call."
)


async def run_agent(ctx, batch):
    """Run bounded concurrent Hanoi episodes while preserving exact model outputs."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Towers of Hanoi requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
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
    n = int(item.record["n"])
    max_moves = int(item.record.get("max_moves", default_max_moves(n)))
    state = initial_state(n)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt},
    ]
    turns: list[AgentTrajectoryTurn] = []
    for step in range(1, max_moves + 1):
        turn_messages = [
            *messages,
            {
                "role": "user",
                "content": f"Move {step} of {max_moves}. Current state:\n{state_to_text(state)}\nCall move now.",
            },
        ]
        tool_choice = {"type": "function", "function": {"name": "move"}}
        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=[MOVE_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=turn_messages,
                response=response,
                tools=[MOVE_TOOL],
                tool_choice=tool_choice,
            )
        )
        assistant_message = _assistant_message(response)
        move = _parse_move(assistant_message)
        if move is None:
            logger.warning("Hanoi model returned no executable move call")
            break
        source, target = move
        if not is_legal_move(state, source, target):
            tool_result = {
                "ok": False,
                "completed": False,
                "move": [source, target],
                "error": illegal_reason(state, source, target),
                "state": state_to_text(state),
            }
            messages.extend(_tool_messages(assistant_message, tool_result))
            await _finish_episode(item, client, messages, turns)
            break
        state = apply_move(state, source, target)
        completed = is_terminal(state, n)
        tool_result = {
            "ok": True,
            "completed": completed,
            "move": [source, target],
            "state": state_to_text(state),
        }
        messages.extend(_tool_messages(assistant_message, tool_result))
        if completed or step == max_moves:
            await _finish_episode(item, client, messages, turns, completed=completed)
            break
    return turns


async def _finish_episode(item, client, messages, turns, *, completed: bool = False) -> None:
    """Append one final non-tool summary turn so the episode has a clean stop."""

    if completed:
        content = "All disks are on peg C. Briefly summarize the solution without calling a tool."
    else:
        content = "The episode is over. Briefly summarize the outcome without calling a tool."
    finish_messages = [*messages, {"role": "user", "content": content}]
    finish_response = await client.chat.completions.create(
        model="policy",
        messages=finish_messages,
        stream=False,
    )
    turns.append(AgentTrajectoryTurn(item=item, messages=finish_messages, response=finish_response))


def _assistant_message(response: dict) -> dict:
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


def _parse_move(message: dict) -> tuple[str, str] | None:
    """Return the single move arguments, or None if the call is not executable.

    A move is executable when the model emitted exactly one ``move`` call with a
    JSON ``source``/``target`` pair drawn from the supported pegs. Moves that are
    parseable but illegal (empty source, larger on smaller) still return here so
    the episode can record the rejection and terminate.
    """

    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "move":
        return None
    try:
        arguments = json.loads(calls[0]["function"].get("arguments") or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arguments, dict):
        return None
    source = arguments.get("source")
    target = arguments.get("target")
    if source not in ("A", "B", "C") or target not in ("A", "B", "C"):
        return None
    return source, target


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    call = assistant_message["tool_calls"][0]
    return [
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": "move",
            "content": json.dumps(tool_result, ensure_ascii=False),
        },
    ]
