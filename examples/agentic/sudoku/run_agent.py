"""Bounded multi-turn agent loop for Sudoku."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import SYSTEM_PROMPT, TOOLS, SudokuEpisode, inspect_candidates  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def run_agent(ctx, batch):
    """Run bounded concurrent Sudoku episodes and preserve exact model outputs."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Sudoku agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
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
    """Run one Sudoku puzzle as a multi-turn tool-call episode."""

    record = item.record
    episode = SudokuEpisode(record["puzzle"], max_actions=int(record.get("max_actions", 120)))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt},
    ]
    turns: list[AgentTrajectoryTurn] = []

    while not episode.is_done():
        turn_messages = [
            *messages,
            {"role": "user", "content": f"Action {episode.actions_taken + 1}/{episode.max_actions}: call one tool now."},
        ]
        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=TOOLS,
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=turn_messages,
                response=response,
                tools=TOOLS,
            )
        )
        assistant_message = _assistant_message(response)
        tool_result = _execute_tool(assistant_message, episode)
        if tool_result is None:
            logger.warning("Sudoku model returned no executable tool call")
            break
        messages.extend(_tool_messages(assistant_message, tool_result))

    return turns


def _assistant_message(response) -> dict:
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


def _execute_tool(assistant_message: dict, episode: SudokuEpisode) -> dict | None:
    """Execute the first tool call from the assistant message on the episode."""

    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return None
    call = calls[0]
    name = call.get("function", {}).get("name")
    raw_args = call.get("function", {}).get("arguments") or "{}"
    try:
        args = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return {"valid": False, "error": "invalid JSON arguments"}
    if not isinstance(args, dict):
        return {"valid": False, "error": "tool arguments must be a JSON object"}

    if name == "inspect_candidates":
        return game_inspect_candidates(episode, args)
    if name == "place_digit":
        return game_place_digit(episode, args)
    if name == "undo":
        return game_undo(episode, args)
    return {"valid": False, "error": f"unknown tool: {name}"}


def game_inspect_candidates(episode: SudokuEpisode, args: dict) -> dict:
    row = int(args.get("row", -1))
    col = int(args.get("col", -1))
    return inspect_candidates(episode.board, row, col)


def game_place_digit(episode: SudokuEpisode, args: dict) -> dict:
    row = int(args.get("row", -1))
    col = int(args.get("col", -1))
    digit = int(args.get("digit", 0))
    return episode.place(row, col, digit)


def game_undo(episode: SudokuEpisode, args: dict) -> dict:
    return episode.undo()


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    call = assistant_message["tool_calls"][0]
    return [
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": call["function"]["name"],
            "content": json.dumps(tool_result),
        },
    ]