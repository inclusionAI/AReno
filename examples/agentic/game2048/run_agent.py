"""Bounded multi-turn agent loop for 2048."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import MOVE_TOOL, DEFAULT_MAX_MOVES, DIRECTIONS, SYSTEM_PROMPT  # noqa: E402
import game  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def run_agent(ctx, batch):
    """Run bounded concurrent 2048 episodes and preserve exact model outputs."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Game2048 requires `openai` and `httpx`. Install them with `pip install openai`."
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
    record = item.record
    board = game.normalize_board(record["board"])
    seed = int(record["seed"])
    max_moves = min(max(int(record.get("max_moves", DEFAULT_MAX_MOVES)), 1), DEFAULT_MAX_MOVES)
    rng = random.Random(seed)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt},
    ]
    turns: list[AgentTrajectoryTurn] = []
    tool_choice = {"type": "function", "function": {"name": "move"}}

    for move_number in range(1, max_moves + 1):
        turn_messages = [
            *messages,
            {"role": "user", "content": f"Step {move_number}/{max_moves}. Call the move tool."},
        ]
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
        tool_result = _execute_move(assistant_message, board, rng)
        if tool_result is None:
            logger.warning("Game2048 model returned no executable move call; penalising and falling back to random")
            tool_result = _invalid_move_result(board, rng)
            board = tool_result["board"]
            messages.append({
                "role": "user",
                "content": (
                    f"No valid move tool call detected. A random move ({tool_result['direction']}) "
                    f"was executed. Board:\n{tool_result['board_text']}"
                ),
            })
        else:
            board = tool_result["board"]
            messages.extend(_tool_messages(assistant_message, tool_result))

        if tool_result["terminal"]:
            finish_messages = [
                *messages,
                {"role": "user", "content": "Game over. Briefly summarize the result. Do not call any tool."},
            ]
            finish_response = await client.chat.completions.create(
                model="policy",
                messages=finish_messages,
                stream=False,
            )
            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=finish_messages,
                    response=finish_response,
                )
            )
            break

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


def _execute_move(assistant_message: dict, board, rng) -> dict | None:
    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "move":
        return None
    try:
        arguments = json.loads(calls[0]["function"].get("arguments") or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arguments, dict) or "direction" not in arguments:
        return None
    direction = str(arguments["direction"]).upper()
    if direction not in DIRECTIONS:
        return None

    new_board, score, valid, terminal = game.move(board, direction, rng)
    return {
        "board": new_board,
        "score": score,
        "valid": valid,
        "terminal": terminal,
        "direction": direction,
        "board_text": game.board_to_text(new_board),
    }


def _invalid_move_result(board, rng) -> dict:
    """Fallback: pick a random legal direction and mark the move as invalid."""
    direction = game.random_action(board, rng)
    new_board, score, valid, terminal = game.move(board, direction, rng)
    return {
        "board": new_board,
        "score": score,
        "valid": False,  # force invalid so reward_fn applies penalty
        "terminal": terminal,
        "direction": direction,
        "board_text": game.board_to_text(new_board),
    }


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    calls = assistant_message.get("tool_calls") or []
    content = json.dumps({
        "valid": tool_result["valid"],
        "score": tool_result["score"],
        "terminal": tool_result["terminal"],
        "board": tool_result["board_text"],
    })
    if not calls:
        return [
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": "fallback",
                "name": "move",
                "content": content,
            },
        ]
    call = calls[0]
    return [
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": "move",
            "content": content,
        },
    ]