"""Single-turn tool-call agent loop for 2048.

Each step is an independent LLM call: the model sees only the system prompt
and the current board. No conversation history is accumulated. The model
calls the move tool with a direction, which is parsed from the tool call.
"""

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

    turns: list[AgentTrajectoryTurn] = []
    tool_choice = {"type": "function", "function": {"name": "move"}}

    for move_number in range(1, max_moves + 1):
        user_content = (
            f"Step {move_number}/{max_moves}.\n"
            f"Current board:\n{game.board_to_text(board)}\n"
            f"Call the move tool with the best direction."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=[MOVE_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=messages,
                response=response,
                tools=[MOVE_TOOL],
                tool_choice=tool_choice,
            )
        )

        direction = _extract_direction(response)
        if direction is None:
            logger.warning("Game2048 no valid move tool call, falling back to random")
            direction = game.random_action(board, rng)

        new_board, score, valid, terminal = game.move(board, direction, rng)
        board = new_board

        if terminal:
            break

    return turns


def _extract_direction(response) -> str | None:
    """Parse direction from the model's tool call response."""

    choice = response.choices[0]
    message = choice.message
    tool_calls = getattr(message, "tool_calls", None) or []
    content = getattr(message, "content", None)
    finish_reason = getattr(choice, "finish_reason", None)
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    logger.warning(
        "Game2048 DEBUG: finish_reason=%s completion_tokens=%s content=%r tool_calls=%r",
        finish_reason,
        completion_tokens,
        content,
        tool_calls,
    )
    if not tool_calls:
        return None
    call = tool_calls[0]
    function = getattr(call, "function", None)
    if function is None or getattr(function, "name", None) != "move":
        return None
    try:
        arguments = json.loads(function.arguments or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arguments, dict) or "direction" not in arguments:
        return None
    direction = str(arguments["direction"]).upper()
    if direction not in DIRECTIONS:
        return None
    return direction