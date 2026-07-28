"""Bounded multi-turn agent loop for Codebreaker."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import GUESS_TOOL, score_guess  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a rigorous codebreaker. On every guessing turn call guess_code exactly once. "
    "Use prior tool clues and never repeat a guess. After the game ends, summarize the outcome without a tool call."
)


async def run_agent(ctx, batch):
    """Run bounded concurrent Codebreaker episodes and preserve exact model outputs."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Codebreaker requires `openai` and `httpx`. Install them with `pip install openai`."
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
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item.prompt}]
    turns = []
    max_guesses = min(max(int(item.record["max_guesses"]), 1), 6)
    for guess_number in range(1, max_guesses + 1):
        turn_messages = [
            *messages,
            {"role": "user", "content": f"Guess {guess_number} of {max_guesses}: call guess_code now."},
        ]
        tool_choice = {"type": "function", "function": {"name": "guess_code"}}
        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=[GUESS_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=turn_messages,
                response=response,
                tools=[GUESS_TOOL],
                tool_choice=tool_choice,
            )
        )
        assistant_message = _assistant_message(response)
        tool_result = _execute_guess(assistant_message, item.record)
        if tool_result is None:
            logger.warning("Codebreaker model returned no executable guess_code call")
            break
        messages.extend(_tool_messages(assistant_message, tool_result))
        game_over = tool_result.get("solved") or not tool_result.get("valid") or guess_number == max_guesses
        if game_over:
            finish_messages = [
                *messages,
                {"role": "user", "content": "The game is over. Briefly summarize the outcome without calling a tool."},
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


def _execute_guess(assistant_message: dict, record: dict) -> dict | None:
    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "guess_code":
        return None
    try:
        arguments = json.loads(calls[0]["function"].get("arguments") or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arguments, dict) or "code" not in arguments:
        return None
    return score_guess(record["secret"], arguments["code"])


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    call = assistant_message["tool_calls"][0]
    return [
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": "guess_code",
            "content": json.dumps(tool_result),
        },
    ]
