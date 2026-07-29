"""Agent entrypoint for water-jug tool-call rollouts.

This module implements ``run_agent(ctx, batch)``, the function AReno calls
during Agentic GSPO rollout. For each puzzle in the batch, the agent:

1. Sends the puzzle prompt to the model with a ``water_jug_action`` tool.
2. Parses the model's tool call, applies the action to the game state.
3. Feeds the new state back to the model as a tool response.
4. Repeats until the puzzle is solved or ``MAX_TURNS`` is reached.

Each model call produces one ``AgentTrajectoryTurn``. AReno internally
concatenates multiple turns into a single training trajectory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a puzzle-solving agent. You solve water-jug puzzles by calling the "
    "`water_jug_action` tool one step at a time. Think carefully about which action "
    "brings you closer to the target. After each tool call you will see the new state. "
    "Stop calling tools once a jug contains the target amount."
)

WATER_JUG_TOOL = {
    "type": "function",
    "function": {
        "name": "water_jug_action",
        "description": "Perform one action on the water jugs. Available actions: fill(i), empty(i), pour(i,j).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "One of: fill(i), empty(i), pour(i,j) where i,j are jug indices.",
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}

MAX_TURNS = 10


async def run_agent(ctx, batch):
    try:
        import httpx
        from openai import AsyncOpenAI
        from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn
    except ImportError as exc:
        raise RuntimeError(
            "The water-jug agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai httpx`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("Water-jug agent start requests=%d max_running_prompts=%d",
                len(items), ctx.max_running_prompts)

    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections,
                            max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(
        base_url=ctx.get_base_url(),
        api_key=ctx.api_key,
        http_client=http_client,
        max_retries=0,
    )

    async def run_one(item):
        record = item.record
        image = record.get("image", record)
        caps = tuple(image.get("capacities", (3, 5)))
        target = int(image.get("target", 4))
        initial = tuple(image.get("initial_state", [0] * len(caps)))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        state = initial
        turns = []

        for _step in range(MAX_TURNS):
            if game.is_goal(state, target):
                break
            try:
                response = await client.chat.completions.create(
                    model="policy",
                    messages=messages,
                    tools=[WATER_JUG_TOOL],
                    tool_choice="auto",
                    stream=False,
                )
            except Exception as exc:
                logger.warning("Model request failed: %s", exc)
                break

            turns.append(AgentTrajectoryTurn(
                item=item,
                messages=list(messages),
                response=response,
                tools=[WATER_JUG_TOOL],
                tool_choice="auto",
            ))

            choice = response.choices[0]
            msg = choice.message
            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                if fn_name != "water_jug_action":
                    continue
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                action = args.get("action", "")
                try:
                    state = game.apply_action(caps, state, action)
                except Exception as e:
                    board = f"Invalid action. Error: {e}"
                else:
                    board = game.format_board(caps, state, target)

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [{
                        "id": tc.id, "type": "function",
                        "function": {"name": fn_name, "arguments": tc.function.arguments},
                    }],
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "name": fn_name, "content": board,
                })

        if not turns:
            turns.append(AgentTrajectoryTurn(
                item=item, messages=messages, response=None,
                tools=[WATER_JUG_TOOL], tool_choice="auto",
            ))
        return turns

    try:
        results = await asyncio.gather(*(run_one(item) for item in items))
        all_turns = []
        for turn_list in results:
            all_turns.extend(turn_list)
        return AgentTrajectory(turns=all_turns)
    finally:
        await http_client.aclose()