"""Agent entrypoint for Hanoi multi-turn tool-call rollouts.

Mirrors ``examples/agentic/shopping/run_agent.py``. For each start board the
model makes one ``move_disk`` call per turn, the agent applies the move and
feeds the updated board back as a tool result, looping until solved or the move
budget is exhausted.  No trainer logic lives here — token rows, loss masks,
and tool-result masking are handled by ``RolloutSession`` / ``LossMaskPolicy``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator
import game

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are an expert at the Towers of Hanoi puzzle. "
    "Call move_disk ONCE per turn with a single {source, target} move. "
    "source and target are integers in {0,1,2}. "
    "Only move a top disk onto an empty peg or a larger disk. "
    "Win when all disks are stacked on peg 2 (largest at the bottom). "
    "Aim for the shortest solution (the optimum is 2**n - 1 moves)."
)

# Tool schema: a single (source, target) move.
MOVE_DISK_TOOL = {
    "type": "function",
    "function": {
        "name": "move_disk",
        "description": "Move one disk from source peg to target peg.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2,
                    "description": "Source peg (0, 1, or 2).",
                },
                "target": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2,
                    "description": "Target peg (0, 1, or 2).",
                },
            },
            "required": ["source", "target"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run multi-turn agentic rollouts for each Hanoi start board."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The Hanoi agentic example requires `openai`. Install it with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info(
        "Hanoi agent start requests=%d max_running_prompts=%d",
        len(items),
        ctx.max_running_prompts,
    )
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(
        base_url=ctx.get_base_url(),
        api_key=ctx.api_key,
        http_client=http_client,
        max_retries=0,
    )

    async def run_one(item):
        state = dataset_generator.record_to_state(item.record["state"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        turns: list[AgentTrajectoryTurn] = []
        tool_choice = {"type": "function", "function": {"name": "move_disk"}}

        for _turn_idx in range(state.max_moves):
            response = await client.chat.completions.create(
                model="policy",
                messages=messages,
                tools=[MOVE_DISK_TOOL],
                tool_choice=tool_choice,
                stream=False,
            )
            turn = AgentTrajectoryTurn(
                item=item,
                messages=list(messages),
                response=response,
                tools=[MOVE_DISK_TOOL],
                tool_choice=tool_choice,
            )
            turns.append(turn)

            message = response.choices[0].message
            raw_calls = message.tool_calls or []
            if not raw_calls:
                break
            call = raw_calls[0]
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                break
            source = args.get("source")
            target = args.get("target")
            if source is None or target is None or not isinstance(source, int) or not isinstance(target, int):
                break

            # Append the assistant message with tool call.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": "move_disk", "arguments": call.function.arguments},
                        }
                    ],
                }
            )

            # Apply the move and check terminal.
            state, _reward, done, _info = game.step(state, (source, target))

            if done:
                break

            # Feed back the current board state as tool result.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": "move_disk",
                    "content": json.dumps(
                        {
                            "pegs": [list(stack) for stack in state.pegs],
                            "legal_moves": [list(mv) for mv in game.legal_moves(state)],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )

        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()
