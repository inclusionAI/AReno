"""Agent entrypoint for multi-turn balance-scale tool-call rollouts."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game as game_module  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are solving a balance-scale odd-ball puzzle. "
    "You have a set of visually identical balls, one of which is heavier or lighter. "
    "Use the weigh tool to compare two equal-size disjoint groups of balls. "
    "When you have identified the odd ball, call the answer tool with its index "
    "and whether it is heavier or lighter. Choose your weighings carefully — "
    "you have a limited budget."
)

BALANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "weigh",
        "description": "Compare two disjoint equal-size groups of balls on a balance scale.",
        "parameters": {
            "type": "object",
            "properties": {
                "left_group": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Ball indices to place on the left side of the scale.",
                },
                "right_group": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Ball indices to place on the right side of the scale.",
                },
            },
            "required": ["left_group", "right_group"],
            "additionalProperties": False,
        },
    },
}

ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Submit the final answer: which ball is odd and whether it is heavier or lighter.",
        "parameters": {
            "type": "object",
            "properties": {
                "ball_index": {
                    "type": "integer",
                    "description": "The index of the odd ball.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["heavier", "lighter"],
                    "description": "Whether the odd ball is heavier or lighter.",
                },
            },
            "required": ["ball_index", "direction"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [BALANCE_TOOL, ANSWER_TOOL]
MAX_TURNS = 20  # Safety cap to prevent infinite loops.

# Max tokens per model call.  Weighing responses are short, but the model
# may produce reasoning text before the tool call.
MAX_NEW_TOKENS_PER_TURN = 256


async def run_agent(ctx, batch):
    """Run multi-turn balance-scale rollouts, one per puzzle.

    For each puzzle the agent loops: it sends the current message history to
    the policy model, receives a tool call (weigh or answer), executes the
    tool locally, appends the tool result to the message history, and repeats
    until the model calls ``answer`` or the weighing budget is exhausted.

    Each model invocation produces one ``AgentTrajectoryTurn``; the full
    multi-turn trajectory is returned as ``AgentTrajectory(turns=[...])``.
    """

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The balance-scale agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info(
        "Balance-scale agent start requests=%d max_running_prompts=%d",
        len(items),
        ctx.max_running_prompts,
    )
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        ),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(
        base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0
    )

    async def run_one(item):
        """Run one puzzle instance: weigh repeatedly, then answer."""

        record = item.record
        puzzle = game_module.BalanceGame(
            num_balls=int(record["num_balls"]),
            odd_ball_index=int(record["odd_ball_index"]),
            odd_ball_direction=record["odd_ball_direction"],
            max_weighings=int(record["max_weighings"]),
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]

        turns: list[AgentTrajectoryTurn] = []

        for turn_idx in range(MAX_TURNS):
            # On the last allowed weighing, tell the model it must answer now.
            if puzzle.weighings_remaining <= 1 and puzzle.weighings_remaining > 0:
                messages.append(
                    {
                        "role": "system",
                        "content": "You have 1 weighing remaining. After this weighing you must call the answer tool.",
                    }
                )

            if puzzle.weighings_remaining <= 0:
                messages.append(
                    {
                        "role": "system",
                        "content": "Your weighing budget is exhausted. You must call the answer tool now.",
                    }
                )

            response = await client.chat.completions.create(
                model="policy",
                messages=messages,
                tools=TOOLS,
                stream=False,
                max_tokens=MAX_NEW_TOKENS_PER_TURN,
            )

            turn = AgentTrajectoryTurn(
                item=item,
                messages=list(messages),
                response=response,
                tools=TOOLS,
            )
            turns.append(turn)

            tool_calls = turn.parsed_tool_calls
            if not tool_calls:
                # Model returned plain text without a tool call — stop.
                break

            # Process tool calls (expect one per turn for this puzzle).
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        fn_args = {}

                if fn_name == "answer":
                    # Game over — return all turns collected so far.
                    return turns

                if fn_name == "weigh":
                    try:
                        result = puzzle.weigh(
                            fn_args.get("left_group", []),
                            fn_args.get("right_group", []),
                        )
                        tool_content = result
                    except ValueError as exc:
                        tool_content = f"error: {exc}"
                    # Append the assistant's tool call and the tool response.
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [tc],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", "call_0"),
                            "name": "weigh",
                            "content": tool_content,
                        }
                    )

        return turns

    try:
        results = await asyncio.gather(*(run_one(item) for item in items))
        all_turns: list[AgentTrajectoryTurn] = []
        for turn_list in results:
            all_turns.extend(turn_list)
        return AgentTrajectory(turns=all_turns)
    finally:
        await client.close()
