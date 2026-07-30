"""Agent entrypoint for multi-turn balance-scale XML rollouts.

This variant does not use OpenAI tool calls.  Instead, the model outputs plain
text containing XML tags such as ``<weigh left="0,1" right="2,3"/>`` and
``<answer ball="3" direction="heavier"/>``.  The agent loop parses these tags,
executes the weighing locally, and appends the result as a user message.
"""

from __future__ import annotations

import asyncio
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
    "Output a weigh tag to compare two equal-size disjoint groups of balls. "
    "Output an answer tag when you have identified the odd ball. "
    "Choose your weighings carefully — you have a limited budget."
)

MAX_TURNS = 20
MAX_NEW_TOKENS_PER_TURN = 256


async def run_agent(ctx, batch):
    """Run multi-turn balance-scale XML rollouts, one per puzzle."""

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
        "Balance-scale XML agent start requests=%d max_running_prompts=%d",
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
        """Run one puzzle instance using XML tags instead of tool calls."""

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
            # Nudge the model when the budget is running low.
            if puzzle.weighings_remaining <= 1 and puzzle.weighings_remaining > 0:
                messages.append(
                    {
                        "role": "user",
                        "content": "You have 1 weighing remaining. Use it wisely, then answer.",
                    }
                )

            if puzzle.weighings_remaining <= 0:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your weighing budget is exhausted. You must output an answer tag now.",
                    }
                )

            response = await client.chat.completions.create(
                model="policy",
                messages=messages,
                stream=False,
                max_tokens=MAX_NEW_TOKENS_PER_TURN,
            )

            turn = AgentTrajectoryTurn(
                item=item,
                messages=list(messages),
                response=response,
            )
            turns.append(turn)

            # Extract the model's text from the response.
            text = _response_text(response)

            # Check for an answer tag first — if found, we are done.
            answer = game_module.parse_xml_answer(text)
            if answer is not None:
                return turns

            # Check for a weigh tag.
            weigh = game_module.parse_xml_weigh(text)
            if weigh is None:
                # No actionable tag — stop to avoid infinite loops.
                break

            left_group, right_group = weigh
            try:
                result = puzzle.weigh(left_group, right_group)
            except ValueError as exc:
                result = f"error: {exc}"

            # Feed the weighing result back to the model as a user message.
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Result: {result}",
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


def _response_text(response) -> str:
    """Extract the assistant text from an OpenAI-compatible response."""

    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""