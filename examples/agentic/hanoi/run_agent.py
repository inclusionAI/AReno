"""Agent entrypoint for Hanoi tool-call rollouts.

Mirrors ``examples/agentic/duelgrid/run_agent.py``. For each start board the
model calls ``move_disk`` with a full move sequence; the rollout session
converts each call into the trainer's token/logprob rows. No trainer logic
lives here — token rows, loss masks, and tool-result masking are handled by
``RolloutSession`` / ``LossMaskPolicy``.

Design choice — single-turn full solution:
    Each board is solved in ONE ``move_disk`` call carrying the whole move
    list, rather than a multi-turn "move → observe → move" loop. This matches
    DuelGrid's single-call shape, keeps the trajectory short (one policy span
    to score), and lets ``reward.py`` score the complete sequence at once via
    the rules engine. A multi-turn variant is a follow-up, not needed for the
    issue's acceptance criteria.
"""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are an expert at the Towers of Hanoi puzzle. "
    "Solve the given board by calling the move_disk tool with an ordered list "
    "of moves. Each move is {source, target} with source, target in {0,1,2}. "
    "The tool name is always move_disk; never use MOVE or another name as the tool. "
    "Only move a top disk onto an empty peg or a larger disk. "
    "Aim for the shortest solution (the optimum is 2**n - 1 moves)."
)

# Tool schema: each move is a fixed [source, target] integer pair. We use
# JSON Schema tuple form (prefixItems + items:false, draft 2020-12) to enforce
# "exactly two ints". Some OpenAI-compatible servers ignore the tuple form and
# only validate the outer array — that is fine: the rules engine re-checks
# every move's legality in reward.py, so schema validation is convenience, not
# a correctness boundary.
MOVE_DISK_TOOL = {
    "type": "function",
    "function": {
        "name": "move_disk",
        "description": "Submit an ordered move sequence to solve the Towers of Hanoi board.",
        "parameters": {
            "type": "object",
            "properties": {
                "moves": {
                    "type": "array",
                    "minItems": 1,
                    "description": "Ordered (source, target) moves to execute, e.g. [[0,2],[0,1]].",
                    "items": {
                        "type": "array",
                        "prefixItems": [{"type": "integer"}, {"type": "integer"}],
                        "minItems": 2,
                        "maxItems": 2,
                        "items": False,
                    },
                },
            },
            "required": ["moves"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call model request for each Hanoi start board."""

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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        # Force the move_disk tool so the model cannot ramble in natural language
        # (a weak base model tends to; reward_fn would then fall back to parsing
        # completion text and usually score 0). tool_choice is honored by the
        # OpenAI-compatible proxy backing the in-training policy.
        tool_choice = {"type": "function", "function": {"name": "move_disk"}}
        response = await client.chat.completions.create(
            model="policy",  # routes to the in-training policy via ctx.get_base_url()
            messages=messages,
            tools=[MOVE_DISK_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[MOVE_DISK_TOOL],
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()
