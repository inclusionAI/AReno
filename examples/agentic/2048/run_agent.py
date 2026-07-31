"""Agent entrypoint for episode-based 2048 tool-call rollouts.

Each prompt is one 2048 starting board. The policy returns a bounded ``moves``
array in a single ``choose_moves`` tool call; the engine (in ``game.py``) replays
the whole episode deterministically at reward time.
"""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Per-sample strategy hints to break symmetry across n_samples.
# Each sample in a prompt group gets a different hint, so they explore
# different first moves → different scores → non-zero advantages.
EXPLORATION_HINTS = [
    "Try starting with a left move to consolidate the row.",
    "Try starting with a right move to push tiles rightward.",
    "Try starting with an up move to merge tiles upward.",
    "Try starting with a down move to bring tiles down.",
    "Try an aggressive strategy: make big merges early.",
    "Try a conservative strategy: keep the board tidy.",
    "Try focusing on keeping the largest tile in a corner.",
    "Try building chains: left, down, left, down.",
    "Always call choose_moves with at least one direction.",
    "Plan 3-5 moves ahead: look for tiles that can chain.",
    "Prioritize merging the largest tile with a neighbour.",
    "Try clearing one row at a time before shifting rows.",
    "Try alternating direction to bunch tiles together.",
    "Try moving away from the largest tile to open space.",
    "Try building toward one corner to keep the board tight.",
    "Watch for adjacent equal tiles and merge them first.",
]

SYSTEM_PROMPT = (
    "You are an expert 2048 player. "
    "Follow the output format described below."
)

CHOOSE_MOVES_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_moves",
        "description": "Choose the 2048 direction sequence to play from the current board.",
        "parameters": {
            "type": "object",
            "properties": {
                "moves": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "description": "Legal directions to play, in order.",
                    "items": {"type": "string", "enum": ["up", "down", "left", "right"]},
                }
            },
            "required": ["moves"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call model request for each 2048 board."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 2048 agentic example requires `openai`. Install it with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("2048 agent start requests=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    # Limit concurrent HTTP requests to avoid overwhelming the engine's
    # single-file command queue.  Without this, the asyncio.gather below
    # submits *all* requests at once; the proxy/engine serialises them, and
    # the worker refill logic can lose/defer responses, causing a permanent
    # hang.  8 is a safe default that keeps the GPU fed without saturation.
    max_concurrent = min(ctx.max_running_prompts, 8)
    max_connections = max(max_concurrent, ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(item):
        hint = EXPLORATION_HINTS[item.sample_index % len(EXPLORATION_HINTS)]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
            {"role": "assistant", "content": hint},
        ]
        tool_choice = {"type": "function", "function": {"name": "choose_moves"}}
        async with semaphore:
            response = await client.chat.completions.create(
                model="policy",
                messages=messages,
                tools=[CHOOSE_MOVES_TOOL],
                tool_choice=tool_choice,
                stream=False,
            )
        # Log model raw output for debugging collapse/format issues.
        choice = response.choices[0]
        raw_text = choice.message.content or ""
        tool_calls = choice.message.tool_calls or []
        finish = choice.finish_reason or "?"
        preview = raw_text[:120].replace("\n", "\\n")
        if tool_calls:
            args_preview = str(tool_calls[0].function.arguments)[:80]
            logger.info(
                "2048 sample prompt=%d sample=%d finish=%s tool=yes args=%s",
                item.prompt_index, item.sample_index, finish, args_preview,
            )
        else:
            logger.warning(
                "2048 sample prompt=%d sample=%d finish=%s tool=NO text=%s",
                item.prompt_index, item.sample_index, finish, preview,
            )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[CHOOSE_MOVES_TOOL],
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()