"""Agent entrypoint for Wordle tool-call rollouts."""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# CRITICAL: Keep prompt SHORT and DIRECT.
# The system prompt must explicitly require tool calls.
SYSTEM_PROMPT = (
    "You are playing Wordle. You MUST call guess_word to submit your guess. "
    "Do not write any text. Only call the guess_word tool with a 5-letter word."
)

GUESS_WORD_TOOL = {
    "type": "function",
    "function": {
        "name": "guess_word",
        "description": "Guess a 5-letter word in Wordle.",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "pattern": "^[a-zA-Z]{5}$",
                    "description": "A valid 5-letter English word to guess.",
                }
            },
            "required": ["word"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call model request for each Wordle game."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The Wordle agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("Wordle agent start requests=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        # Use tool_choice="required" to force the model to always call a tool
        tool_choice = "required"
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=[GUESS_WORD_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[GUESS_WORD_TOOL],
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()