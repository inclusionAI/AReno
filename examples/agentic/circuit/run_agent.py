"""Agent entrypoint for circuit-diagnosis tool-call rollouts (issue #193).

The agent interacts with a faulty logic circuit through two tools:
- ``probe``: Set input values and inspect a wire's output.
- ``submit``: Submit the guessed faulty gate ID.

The agent sees the circuit structure but not which gate is faulty.
"""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a digital circuit diagnosis expert. "
    "A logic circuit has one faulty gate (stuck-at-0 or stuck-at-1). "
    "Use the 'probe' tool to set input values and inspect wire outputs. "
    "Use the 'submit' tool to identify the faulty gate. "
    "Think step by step: compare expected vs observed outputs to narrow down the fault."
)

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "probe",
        "description": "Set circuit inputs and inspect a wire's output value.",
        "parameters": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "array",
                    "items": {"type": "boolean"},
                    "description": "Boolean values for each input gate, in order.",
                },
                "wire_id": {
                    "type": "integer",
                    "description": "The wire index to inspect (0 to num_gates-1).",
                },
            },
            "required": ["inputs", "wire_id"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit the gate ID you believe is faulty.",
        "parameters": {
            "type": "object",
            "properties": {
                "gate_id": {
                    "type": "integer",
                    "description": "The gate index that you believe is faulty.",
                }
            },
            "required": ["gate_id"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [PROBE_TOOL, SUBMIT_TOOL]


async def run_agent(ctx, batch):
    """Run one tool-call model request for each circuit diagnosis task."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The circuit-diagnosis agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info(
        "Circuit diagnosis agent start requests=%d max_running_prompts=%d",
        len(items),
        ctx.max_running_prompts,
    )
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
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=TOOLS,
            tool_choice="auto",
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()
