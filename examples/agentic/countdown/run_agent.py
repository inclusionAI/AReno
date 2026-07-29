"""Bounded multi-turn agent loop for Countdown arithmetic game.

This module defines the agentic environment AReno trains against. Each
episode:

1. Loads a Countdown puzzle (numbers + target) from the dataset.
2. Sends the puzzle to the policy model with 5 available tools
   (add, subtract, multiply, divide, finish).
3. Lets the model call tools one at a time, up to ``max_steps`` times.
4. Records every (messages, response) pair as an ``AgentTrajectoryTurn`` so
   AReno can compute losses over the full trajectory.

The tools are executed locally in-process (no external services); the
``finish`` tool is what triggers reward computation by ending the episode.

Key design choices:
- Bounded concurrency: we cap the number of in-flight OpenAI requests by
  ``ctx.max_running_prompts`` so we don't saturate the serving backend.
- No stateful environment: each tool call is pure (a + b, etc.), and the
  model is responsible for tracking which numbers it has used. This keeps
  the env trivial to reproduce and debug.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
# httpx is chatty at INFO level; silence it so the training log is readable.
logging.getLogger("httpx").setLevel(logging.WARNING)

# The system prompt sets the game rules and tells the model how to interact
# with the tools. Small base models often need this to be explicit about
# "call one tool at a time" and "call finish when done".
SYSTEM_PROMPT = (
    "You are solving a Countdown arithmetic puzzle. Use the available numbers and basic operations "
    "(+, -, *, /) to reach the target number. Each number can only be used once. "
    "Think step by step, calling one tool at a time. When you have the final answer, call 'finish'."
)

# Tool definitions
ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers: a + b",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    }
}

SUBTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "subtract",
        "description": "Subtract two numbers: a - b",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    }
}

MULTIPLY_TOOL = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two numbers: a * b",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    }
}

DIVIDE_TOOL = {
    "type": "function",
    "function": {
        "name": "divide",
        "description": "Divide two numbers: a / b (only if b != 0 and result is an integer)",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Dividend"},
                "b": {"type": "number", "description": "Divisor (non-zero)"}
            },
            "required": ["a", "b"]
        }
    }
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Submit the final answer to the puzzle",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "number", "description": "The final calculated answer"},
                "reasoning": {"type": "string", "description": "Brief explanation of the solution steps"}
            },
            "required": ["answer"]
        }
    }
}

ALL_TOOLS = [ADD_TOOL, SUBTRACT_TOOL, MULTIPLY_TOOL, DIVIDE_TOOL, FINISH_TOOL]


async def run_agent(ctx, batch):
    """Run bounded concurrent Countdown episodes.

    This is the entry point AReno calls during rollout. We receive a batch
    of puzzle samples and run them concurrently against the policy model
    served at ``ctx.get_base_url()``.

    Args:
        ctx: AReno agent context. Provides the OpenAI-compatible base URL,
            API key, and ``max_running_prompts`` (concurrency cap).
        batch: Iterable of puzzle samples (each has ``prompt``, ``numbers``,
            ``target``, ``id``).

    Returns:
        An ``AgentTrajectory`` flattening all turns from all episodes in the
        batch. AReno uses this to compute per-token losses during training.
    """

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Countdown requires `openai` and `httpx`. Install them with `pip install openai httpx`."
        ) from exc

    items = list(batch.iter_samples())
    # Size the httpx connection pool to the concurrency cap so we never block
    # on connection acquisition during rollout.
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    try:
        # Fan out all episodes concurrently; each returns its own list of
        # turns, which we flatten into a single trajectory for AReno.
        grouped = await asyncio.gather(*(_run_episode(item, client) for item in items))
        return AgentTrajectory(turns=[turn for episode in grouped for turn in episode])
    finally:
        await client.close()


async def _run_episode(item, client) -> list[AgentTrajectoryTurn]:
    """Run a single Countdown episode: puzzle -> up to max_steps tool calls.

    The loop alternates between model generation and tool execution. Each
    iteration appends one ``AgentTrajectoryTurn`` so AReno can score the
    model's generation. The episode ends when:
    - the model calls ``finish`` (success), or
    - the model returns no valid tool call (we bail early), or
    - ``max_steps`` is reached (timeout).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt}
    ]
    turns = []
    max_steps = 20  # Maximum number of tool calls allowed per episode
    finished = False

    for step in range(1, max_steps + 1):
        # Re-inject a step counter as a user message so the model has a
        # sense of remaining budget. This helps small models decide to
        # call ``finish`` before hitting the cap.
        turn_messages = [
            *messages,
            {"role": "user", "content": f"Step {step} of {max_steps}: Call a tool to progress toward the target."}
        ]

        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=ALL_TOOLS,
            tool_choice="auto",
            stream=False,
        )

        # Record the (messages, response) pair so AReno can later compute
        # per-token losses over the model's generation.
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=turn_messages,
                response=response,
                tools=ALL_TOOLS,
                tool_choice="auto",
            )
        )

        assistant_message = _assistant_message(response)
        tool_result = _execute_tool(assistant_message)

        if tool_result is None:
            # The model didn't emit a valid tool call this turn. Rather than
            # inventing one, we stop the episode -- AReno will still score
            # whatever turns we already collected (typically with a 0 reward
            # via reward.py's "no finish call" branch).
            logger.warning("Countdown model returned no valid tool call")
            break

        messages.extend(_tool_messages(assistant_message, tool_result))

        # Check if finish was called
        if tool_result.get("name") == "finish":
            finished = True
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


def _execute_tool(assistant_message: dict) -> dict | None:
    """Execute the first tool call in the assistant message in-process.

    Countdown's tools are pure arithmetic, so we evaluate them directly
    rather than routing through an external service. This keeps the env
    deterministic and cheap.

    Returns:
        A dict describing the tool result. The shape is tool-specific:
        - arithmetic tools: ``{"name": ..., "result": <number>, "error": <str|None>, "arguments": {...}}``
        - finish: ``{"name": "finish", "answer": <number>, "reasoning": <str>}``
        - invalid tool call / bad JSON: ``{"name": ..., "error": ..., "result": None}``
        - no tool call at all: ``None`` (caller treats this as end-of-episode)
    """
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return None

    # Only the first tool call is honored; the model is instructed to call
    # one tool at a time in SYSTEM_PROMPT.
    call = calls[0]
    func = call.get("function", {})
    name = func.get("name")
    arguments_str = func.get("arguments", "{}")

    try:
        arguments = json.loads(arguments_str)
    except json.JSONDecodeError:
        return {"name": name, "error": "Invalid JSON arguments", "result": None}

    if not isinstance(arguments, dict):
        return {"name": name, "error": "Arguments must be an object", "result": None}

    # Execute the tool
    result = None
    error = None

    try:
        if name == "add":
            a = float(arguments.get("a", 0))
            b = float(arguments.get("b", 0))
            result = a + b
        elif name == "subtract":
            a = float(arguments.get("a", 0))
            b = float(arguments.get("b", 0))
            result = a - b
        elif name == "multiply":
            a = float(arguments.get("a", 0))
            b = float(arguments.get("b", 0))
            result = a * b
        elif name == "divide":
            a = float(arguments.get("a", 0))
            b = float(arguments.get("b", 0))
            if b == 0:
                error = "Cannot divide by zero"
            else:
                result = a / b
                # Only allow integer results for Countdown rules
                if result != int(result):
                    error = "Result must be an integer"
                else:
                    result = int(result)
        elif name == "finish":
            answer = arguments.get("answer")
            return {"name": name, "answer": answer, "reasoning": arguments.get("reasoning", "")}
        else:
            error = f"Unknown tool: {name}"
    except Exception as e:
        error = str(e)

    return {"name": name, "result": result, "error": error, "arguments": arguments}


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    call = assistant_message["tool_calls"][0]
    content = json.dumps(tool_result)
    return [
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": tool_result.get("name", "unknown"),
            "content": content,
        },
    ]
