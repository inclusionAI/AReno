"""Bounded multi-turn agent loop for Countdown arithmetic game."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

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
    """Run bounded concurrent Countdown episodes."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Countdown requires `openai` and `httpx`. Install them with `pip install openai httpx`."
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt}
    ]
    turns = []
    max_steps = 20  # Maximum number of tool calls allowed
    finished = False

    for step in range(1, max_steps + 1):
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
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return None

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
