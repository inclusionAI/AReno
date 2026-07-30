"""Agent entrypoint for multi-turn odd-ball balance-scale rollouts.

The agent loops for at most ``max_weighings + 1`` turns. Each turn the model
chooses between the ``weigh`` tool (compare two equal-size disjoint ball
groups) and the ``submit_answer`` tool (final answer with ball index and
direction). When the weighing budget is exhausted the agent is forced to
submit an answer.

The orchestration logic is split into :func:`_run_puzzle_loop` (pure Python,
no torch/openai dependency) and :func:`run_agent` (async entrypoint that
wires the loop to an OpenAI-compatible rollout proxy).  This separation
allows the multi-turn tool-call flow, budget enforcement, and tool dispatch
to be tested on CPU without GPU infrastructure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = game.format_system_prompt()

WEIGH_TOOL = {
    "type": "function",
    "function": {
        "name": "weigh",
        "description": "Compare two equal-size disjoint groups of balls on a balance scale.",
        "parameters": {
            "type": "object",
            "properties": {
                "left": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "description": "Ball indices on the left side of the scale.",
                },
                "right": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "description": "Ball indices on the right side of the scale.",
                },
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the final answer: which ball is odd and whether it is heavier or lighter.",
        "parameters": {
            "type": "object",
            "properties": {
                "ball_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "The index of the odd ball.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["heavier", "lighter"],
                    "description": "Whether the odd ball is heavier or lighter than the rest.",
                },
            },
            "required": ["ball_index", "direction"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [WEIGH_TOOL, SUBMIT_ANSWER_TOOL]
TOOL_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS}

# Type alias for the model-call callback used by _run_puzzle_loop.
# The callback receives (messages, tools, tool_choice) and returns a dict
# with keys: content (str|None), tool_calls (list[dict] in the normalized
# {"id", "type", "function": {"name", "arguments"}} format).
ModelCallback = Callable[[list[dict], list[dict], object], Awaitable[dict]]


async def _call_model(client, messages, tools, tool_choice):
    """Send a chat completion request and normalise the response.

    When the proxy does not parse tool_calls from the model's text output
    (common with untrained models on AReno's native rollout), fall back to
    extracting tool calls from the response content text and inject them
    into the response object so AgentTrajectoryTurn can pick them up.

    Returns a dict with keys: response, content, tool_calls.
    """

    import uuid

    response = await client.chat.completions.create(
        model="policy",
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        stream=False,
    )
    message = response.choices[0].message
    tool_calls_raw = message.tool_calls or []
    tool_calls = [
        {
            "id": call.id,
            "type": call.type,
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in tool_calls_raw
    ]

    # Fallback: if proxy returned no tool_calls, parse from content text
    if not tool_calls:
        if message.content:
            parsed = _parse_tool_call_from_text(message.content, tool_choice)
        else:
            parsed = None
        if parsed:
            parsed_call_id = f"parsed_{uuid.uuid4().hex[:8]}"
            tool_calls = [{
                "id": parsed_call_id,
                "type": "function",
                "function": {
                    "name": parsed["name"],
                    "arguments": parsed["arguments"],
                },
            }]
            # Inject into the response object so AgentTrajectoryTurn
            # __post_init__ can parse tool_calls from it.
            _inject_tool_calls_into_response(response, parsed_call_id, parsed["name"], parsed["arguments"])

    return {
        "response": response,
        "content": message.content,
        "tool_calls": tool_calls,
    }


async def run_agent(ctx, batch):
    """Run multi-turn weigh/submit_answer rollouts for each puzzle."""

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
        "Balance-scale agent start tasks=%d max_running_prompts=%d",
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
        record = item.record
        ball_set = game.BallSet(
            num_balls=record["num_balls"],
            odd_ball_index=record["odd_ball_index"],
            direction=record["direction"],
            max_weighings=record["max_weighings"],
        )
        turns, _messages = await _run_puzzle_loop(
            item, ball_set,
            lambda msgs, tls, tc: _call_model(client, msgs, tls, tc),
        )
        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


async def _run_puzzle_loop(
    item,
    ball_set: game.BallSet,
    call_model: ModelCallback,
) -> tuple[list, list[dict]]:
    """Execute the multi-turn weigh/submit_answer loop for one puzzle.

    This is the pure-orchestration core extracted from :func:`run_agent`.
    It accepts a ``call_model`` callback so that the loop logic — budget
    enforcement, tool dispatch, message accumulation — can be tested
    without GPU or network dependencies.

    Returns ``(turns, messages)`` where ``turns`` is a list of
    ``AgentTrajectoryTurn`` and ``messages`` is the full conversation
    history.
    """

    turns: list[AgentTrajectoryTurn] = []
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt},
    ]
    weighings_used = 0

    # Allow max_weighings weighings + extra turns for invalid weighings
    # that don't consume budget, plus 1 forced submit turn.
    max_turns = ball_set.max_weighings * 2 + 1
    for _turn_idx in range(max_turns):
        if weighings_used >= ball_set.max_weighings:
            # Budget exhausted: force submit_answer
            tools = [SUBMIT_ANSWER_TOOL]
            tool_choice: object = {"type": "function", "function": {"name": "submit_answer"}}
            hint = (
                f"You have used all {ball_set.max_weighings} weighings. "
                "You must now call submit_answer with your best guess."
            )
        elif weighings_used == 0 and _turn_idx == 0:
            # First turn: force weigh to bootstrap tool usage
            tools = [WEIGH_TOOL]
            tool_choice = {"type": "function", "function": {"name": "weigh"}}
            hint = None
        else:
            # Subsequent turns: allow either weigh or submit_answer
            tools = TOOLS
            tool_choice = "auto"
            hint = None

        turn_messages = [*messages]
        if hint:
            turn_messages.append({"role": "user", "content": hint})

        model_output = await call_model(turn_messages, tools, tool_choice)

        tool_calls = model_output.get("tool_calls", [])
        response = model_output.get("response")
        content = model_output.get("content")

        turn = AgentTrajectoryTurn(
            item=item,
            messages=turn_messages,
            response=response,
            tools=tools,
            tool_choice=tool_choice,
        )

        # If fallback parsing found tool_calls but the framework didn't
        # (because response object is immutable), override parsed_tool_calls
        if not turn.parsed_tool_calls and tool_calls:
            turn.parsed_tool_calls = list(tool_calls)

        turns.append(turn)

        if not tool_calls:
            break

        call = tool_calls[0]
        name = call["function"]["name"]
        args_str = call["function"]["arguments"] or "{}"

        assistant_message = {
            "role": "assistant",
            "content": content,
            "tool_calls": [call],
        }
        messages.append(assistant_message)

        if name == "submit_answer":
            tool_result = _run_submit_answer(args_str)
            messages.append(_tool_result_message(call, tool_result))
            break
        elif name == "weigh":
            tool_result, did_weigh = _run_weigh(args_str, ball_set, weighings_used)
            messages.append(_tool_result_message(call, tool_result))
            if did_weigh:
                weighings_used += 1
        else:
            tool_result = {"error": f"unknown tool: {name}"}
            messages.append(_tool_result_message(call, tool_result))

    return turns, messages


def _run_weigh(args_str: str, ball_set: game.BallSet, weighings_used: int) -> tuple[dict, bool]:
    """Execute a weigh tool call. Returns (result_dict, did_weigh_succeed)."""

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return {"error": "invalid JSON arguments"}, False

    left = args.get("left")
    right = args.get("right")
    if not isinstance(left, list) or not isinstance(right, list):
        return {"error": "left and right must be lists of ball indices"}, False

    try:
        result = game.weigh(ball_set, left, right, weighings_used=weighings_used)
    except ValueError as exc:
        return {"error": str(exc)}, False

    return {"result": result, "weighings_used": weighings_used + 1}, True


def _run_submit_answer(args_str: str) -> dict:
    """Execute a submit_answer tool call."""

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return {"error": "invalid JSON arguments"}

    ball_index = args.get("ball_index")
    direction = args.get("direction")
    if ball_index is None or direction is None:
        return {"error": "ball_index and direction are required"}

    try:
        ball_index = int(ball_index)
    except (TypeError, ValueError):
        return {"error": f"ball_index must be an integer, got {ball_index!r}"}

    if direction not in game.DIRECTIONS:
        return {"error": f"direction must be one of {game.DIRECTIONS}, got {direction!r}"}

    return {"submitted": True, "ball_index": ball_index, "direction": direction}


def _tool_result_message(call: dict, result: dict) -> dict:
    """Build a tool-role message carrying the result of a tool call."""

    return {
        "role": "tool",
        "tool_call_id": call["id"],
        "name": call["function"]["name"],
        "content": json.dumps(result, ensure_ascii=False),
    }


def _inject_tool_calls_into_response(response: object, call_id: str, name: str, arguments: str) -> None:
    """Inject parsed tool_calls into an OpenAI response object.

    AReno's AgentTrajectoryTurn.__post_init__ calls _chat_response_message_tool_calls
    which reads response.choices[0].message.tool_calls. When the proxy doesn't
    populate this field, we inject our parsed tool call so the framework can
    track it in tool_calls stats and reward records.

    This is best-effort: if the response object is immutable (e.g. OpenAI SDK's
    pydantic ChatCompletionMessage), injection silently fails and we fall back
    to overriding ``turn.parsed_tool_calls`` directly in the caller.
    """

    try:
        choices = _response_get(response, "choices")
        if not isinstance(choices, list) or not choices:
            return
        message = _response_get(choices[0], "message")
        if message is None:
            return

        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

        if isinstance(message, dict):
            message["tool_calls"] = [tool_call]
        else:
            # OpenAI SDK returns a pydantic model (ChatCompletionMessage) which
            # may be frozen/immutable. setattr will raise AttributeError or
            # TypeError — that's expected, the caller handles this by setting
            # turn.parsed_tool_calls directly.
            try:
                message.tool_calls = [tool_call]
            except (AttributeError, TypeError):
                pass
    except Exception:
        # Last-resort: if any unexpected error occurs, the fallback tool_calls
        # in the return dict of _call_model still work for _run_puzzle_loop.
        pass


def _response_get(obj: object, key: str):
    """Helper: get attribute or dict key from a response-like object."""

    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _parse_tool_call_from_text(content: str, tool_choice: object) -> dict | None:
    """Fallback parser: extract a tool call from model text output.

    AReno's rollout proxy may not parse structured tool_calls from the model's
    text. This function scans the generated text for JSON-like patterns that
    resemble weigh or submit_answer arguments.

    Patterns are tried in priority order:
      1. Explicit {"name": "...", "arguments": {...}} — most precise.
      2. Weigh args {"left": [...], "right": [...]} — common JSON shorthand.
      3. Submit args {"ball_index": N, "direction": "..."} — common JSON shorthand.
      4. Forced weigh: extract any two equal-length arrays (last resort).
      5. Forced submit: parse natural language "ball N is heavier/lighter".

    Returns ``{"name": str, "arguments": str(json)}`` or ``None``.
    """

    import re

    # Determine expected tool name from tool_choice if forced
    expected_name = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        expected_name = fn.get("name")

    # Pattern 1: explicit tool call with name + arguments
    # e.g. {"name": "weigh", "arguments": {"left": [0, 1], "right": [2, 3]}}
    m = re.search(
        r'"name"\s*:\s*"(weigh|submit_answer)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})',
        content,
    )
    if m:
        return {"name": m.group(1), "arguments": m.group(2)}

    # Pattern 2: weigh with left/right arrays
    # e.g. {"left": [0, 1], "right": [2, 3]}
    # or left: [0, 1], right: [2, 3]
    m = re.search(
        r'"?left"?\s*:\s*\[([0-9,\s]+)\]\s*,\s*"?right"?\s*:\s*\[([0-9,\s]+)\]',
        content,
    )
    if m:
        left = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
        right = [int(x.strip()) for x in m.group(2).split(",") if x.strip().isdigit()]
        if left and right:
            return {
                "name": "weigh",
                "arguments": json.dumps({"left": left, "right": right}),
            }

    # Pattern 3: submit_answer with ball_index + direction
    # e.g. {"ball_index": 5, "direction": "heavier"}
    # or ball_index: 5, direction: "heavier"
    # also matches when wrapped in other text
    m = re.search(
        r'"?ball_index"?\s*:\s*(\d+)[^}]*?"?direction"?\s*:\s*"?(\w+)"?',
        content,
    )
    if m:
        direction = m.group(2).lower()
        if "heav" in direction:
            direction = "heavier"
        elif "light" in direction:
            direction = "lighter"
        return {
            "name": "submit_answer",
            "arguments": json.dumps({
                "ball_index": int(m.group(1)),
                "direction": direction,
            }),
        }

    # Pattern 4: if tool_choice forces a specific tool, try to extract any JSON
    if expected_name == "weigh":
        # Look for any array-like patterns that could be left/right
        arrays = re.findall(r'\[([0-9,\s]+)\]', content)
        if len(arrays) >= 2:
            left = [int(x.strip()) for x in arrays[0].split(",") if x.strip().isdigit()]
            right = [int(x.strip()) for x in arrays[1].split(",") if x.strip().isdigit()]
            if left and right and len(left) == len(right):
                return {
                    "name": "weigh",
                    "arguments": json.dumps({"left": left, "right": right}),
                }

    # Pattern 5: if tool_choice forces submit_answer, try to extract ball + direction
    # from natural language like "ball 5 is heavier" or "the odd ball is 3, lighter"
    if expected_name == "submit_answer":
        # Look for "ball N" pattern
        m = re.search(r'ball\s*(?:index\s*)?(?:is\s*)?(\d+)', content, re.IGNORECASE)
        if m:
            ball_idx = int(m.group(1))
            # Determine direction from text
            direction = None
            if re.search(r'heavier|heavy', content, re.IGNORECASE):
                direction = "heavier"
            elif re.search(r'lighter|light', content, re.IGNORECASE):
                direction = "lighter"
            if direction:
                return {
                    "name": "submit_answer",
                    "arguments": json.dumps({"ball_index": ball_idx, "direction": direction}),
                }

    return None
