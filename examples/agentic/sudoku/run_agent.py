"""Multi-turn Sudoku agent entrypoint for AReno agentic rollouts.

Unlike the single-step Tic-Tac-Toe example, Sudoku is multi-turn: the policy
calls ``inspect_candidates`` / ``place_digit`` / ``undo`` repeatedly until the
board is solved or the action budget is exhausted. The loop mirrors the
coding-agent loop (lockstep turns so rollout requests batch together) but the
"environment" is an in-memory :class:`SudokuEnv` instead of a filesystem
workspace. Tool results are JSON-serialized into ``role: tool`` messages.

The solution is never placed in any message; only the visible board and
constraint-based feedback are returned to the policy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from areno.agent.agent_loop import create_chat_completion_with_retry  # noqa: E402
from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn  # noqa: E402

import sudoku  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_MAX_TURNS = 30

SYSTEM_PROMPT = """You are solving a Sudoku puzzle by calling tools. One tool call per turn.

Tools:
- inspect_candidates(row, col): list legal digits for an empty cell. row/col are 1-based (1..9).
- place_digit(row, col, digit): put a digit (1..9) into an empty cell. row/col are 1-based.
- undo(): revert your most recent placement.

Strategy — ALWAYS BE PLACING:
- Pick an empty cell, call inspect_candidates on it, then IMMEDIATELY call
  place_digit with one of the returned candidates. Do not just inspect cells
  repeatedly without placing — you make progress only by placing digits.
- Best order: find a cell whose candidates list has length 1 (a forced move)
  and place that digit first; then handle cells with 2-3 candidates.
- If a placement is rejected (digit conflicts), it is not fatal — pick a
  different candidate or a different cell and place again. Keep placing.
- The board is uniquely solvable. Solve it before the action budget runs out.

Example turn (you output exactly one tool call):
  place_digit(row=1, col=5, digit=4)   # places 4 into row 1, col 5

Remember: every turn must place, inspect-then-place, or undo. Never produce
text without a tool call."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_candidates",
            "description": "List the legal candidate digits for an empty cell (1-based row/col).",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer", "minimum": 1, "maximum": 9},
                    "col": {"type": "integer", "minimum": 1, "maximum": 9},
                },
                "required": ["row", "col"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_digit",
            "description": "Place a digit (1..9) into an empty cell (1-based row/col).",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer", "minimum": 1, "maximum": 9},
                    "col": {"type": "integer", "minimum": 1, "maximum": 9},
                    "digit": {"type": "integer", "minimum": 1, "maximum": 9},
                },
                "required": ["row", "col", "digit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo",
            "description": "Undo the most recent successful placement.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


async def run_agent(ctx, batch) -> AgentTrajectory:
    """Run the Sudoku agent loop for every expanded prompt/sample item."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The Sudoku agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("Sudoku agent start tasks=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    try:
        turns = await _run_episodes(client=client, items=items, model="policy")
        return AgentTrajectory(turns=turns)
    finally:
        await client.close()
        await http_client.aclose()


async def _run_episodes(*, client: Any, items: list[Any], model: str) -> list[AgentTrajectoryTurn]:
    """Run all episodes in lockstep turns so requests batch together."""

    states = [_init_state(item) for item in items]
    turn_cap = max((int(state["turn_limit"]) for state in states), default=DEFAULT_MAX_TURNS)
    turns: list[AgentTrajectoryTurn] = []

    for turn_idx in range(turn_cap):
        active = [s for s in states if not s["done"] and turn_idx < s["turn_limit"]]
        if not active:
            break
        responses = await asyncio.gather(
            *(
                create_chat_completion_with_retry(
                    client,
                    model=model,
                    messages=s["messages"],
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=False,
                )
                for s in active
            )
        )
        for state, response in zip(active, responses, strict=True):
            turns.append(
                AgentTrajectoryTurn(item=state["item"], messages=list(state["messages"]), response=response, tools=TOOLS)
            )
            assistant = _assistant_message(response)
            state["messages"].append(assistant)
            call = _first_tool_call(assistant)
            if call is None:
                state["messages"].append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not include a tool call. Continue by calling exactly "
                            "one tool: inspect_candidates, place_digit, or undo."
                        ),
                    }
                )
                continue
            result = _execute_tool(state["env"], call)
            # The env's action_budget is loose; the binding cap is max_turns.
            # Surface how many future turns the policy still has so it can budget
            # its inspect/undo/recovery calls — actions_remaining alone (from the
            # loose budget) overstates them. turn_idx is this state's (turn_idx)th
            # response, so after this call it has used turn_idx + 1 turns.
            result["turns_remaining"] = max(0, state["turn_limit"] - (turn_idx + 1))
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
                }
            )
            if state["env"].is_terminal():
                state["done"] = True
                env = state["env"]
                logger.info(
                    "episode done solved=%s difficulty=%s actions_used=%d invalid_actions=%d "
                    "prompt_idx=%d sample_idx=%d",
                    env.is_solved(),
                    env.difficulty,
                    env.actions_used,
                    env.invalid_actions,
                    state["item"].prompt_index,
                    state["item"].sample_index,
                )

    # Episodes that ran out of turns without reaching a terminal state
    # (truncated, not solved) — log them too so solve-rate is countable.
    for state in states:
        if state["done"]:
            continue
        env = state["env"]
        logger.info(
            "episode truncated solved=%s difficulty=%s actions_used=%d invalid_actions=%d "
            "prompt_idx=%d sample_idx=%d",
            env.is_solved(),
            env.difficulty,
            env.actions_used,
            env.invalid_actions,
            state["item"].prompt_index,
            state["item"].sample_index,
        )
    return turns


def _init_state(item: Any) -> dict[str, Any]:
    record = item.record
    puzzle = sudoku._normalize_puzzle(record["puzzle"])  # noqa: SLF001
    env = sudoku.SudokuEnv.from_puzzle(
        puzzle,
        difficulty=record.get("difficulty", sudoku.DEFAULT_DIFFICULTY),
        seed=int(record.get("seed", 0)),
        action_budget=int(record.get("action_budget", sudoku.DEFAULT_ACTION_BUDGET)),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt},
    ]
    turn_limit = int(record.get("max_turns") or DEFAULT_MAX_TURNS)
    return {"item": item, "env": env, "messages": messages, "turn_limit": turn_limit, "done": False}


def _assistant_message(response: Any) -> dict[str, Any]:
    message = response.choices[0].message
    # The OpenAI SDK returns a pydantic-like object; coerce to a plain dict so
    # we can re-feed it into messages and mutate it safely. We keep ONLY the
    # first tool call: the loop executes exactly one tool per turn and appends
    # exactly one tool-result message, so keeping all of a multi-call response
    # would make the conversation self-inconsistent (model emits K calls but
    # sees only 1 result next turn), which corrupts the training signal.
    assistant: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    raw_calls = getattr(message, "tool_calls", None) or []
    if raw_calls:
        call = raw_calls[0]
        call_id = getattr(call, "id", None) or "call_0"
        assistant["tool_calls"] = [
            {
                "id": call_id,
                "type": getattr(call, "type", "function"),
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
        ]
    return assistant


def _first_tool_call(assistant: dict[str, Any]) -> dict[str, Any] | None:
    calls = assistant.get("tool_calls") or []
    return calls[0] if calls else None


def _execute_tool(env: sudoku.SudokuEnv, call: dict[str, Any]) -> dict[str, Any]:
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return {"action": name, "error": "invalid_json_arguments", "solved": env.is_solved()}

    try:
        if name == "inspect_candidates":
            row, col = sudoku.parse_coord([args.get("row"), args.get("col")])
            return _with_board(env, env.inspect_candidates(row, col))
        if name == "place_digit":
            row, col = sudoku.parse_coord([args.get("row"), args.get("col")])
            digit = args.get("digit")
            if digit is None:
                raise sudoku.SudokuError("missing required argument 'digit'")
            try:
                digit = int(digit)
            except (TypeError, ValueError):
                raise sudoku.SudokuError(f"digit must be 1-9, got {digit!r}") from None
            return _with_board(env, env.place_digit(row, col, digit))
        if name == "undo":
            return _with_board(env, env.undo())
        return {"action": name, "error": f"unknown_tool:{name}", "solved": env.is_solved()}
    except sudoku.SudokuError as exc:
        # Illegal input (bad/missing coord or digit, non-empty cell, undo at
        # start, terminal). Surface the message back to the policy without
        # leaking the solution, so the loop continues instead of crashing.
        return {"action": name, "error": str(exc), "solved": env.is_solved()}


def _with_board(env: sudoku.SudokuEnv, result: dict[str, Any]) -> dict[str, Any]:
    """Attach lightweight state to a tool result (never the solution).

    We echo a *compact* one-line board each turn (``board_compact``, ~30
    tokens) plus the action outcome and a few progress integers, but NOT the
    full ``board_text`` rendering (that would re-send ~500 tokens/turn and blow
    a multi-turn context budget). The compact echo lets a small policy see the
    current board instead of tracking every placement mentally, which a 0.6B
    model cannot do reliably across turns. The solution digit is never
    included — ``board_compact`` only shows the agent-visible board.
    """

    result["is_terminal"] = env.is_terminal()
    result["actions_remaining"] = env._actions_remaining()  # noqa: SLF001
    result["board_compact"] = env.board_compact()
    return result