"""Agent entrypoint for the competition agentic example.

Two agents compete to generate the best sandwich feedback for a user's diary.
Each agent calls fetch_profile, generate_content, self_score, and peer_score.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    INITIAL_SHARES,
    TOOL_BY_NAME,
    get_max_tokens,
    get_compute_prompt_hint,
    run_tool,
    simulate_user_score,
    transfer_compute,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a daily-summary assistant competing against another assistant. "
    "Your goal is to generate the best sandwich feedback for the user: "
    "affirm their effort, gently point out one area for improvement with a specific suggestion, "
    "then affirm again. "
    "Call fetch_profile first, then generate_content, then self_score, then peer_score. "
    "Be specific and reference the user's actual diary events."
)

TURN_PROMPTS = {
    "fetch_profile": "Turn 1: Call fetch_profile to get the user's profile.",
    "generate_content": "Turn 2: Call generate_content with your sandwich feedback.",
    "self_score": "Turn 3: Call self_score to rate your own content honestly.",
    "peer_score": "Turn 4: Call peer_score to rate the opponent's content fairly.",
}


async def run_agent(ctx, batch):
    """Run the 4-turn tool-call sequence for each sample."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The competition agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("Competition agent start tasks=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)

    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    try:
        groups = _group_by_prompt(items)
        grouped = await asyncio.gather(*(_run_competition_group(group, client) for group in groups))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


def _group_by_prompt(items: list) -> list[list]:
    groups_by_index = {}
    for item in items:
        groups_by_index.setdefault(item.prompt_index, []).append(item)
    return [groups_by_index[index] for index in sorted(groups_by_index)]


async def _run_competition_group(items: list, client) -> list[AgentTrajectoryTurn]:
    turns = []
    states = {item.sample_index: _initial_state(item) for item in items}
    active = set(states)
    shares = _initial_shares(len(items))

    for sample_index, state in states.items():
        share = shares[min(sample_index, len(shares) - 1)]
        state["messages"][0]["content"] = (
            f"{SYSTEM_PROMPT} You are agent {sample_index}. {get_compute_prompt_hint(share)}"
        )

    for tool_name in ("fetch_profile", "generate_content"):
        for item in items:
            if item.sample_index not in active:
                continue
            state = states[item.sample_index]
            max_tokens = get_max_tokens(shares[min(item.sample_index, len(shares) - 1)])
            assistant_message, turn = await _call_model(
                item,
                client,
                state["messages"],
                tool_name,
                max_tokens=max_tokens,
            )
            turns.append(turn)
            tool_result = _execute_tool(tool_name, assistant_message, item.record)
            if tool_result is None:
                logger.warning("Competition model returned no executable %s call", tool_name)
                active.remove(item.sample_index)
                continue
            state["messages"].extend(_tool_messages(assistant_message, tool_result))
            if tool_name == "generate_content":
                state["content"] = _tool_arguments(assistant_message, tool_name).get("content", "")

    for item in items:
        if item.sample_index not in active:
            continue
        state = states[item.sample_index]
        assistant_message, turn = await _call_model(item, client, state["messages"], "self_score")
        turns.append(turn)
        tool_result = _execute_tool("self_score", assistant_message, item.record)
        if tool_result is None:
            logger.warning("Competition model returned no executable self_score call")
            active.remove(item.sample_index)
            continue
        state["messages"].extend(_tool_messages(assistant_message, tool_result))
        state["self_score"] = _tool_arguments(assistant_message, "self_score").get("score", 0.5)

    for item in items:
        if item.sample_index not in active:
            continue
        opponent = _opponent_state(item.sample_index, states)
        opponent_content = opponent.get("content", "") if opponent is not None else ""
        prompt = (
            f"{TURN_PROMPTS['peer_score']}\n\n"
            f"Opponent feedback to evaluate:\n{opponent_content or '(opponent did not submit feedback)'}"
        )
        state = states[item.sample_index]
        assistant_message, turn = await _call_model(item, client, state["messages"], "peer_score", prompt=prompt)
        turns.append(turn)
        tool_result = _execute_tool("peer_score", assistant_message, item.record)
        if tool_result is None:
            logger.warning("Competition model returned no executable peer_score call")
            active.remove(item.sample_index)
            continue
        state["messages"].extend(_tool_messages(assistant_message, tool_result))
        state["peer_score_given"] = _tool_arguments(assistant_message, "peer_score").get("score", 0.5)

    _record_competition_results(items, states, shares)
    return turns


def _initial_state(item) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ],
        "content": "",
        "self_score": 0.5,
        "peer_score_given": 0.5,
    }


def _initial_shares(agent_count: int) -> list[int]:
    if agent_count == 2:
        return list(INITIAL_SHARES)
    if agent_count <= 0:
        return []
    even_share = max(1, 100 // agent_count)
    return [even_share for _ in range(agent_count)]


async def _call_model(
    item,
    client,
    messages: list[dict],
    tool_name: str,
    *,
    prompt: str | None = None,
    max_tokens: int | None = None,
):
    turn_messages = [*messages, {"role": "user", "content": prompt or TURN_PROMPTS[tool_name]}]
    tool = TOOL_BY_NAME[tool_name]
    tool_choice = {"type": "function", "function": {"name": tool_name}}
    kwargs = {
        "model": "policy",
        "messages": turn_messages,
        "tools": [tool],
        "tool_choice": tool_choice,
        "stream": False,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    response = await client.chat.completions.create(**kwargs)
    return _assistant_message(response), AgentTrajectoryTurn(
        item=item,
        messages=turn_messages,
        response=response,
        tools=[tool],
        tool_choice=tool_choice,
    )


def _assistant_message(response) -> dict:
    message = response.choices[0].message
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in (message.tool_calls or [])
        ],
    }


def _execute_tool(tool_name: str, assistant_message: dict, record: dict) -> dict | None:
    """Execute a tool call and return the result."""
    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != tool_name:
        return None

    call = calls[0]
    try:
        args = json.loads(call["function"].get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {"error": "invalid JSON arguments"}
    if not isinstance(args, dict):
        return {"error": "tool arguments must be an object"}

    return run_tool(tool_name, args, record)


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    call = assistant_message["tool_calls"][0]
    return [
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": call["function"]["name"],
            "content": json.dumps(tool_result, ensure_ascii=False),
        },
    ]


def _tool_arguments(assistant_message: dict, tool_name: str) -> dict:
    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != tool_name:
        return {}
    try:
        args = json.loads(calls[0]["function"].get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}


def _opponent_state(sample_index: int, states: dict[int, dict]) -> dict | None:
    for other_index, state in states.items():
        if other_index != sample_index:
            return state
    return None


def _record_competition_results(items: list, states: dict[int, dict], initial_shares: list[int]) -> None:
    if not items:
        return
    profile = items[0].record.get("user_profile") or {}
    diary = str(items[0].record.get("diary", ""))
    user_scores = {
        str(index): simulate_user_score(str(state.get("content", "")), diary, profile)
        for index, state in states.items()
    }
    self_scores = {str(index): _clamp_score(state.get("self_score", 0.5)) for index, state in states.items()}
    peer_given = {str(index): _clamp_score(state.get("peer_score_given", 0.5)) for index, state in states.items()}
    peer_received = {}
    for index in states:
        opponent = _opponent_index(index, states)
        peer_received[str(index)] = peer_given.get(str(opponent), 0.5) if opponent is not None else 0.5

    base_scores = {
        str(index): user_scores[str(index)] * 0.5 + self_scores[str(index)] * 0.2 + peer_received[str(index)] * 0.3
        for index in states
    }
    winner = max(states, key=lambda index: base_scores[str(index)])
    final_shares = transfer_compute(initial_shares, winner) if len(initial_shares) == 2 else list(initial_shares)
    compute_gains = {}
    for index in states:
        if not initial_shares:
            continue
        final_index = min(index, len(final_shares) - 1)
        initial_index = min(index, len(initial_shares) - 1)
        compute_gains[str(index)] = (final_shares[final_index] - initial_shares[initial_index]) * 0.01
    result = {
        "user_scores": user_scores,
        "self_scores": self_scores,
        "peer_scores_given": peer_given,
        "peer_scores_received": peer_received,
        "base_scores": base_scores,
        "winner": winner,
        "initial_shares": initial_shares,
        "final_shares": final_shares,
        "compute_gains": compute_gains,
    }
    for item in items:
        item.record["_competition_result"] = result


def _opponent_index(sample_index: int, states: dict[int, dict]) -> int | None:
    for other_index in states:
        if other_index != sample_index:
            return other_index
    return None


def _clamp_score(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5
