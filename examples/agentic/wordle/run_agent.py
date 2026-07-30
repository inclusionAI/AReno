"""Bounded multi-turn agent loop for Wordle."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import GUESS_TOOL, WORDLE_WORDS, score_guess  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Pre-build a regex to find any valid 5-letter word in model output text.
_WORDLE_WORD_RE = None

SYSTEM_PROMPT = (
    "You are a Wordle solver. On every guessing turn call guess_word exactly once. "
    "Use the exact/present/absent feedback from prior guesses to deduce the hidden word. "
    "Never repeat a guess. After the game ends, summarize the outcome without a tool call."
)


async def run_agent(ctx, batch):
    """Run bounded concurrent Wordle episodes and preserve exact model outputs."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Wordle requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        ),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(
        base_url=ctx.get_base_url(),
        api_key=ctx.api_key,
        http_client=http_client,
        max_retries=0,
    )
    try:
        grouped = await asyncio.gather(
            *(_run_episode(item, client) for item in items)
        )
        return AgentTrajectory(
            turns=[turn for episode in grouped for turn in episode]
        )
    finally:
        await client.close()


async def _run_episode(item, client) -> list[AgentTrajectoryTurn]:
    """One Wordle episode: up to max_guesses tool calls + a final summary."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt},
    ]
    turns = []
    max_guesses = min(max(int(item.record["max_guesses"]), 1), 6)

    for guess_number in range(1, max_guesses + 1):
        turn_messages = [
            *messages,
            {"role": "user", "content": f"Guess {guess_number} of {max_guesses}: call guess_word now."},
        ]
        tool_choice = {"type": "function", "function": {"name": "guess_word"}}
        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=[GUESS_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=turn_messages,
                response=response,
                tools=[GUESS_TOOL],
                tool_choice=tool_choice,
            )
        )

        assistant_message = _assistant_message(response)
        tool_result = _execute_guess(assistant_message, item.record)

        if tool_result is None:
            # Fallback: small models may emit a tool_call with missing/invalid
            # arguments, or no tool_call at all.  Try to extract a guess from
            # the response text or tool_call arguments instead.
            content = response.choices[0].message.content or ""
            tool_call_args = ""
            calls = assistant_message.get("tool_calls") or []
            if calls and calls[0].get("function", {}).get("arguments"):
                tool_call_args = calls[0]["function"]["arguments"]
            fallback_word = _extract_fallback_guess(content) or _extract_fallback_guess(tool_call_args)
            if fallback_word is not None:
                assistant_message = _synth_tool_call_message(content, fallback_word)
                tool_result = _execute_guess(assistant_message, item.record)
                if tool_result is not None:
                    logger.info("Wordle fallback extracted guess: %s", fallback_word)

        if tool_result is None:
            # DEBUG: dump raw model response to diagnose why tool_call parsing fails
            raw_msg = response.choices[0].message
            areno_meta = getattr(response, "areno", None) or {}
            usage = getattr(response, "usage", None)
            max_seq_len = getattr(usage, "max_sequence_len", None) if usage else None
            resp_tokens = areno_meta.get("response_tokens", []) if isinstance(areno_meta, dict) else []
            logger.warning(
                "Wordle raw response: content=%r, tool_calls=%r, finish_reason=%r, "
                "response_token_count=%d, max_sequence_len=%r, prompt_tokens=%r",
                raw_msg.content,
                [(c.function.name, c.function.arguments) for c in (raw_msg.tool_calls or [])],
                getattr(response.choices[0], "finish_reason", None),
                len(resp_tokens),
                max_seq_len,
                getattr(usage, "prompt_tokens", None) if usage else None,
            )
            logger.warning("Wordle model returned no executable guess_word call")
            break

        messages.extend(_tool_messages(assistant_message, tool_result))

        game_over = (
            tool_result.get("solved")
            or not tool_result.get("valid")
            or guess_number == max_guesses
        )
        if game_over:
            finish_messages = [
                *messages,
                {"role": "user", "content": "The game is over. Briefly summarize the outcome without calling a tool."},
            ]
            finish_response = await client.chat.completions.create(
                model="policy",
                messages=finish_messages,
                stream=False,
            )
            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=finish_messages,
                    response=finish_response,
                )
            )
            break

    return turns


def _synth_tool_call_message(content: str, word: str) -> dict:
    """Synthesize a tool-call message for a fallback-extracted guess."""

    import uuid

    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": "guess_word",
                    "arguments": json.dumps({"word": word}),
                },
            }
        ],
    }


def _build_wordle_word_re():
    """Build a regex that matches any word from the Wordle word list."""

    import re

    global _WORDLE_WORD_RE
    if _WORDLE_WORD_RE is None:
        # Match any valid 5-letter word as a whole word (case-insensitive).
        alternation = "|".join(re.escape(w) for w in WORDLE_WORDS)
        _WORDLE_WORD_RE = re.compile(rf"\b({alternation})\b", re.IGNORECASE)
    return _WORDLE_WORD_RE


def _extract_fallback_guess(content: str | None) -> str | None:
    """Extract a Wordle guess from natural-language model output.

    Small models without tool-calling fine-tuning often emit a sentence like
    "I'll guess apple" instead of a structured tool call.  This fallback scans
    the text for any known 5-letter word and returns the last match (the final
    guess is most likely the model's intended answer).
    """

    if not content:
        return None
    matches = _build_wordle_word_re().findall(content)
    return matches[-1].lower() if matches else None


def _assistant_message(response) -> dict:
    """Extract the assistant message from an OpenAI chat completion response."""

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


def _execute_guess(assistant_message: dict, record: dict) -> dict | None:
    """Execute the guess_word tool call and return the result dict."""

    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "guess_word":
        return None
    try:
        arguments = json.loads(calls[0]["function"].get("arguments") or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arguments, dict) or "word" not in arguments:
        return None
    return score_guess(record["secret"], arguments["word"])


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    """Build the assistant + tool messages to append to the conversation."""

    call = assistant_message["tool_calls"][0]
    return [
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": "guess_word",
            "content": json.dumps(tool_result),
        },
    ]

