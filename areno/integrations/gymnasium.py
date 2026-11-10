"""Gymnasium rollout factory for AReno agentic training.

``make_gymnasium_rollout`` connects a Gymnasium-style environment to the
agentic rollout API. ``run_agent`` plays one episode per prompt/sample item and
returns trajectory turns; the paired ``reward_fn`` scores each episode by its
cumulative return. Load both from the same task file with ``--agent-fn`` and
``--reward-fn-path``.

The CLI executes the task file once per hook, so ``run_agent`` and ``reward_fn``
end up in separate module instances. Episode statistics therefore travel on the
sample record -- ``run_agent`` writes them under :data:`EPISODE_STATS_KEY` and
``reward_fn`` reads them back from ``record.source_record`` -- instead of
closure state.

The environment must follow the Gymnasium API (``reset(seed=...) -> (obs, info)``
and ``step(action) -> (obs, reward, terminated, truncated, info)``). This
module deliberately does not import ``gymnasium``: environments come from the
task file's ``make_env`` factory, so no dependency is added here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

EPISODE_STATS_KEY = "gymnasium_episode"


def make_gymnasium_rollout(
    make_env: Callable[[], Any],
    encode_observation: Callable[[Any], str | list[dict[str, Any]]],
    decode_action: Callable[[str], Any],
    max_episode_steps: int = 128,
) -> tuple[Callable[..., Any], Callable[[Any], float]]:
    """Build the ``run_agent`` / ``reward_fn`` pair for one Gymnasium task.

    Args:
        make_env: zero-argument factory returning a Gymnasium-style
            environment. Every episode gets its own instance so samples stay
            isolated.
        encode_observation: maps an observation to user-message content: either
            a string or a multimodal content-part list (for image observations).
        decode_action: maps the model response text to an environment action.
            A raised exception ends the episode as an invalid action; an action
            rejected by ``env.step`` ends it the same way.
        max_episode_steps: per-episode step cap. Reaching the cap marks the
            episode truncated unless it already terminated.

    Returns:
        ``(run_agent, reward_fn)`` for ``--agent-fn`` and ``--reward-fn-path``.
    """

    if int(max_episode_steps) < 1:
        raise ValueError("max_episode_steps must be >= 1")

    async def run_one(item: Any, client: Any, model: str = "policy") -> list[AgentTrajectoryTurn]:
        env = make_env()
        turns: list[AgentTrajectoryTurn] = []
        episode_return = 0.0
        length = 0
        terminated = False
        truncated = False
        invalid_action = False
        seed = item.record.get("seed")
        try:
            obs, _ = env.reset(seed=seed)
            messages = [{"role": "user", "content": encode_observation(obs)}]
            while True:
                response = await client.chat.completions.create(model=model, messages=list(messages), stream=False)
                turns.append(AgentTrajectoryTurn(item=item, messages=list(messages), response=response))
                try:
                    action = decode_action(_response_text(response))
                    obs, reward, terminated, truncated, _ = env.step(action)
                except Exception:
                    invalid_action = True
                    break
                length += 1
                episode_return += float(reward)
                if terminated or truncated or length >= max_episode_steps:
                    break
                messages.append({"role": "assistant", "content": _response_text(response)})
                messages.append({"role": "user", "content": encode_observation(obs)})
        finally:
            item.record[EPISODE_STATS_KEY] = {
                "episode_return": float(episode_return),
                "length": int(length),
                "seed": seed,
                "terminated": bool(terminated),
                "truncated": bool(truncated or (length >= max_episode_steps and not terminated)),
                "invalid_action": bool(invalid_action),
            }
            env.close()
        return turns

    async def run_agent(ctx: Any, batch: Any) -> AgentTrajectory:
        try:
            import httpx
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The Gymnasium rollout requires `openai` and `httpx`. Install them with `pip install openai httpx`."
            ) from exc

        items = list(batch.iter_samples())
        max_connections = max(len(items), ctx.max_running_prompts)
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            timeout=httpx.Timeout(900.0, connect=30.0),
        )
        client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
        try:
            episodes = await asyncio.gather(*(run_one(item, client) for item in items))
            return AgentTrajectory(turns=[turn for episode in episodes for turn in episode])
        finally:
            await client.close()
            await http_client.aclose()

    def reward_fn(record: Any) -> float:
        stats = dict(getattr(record, "source_record", None) or {}).get(EPISODE_STATS_KEY)
        if not stats:
            raise ValueError(
                "record carries no gymnasium episode stats; use the reward_fn paired with run_agent "
                "from make_gymnasium_rollout on the same task file"
            )
        return float(stats["episode_return"])

    return run_agent, reward_fn


def _response_text(response: Any) -> str:
    return response.choices[0].message.content or ""
