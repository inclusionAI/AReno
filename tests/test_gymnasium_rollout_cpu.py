"""CPU tests for the Gymnasium rollout interface (issue #669 acceptance).

Covers fixed-seed parity with native Gymnasium, termination vs truncation,
final-step reward, invalid actions, and sample isolation. ``gymnasium`` is only
required for the parity test; the scripted-env tests are dependency-free.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from areno.integrations.gymnasium import EPISODE_STATS_KEY, make_gymnasium_rollout


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        areno={"input_tokens": [101], "response_tokens": [202, 203], "response_logprobs": [-0.5, -0.7]},
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class ScriptedEnv:
    """Deterministic Gymnasium-API env driven by a per-step script."""

    def __init__(self, script, *, instances=None):
        self.script = list(script)
        if instances is not None:
            instances.append(self)
        self.step_count = 0
        self.reset_seed = "unset"
        self.closed = False

    def reset(self, *, seed=None, options=None):
        del options
        self.reset_seed = seed
        self.step_count = 0
        return {"step": 0, "seed": seed}, {}

    def step(self, action):
        entry = self.script[self.step_count]
        self.step_count += 1
        if entry.get("reject_action") is not None and action == entry["reject_action"]:
            raise ValueError(f"invalid action {action!r}")
        obs = {"step": self.step_count, "seed": self.reset_seed}
        return (
            obs,
            float(entry.get("reward", 0.0)),
            bool(entry.get("terminated", False)),
            bool(entry.get("truncated", False)),
            {},
        )

    def close(self):
        self.closed = True


def _item(*, seed=None, prompt_index=0, sample_index=0):
    return SimpleNamespace(
        prompt="gymnasium episode",
        record={"prompt": "gymnasium episode", "seed": seed},
        prompt_index=prompt_index,
        sample_index=sample_index,
    )


def _client_stubs(completions):
    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = SimpleNamespace(completions=completions)

        async def close(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            del kwargs

        async def aclose(self):
            return None

    openai_stub = SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
    httpx_stub = SimpleNamespace(
        AsyncClient=FakeAsyncClient,
        Limits=lambda **kwargs: None,
        Timeout=lambda *args, **kwargs: None,
    )
    return openai_stub, httpx_stub


def _run_batch(run_agent, items, responses):
    completions = FakeCompletions(responses)
    ctx = SimpleNamespace(
        get_base_url=lambda: "http://127.0.0.1:1/v1",
        api_key="areno-agentic",
        max_running_prompts=2,
    )
    openai_stub, httpx_stub = _client_stubs(completions)
    previous_openai = sys.modules.get("openai")
    previous_httpx = sys.modules.get("httpx")
    sys.modules["openai"] = openai_stub
    sys.modules["httpx"] = httpx_stub
    try:
        batch = SimpleNamespace(iter_samples=lambda: iter(items))
        trajectory = asyncio.run(run_agent(ctx, batch))
    finally:
        sys.modules.pop("openai", None)
        sys.modules.pop("httpx", None)
        if previous_openai is not None:
            sys.modules["openai"] = previous_openai
        if previous_httpx is not None:
            sys.modules["httpx"] = previous_httpx
    return trajectory, completions


def test_fixed_seed_actions_match_native_gymnasium():
    gym = pytest.importorskip("gymnasium")
    actions = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    cap = len(actions)

    def encode(obs):
        return " ".join(f"{value:.6f}" for value in obs)

    run_agent, reward_fn = make_gymnasium_rollout(
        make_env=lambda: gym.make("CartPole-v1"),
        encode_observation=encode,
        decode_action=lambda text: int(text),
        max_episode_steps=cap,
    )
    item = _item(seed=7)
    trajectory, completions = _run_batch(run_agent, [item], [_response(str(action)) for action in actions])

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=7)
    native_encoded = [encode(obs)]
    total = 0.0
    length = 0
    terminated = truncated = False
    for action in actions:
        obs, reward, terminated, truncated, _ = env.step(action)
        length += 1
        total += reward
        if terminated or truncated or length >= cap:
            break
        native_encoded.append(encode(obs))
    env.close()

    stats = item.record[EPISODE_STATS_KEY]
    assert stats["seed"] == 7
    assert stats["episode_return"] == pytest.approx(total)
    assert stats["length"] == length
    assert stats["terminated"] is bool(terminated)
    assert stats["truncated"] is bool(truncated or (length >= cap and not terminated))
    assert stats["invalid_action"] is False
    assert len(trajectory.turns) == length
    assert reward_fn(SimpleNamespace(source_record=dict(item.record))) == pytest.approx(total)

    sent = [request["messages"] for request in completions.requests]
    assert len(sent) == length
    for index, messages in enumerate(sent):
        assert messages[-1]["content"] == native_encoded[index]
        if index:
            assert messages[-2]["role"] == "assistant"


def test_termination_truncation_and_final_step_reward():
    instances = []
    run_agent, _ = make_gymnasium_rollout(
        make_env=lambda: ScriptedEnv(
            [{"reward": 1.0}, {"reward": 1.5}, {"reward": 5.0, "terminated": True}, {"reward": 100.0}],
            instances=instances,
        ),
        encode_observation=lambda obs: f"obs {obs['step']}",
        decode_action=lambda text: int(text),
        max_episode_steps=16,
    )
    item = _item(seed=0)
    trajectory, _ = _run_batch(run_agent, [item], [_response("0")] * 4)
    stats = item.record[EPISODE_STATS_KEY]
    assert stats["episode_return"] == pytest.approx(7.5)  # includes the final step's 5.0
    assert stats["length"] == 3
    assert stats["terminated"] is True
    assert stats["truncated"] is False
    assert stats["invalid_action"] is False
    assert len(trajectory.turns) == 3
    assert instances[0].closed is True


def test_env_truncation_and_step_cap_both_mark_truncated():
    run_agent, _ = make_gymnasium_rollout(
        make_env=lambda: ScriptedEnv([{"reward": 1.0}, {"reward": 2.0, "truncated": True}]),
        encode_observation=lambda obs: f"obs {obs['step']}",
        decode_action=lambda text: int(text),
        max_episode_steps=16,
    )
    item = _item(seed=1)
    _run_batch(run_agent, [item], [_response("0")] * 4)
    stats = item.record[EPISODE_STATS_KEY]
    assert stats["truncated"] is True
    assert stats["terminated"] is False
    assert stats["episode_return"] == pytest.approx(3.0)
    assert stats["length"] == 2

    run_agent, _ = make_gymnasium_rollout(
        make_env=lambda: ScriptedEnv([{"reward": 1.0}] * 10),
        encode_observation=lambda obs: f"obs {obs['step']}",
        decode_action=lambda text: int(text),
        max_episode_steps=3,
    )
    item = _item(seed=2)
    trajectory, _ = _run_batch(run_agent, [item], [_response("0")] * 10)
    stats = item.record[EPISODE_STATS_KEY]
    assert stats["truncated"] is True
    assert stats["terminated"] is False
    assert stats["length"] == 3
    assert stats["episode_return"] == pytest.approx(3.0)
    assert len(trajectory.turns) == 3


def test_invalid_actions_end_episode_with_partial_return():
    def decode(text):
        if text == "bad":
            raise ValueError("unparseable action")
        return int(text)

    run_agent, _ = make_gymnasium_rollout(
        make_env=lambda: ScriptedEnv([{"reward": 1.0}, {"reward": 1.0}, {"reward": 9.0}]),
        encode_observation=lambda obs: f"obs {obs['step']}",
        decode_action=decode,
        max_episode_steps=16,
    )
    item = _item(seed=3)
    trajectory, _ = _run_batch(run_agent, [item], [_response("0"), _response("bad"), _response("0")])
    stats = item.record[EPISODE_STATS_KEY]
    assert stats["invalid_action"] is True
    assert stats["episode_return"] == pytest.approx(1.0)  # partial return kept
    assert stats["length"] == 1
    assert len(trajectory.turns) == 2  # the failing model call is still a trainable turn

    run_agent, _ = make_gymnasium_rollout(
        make_env=lambda: ScriptedEnv([{"reward": 1.0}, {"reward": 1.0, "reject_action": 5}]),
        encode_observation=lambda obs: f"obs {obs['step']}",
        decode_action=lambda text: int(text),
        max_episode_steps=16,
    )
    item = _item(seed=4)
    _run_batch(run_agent, [item], [_response("0"), _response("5")])
    stats = item.record[EPISODE_STATS_KEY]
    assert stats["invalid_action"] is True
    assert stats["episode_return"] == pytest.approx(1.0)
    assert stats["length"] == 1


def test_sample_isolation_seeded_resets_and_per_item_stats():
    instances = []

    def make_env():
        return ScriptedEnv([{"reward": 1.0}, {"reward": 2.0, "terminated": True}], instances=instances)

    run_agent, _ = make_gymnasium_rollout(
        make_env=make_env,
        encode_observation=lambda obs: f"obs {obs['step']} seed {obs['seed']}",
        decode_action=lambda text: int(text),
        max_episode_steps=16,
    )
    item_a = _item(seed=11, sample_index=0)
    item_b = _item(seed=22, sample_index=1)
    _run_batch(run_agent, [item_a, item_b], [_response("0")] * 4)

    assert item_a.record is not item_b.record
    assert len(instances) == 2  # one env per episode
    assert instances[0] is not instances[1]
    assert instances[0].reset_seed == 11
    assert instances[1].reset_seed == 22
    assert all(env.closed for env in instances)
    assert item_a.record[EPISODE_STATS_KEY]["seed"] == 11
    assert item_b.record[EPISODE_STATS_KEY]["seed"] == 22
    for item in (item_a, item_b):
        assert item.record[EPISODE_STATS_KEY]["episode_return"] == pytest.approx(3.0)

    item_c = _item(seed=11, sample_index=2)
    _run_batch(run_agent, [item_c], [_response("0")] * 2)
    assert item_c.record[EPISODE_STATS_KEY] == item_a.record[EPISODE_STATS_KEY]


def test_reward_fn_scores_episode_return_and_rejects_missing_stats():
    _, reward_fn = make_gymnasium_rollout(
        make_env=lambda: ScriptedEnv([]),
        encode_observation=str,
        decode_action=int,
    )
    record = SimpleNamespace(source_record={EPISODE_STATS_KEY: {"episode_return": 3.5}})
    assert reward_fn(record) == 3.5
    with pytest.raises(ValueError, match="gymnasium episode stats"):
        reward_fn(SimpleNamespace(source_record={}))


def test_multimodal_observation_content_reaches_model_messages():
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

    def encode(obs):
        return [{"type": "text", "text": f"frame {obs['step']}"}, dict(image_part)]

    run_agent, _ = make_gymnasium_rollout(
        make_env=lambda: ScriptedEnv([{"reward": 0.0}, {"reward": 1.0, "terminated": True}]),
        encode_observation=encode,
        decode_action=lambda text: int(text),
        max_episode_steps=16,
    )
    item = _item(seed=5)
    _, completions = _run_batch(run_agent, [item], [_response("0"), _response("0")])
    assert completions.requests[0]["messages"][0]["content"] == [{"type": "text", "text": "frame 0"}, image_part]
    assert completions.requests[1]["messages"][-1]["content"] == [{"type": "text", "text": "frame 1"}, image_part]


def test_max_episode_steps_must_be_positive():
    with pytest.raises(ValueError, match="max_episode_steps"):
        make_gymnasium_rollout(
            make_env=lambda: ScriptedEnv([]),
            encode_observation=str,
            decode_action=int,
            max_episode_steps=0,
        )
