from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "competition"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    helper_names = ["dataset_generator", "dataset_loader", "game", "reward"]
    previous_helpers = {helper_name: sys.modules.pop(helper_name, None) for helper_name in helper_names}
    previous_agentic = sys.modules.get("areno.api.agentic")
    if name == "run_agent":
        sys.modules["areno.api.agentic"] = SimpleNamespace(
            AgentTrajectory=type("AgentTrajectory", (), {}),
            AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
        )
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_competition_{name}_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        for helper_name in helper_names:
            sys.modules.pop(helper_name, None)
            if previous_helpers[helper_name] is not None:
                sys.modules[helper_name] = previous_helpers[helper_name]
        if name == "run_agent":
            sys.modules.pop("areno.api.agentic", None)
            if previous_agentic is not None:
                sys.modules["areno.api.agentic"] = previous_agentic


def test_generator_is_reproducible_and_loader_attaches_profile():
    generator = _load_module("dataset_generator")
    loader = _load_module("dataset_loader")

    rows = generator.generate_records(6, seed=8)
    records = loader.load_training_dataset("unused", default_loader=lambda _: rows)

    assert rows == generator.generate_records(6, seed=8)
    assert len(records) == 6
    assert all("user_profile" in record for record in records)
    assert "Generate a sandwich feedback" in records[0]["prompt"]
    assert records[0]["diary"] in records[0]["prompt"]


def test_tool_schemas_are_closed_and_scores_are_bounded():
    game = _load_module("game")

    for tool in game.TOOLS:
        parameters = tool["function"]["parameters"]
        assert parameters["additionalProperties"] is False

    assert game.SELF_SCORE_TOOL["function"]["parameters"]["properties"]["score"]["minimum"] == 0.0
    assert game.SELF_SCORE_TOOL["function"]["parameters"]["properties"]["score"]["maximum"] == 1.0
    assert game.check_sandwich_structure("今天做得好，不过建议下次先列一个小清单，继续保持。") == 1.0
    assert game.check_sandwich_structure("你很棒，继续加油。") == 0.1


def test_agent_executes_strict_named_tool_and_does_not_fabricate_calls():
    run_agent = _load_module("run_agent")

    valid = {
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "self_score", "arguments": json.dumps({"score": 2, "reason": "too high"})},
            }
        ]
    }

    assert run_agent._execute_tool("self_score", valid, {}) == {"recorded": True, "score": 1.0}
    assert run_agent._execute_tool("self_score", {"tool_calls": []}, {}) is None
    assert run_agent._execute_tool("self_score", {"tool_calls": [valid["tool_calls"][0], valid["tool_calls"][0]]}, {}) is None
    assert (
        run_agent._execute_tool(
            "self_score",
            {"tool_calls": [{"function": {"name": "peer_score", "arguments": "{}"}}]},
            {},
        )
        is None
    )
    assert run_agent._execute_tool(
        "self_score",
        {"tool_calls": [{"function": {"name": "self_score", "arguments": "not-json"}}]},
        {},
    ) == {"error": "invalid JSON arguments"}


def test_competition_group_preserves_tool_order_and_peer_context():
    run_agent = _load_module("run_agent")
    captured = []

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.append(kwargs)
            tool_name = kwargs["tool_choice"]["function"]["name"]
            system_content = kwargs["messages"][0]["content"]
            agent_index = 1 if "agent 1" in system_content else 0
            arguments = {}
            if tool_name == "generate_content":
                arguments = {
                    "content": (
                        f"Agent {agent_index}: 今天做得好，不过建议明天先设一个小目标，"
                        "继续认可自己的推进。"
                    )
                }
            elif tool_name == "self_score":
                arguments = {"score": 0.8 if agent_index == 0 else 0.4, "reason": "calibrated"}
            elif tool_name == "peer_score":
                arguments = {"score": 0.2 if agent_index == 0 else 0.9, "reason": "peer review"}
            call = SimpleNamespace(
                id=f"call-{agent_index}-{tool_name}",
                type="function",
                function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments, ensure_ascii=False)),
            )
            message = SimpleNamespace(content=None, tool_calls=[call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    shared_record = {
        "diary": "今天跑通了AReno训练，下午改简历改了很久。",
        "mood": "充实但累",
        "user_profile": {"name": "User", "age": 20, "personality": [], "preferences": []},
    }
    items = [
        SimpleNamespace(prompt="daily prompt", record=shared_record, prompt_index=0, sample_index=0),
        SimpleNamespace(prompt="daily prompt", record=shared_record, prompt_index=0, sample_index=1),
    ]
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    turns = asyncio.run(run_agent._run_competition_group(items, client))

    assert len(turns) == 8
    assert [kwargs["tool_choice"]["function"]["name"] for kwargs in captured] == [
        "fetch_profile",
        "fetch_profile",
        "generate_content",
        "generate_content",
        "self_score",
        "self_score",
        "peer_score",
        "peer_score",
    ]
    assert "Agent 1:" in captured[-2]["messages"][-1]["content"]
    assert "Agent 0:" in captured[-1]["messages"][-1]["content"]
    assert shared_record["_competition_result"]["winner"] == 0
    assert shared_record["_competition_result"]["peer_scores_received"]["0"] == 0.9
    assert shared_record["_competition_result"]["compute_gains"]["0"] == 0.1


def test_competition_group_stops_sample_on_missing_required_tool_call():
    run_agent = _load_module("run_agent")

    class MissingCompletions:
        async def create(self, **kwargs):
            del kwargs
            message = SimpleNamespace(content="plain text", tool_calls=[])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    item = SimpleNamespace(prompt="daily prompt", record={"diary": "x"}, prompt_index=0, sample_index=0)
    client = SimpleNamespace(chat=SimpleNamespace(completions=MissingCompletions()))

    turns = asyncio.run(run_agent._run_competition_group([item], client))

    assert len(turns) == 1
    assert "_competition_result" in item.record


def test_reward_separates_missing_partial_and_competitive_paths():
    reward = _load_module("reward")
    source = {
        "diary": "今天跑通了AReno训练，下午改简历改了很久。",
        "_competition_result": {
            "peer_scores_received": {"0": 0.9, "1": 0.2},
            "compute_gains": {"0": 0.1, "1": -0.1},
        },
    }

    assert reward.reward_fn(SimpleNamespace(source_record=source, tool_calls=[], metadata={"sample_index": 0})) == -1.0

    praise_only = [
        {"name": "generate_content", "arguments": json.dumps({"content": "你今天很棒，继续加油。"})},
        {"name": "self_score", "arguments": json.dumps({"score": 0.9, "reason": "good"})},
    ]
    strong = [
        {
            "name": "generate_content",
            "arguments": json.dumps(
                {"content": "今天跑通AReno训练做得好，不过改简历太久可能卡在完美主义，建议先投一版再迭代，继续保持。"}
            ),
        },
        {"name": "self_score", "arguments": json.dumps({"score": 0.8, "reason": "specific"})},
    ]

    weak_score = reward.reward_fn(SimpleNamespace(source_record=source, tool_calls=praise_only, metadata={"sample_index": 1}))
    strong_score = reward.reward_fn(SimpleNamespace(source_record=source, tool_calls=strong, metadata={"sample_index": 0}))

    assert weak_score < strong_score
    assert strong_score > 0.8


def test_eval_feedback_scores_and_summarizes_outputs(tmp_path):
    eval_feedback = _load_module("eval_feedback")

    record = {
        "id": 7,
        "diary": "今天跑通了AReno训练，下午改简历改了很久。",
        "mood": "充实但累",
        "user_profile": {"name": "User", "age": 20, "personality": [], "preferences": []},
    }
    strong = "今天跑通AReno训练做得好，不过改简历太久可能卡在完美主义，建议先投一版再迭代，继续保持。"
    empty_metrics = eval_feedback.score_feedback(record, "")
    strong_metrics = eval_feedback.score_feedback(record, strong)

    assert empty_metrics["reward"] == -1.0
    assert strong_metrics["structure_score"] == 1.0
    assert strong_metrics["reward"] > empty_metrics["reward"]

    rows = [
        {
            "label": "before",
            "id": record["id"],
            "candidate_index": 0,
            "diary": record["diary"],
            "mood": record["mood"],
            "content": strong,
            "metrics": strong_metrics,
            "error": None,
        }
    ]
    output = tmp_path / "eval.jsonl"
    report = tmp_path / "report.md"

    eval_feedback.write_jsonl(output, rows)
    loaded = eval_feedback.read_jsonl(output)
    eval_feedback.write_report(report, loaded, label="after", baseline_rows=[{**rows[0], "metrics": empty_metrics}])

    assert loaded[0]["metrics"]["reward"] == strong_metrics["reward"]
    assert eval_feedback.summarize(loaded)["structure_score"] == 1.0
    report_text = report.read_text(encoding="utf-8")
    assert "Competition Feedback Evaluation: after" in report_text
    assert "Delta Vs Baseline" in report_text


def test_eval_feedback_runs_without_endpoint_and_records_empty_candidates():
    eval_feedback = _load_module("eval_feedback")
    records = [{"id": 1, "diary": "今天调小batch-size避免OOM。", "mood": "谨慎"}]

    rows = eval_feedback.run_evaluation(
        records,
        base_url=None,
        model="policy",
        api_key="token",
        label="dry",
        candidates=2,
    )

    assert len(rows) == 2
    assert rows[0]["label"] == "dry"
    assert rows[0]["content"] == ""
    assert rows[0]["metrics"]["reward"] == -1.0


def test_eval_feedback_strips_qwen_reasoning_traces():
    eval_feedback = _load_module("eval_feedback")

    assert eval_feedback.strip_reasoning_traces("<think>hidden</think>\n\n今天先复盘一个报错。") == "今天先复盘一个报错。"
    assert eval_feedback.strip_reasoning_traces("prefix <think>hidden</think> suffix") == "prefix  suffix"
    assert eval_feedback.strip_reasoning_traces("<think>unfinished reasoning") == ""
