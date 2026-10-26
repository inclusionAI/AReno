"""SVG training rewards follow task accuracy rather than visual polish."""

import asyncio
import json

import pytest

from examples.agentic.pi.frontend_dataset import FrontendDataset
from examples.agentic.pi.generate_svg_dataset import task_record
from examples.agentic.pi.svg_dataset import SVGAccuracyJudge, SVGIconDataset
from examples.agentic.pi.visual_reward import VLMJudge


@pytest.mark.parametrize("score", [0, 27.5, 100])
def test_svg_reward_and_judge_prompt_use_only_accuracy(monkeypatch, score):
    requests = []

    def request(self, payload):
        requests.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "dimensions": {"semantic_match": {"score": score, "reason": "Observed task match"}},
                                "checks": {},
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(VLMJudge, "_request", request)
    dataset = SVGIconDataset()
    result = asyncio.run(
        dataset.judge_class("https://judge.invalid/v1", "fixture", "fixture").score(
            instruction="A black circle with a white X in its center.",
            images={"Candidate icon": b"\x89PNG\r\n\x1a\nfixture"},
            dimensions=dataset.dimensions,
        )
    )
    payload = requests[0]
    rubric = json.loads(payload["messages"][1]["content"][0]["text"])["rubric"]
    assert list(rubric) == ["semantic_match"]
    assert result["reward"] == pytest.approx(score / 100)
    assert payload["messages"][0]["content"] == SVGAccuracyJudge.system_prompt
    assert "Evaluate only how accurately" in payload["messages"][0]["content"]
    assert FrontendDataset.judge_class is VLMJudge


def test_svg_task_prompts_describe_accuracy_and_the_actual_turn_budget(tmp_path):
    record = task_record(
        {"id": "cross", "svg": "<svg/>", "description": "A black circle with a white X.", "detail": ""},
        reference=tmp_path / "reference.png",
        source={},
        max_turns=6,
    )
    assert "at most 6 assistant turns" in record["prompt"]
    assert "scored only for accuracy" in record["prompt"]
    assert "Judge only accuracy" in record["grading_instruction"]
    assert "composition, geometry, style and aesthetics" not in record["prompt"]
    assert record["source_prompt"] == "A black circle with a white X."
