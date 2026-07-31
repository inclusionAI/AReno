"""Focused CPU tests for the agentic trajectory detail page in the dashboard server.

Tests cover:
  - Core logic: build_trajectory_detail success path
  - Invalid input: non-agentic samples, missing fields, malformed data
  - Boundary values: empty messages, large tool output, truncated loss mask
  - Privacy-safe defaults: raw sample is included but not executed
  - Deterministic output for the same input
  - Integration: trajectory_detail resolves samples from registered jobs,
    validates query parameters, and controls raw-sample exposure via
    include_raw
"""

from __future__ import annotations

import json

import pytest

from areno.dashboard.server import (
    STATE,
    Job,
    build_trajectory_detail,
    trajectory_detail,
    validate_trajectory_sample,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _agentic_sample(**overrides):
    """Create a minimal valid agentic rollout sample."""
    base = {
        "kind": "agentic",
        "step": 1,
        "prompt_idx": 0,
        "sample_idx": 0,
        "prompt": "What is 2+2?",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "calculate", "arguments": '{"expr": "2+2"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "4"},
            {"role": "assistant", "content": "The answer is 4."},
        ],
        "final_answer": "The answer is 4.",
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "calculate", "arguments": '{"expr": "2+2"}'}}
        ],
        "tool_results": [
            {"name": "calculate", "ok": True, "result": 4},
        ],
        "loss_mask_true": 8,
        "loss_mask_total": 12,
        "first_loss_idx": 4,
        "loss_mask": [False, False, False, False, True, True, True, True, True, True, True, True],
        "tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "prompt_tokens": [1, 2, 3, 4],
        "response_tokens": [5, 6, 7, 8, 9, 10, 11, 12],
        "end_reason": "completed",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# validate_trajectory_sample
# ---------------------------------------------------------------------------


class TestValidateTrajectorySample:
    def test_valid_agentic_sample_returns_no_errors(self):
        sample = _agentic_sample()
        assert validate_trajectory_sample(sample) == []

    def test_non_agentic_kind_is_invalid(self):
        sample = _agentic_sample(kind="rollout")
        errors = validate_trajectory_sample(sample)
        assert len(errors) == 1
        assert "not 'agentic'" in errors[0]

    def test_missing_kind_is_invalid(self):
        sample = _agentic_sample()
        del sample["kind"]
        errors = validate_trajectory_sample(sample)
        assert "missing" in errors[0]

    def test_empty_messages_is_invalid(self):
        sample = _agentic_sample(messages=[])
        errors = validate_trajectory_sample(sample)
        assert any("messages" in e for e in errors)

    def test_non_dict_input_is_invalid(self):
        errors = validate_trajectory_sample("not a dict")  # type: ignore[arg-type]
        assert len(errors) == 1
        assert "not a dict" in errors[0]


# ---------------------------------------------------------------------------
# build_trajectory_detail — success path
# ---------------------------------------------------------------------------


class TestBuildTrajectoryDetailSuccess:
    def test_returns_valid_flag(self):
        detail = build_trajectory_detail(_agentic_sample())
        assert detail["valid"] is True

    def test_preserves_step_and_indices(self):
        sample = _agentic_sample(step=7, prompt_idx=3, sample_idx=2)
        detail = build_trajectory_detail(sample)
        assert detail["step"] == 7
        assert detail["prompt_idx"] == 3
        assert detail["sample_idx"] == 2

    def test_events_are_in_chronological_order(self):
        detail = build_trajectory_detail(_agentic_sample())
        roles = [event["role"] for event in detail["events"]]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_tool_calls_are_attached_to_assistant_message(self):
        detail = build_trajectory_detail(_agentic_sample())
        assistant_event = detail["events"][1]
        assert "tool_calls" in assistant_event
        assert assistant_event["tool_calls"][0]["function"]["name"] == "calculate"

    def test_tool_result_is_interleaved_with_tool_message(self):
        detail = build_trajectory_detail(_agentic_sample())
        tool_event = detail["events"][2]
        assert tool_event["role"] == "tool"
        assert "tool_result" in tool_event
        assert tool_event["tool_result"]["name"] == "calculate"

    def test_final_answer_is_extracted(self):
        detail = build_trajectory_detail(_agentic_sample())
        assert detail["final_answer"] == "The answer is 4."

    def test_token_counts_are_computed(self):
        sample = _agentic_sample(prompt_tokens=[1, 2, 3], response_tokens=[4, 5])
        detail = build_trajectory_detail(sample)
        assert detail["token_counts"]["prompt_tokens"] == 3
        assert detail["token_counts"]["response_tokens"] == 2

    def test_training_mask_summary(self):
        detail = build_trajectory_detail(_agentic_sample(loss_mask_true=8, loss_mask_total=12, first_loss_idx=4))
        assert detail["training_mask"]["loss_mask_true"] == 8
        assert detail["training_mask"]["loss_mask_total"] == 12
        assert detail["training_mask"]["first_loss_idx"] == 4

    def test_end_reason_is_preserved(self):
        detail = build_trajectory_detail(_agentic_sample(end_reason="max_tokens"))
        assert detail["end_reason"] == "max_tokens"

    def test_tool_call_and_result_counts(self):
        detail = build_trajectory_detail(_agentic_sample())
        assert detail["tool_call_count"] == 1
        assert detail["tool_result_count"] == 1

    def test_default_end_reason_when_missing(self):
        sample = _agentic_sample()
        del sample["end_reason"]
        detail = build_trajectory_detail(sample)
        assert detail["end_reason"] == "completed"


# ---------------------------------------------------------------------------
# build_trajectory_detail — invalid input
# ---------------------------------------------------------------------------


class TestBuildTrajectoryDetailInvalid:
    def test_non_agentic_returns_error(self):
        detail = build_trajectory_detail({"kind": "rollout", "messages": [{"role": "user", "content": "hi"}]})
        assert detail["valid"] is False
        assert "error" in detail

    def test_missing_messages_returns_error(self):
        detail = build_trajectory_detail({"kind": "agentic", "prompt": "p"})
        assert detail["valid"] is False
        assert "messages" in detail["error"]


# ---------------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------------


class TestBoundaryValues:
    def test_empty_tool_results_with_tool_messages(self):
        sample = _agentic_sample(
            tool_results=[],
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        detail = build_trajectory_detail(sample)
        assert detail["valid"] is True
        assert detail["tool_result_count"] == 0

    def test_truncated_loss_mask_is_marked(self):
        sample = _agentic_sample(loss_mask=[True, True], loss_mask_true=2, loss_mask_total=100)
        detail = build_trajectory_detail(sample)
        assert detail["training_mask"]["truncated"] is True

    def test_non_truncated_loss_mask(self):
        sample = _agentic_sample(
            loss_mask=[True] * 12,
            loss_mask_true=12,
            loss_mask_total=12,
        )
        detail = build_trajectory_detail(sample)
        assert detail["training_mask"]["truncated"] is False

    def test_large_tool_result_does_not_crash(self):
        big_result = {"name": "search", "ok": True, "data": "x" * 5000}
        sample = _agentic_sample(tool_results=[big_result])
        detail = build_trajectory_detail(sample)
        assert detail["valid"] is True
        assert detail["tool_result_count"] == 1

    def test_no_final_answer(self):
        sample = _agentic_sample(final_answer="")
        detail = build_trajectory_detail(sample)
        assert detail["final_answer"] == ""


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_produces_same_output(self):
        sample = _agentic_sample()
        detail1 = build_trajectory_detail(json.loads(json.dumps(sample)))
        detail2 = build_trajectory_detail(json.loads(json.dumps(sample)))
        assert detail1 == detail2


# ---------------------------------------------------------------------------
# Privacy-safe defaults
# ---------------------------------------------------------------------------


class TestPrivacySafeDefaults:
    def test_never_includes_full_training_data_in_detail_fields(self):
        sample = _agentic_sample()
        detail = build_trajectory_detail(sample)
        # Detail should not contain raw tokens in the structured output
        # (only counts). The "raw" key is added by trajectory_detail(), not
        # build_trajectory_detail().
        assert "tokens" not in detail
        assert "loss_mask" not in detail

    def test_loss_mask_only_exposed_as_counts(self):
        detail = build_trajectory_detail(_agentic_sample())
        assert "loss_mask" not in detail
        assert "training_mask" in detail
        assert isinstance(detail["training_mask"]["loss_mask_true"], int)


# ---------------------------------------------------------------------------
# Integration: trajectory_detail (job + sample resolution, include_raw)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _registered_job():
    """Register a job with one agentic sample in the global STATE, clean up after."""
    job = Job(
        kind="train",
        name="test trajectory job",
        command=["echo", "noop"],
        config={},
        metrics_dir=None,
    )
    sample = _agentic_sample()
    job.samples.append(sample)
    job._sample_keys.add((sample["step"], sample["prompt_idx"], sample["sample_idx"]))
    STATE.jobs[job.id] = job
    yield job
    STATE.jobs.pop(job.id, None)


class TestTrajectoryDetailIntegration:
    def test_resolves_sample_from_registered_job(self, _registered_job):
        detail = trajectory_detail(_registered_job.id, step=1, prompt_idx=0, sample_idx=0)
        assert detail["valid"] is True
        assert detail["step"] == 1
        assert detail["prompt_idx"] == 0
        assert detail["sample_idx"] == 0

    def test_job_not_found_returns_error(self):
        detail = trajectory_detail("nonexistent-job-id", step=0, prompt_idx=0, sample_idx=0)
        assert detail["valid"] is False
        assert "job not found" in detail["error"]

    def test_sample_not_found_returns_error(self, _registered_job):
        detail = trajectory_detail(_registered_job.id, step=999, prompt_idx=0, sample_idx=0)
        assert detail["valid"] is False
        assert "trajectory sample not found" in detail["error"]

    def test_raw_omitted_by_default(self, _registered_job):
        detail = trajectory_detail(_registered_job.id, step=1, prompt_idx=0, sample_idx=0)
        assert "raw" not in detail

    def test_raw_included_when_requested(self, _registered_job):
        detail = trajectory_detail(_registered_job.id, step=1, prompt_idx=0, sample_idx=0, include_raw=True)
        assert "raw" in detail
        assert detail["raw"]["kind"] == "agentic"
        # Raw contains the full training data (tokens, loss_mask) that the
        # structured detail deliberately omits.
        assert "tokens" in detail["raw"]
        assert "loss_mask" in detail["raw"]

    def test_non_agentic_sample_in_job_returns_invalid(self, _registered_job):
        _registered_job.samples.append(
            {
                "kind": "rollout",
                "step": 2,
                "prompt_idx": 0,
                "sample_idx": 0,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        _registered_job._sample_keys.add((2, 0, 0))
        detail = trajectory_detail(_registered_job.id, step=2, prompt_idx=0, sample_idx=0)
        assert detail["valid"] is False
        assert "not 'agentic'" in detail["error"]

    def test_large_trace_does_not_crash(self, _registered_job):
        """A sample with many messages and tool results resolves without error."""
        messages = [{"role": "user", "content": "start"}]
        tool_calls = []
        tool_results = []
        for i in range(50):
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{i}",
                            "type": "function",
                            "function": {"name": f"tool_{i}", "arguments": "{}"},
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": f"call-{i}", "content": f"result {i}"})
            tool_calls.append(
                {"id": f"call-{i}", "type": "function", "function": {"name": f"tool_{i}", "arguments": "{}"}}
            )
            tool_results.append({"name": f"tool_{i}", "ok": True, "result": f"result {i}"})
        messages.append({"role": "assistant", "content": "done"})
        sample = _agentic_sample(
            step=3,
            messages=messages,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        _registered_job.samples.append(sample)
        _registered_job._sample_keys.add((3, 0, 0))
        detail = trajectory_detail(_registered_job.id, step=3, prompt_idx=0, sample_idx=0)
        assert detail["valid"] is True
        assert detail["tool_call_count"] == 50
        assert detail["tool_result_count"] == 50
        assert len(detail["events"]) == 102  # 50 assistant + 50 tool + 1 user + 1 final assistant
