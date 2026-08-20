import logging
from types import SimpleNamespace

from areno.api.rewards import RewardRecord
from areno.api.trainers.policy_only import PolicyOnlyTrainer, _dashboard_safe_value


class _SampleRecorder:
    def __init__(self):
        self.samples = []

    def record_rollout_sample(self, sample):
        self.samples.append(sample)


def test_agentic_dashboard_sample_includes_output_prompt_and_record(monkeypatch):
    monkeypatch.setenv("ARENO_LOG_COMPLETIONS", "1")
    recorder = _SampleRecorder()
    trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
    trainer.areno = recorder
    trainer.logger = logging.getLogger("test.dashboard.sample")
    reward_record = RewardRecord(
        prompt="What happens?",
        completion='<tool_call name="report">walking</tool_call>',
        rendered_completion="assistant: report(walking)",
        final_answer="",
        messages=[
            {"role": "system", "content": "Recognize the event."},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "/data/clip.mp4"}},
                    {"type": "text", "text": "What happens?"},
                ],
            },
            {"role": "assistant", "content": "", "tool_calls": [{"name": "report"}]},
        ],
        source_record={"video_path": "/data/clip.mp4", "label": "walking"},
        metadata={"prompt_index": 0, "sample_index": 0},
    )
    batch = SimpleNamespace(reward_records=[reward_record], loss_masks=[[False, True]], token_rows=[[1, 2]])

    trainer._log_agentic_sample_completions(epoch=0, step=3, agent_batch=batch)

    sample = recorder.samples[0]
    assert sample["completion"] == reward_record.completion
    assert sample["rendered_completion"] == reward_record.rendered_completion
    assert sample["prompt_messages"] == reward_record.messages[:-1]
    assert sample["source_record"] == reward_record.source_record


def test_dashboard_safe_value_summarizes_large_binary_and_tensor_like_values():
    tensor_like = SimpleNamespace(shape=(2, 3), dtype="float32")
    value = _dashboard_safe_value({"audio_base64": "a" * 300, "features": tensor_like})

    assert value["audio_base64"] == "<base64 data: 300 characters>"
    assert value["features"] == {"type": "SimpleNamespace", "shape": [2, 3], "dtype": "float32"}


def test_non_agentic_dashboard_sample_includes_source_record(monkeypatch):
    monkeypatch.setenv("ARENO_LOG_COMPLETIONS", "1")
    recorder = _SampleRecorder()
    trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
    trainer.areno = recorder
    trainer.logger = logging.getLogger("test.dashboard.non_agentic")
    tokenizer = SimpleNamespace(decode=lambda tokens: " ".join(str(token) for token in tokens))
    item = SimpleNamespace(
        prompt="Describe the image.",
        input_tokens=[1, 2],
        record={"image_path": "/data/frame.png", "label": "board"},
    )
    prompt_batch = SimpleNamespace(items=[item])
    sequence = SimpleNamespace(resp_tokens=[3], resp_logprobs=[-0.1])
    rollout_results = [SimpleNamespace(sequences=[sequence])]

    trainer._record_sample_completions(tokenizer, 0, 1, prompt_batch, rollout_results)

    assert recorder.samples[0]["completion"] == "3"
    assert recorder.samples[0]["source_record"] == item.record
