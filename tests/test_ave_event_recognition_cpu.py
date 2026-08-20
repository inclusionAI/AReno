from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "multimodal" / "ave_event_recognition"


def _load(name: str):
    previous_common = sys.modules.pop("common", None)
    sys.path.insert(0, str(EXAMPLE))
    try:
        spec = importlib.util.spec_from_file_location(f"ave_event_recognition_{name}", EXAMPLE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.modules.pop("common", None)
        if previous_common is not None:
            sys.modules["common"] = previous_common


def test_prompt_asks_for_events_in_interval_without_leaking_labels():
    common = _load("common")

    prompt = common.prompt_text(1.5, 7)

    assert "between 1.5 and 7 seconds" in prompt
    assert "Bell" not in prompt


def test_annotations_skip_zero_duration_events(tmp_path):
    common = _load("common")
    annotations = tmp_path / "Annotations.txt"
    annotations.write_text(
        "Category&VideoID&Quality&StartTime&EndTime\nBell&abc&good&0&0\nBell&def&good&1&3\n",
        encoding="utf-8",
    )

    assert [row.video_id for row in common.read_annotations(annotations, has_header=True)] == ["def"]


def test_generator_aggregates_overlapping_events_by_interval(tmp_path, monkeypatch):
    generator = _load("dataset_generator")
    (tmp_path / "Annotations.txt").write_text(
        "Category&VideoID&Quality&StartTime&EndTime\nBell&abc&good&1&5\nSpeech&abc&good&3&8\nCat&def&good&0&10\n",
        encoding="utf-8",
    )
    (tmp_path / "trainSet.txt").write_text(
        "Bell&abc&good&1&5\nSpeech&abc&good&3&8\nCat&def&good&0&10\n",
        encoding="utf-8",
    )
    (tmp_path / "videos").mkdir()
    for video_id in ("abc", "def"):
        (tmp_path / "videos" / f"{video_id}.mp4").write_bytes(b"video")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    def fake_audio(_audio_dir, _video, video_id):
        audio = audio_dir / f"{video_id}.wav"
        audio.write_bytes(b"audio")
        return audio

    monkeypatch.setattr(generator, "_ensure_audio", fake_audio)

    records = generator.generate_manifest(tmp_path, tmp_path / "train.jsonl", split="train", seed=7)

    by_interval = {(record["video_id"], record["start_seconds"], record["end_seconds"]): record for record in records}
    assert len(records) == 3
    assert by_interval[("abc", 1.0, 5.0)]["event_classes"] == ["Bell", "Speech"]
    assert by_interval[("abc", 3.0, 8.0)]["event_classes"] == ["Bell", "Speech"]
    assert by_interval[("def", 0.0, 10.0)]["event_classes"] == ["Cat"]
    assert all("event_class" not in record for record in records)
    assert {record["video_path"] for record in records} == {"videos/abc.mp4", "videos/def.mp4"}


def test_dataset_loader_formats_event_list_schema(tmp_path):
    loader = _load("dataset_loader")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "abc:events:1-5",
                "dataset_root": str(tmp_path),
                "video_path": "videos/abc.mp4",
                "audio_path": "audio/abc.wav",
                "event_classes": ["Bell", "Speech"],
                "start_seconds": 1,
                "end_seconds": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = loader.load_training_dataset(str(manifest))[0]

    assert "between 1 and 5 seconds" in record["prompt"]
    assert "Bell" not in record["prompt"]
    assert json.loads(record["response"]) == {"events": ["Bell", "Speech"]}
    assert record["reference"] == ["Bell", "Speech"]


def test_dataset_loader_rejects_old_single_label_schema(tmp_path):
    loader = _load("dataset_loader")
    manifest = tmp_path / "old.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "event_class": "Bell",
                "video_path": "videos/abc.mp4",
                "audio_path": "audio/abc.wav",
                "start_seconds": 1,
                "end_seconds": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no valid event_classes"):
        loader.load_training_dataset(str(manifest))


def test_reward_reads_exactly_one_report_events_tool_call(monkeypatch):
    reward = _load("reward")
    monkeypatch.setattr(
        reward,
        "_judge_event_similarity",
        lambda expected, predicted: 8.75 if expected == ("Bark",) and predicted == ("dog barking",) else 0.0,
    )
    call = {"name": "report_events", "arguments": json.dumps({"events": ["dog barking"]})}
    record = SimpleNamespace(source_record={"event_classes": ["Bark"]}, tool_calls=[call])

    assert reward.reward_fn(record) == 0.875

    record.tool_calls.append(call)
    assert reward.reward_fn(record) == 0.0
    record.tool_calls = [{"name": "report_events", "arguments": json.dumps({"events": []})}]
    assert reward.reward_fn(record) == 0.0


def test_reward_reads_judge_configuration_from_environment(monkeypatch):
    reward = _load("reward")
    seen = {}

    def fake_request(base_url, model, api_key, expected, predicted):
        seen.update(
            base_url=base_url,
            model=model,
            api_key=api_key,
            expected=expected,
            predicted=predicted,
        )
        return 8.25

    monkeypatch.setenv("JUDGE_BASE_URL", "http://judge.example/v1")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_API_KEY", "secret")
    monkeypatch.setattr(reward, "_judge_request", fake_request)

    assert reward._judge_event_similarity(("Bark",), ("dog barking",)) == 8.25
    assert seen == {
        "base_url": "http://judge.example/v1",
        "model": "judge-model",
        "api_key": "secret",
        "expected": ("Bark",),
        "predicted": ("dog barking",),
    }


def test_reward_exposes_configured_parallel_workers(monkeypatch):
    monkeypatch.setenv("JUDGE_MAX_WORKERS", "7")

    reward = _load("reward")

    assert reward.reward_fn.parallel_workers == 7


def test_reward_fails_loudly_without_judge_configuration(monkeypatch):
    reward = _load("reward")
    for name in ("JUDGE_BASE_URL", "JUDGE_MODEL", "JUDGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)

    with pytest.raises(RuntimeError, match="JUDGE_BASE_URL"):
        reward._judge_event_similarity(("Bark",), ("Bark",))


def test_reward_logs_remote_judge_response(monkeypatch, caplog):
    reward = _load("reward")
    requests = []

    class FakeClient:
        def __init__(self, **_kwargs):
            def create(**request):
                requests.append(request)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{“score”:8.375}"))])

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))
    caplog.set_level(logging.INFO, logger=reward.__name__)
    reward._judge_request.cache_clear()

    score = reward._judge_request("http://judge/v1", "judge", "secret", ("Bark",), ("dog barking",))

    assert score == 8.375
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "response='{“score”:8.375}' score=8.375 normalized_reward=0.838" in caplog.text
