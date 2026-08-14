from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE = Path(__file__).parents[1] / "examples" / "agentic" / "ave_temporal_grounding"


def _load(name: str):
    previous_common = sys.modules.pop("common", None)
    sys.path.insert(0, str(EXAMPLE))
    try:
        spec = importlib.util.spec_from_file_location(f"ave_temporal_{name}", EXAMPLE / f"{name}.py")
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


def test_timestamp_reward_is_dense_and_rejects_invalid_ranges():
    common = _load("common")

    assert common.timestamp_reward(2, 6, 2, 6) == 1.0
    assert -1 < common.timestamp_reward(3, 7, 2, 6) < 1
    assert common.timestamp_reward(0, 10, 2, 6) <= 0.0
    assert -1 < common.timestamp_reward(7, 8, 2, 6) < 0
    assert common.timestamp_reward(6, 2, 2, 6) == -1.0
    assert common.timestamp_reward(-1, 6, 2, 6) == -1.0


def test_timestamp_reward_is_strict_without_collapsing_valid_ranges():
    common = _load("common")

    assert common.timestamp_reward(1, 9, 0, 10) == 0.365
    assert -1 < common.timestamp_reward(7, 8, 2, 6) < 0
    assert common.timestamp_reward(3, 7, 2, 6) < 0


def test_reward_reads_report_event_range_tool_call():
    reward = _load("reward")
    record = SimpleNamespace(
        source_record={"start_seconds": 2, "end_seconds": 6},
        completion="",
        tool_calls=[
            {
                "name": "report_event_range",
                "arguments": json.dumps({"start_seconds": 2, "end_seconds": 6}),
            }
        ],
    )

    assert reward.reward_fn(record) == 1.0

    record.tool_calls.append(record.tool_calls[0])
    assert reward.reward_fn(record) == -1.0


def test_annotations_skip_zero_duration_events(tmp_path):
    common = _load("common")
    annotations = tmp_path / "Annotations.txt"
    annotations.write_text(
        "Category&VideoID&Quality&StartTime&EndTime\nBell&abc&good&0&0\nBell&def&good&1&3\n",
        encoding="utf-8",
    )

    assert [row.video_id for row in common.read_annotations(annotations, has_header=True)] == ["def"]


def test_generator_emits_one_record_per_split_annotation(tmp_path, monkeypatch):
    generator = _load("dataset_generator")
    (tmp_path / "Annotations.txt").write_text(
        "Category&VideoID&Quality&StartTime&EndTime\nBell&abc&good&1&3\nSpeech&abc&good&5&8\n",
        encoding="utf-8",
    )
    (tmp_path / "trainSet.txt").write_text("Bell&abc&good&1&3\nSpeech&abc&good&5&8\n", encoding="utf-8")
    (tmp_path / "videos").mkdir()
    video = tmp_path / "videos" / "abc.mp4"
    video.write_bytes(b"video")
    audio = tmp_path / "audio" / "abc.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"audio")
    monkeypatch.setattr(generator, "_ensure_audio", lambda *_: audio)

    records = generator.generate_manifest(tmp_path, tmp_path / "train.jsonl", split="train", seed=7)

    assert len(records) == 2
    assert {record["event_class"] for record in records} == {"Bell", "Speech"}
    assert {record["video_path"] for record in records} == {"videos/abc.mp4"}
    assert {(record["start_seconds"], record["end_seconds"]) for record in records} == {(1.0, 3.0), (5.0, 8.0)}
