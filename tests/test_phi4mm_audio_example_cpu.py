from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "multimodal" / "phi4mm_audio"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"phi4mm_audio_example_{name}", EXAMPLE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_request_embeds_local_audio_in_openai_message(tmp_path) -> None:
    request = _load("request")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-test-audio")

    messages = request.build_messages(audio, "Describe it")

    content = messages[0]["content"]
    assert messages[0]["role"] == "user"
    assert content[0]["type"] == "input_audio"
    assert content[0]["input_audio"]["format"] == "wav"
    assert base64.b64decode(content[0]["input_audio"]["data"]) == audio.read_bytes()
    assert content[1] == {"type": "text", "text": "Describe it"}


def test_request_rejects_unsupported_input_audio_format(tmp_path) -> None:
    request = _load("request")
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"audio")

    with pytest.raises(ValueError, match=".wav or .mp3"):
        request.build_messages(audio, "Describe it")


def test_dataset_loader_resolves_audio_relative_to_manifest(tmp_path) -> None:
    loader = _load("dataset_loader")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": "audio/one.wav", "prompt": "Transcribe", "response": "Hello"}) + "\n",
        encoding="utf-8",
    )

    record = loader.load_training_dataset(str(manifest))[0]

    part = record["messages"][0]["content"][0]
    assert part == {"type": "audio_url", "audio_url": {"url": str(tmp_path / "audio" / "one.wav")}}
    assert record["solutions"] == ["Hello"]


def test_dataset_loader_reports_missing_required_fields(tmp_path) -> None:
    loader = _load("dataset_loader")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"audio_path": "one.wav", "prompt": "Transcribe"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field 'response'"):
        loader.load_training_dataset(str(manifest))
