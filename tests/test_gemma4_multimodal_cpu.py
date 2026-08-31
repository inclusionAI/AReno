from __future__ import annotations

from types import SimpleNamespace

import torch

import areno.api.multimodal as multimodal
from areno.api.multimodal import encode_processor_messages, record_has_multimodal
from areno.engine.data.batch import to_device
from areno.engine.data.rollout_state import InferenceBatchState


class _NativeProcessor:
    image_token_id = 101
    audio_token_id = 102
    video_token_id = 103

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 101, 102, 103, 2]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 3, 2, 0]]),
            "pixel_values": torch.ones(1, 2, 3),
            "image_position_ids": torch.zeros(1, 2, 2, dtype=torch.long),
            "input_features": torch.ones(1, 4, 3),
            "input_features_mask": torch.ones(1, 4, dtype=torch.bool),
            "pixel_values_videos": torch.ones(1, 1, 2, 3),
            "video_position_ids": torch.zeros(1, 1, 2, 2, dtype=torch.long),
        }


def test_native_processor_normalizes_openai_media_parts():
    processor = _NativeProcessor()
    tokens, features = encode_processor_messages(
        processor,
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA=="}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
    )

    assert tokens == [1, 101, 102, 103, 2]
    assert features["modality_token_ids"] == {"image": 101, "video": 103, "audio": 102}
    parts = processor.messages[0]["content"]
    assert parts[0] == {"type": "image", "url": "data:image/png;base64,AA=="}
    assert parts[1] == {"type": "audio", "url": "data:audio/wav;base64,AA=="}
    assert parts[2] == {"type": "video", "url": "data:video/mp4;base64,AA=="}


def test_record_detects_media_inside_messages():
    assert record_has_multimodal(
        {"messages": [{"role": "user", "content": [{"type": "audio", "audio": "/tmp/a.wav"}]}]}
    )


def test_gemma4_torchvision_video_fps_is_backfilled(monkeypatch):
    from transformers import video_utils

    def read_video(*args, **kwargs):
        return torch.zeros(2, 3, 4, 4), torch.empty(0), {}

    fake_io = SimpleNamespace(
        read_video=read_video,
        read_video_timestamps=lambda *args, **kwargs: ([0, 1], 24.0),
    )
    processor_type = type("Gemma4Processor", (), {})
    processor_type.__module__ = "transformers.models.gemma4.processing_gemma4"
    monkeypatch.setattr(video_utils, "torchvision_io", fake_io, raising=False)
    monkeypatch.setattr(multimodal, "_VIDEO_FPS_PATCHED", False)

    multimodal._ensure_gemma4_torchvision_video_fps(processor_type())
    _, _, info = fake_io.read_video("capture.mp4")

    assert info["video_fps"] == 24.0


def test_torchvision_video_fps_uses_safe_default():
    fake_io = SimpleNamespace(read_video_timestamps=lambda *args, **kwargs: ([], None))

    assert multimodal._read_torchvision_fps(fake_io, "capture.mp4") == 30.0


def test_rollout_chunks_track_each_modality_offset():
    prompt = [1, 101, 101, 2, 102, 102, 102, 3, 103, 4]
    features = {
        "modality_token_ids": {"image": 101, "audio": 102, "video": 103},
        "pixel_values": torch.ones(1, 2, 3),
        "image_position_ids": torch.zeros(1, 2, 2, dtype=torch.long),
        "input_features": torch.ones(1, 4, 3),
        "input_features_mask": torch.ones(1, 4, dtype=torch.bool),
        "pixel_values_videos": torch.ones(1, 1, 2, 3),
        "video_position_ids": torch.zeros(1, 1, 2, 2, dtype=torch.long),
    }
    state = InferenceBatchState(
        [prompt],
        max_new_tokens=1,
        max_prefill_tokens=6,
        max_cache_len=32,
        kv_block_size=4,
        num_cache_blocks=8,
        prompt_features=[features],
    )

    first = state.build_prefill_payload()
    first_row = first["features"]["image_feature_rows"][0]
    assert first_row["modality_token_offsets"] == {"image": 0, "audio": 0, "video": 0}
    assert first_row["modality_token_counts"] == {"image": 2, "audio": 2, "video": 0}

    second = state.build_prefill_payload()
    second_row = second["features"]["image_feature_rows"][0]
    assert second_row["modality_token_offsets"] == {"image": 2, "audio": 2, "video": 0}
    assert second_row["modality_token_counts"] == {"image": 0, "audio": 1, "video": 1}


def test_gemma4_embedding_ids_replace_all_media_tokens():
    from areno.models.gemma4_utils import text_embedding_ids

    ids = torch.tensor([[1, 101, 102, 103, 2]])
    result = text_embedding_ids(ids, modality_token_ids=(101, 102, 103), pad_token_id=0)

    assert result.tolist() == [[1, 0, 0, 0, 2]]


def test_gemma4_frozen_media_modules_stay_in_eval_mode():
    from areno.models.gemma4_utils import keep_frozen_modules_in_eval

    modules = [torch.nn.Dropout(0.5), torch.nn.Linear(2, 2)]
    for module in modules:
        module.train()

    keep_frozen_modules_in_eval(modules)

    assert all(not module.training for module in modules)


def test_to_device_preserves_repeated_media_tensor_aliases():
    media = torch.ones(2, 3)

    moved = to_device([{"pixel_values": media}, {"pixel_values": media}], torch.device("meta"))

    assert moved[0]["pixel_values"] is moved[1]["pixel_values"]
