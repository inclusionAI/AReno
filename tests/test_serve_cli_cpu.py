from __future__ import annotations

from types import SimpleNamespace

from areno.cli import serve as serve_mod


def test_create_app_passes_eager_decode_runtime_config(monkeypatch):
    captured = {}

    class FakeEngine:
        config = SimpleNamespace(model=SimpleNamespace(max_position_embeddings=1024))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args
            captured["runtime_config"] = kwargs["runtime_config"]
            return cls()

    monkeypatch.setattr(serve_mod, "load_tokenizer", lambda model_path: SimpleNamespace(eos_token_id=1))
    monkeypatch.setattr(serve_mod, "ArenoEngine", FakeEngine)

    serve_mod.create_app(
        model_path="model",
        tp_size=1,
        world_size=1,
        max_running_prompts=4,
        default_max_tokens=16,
        decode_progress_interval_s=0.0,
        eager_decode=True,
    )

    assert captured["runtime_config"].eager_decode is True
