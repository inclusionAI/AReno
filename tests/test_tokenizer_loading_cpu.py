"""Serialized tokenizers must load independently of native model configs."""

import json

import pytest

from areno.engine.data.tokenizer import load_tokenizer


@pytest.mark.parametrize("tokenizer_class", ["PreTrainedTokenizerFast", "TokenizersBackend"])
def test_generic_fast_tokenizer_loads_phi_longrope_checkpoint(tmp_path, monkeypatch, tokenizer_class):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    expected = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    expected.chat_template = "{{ messages[0]['content'] }}"
    expected.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = tokenizer_class
    config_path.write_text(json.dumps(config))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "phi4mm",
                "max_position_embeddings": 4096,
                "rope_scaling": {"type": "longrope", "short_factor": [1.0], "long_factor": [1.0]},
            }
        )
    )

    def broken_auto(*args, **kwargs):
        raise AttributeError("'PreTrainedConfig' object has no attribute 'max_position_embeddings'")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", broken_auto)
    actual = load_tokenizer(tmp_path)
    assert actual.encode("hello") == expected.encode("hello") == [2]
    assert actual.eos_token_id == expected.eos_token_id
    assert actual.chat_template == expected.chat_template


@pytest.mark.parametrize(
    "metadata",
    [
        {"tokenizer_class": "CustomTokenizer"},
        {"tokenizer_class": "PreTrainedTokenizerFast", "auto_map": {"AutoTokenizer": ["custom.Slow", "custom.Fast"]}},
        {"tokenizer_class": "TokenizersBackend", "auto_map": ["custom.Slow", "custom.Fast"]},
    ],
)
def test_custom_tokenizer_dispatch_still_uses_auto(tmp_path, monkeypatch, metadata):
    from transformers import AutoTokenizer

    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    (tmp_path / "tokenizer.json").write_text("{}")
    sentinel = object()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: sentinel)
    assert load_tokenizer(tmp_path) is sentinel
