"""Write tokenizer assets without a converter or a tokenizer download."""

from __future__ import annotations

import json
import re
from pathlib import Path

QWEN_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
LLAMA3_REGEX = QWEN_REGEX.replace(r"|\p{N}|", r"|\p{N}{1,3}|")
QWEN35_REGEX = QWEN_REGEX.replace(r"?\p{L}+", r"?[\p{L}\p{M}]+").replace(r"\p{L}\p{N}]+", r"\p{L}\p{M}\p{N}]+")
BAILING_REGEX = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)
QWEN_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{% endif %}"
)


def _bpe_profile(tokenizer: dict, required: str) -> str:
    pre = tokenizer.get("pre_tokenizer", {})
    if pre.get("type") == "ByteLevel" and pre.get("use_regex", True) and not pre.get("add_prefix_space", False):
        profile = "olmo" if required == "olmo" else "gpt-2"
    else:
        parts = pre.get("pretokenizers", [])
        if not (
            pre.get("type") == "Sequence"
            and len(parts) == 2
            and parts[0].get("type") == "Split"
            and parts[0].get("behavior") == "Isolated"
            and not parts[0].get("invert", False)
            and parts[1].get("type") == "ByteLevel"
            and not parts[1].get("add_prefix_space", False)
            and not parts[1].get("use_regex", True)
        ):
            raise ValueError("unrecognized BPE pre-tokenizer; refusing to change tokenization")
        pattern = parts[0].get("pattern", {}).get("Regex")
        profiles = {
            QWEN_REGEX: "qwen2",
            LLAMA3_REGEX: "llama-bpe",
            QWEN35_REGEX: "qwen35",
            BAILING_REGEX: "bailingmoe2",
        }
        if pattern not in profiles:
            raise ValueError("unrecognized BPE pre-tokenizer regex")
        profile = profiles[pattern]
    allowed = {"qwen2", "qwen35"} if required == "qwen35" else {required}
    if tokenizer.get("normalizer") not in (None, {"type": "NFC"}) or required != "auto" and profile not in allowed:
        raise ValueError(f"unsupported tokenizer normalization or pre-tokenizer for {required}")
    return profile


def write_tokenizer(writer, model_path: Path, architecture) -> str:
    import gguf

    config = architecture.config
    count = int(config["vocab_size"])
    data = json.loads((model_path / "tokenizer.json").read_text())
    model = data["model"]
    tokens = [f"[PAD{index}]" for index in range(count)]
    types = [int(gguf.TokenType.UNUSED)] * count
    scores = [-1000.0] * count
    sentencepiece = model.get("byte_fallback", False) or model["type"] == "Unigram"
    if model["type"] not in {"BPE", "Unigram"}:
        raise ValueError(f"unsupported GGUF tokenizer model: {model['type']}")
    if sentencepiece:
        if architecture.tokenizer_profile not in {"auto", "gemma4"}:
            raise ValueError("this architecture requires a byte-level BPE tokenizer")
        tokenizer_model = "gemma4" if architecture.tokenizer_profile == "gemma4" else "llama"
        writer.add_tokenizer_pre("default")
        entries = (
            enumerate(model["vocab"])
            if model["type"] == "Unigram"
            else ((index, (token, -1000.0)) for token, index in model["vocab"].items())
        )
    else:
        tokenizer_model = "gpt2"
        writer.add_tokenizer_pre(_bpe_profile(data, architecture.tokenizer_profile))
        if data.get("normalizer") == {"type": "NFC"}:
            writer.add_bool("tokenizer.ggml.normalizer.nfc", True)
        entries = ((index, (token, -1000.0)) for token, index in model["vocab"].items())
    for index, (token, score) in entries:
        if not 0 <= index < count:
            raise ValueError(f"token id {index} is outside model vocabulary")
        tokens[index], scores[index] = token, float(score)
        types[index] = int(
            gguf.TokenType.BYTE
            if sentencepiece and re.fullmatch(r"<0x[0-9a-fA-F]{2}>", token)
            else gguf.TokenType.NORMAL
        )
        if index == model.get("unk_id"):
            types[index] = int(gguf.TokenType.UNKNOWN)
    for token in data.get("added_tokens", []):
        index = int(token["id"])
        if not 0 <= index < count:
            raise ValueError(f"added token id {index} is outside model vocabulary")
        tokens[index] = token["content"]
        if types[index] != int(gguf.TokenType.BYTE):
            types[index] = int(gguf.TokenType.CONTROL if token.get("special") else gguf.TokenType.USER_DEFINED)
    writer.add_tokenizer_model(tokenizer_model)
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    if sentencepiece:
        writer.add_token_scores(scores)
    special = gguf.SpecialVocab(model_path, load_merges=not sentencepiece, n_vocab=count)
    if not sentencepiece and not special.merges:
        raise ValueError("BPE tokenizer has no merges")
    if architecture.thinking_off:
        special.chat_template = QWEN_TEMPLATE
        special.add_special_token.update(bos=False, eos=False)
    special.add_to_gguf(writer)
    if architecture.tokenizer_profile == "gemma4":
        writer.add_add_space_prefix(False)
        writer.add_add_bos_token(True)
    return "text, thinking disabled" if architecture.thinking_off else "source tokenizer chat template"


def modelfile(architecture) -> str:
    result = "FROM ./model.gguf\n"
    if architecture.thinking_off:
        result += (
            'TEMPLATE """{{ range .Messages }}<|im_start|>{{ .Role }}\n'
            '{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n<think>\n\n</think>\n\n"""\n'
            'PARAMETER stop "<|im_end|>"\nPARAMETER stop "<|endoftext|>"\n'
        )
    return result + f"PARAMETER num_ctx {min(architecture.context_length, 4096)}\n"
