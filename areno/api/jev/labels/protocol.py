"""Text-only SGLang 297e54f decision prompt format v1 (no engine dependency)."""

import hashlib
import json
import math
import string
from itertools import product
from pathlib import Path

from areno.api.models import TrainSequence

ENGINE_REVISION = "297e54fd105e974cdf401e5b4d20253a22c16f90"
PROMPT_VERSION = 1
MAX_CHOICE_OPTIONS = 255
_PAIR_LABELS = tuple(a + b for a, b in product(string.ascii_uppercase, repeat=2))


def render_text(value):
    return (
        value
        if isinstance(value, str)
        else ""
        if value is None
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )


def _chat_prompt(tokenizer, content, system):
    messages = [{"role": "system", "content": system}] if system is not None else []
    messages.append({"role": "user", "content": content})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _label_context(tokenizer, prompt, ids):
    # Match SGLang: an added token isolates the short answer suffix from the input.
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    added_tokens = {token_id: token for token, token_id in get_added_vocab().items()} if get_added_vocab else {}
    for index in range(len(ids) - 1, -1, -1):
        token = added_tokens.get(ids[index])
        if token is not None:
            position = prompt.rfind(token)
            if position >= 0:
                suffix = prompt[position + len(token) :]
                suffix_ids = ids[index + 1 :]
                if tokenizer.encode(suffix, add_special_tokens=False) == suffix_ids:
                    return suffix, suffix_ids
            break
    return None


def _label_token_id(tokenizer, text, ids, label):
    extended = tokenizer.encode(text + label, add_special_tokens=False)
    if len(extended) == len(ids) + 1 and extended[:-1] == ids:
        return extended[-1]
    return None


def _pair_labels(tokenizer, count, system):
    if tokenizer is None:
        raise ValueError("choice with more than 26 options requires a tokenizer")
    contexts = []
    for message in ("x", "y"):
        prompt = _chat_prompt(tokenizer, message, system)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        contexts.append(_label_context(tokenizer, prompt, ids))
    context, other = contexts
    if context is None or context != other:
        raise ValueError(
            "choice with more than 26 options requires an isolated chat answer suffix after an added token"
        )

    text, ids = context
    labels, seen = [], set()
    for label in _PAIR_LABELS:
        token_id = _label_token_id(tokenizer, text, ids, label)
        if token_id is not None and token_id not in seen:
            labels.append(label)
            seen.add(token_id)
            if len(labels) == count:
                return labels
    raise ValueError(
        f"choice has {count} options, but this tokenizer and chat template support only {max(26, len(labels))}"
    )


def render_question(state, question, *, tokenizer=None, system=None):
    kind, criteria = question["type"], question.get("criteria")
    text = render_text(question.get("instructions"))
    if kind == "choice":
        names = list(criteria)
        count = len(names)
        if not 2 <= count <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice requires 2–{MAX_CHOICE_OPTIONS} options")
        labels = list(string.ascii_uppercase[:count]) if count <= 26 else _pair_labels(tokenizer, count, system)
        lines = [f"Question: {text}"] if text.strip() else []
        for label, name in zip(labels, names, strict=True):
            detail = render_text(criteria[name])
            lines.append(f"{label}: {name} - {detail}" if detail else f"{label}: {name}")
        lines.append("Answer with the letter of one option only.")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("score requires 2–10 levels")
        names = labels = [str(i) for i in range(len(criteria))]
        lines = [f"Question: {text}"] if text.strip() else []
        lines.extend(f"{label}: {render_text(level)}" for label, level in zip(labels, criteria, strict=True))
        lines.append("Answer with the number of one level only.")
    elif kind == "noul":
        names, labels = ["true", "false"], ["yes", "no"]
        lines = [f"Is the following true? {text}" if text.strip() else "Is the following true?"]
        for name, label in zip(names, labels, strict=True):
            detail = render_text((criteria or {}).get(name))
            if detail:
                lines.append(f"{label}: {detail}")
        lines.append("Answer with yes or no only.")
    else:
        raise ValueError(f"unsupported question type: {kind}")
    return "\n".join([render_text(state), "", *lines]), names, labels


def encode_question(state, question, tokenizer, *, target=None, max_seq_len=4096, system=None):
    content, names, labels = render_question(state, question, tokenizer=tokenizer, system=system)
    prompt = _chat_prompt(tokenizer, content, system)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_seq_len:
        raise ValueError(
            "prompt exceeds max_seq_len; v1 rejects truncation to preserve all options and answer boundary"
        )
    text, prefix_ids = prompt, ids
    if question["type"] == "choice" and len(labels) > 26:
        context = _label_context(tokenizer, prompt, ids)
        if context is None:
            raise ValueError("choice with more than 26 options requires an isolated chat answer suffix")
        text, prefix_ids = context
    label_ids, seen = [], set()
    for label in labels:
        token_id = _label_token_id(tokenizer, text, prefix_ids, label)
        if token_id is None or token_id in seen:
            raise ValueError(f"label {label!r} must add one distinct token at the answer boundary")
        label_ids.append(token_id)
        seen.add(token_id)
    if target is None:
        distribution = None
    elif isinstance(target, dict):
        distribution = [float(target[name]) for name in names]
    else:
        name = str(target).lower() if isinstance(target, bool) else str(target)
        if name not in names:
            raise ValueError("hard target is not a candidate name")
        distribution = [float(name == candidate) for candidate in names]
    if distribution is not None and (
        any(not math.isfinite(p) or p < 0 for p in distribution) or not math.isclose(sum(distribution), 1, abs_tol=1e-5)
    ):
        raise ValueError("target must be a finite probability distribution summing to one")
    return {
        "prompt_token_ids": ids,
        "label_token_ids": label_ids,
        "names": names,
        "labels": labels,
        "target": distribution,
    }


def training_row(encoded, *, temperature=1.0, bi_attn=False):
    if not math.isfinite(temperature) or temperature <= 0 or encoded["target"] is None:
        raise ValueError("training needs a target and finite positive temperature")
    return TrainSequence(
        tokens=encoded["prompt_token_ids"],
        features={
            "label_jev": {
                "ids": encoded["label_token_ids"],
                "target": encoded["target"],
                "temperature": temperature,
                "bi_attn": bi_attn,
            }
        },
    )


def save_protocol(path, tokenizer, *, temperature, max_seq_len, bi_attn=False):
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "areno-label-jev-v1",
        "prompt_version": PROMPT_VERSION,
        "sglang_revision": ENGINE_REVISION,
        "temperature": temperature,
        "max_seq_len": max_seq_len,
        "truncation": "reject",
        "bi_attn": bool(bi_attn),
        "attention_scope": "full_attention_only",
        "thinking": False,
        "choice": "A–Z up to 26; tokenizer-supported two-letter labels in criteria insertion order up to 255",
        "noul": {"yes": "true", "no": "false"},
        "score": "0–9 in level order",
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "resume": "adapter weights only; optimizer/RNG/data cursor not restored",
    }
    (destination / "label_jev_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    config_path = destination / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config.get("text_config", config)["is_causal"] = not bi_attn
        config_path.write_text(json.dumps(config, indent=2) + "\n")


def corpus_record(row):
    """Convert the flat distill-v3 schema without scalar-head serialization."""
    kind = row["kind"]
    options = [str(value).lower() if isinstance(value, bool) else str(value) for value in row["options"]]
    if kind == "choice":
        criteria = {name: name for name in options}
        target = dict(zip(options, row["target"], strict=True))
    elif kind == "score":
        criteria = options
        target = {str(i): value for i, value in enumerate(row["target"])}
    elif kind == "noul":
        criteria = None
        mapping = {"true": "true", "yes": "true", "1": "true", "false": "false", "no": "false", "0": "false"}
        if (
            len(options) != 2
            or any(name not in mapping for name in options)
            or len({mapping[name] for name in options}) != 2
        ):
            raise ValueError("noul options must identify distinct true and false outcomes")
        target = {mapping[name]: value for name, value in zip(options, row["target"], strict=True)}
    else:
        raise ValueError(f"unsupported corpus question type: {kind}")
    return {
        "id": row["id"],
        "split": row.get("split", "train"),
        "request": {
            "state": row["state"],
            "questions": {"decision": {"type": kind, "instructions": row["question"], "criteria": criteria}},
        },
        "targets": {"decision": target},
    }
