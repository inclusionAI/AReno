"""Model-aware tool-call parsers for the agentic OpenAI proxy.

This follows the same shape as SGLang's function-call parser layer: a small
registry maps parser names to model/template-specific detectors, and the proxy
selects one parser once per rollout session. The parser turns model-native
tool-call text into OpenAI-compatible ``message.tool_calls`` objects.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class ToolCallParseResult:
    """Parsed tool-call output plus leftover assistant text."""

    normal_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class ToolCallParser(Protocol):
    """Protocol implemented by model-specific tool-call parsers."""

    name: str

    def parse(self, content: str, tools: list[dict[str, Any]], tool_choice: Any) -> ToolCallParseResult:
        """Parse generated text into OpenAI-compatible tool calls."""


_PARSERS: dict[str, type[ToolCallParser]] = {}


def register_tool_call_parser(name: str, parser_cls: type[ToolCallParser]) -> None:
    """Register a parser by public name."""

    _PARSERS[name] = parser_cls


def get_tool_call_parser(name: str | None) -> ToolCallParser:
    """Return a parser instance, falling back to the generic JSON parser."""

    parser_cls = _PARSERS.get((name or "json").lower(), JsonToolCallParser)
    return parser_cls()


def infer_tool_call_parser_name(trainer: Any) -> str:
    """Infer the parser from model path and tokenizer template."""

    tokenizer = _safe_tokenizer(trainer)
    template = str(getattr(tokenizer, "chat_template", "") or "").lower()
    model_path = str(
        getattr(trainer, "_model_path", "")
        or getattr(getattr(trainer, "_ctx", None), "model_path", "")
        or getattr(tokenizer, "name_or_path", "")
        or ""
    ).lower()
    model_type = _read_model_type(model_path)
    haystack = " ".join([template, model_path, model_type])

    if "<|tool_call>" in template or "gemma4" in haystack or "gemma-4" in haystack:
        return "gemma4"
    if "minicpm" in haystack:
        return "minicpm"
    if "<tool_call>" in template or "qwen" in haystack:
        return "qwen"
    return "json"


class JsonToolCallParser:
    """Generic JSON parser for Llama-style and schema-only tool calls."""

    name = "json"

    def parse(self, content: str, tools: list[dict[str, Any]], tool_choice: Any) -> ToolCallParseResult:
        if not tools:
            return ToolCallParseResult(normal_text=content)
        chosen_name = _chosen_tool_name(tools, tool_choice)
        calls = _parse_json_tool_calls(content, tools, chosen_name)
        if calls:
            return ToolCallParseResult(normal_text="", tool_calls=calls)

        if chosen_name is None:
            return ToolCallParseResult(normal_text=content)
        args = _parse_explicit_arguments_text(content, tools, chosen_name)
        if args is None:
            return ToolCallParseResult(normal_text=content)
        return ToolCallParseResult(normal_text="", tool_calls=[_openai_tool_call(chosen_name, args)])


class QwenToolCallParser(JsonToolCallParser):
    """Parser for Qwen-family ``<tool_call> ... </tool_call>`` output."""

    name = "qwen"
    _block_re = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def parse(self, content: str, tools: list[dict[str, Any]], tool_choice: Any) -> ToolCallParseResult:
        calls: list[dict[str, Any]] = []
        for block in self._block_re.findall(content):
            calls.extend(_parse_json_tool_calls(block, tools, _chosen_tool_name(tools, tool_choice)))
            calls.extend(_parse_angle_tool_calls(block, tools, tool_choice))
        if calls:
            normal = content[: content.find("<tool_call>")].strip()
            return ToolCallParseResult(normal_text=normal, tool_calls=calls)
        return super().parse(content, tools, tool_choice)


class Gemma4ToolCallParser(JsonToolCallParser):
    """Parser for Gemma4 ``<|tool_call>call:name{args}<tool_call|>`` output."""

    name = "gemma4"
    _start = "<|tool_call>"
    _end = "<tool_call|>"

    def parse(self, content: str, tools: list[dict[str, Any]], tool_choice: Any) -> ToolCallParseResult:
        calls: list[dict[str, Any]] = []
        search_from = 0
        while True:
            start = content.find(self._start, search_from)
            if start == -1:
                break
            end = content.find(self._end, start)
            if end == -1:
                break
            inner = content[start + len(self._start) : end].strip()
            call = _parse_gemma4_call(inner)
            if call is not None:
                name, args = call
                if _tool_name_allowed(name, tools, tool_choice):
                    calls.append(_openai_tool_call(name, args))
            search_from = end + len(self._end)
        if calls:
            normal = content[: content.find(self._start)].strip()
            return ToolCallParseResult(normal_text=normal, tool_calls=calls)
        return super().parse(content, tools, tool_choice)


class MiniCPMToolCallParser(JsonToolCallParser):
    """Parser for MiniCPM XML-style function calls."""

    name = "minicpm"
    _function_re = re.compile(r"<function\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</function>", re.DOTALL | re.IGNORECASE)
    _param_re = re.compile(r"<param\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</param>", re.DOTALL | re.IGNORECASE)

    def parse(self, content: str, tools: list[dict[str, Any]], tool_choice: Any) -> ToolCallParseResult:
        calls: list[dict[str, Any]] = []
        calls.extend(_parse_angle_tool_call_blocks(content, tools, tool_choice))
        for name, body in self._function_re.findall(content):
            if not _tool_name_allowed(name, tools, tool_choice):
                continue
            args = {
                key.strip(): _parse_minicpm_param_value(value.strip()) for key, value in self._param_re.findall(body)
            }
            calls.append(_openai_tool_call(name.strip(), args))
        if calls:
            first_tool = _first_nonnegative(content.find("<tool_call>"), content.find("<function"))
            normal = content[:first_tool].strip() if first_tool is not None else ""
            return ToolCallParseResult(normal_text=normal, tool_calls=calls)
        return super().parse(content, tools, tool_choice)


def _safe_tokenizer(trainer: Any) -> Any:
    try:
        return trainer.get_tokenizer()
    except Exception:
        return None


def _read_model_type(model_path: str) -> str:
    if not model_path:
        return ""
    path = Path(model_path)
    config_path = path / "config.json"
    if not config_path.exists():
        return ""
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    model_type = str(data.get("model_type", "")).lower()
    text_type = str((data.get("text_config") or {}).get("model_type", "")).lower()
    architectures = " ".join(str(item).lower() for item in data.get("architectures") or [])
    return " ".join(part for part in (model_type, text_type, architectures) if part)


def _parse_json_tool_calls(content: str, tools: list[dict[str, Any]], chosen_name: str | None) -> list[dict[str, Any]]:
    obj = _load_json_object(content)
    if obj is None:
        return []
    raw_calls: list[Any]
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        raw_calls = obj["tool_calls"]
    elif isinstance(obj, list):
        raw_calls = obj
    else:
        raw_calls = [obj]

    calls: list[dict[str, Any]] = []
    for raw in raw_calls:
        parsed = _parse_json_call_object(raw, chosen_name)
        if parsed is None:
            continue
        name, args = parsed
        if _tool_name_allowed(name, tools, _forced_tool_choice(chosen_name)):
            calls.append(_openai_tool_call(name, args))
    return calls


def _parse_json_call_object(obj: Any, chosen_name: str | None) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(obj, dict):
        return None
    function = obj.get("function")
    if isinstance(function, dict):
        name = function.get("name") or obj.get("name") or chosen_name
        args = function.get("arguments", obj.get("arguments", {}))
    else:
        name = obj.get("name") or chosen_name
        args = obj.get("arguments") if "arguments" in obj else obj
    if not isinstance(name, str) or not name:
        return None
    if isinstance(args, str):
        loaded = _load_json_object(args)
        if not isinstance(loaded, dict):
            return None
        args = loaded
    if not isinstance(args, dict):
        return None
    return name, dict(args)


def _load_json_object(text: str) -> Any | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Some models emit prose before/after a JSON object. Only accept a balanced
    # JSON-looking span, never arbitrary text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _parse_explicit_arguments_text(content: str, tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    enum_arg = _single_enum_argument(tools, name)
    if enum_arg is None:
        return None
    key, values = enum_arg
    parsed = _single_enum_value_from_text(content, key, values)
    if parsed is None:
        return None
    return {key: parsed}


def _chosen_tool_name(tools: list[dict[str, Any]], tool_choice: Any) -> str | None:
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
    if len(tools) == 1:
        function = _tool_function(tools[0])
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
    return None


def _forced_tool_choice(name: str | None) -> dict[str, Any] | None:
    if name is None:
        return None
    return {"type": "function", "function": {"name": name}}


def _tool_name_allowed(name: str, tools: list[dict[str, Any]], tool_choice: Any) -> bool:
    chosen = _chosen_tool_name(tools, tool_choice)
    if chosen is not None:
        return name == chosen
    return name in _tool_names(tools)


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names = set()
    for tool in tools:
        function = _tool_function(tool)
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def _single_enum_argument(tools: list[dict[str, Any]], name: str) -> tuple[str, list[Any]] | None:
    for tool in tools:
        function = _tool_function(tool)
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        if not isinstance(properties, dict):
            return None
        enum_fields = []
        for key, spec in properties.items():
            enum = spec.get("enum") if isinstance(spec, dict) else None
            if isinstance(enum, list) and enum:
                enum_fields.append((key, enum))
        return enum_fields[0] if len(enum_fields) == 1 else None
    return None


def _tool_function(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Return an OpenAI function schema from chat or flat tool syntax."""

    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if isinstance(function, dict):
        return function
    if isinstance(tool.get("name"), str):
        return {
            "name": tool["name"],
            "description": tool.get("description"),
            "parameters": tool.get("parameters"),
        }
    return None


def _single_enum_value_from_text(text: str, key: str, values: list[Any]) -> Any | None:
    value_by_lower = {str(value).lower(): value for value in values}
    if not value_by_lower:
        return None
    alternatives = "|".join(re.escape(value) for value in value_by_lower)
    patterns = [
        rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*["\']?\b({alternatives})\b',
        rf"<(?:{re.escape(key)}|move|action)>\s*({alternatives})\s*</(?:{re.escape(key)}|move|action)>",
        rf"\b(?:choose|select|play|move|slide|go|answer|direction)\s*(?:is|:|=|to)?\s*\b({alternatives})\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return value_by_lower[str(matches[-1]).lower()]
    return None


def _parse_gemma4_call(inner: str) -> tuple[str, dict[str, Any]] | None:
    if not inner.startswith("call:"):
        return None
    brace = inner.find("{")
    if brace == -1:
        return None
    name = inner[len("call:") : brace].strip()
    args_text = inner[brace + 1 :]
    end = _find_matching_brace(args_text)
    if end != -1:
        args_text = args_text[:end]
    return name, _parse_gemma4_args(args_text)


def _find_matching_brace(text: str) -> int:
    depth = 1
    i = 0
    while i < len(text) and depth > 0:
        if text.startswith('<|"|>', i):
            i += len('<|"|>')
            end = text.find('<|"|>', i)
            if end == -1:
                return -1
            i = end + len('<|"|>')
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1


def _parse_gemma4_args(args_text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in _split_gemma4_top_level(args_text.strip()):
        if ":" not in part:
            continue
        key, value = _split_gemma4_key_value(part)
        if key:
            result[key] = _parse_gemma4_value(value)
    return result


def _parse_gemma4_value(value: str) -> Any:
    value = value.strip()
    if value.startswith('<|"|>') and value.endswith('<|"|>'):
        return value[len('<|"|>') : -len('<|"|>')]
    if value.startswith("{") and value.endswith("}"):
        return _parse_gemma4_args(value[1:-1])
    if value.startswith("[") and value.endswith("]"):
        return [_parse_gemma4_value(item) for item in _split_gemma4_top_level(value[1:-1])]
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _split_gemma4_key_value(part: str) -> tuple[str, str]:
    key, value = part.split(":", 1)
    return _strip_gemma4_key(key), value.strip()


def _strip_gemma4_key(key: str) -> str:
    key = key.strip()
    if key.startswith('<|"|>') and key.endswith('<|"|>'):
        return key[len('<|"|>') : -len('<|"|>')]
    return key.strip("\"'")


def _split_gemma4_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    i = 0
    while i < len(text):
        if text.startswith('<|"|>', i):
            i += len('<|"|>')
            end = text.find('<|"|>', i)
            if end == -1:
                break
            i = end + len('<|"|>')
            continue
        char = text[i]
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            part = text[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_minicpm_param_value(value: str) -> Any:
    value = re.sub(r"<\|[^>]+?\|>", "", value).strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


_ANGLE_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_ANGLE_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>\s*(.*?)</function>", re.DOTALL | re.IGNORECASE)
_ANGLE_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)</parameter>", re.DOTALL | re.IGNORECASE)


def _parse_angle_tool_call_blocks(content: str, tools: list[dict[str, Any]], tool_choice: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in _ANGLE_TOOL_BLOCK_RE.findall(content):
        calls.extend(_parse_angle_tool_calls(block, tools, tool_choice))
    return calls


def _parse_angle_tool_calls(content: str, tools: list[dict[str, Any]], tool_choice: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for name, body in _ANGLE_FUNCTION_RE.findall(content):
        if not _tool_name_allowed(name, tools, tool_choice):
            continue
        args = {key.strip(): _parse_minicpm_param_value(value.strip()) for key, value in _ANGLE_PARAM_RE.findall(body)}
        calls.append(_openai_tool_call(name.strip(), args))
    return calls


def _first_nonnegative(*values: int) -> int | None:
    matches = [value for value in values if value >= 0]
    return min(matches) if matches else None


def _openai_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


register_tool_call_parser("json", JsonToolCallParser)
register_tool_call_parser("qwen", QwenToolCallParser)
register_tool_call_parser("qwen3", QwenToolCallParser)
register_tool_call_parser("qwen3_5", QwenToolCallParser)
register_tool_call_parser("qwen3_5_moe", QwenToolCallParser)
register_tool_call_parser("qwen3_moe", QwenToolCallParser)
register_tool_call_parser("minicpm", MiniCPMToolCallParser)
register_tool_call_parser("minicpmv46", MiniCPMToolCallParser)
register_tool_call_parser("gemma4", Gemma4ToolCallParser)
register_tool_call_parser("llama", JsonToolCallParser)
register_tool_call_parser("bailing_moe_linear_v2", JsonToolCallParser)
