"""Tokenizer 对齐检查器 (alignment inspector)。

本模块提供聚焦、可独立评审的 tokenizer 对齐检查能力:把原始片段与
token IDs、token pieces、special-token 标记、EOS 位置、roles、loss-mask
spans 并排展示,并把 tokenizer 词表大小与模型 ``config.json`` 的
``vocab_size`` 对齐比较。

设计原则:
- 纯函数 + 不可变 dataclass,便于 CPU 单测,不依赖 torch/engine。
- 复用 ``areno.api.tokenizer`` 的现有编码契约(``encode_generation_prompt``、
  ``apply_chat_template_with_options``、``eos_token_ids`` 等),不重造 tokenizer 路径。
- 只读:绝不修改 tokenizer 对象的加载路径或持久配置;仅在用户显式传入
  ``enable_thinking`` 时才会在内存中设置对应属性(与现有 API 行为一致)。
- model 侧 ``vocab_size`` 通过内联读取 ``config.json`` 获取(参考
  ``areno/api/tokenizer.py`` 中读取 EOS 的模式),避免引入 ``areno.models.registry``
  这类会拉起 torch/engine 的重链。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from areno.api.tokenizer import (
    apply_chat_template_with_options,
    configure_chat_template_enable_thinking,
    eos_token_ids,
    normalize_token_ids,
)

# 对齐状态标签:OK=一致,FAIL=不一致(需关注),SKIP=无法比较(缺数据)
ALIGN_OK = "OK"
ALIGN_FAIL = "FAIL"
ALIGN_SKIP = "SKIP"


@dataclass(frozen=True)
class Segment:
    """单个 token 的对齐视图。

    ``text`` / ``pieces`` 取自 ``convert_ids_to_tokens``;``is_special`` 标记
    special token(如 BOS/EOS/PAD/模板控制符);``is_eos`` 标记停顿 token;
    ``role`` / ``loss_mask`` 来自所属 chat turn(纯 prompt 时为 None / False)。
    """

    text: str
    token_ids: list[int]
    pieces: list[str]
    role: str | None
    is_special: bool
    loss_mask: bool
    is_eos: bool

    def to_dict(self) -> dict[str, Any]:
        # 序列化为 JSON 友好的 dict(供 CLI --json 输出)
        return {
            "text": self.text,
            "token_ids": list(self.token_ids),
            "pieces": list(self.pieces),
            "role": self.role,
            "is_special": self.is_special,
            "loss_mask": self.loss_mask,
            "is_eos": self.is_eos,
        }


@dataclass(frozen=True)
class RoundTrip:
    """encode→decode 还原比对结果。

    ``ok`` 为 False 时 ``diff_note`` 描述差异类型(空白/长度/内容),不回显
    完整原文,避免在日志中泄露训练样本。
    """

    ok: bool
    decoded: str
    diff_note: str

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "decoded": self.decoded, "diff_note": self.diff_note}


@dataclass(frozen=True)
class VocabAlignment:
    """tokenizer 词表与模型 ``config.json`` 词表的对齐结果。"""

    status: str
    tokenizer_size: int | None
    model_vocab_size: int | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "tokenizer_size": self.tokenizer_size,
            "model_vocab_size": self.model_vocab_size,
            "note": self.note,
        }


@dataclass(frozen=True)
class InspectionReport:
    """一次对齐检查的完整结果。"""

    kind: str  # "prompt" | "messages" | "tool_call"
    raw: str  # 用于 round-trip 比对的原始文本
    segments: list[Segment]
    eos_positions: list[int]
    round_trip: RoundTrip
    vocab_alignment: VocabAlignment
    truncated: bool
    has_unknown: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "raw": self.raw,
            "segments": [s.to_dict() for s in self.segments],
            "eos_positions": list(self.eos_positions),
            "round_trip": self.round_trip.to_dict(),
            "vocab_alignment": self.vocab_alignment.to_dict(),
            "truncated": self.truncated,
            "has_unknown": self.has_unknown,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #


def inspect_prompt(
    tokenizer,
    prompt: str,
    *,
    model_path: str | Path | None = None,
    max_length: int | None = None,
) -> InspectionReport:
    """检查一段 plain prompt 的 token 对齐。

    直接 ``tokenizer.encode`` 编码(不套 chat 模板),与 chat messages 路径区分;
    这样 round-trip 比对的是纯分词还原而非模板渲染文本。纯 prompt 不属于任何
    turn,因此所有 token 的 ``role`` 为 None、``loss_mask`` 为 False。
    """

    # 直接 encode,不套 chat 模板;plain prompt 与 chat messages 路径区分。
    ids = normalize_token_ids(tokenizer.encode(prompt))
    return _build_report(
        kind="prompt",
        raw_text=prompt,
        tokenizer=tokenizer,
        ids=ids,
        model_path=model_path,
        max_length=max_length,
        role_by_index={},
    )


def inspect_messages(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    model_path: str | Path | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
    max_length: int | None = None,
) -> InspectionReport:
    """检查 chat messages 的 token 对齐,标注每段 turn 的 role 与 loss-mask。

    role / loss_mask 通过**前缀差分法**确定:对每个前缀 ``messages[:i]`` 渲染
    一次模板,相邻前缀的 token 长度差即为第 i 条消息贡献的 token 区间。
    ``loss_mask`` 仅对 ``assistant`` turn 为 True(计入训练 loss);末尾由
    ``add_generation_prompt`` 追加的 assistant 头部段 role 为 None、不计 loss。
    """

    # 仅当用户显式指定 thinking 开关时才设置内存属性;None 保持默认,不改 tokenizer。
    configure_chat_template_enable_thinking(tokenizer, enable_thinking)

    full_ids, role_by_index = _message_roles(tokenizer, messages, add_generation_prompt)
    raw_text = _render_text(tokenizer, messages, add_generation_prompt)
    return _build_report(
        kind="messages",
        raw_text=raw_text,
        tokenizer=tokenizer,
        ids=full_ids,
        model_path=model_path,
        max_length=max_length,
        role_by_index=role_by_index,
        # messages 的 raw 是含 special 字面的模板文本,需保留 special 还原比对。
        round_trip_skip_special=False,
    )


def inspect_tool_call(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    model_path: str | Path | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = None,
    max_length: int | None = None,
) -> InspectionReport:
    """检查 tool-call 场景的 token 对齐。

    这是 ``inspect_messages`` 的便利包装:期望 ``messages`` 中包含带 ``tool_calls``
    的 assistant turn 以及 ``tool`` 角色的工具返回。``kind`` 标记为 ``tool_call``,
    默认不追加 generation prompt(工具调用通常已自包含)。
    """

    report = inspect_messages(
        tokenizer,
        messages,
        model_path=model_path,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        max_length=max_length,
    )
    return replace(report, kind="tool_call")


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #


def _build_report(
    *,
    kind: str,
    raw_text: str,
    tokenizer,
    ids: list[int],
    model_path: str | Path | None,
    max_length: int | None,
    role_by_index: dict[int, tuple[str | None, bool]],
    round_trip_skip_special: bool = True,
) -> InspectionReport:
    """组装 InspectionReport:截断、分段、EOS、round-trip、vocab 对齐。"""

    warnings: list[str] = []
    truncated = False
    used_ids = list(ids)

    # 截断探测:仅在用户给出 max_length 且超长时触发,展示裁剪到 max_length 的前缀。
    if max_length is not None and len(used_ids) > max_length:
        truncated = True
        dropped = len(used_ids) - max_length
        used_ids = used_ids[:max_length]
        warnings.append(f"input truncated to max_length={max_length}, dropped {dropped} trailing tokens")

    eos_set = _eos_set(tokenizer, model_path)
    special_set = set(getattr(tokenizer, "all_special_ids", []) or [])
    segments = _segments_from_ids(tokenizer, used_ids, role_by_index, eos_set, special_set)

    eos_positions = [idx for idx, seg in enumerate(segments) if seg.is_eos]
    has_unknown = _has_unknown(tokenizer, used_ids)
    round_trip = _decode_round_trip(tokenizer, used_ids, raw_text, skip_special=round_trip_skip_special)
    vocab_alignment = _vocab_alignment(tokenizer, model_path)

    # 词表不一致是值得关注的对齐问题,作为 warning 显式提示。
    if vocab_alignment.status == ALIGN_FAIL:
        warnings.append(f"vocab size mismatch: {vocab_alignment.note}")

    return InspectionReport(
        kind=kind,
        raw=raw_text,
        segments=segments,
        eos_positions=eos_positions,
        round_trip=round_trip,
        vocab_alignment=vocab_alignment,
        truncated=truncated,
        has_unknown=has_unknown,
        warnings=warnings,
    )


def _message_roles(
    tokenizer,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
) -> tuple[list[int], dict[int, tuple[str | None, bool]]]:
    """前缀差分法:返回完整 ids 与每个 token index 的 (role, loss_mask)。"""

    n = len(messages)
    role_by_index: dict[int, tuple[str | None, bool]] = {}
    prev_end = 0

    # 逐前缀渲染(不含 generation prompt),用长度差定位每条消息的 token 区间。
    for i in range(1, n + 1):
        prefix_ids = normalize_token_ids(
            apply_chat_template_with_options(
                tokenizer,
                messages[:i],
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        end = len(prefix_ids)
        role = messages[i - 1].get("role")
        loss_mask = role == "assistant"
        for idx in range(prev_end, end):
            role_by_index[idx] = (role, loss_mask)
        prev_end = end

    # 完整渲染(可选追加 generation prompt),得到最终 ids 序列。
    full_ids = normalize_token_ids(
        apply_chat_template_with_options(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )
    # add_generation_prompt 追加的 assistant 头部段不属于任何消息,不计 loss。
    if add_generation_prompt and len(full_ids) > prev_end:
        for idx in range(prev_end, len(full_ids)):
            role_by_index[idx] = (None, False)

    return full_ids, role_by_index


def _render_text(tokenizer, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    """渲染 messages 为文本(tokenize=False),供 round-trip 比对。"""

    text = apply_chat_template_with_options(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return text if isinstance(text, str) else str(text)


def _segments_from_ids(
    tokenizer,
    ids: list[int],
    role_by_index: dict[int, tuple[str | None, bool]],
    eos_set: set[int],
    special_set: set[int],
) -> list[Segment]:
    """将 id 序列转换为逐 token 的 Segment 列表。"""

    # convert_ids_to_tokens 是 HF tokenizer 的标准接口;缺失时退回逐 id decode。
    pieces = _convert_ids_to_tokens(tokenizer, ids)
    segments: list[Segment] = []
    for idx, (tid, piece) in enumerate(zip(ids, pieces)):
        role, loss_mask = role_by_index.get(idx, (None, False))
        segments.append(
            Segment(
                text=piece,
                token_ids=[tid],
                pieces=[piece],
                role=role,
                is_special=tid in special_set,
                loss_mask=loss_mask,
                is_eos=tid in eos_set,
            )
        )
    return segments


def _convert_ids_to_tokens(tokenizer, ids: list[int]) -> list[str]:
    """安全获取每个 id 的 piece 字符串。"""

    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is not None:
        return list(convert(ids))
    # 退化路径:逐 token decode,保留 special tokens 以反映模板控制符。
    decode = getattr(tokenizer, "decode", None)
    if decode is not None:
        return [decode([tid]) for tid in ids]
    return [str(tid) for tid in ids]


def _eos_set(tokenizer, model_path: str | Path | None) -> set[int]:
    """收集所有 EOS id:优先复用 eos_token_ids(读 tokenizer + config.json)。"""

    if model_path is not None:
        try:
            return set(eos_token_ids(model_path, tokenizer))
        except (OSError, ValueError, KeyError):
            # config.json 读取失败时退化为仅用 tokenizer 自身 EOS,不阻断检查。
            pass
    eos_id = getattr(tokenizer, "eos_token_id", None)
    return {int(eos_id)} if eos_id is not None else set()


def _has_unknown(tokenizer, ids: list[int]) -> bool:
    """是否存在 unknown token。"""

    unk_id = getattr(tokenizer, "unk_token_id", None)
    return unk_id is not None and int(unk_id) in ids


def _decode_round_trip(tokenizer, ids: list[int], original: str, *, skip_special: bool = True) -> RoundTrip:
    """encode→decode 还原比对,差异描述不回显完整原文。

    ``skip_special=True`` 去掉 special token 后比对纯文本(适合 plain prompt,
    其 original 是用户原文);``skip_special=False`` 保留 special token 字面
    比对完整模板文本(适合 messages,其 original 是 apply_chat_template 渲染出的
    含 special 字面的文本)。不支持的旧 API 退化为默认解码。
    """

    decode = getattr(tokenizer, "decode", None)
    if decode is not None:
        try:
            decoded = decode(ids, skip_special_tokens=skip_special)
        except TypeError:
            decoded = decode(ids)  # 不支持 skip_special_tokens 的旧 API
    else:
        decoded = ""
    if decoded == original:
        return RoundTrip(ok=True, decoded=decoded, diff_note="")
    note = _describe_diff(original, decoded)
    return RoundTrip(ok=False, decoded=decoded, diff_note=note)


def _describe_diff(original: str, decoded: str) -> str:
    """分类描述 round-trip 差异(空白/长度/内容)。"""

    if original.strip() == decoded.strip():
        return "whitespace-only round-trip difference (leading/trailing)"
    if len(original) != len(decoded):
        return f"length differs: {len(original)} -> {len(decoded)}"
    return "content differs after encode/decode"


def _vocab_alignment(tokenizer, model_path: str | Path | None) -> VocabAlignment:
    """tokenizer 实际长度与模型 config.json vocab_size 的对齐比较。

    用 ``len(tokenizer)``(含 added tokens,即 tokenizer 实际会用到的最大
    token id + 1)与模型 ``config.json`` 的 ``vocab_size``(embedding 行数)比较:
    超过 → FAIL(token id 落在 embedding 之外);等于 → OK;小于 → OK(模型侧
    预留了 padded/reserved 行,常见于 Qwen 等为 TP 分片对齐而上取整的配置)。
    """

    tokenizer_len = _safe_len(tokenizer)

    if model_path is None:
        return VocabAlignment(
            status=ALIGN_SKIP,
            tokenizer_size=tokenizer_len,
            model_vocab_size=None,
            note="no model_path provided; vocab alignment skipped",
        )

    model_vocab = _read_model_vocab_size(model_path)
    if model_vocab is None:
        return VocabAlignment(
            status=ALIGN_SKIP,
            tokenizer_size=tokenizer_len,
            model_vocab_size=None,
            note=f"config.json at {model_path} has no vocab_size; vocab alignment skipped",
        )
    if tokenizer_len is None:
        return VocabAlignment(
            status=ALIGN_SKIP,
            tokenizer_size=None,
            model_vocab_size=model_vocab,
            note="tokenizer length unavailable; vocab alignment skipped",
        )

    if tokenizer_len > model_vocab:
        return VocabAlignment(
            status=ALIGN_FAIL,
            tokenizer_size=tokenizer_len,
            model_vocab_size=model_vocab,
            note=(
                f"tokenizer length {tokenizer_len} exceeds model config vocab_size {model_vocab}; "
                "token ids would fall outside embedding rows"
            ),
        )
    if tokenizer_len == model_vocab:
        return VocabAlignment(
            status=ALIGN_OK,
            tokenizer_size=tokenizer_len,
            model_vocab_size=model_vocab,
            note=f"vocab sizes match ({model_vocab})",
        )
    return VocabAlignment(
        status=ALIGN_OK,
        tokenizer_size=tokenizer_len,
        model_vocab_size=model_vocab,
        note=(
            f"tokenizer length {tokenizer_len} within model config vocab_size {model_vocab} "
            f"({model_vocab - tokenizer_len} reserved/padded rows)"
        ),
    )


def _safe_len(tokenizer) -> int | None:
    """安全的 tokenizer 长度(含 added tokens),不可 len 时返回 None。"""

    try:
        return len(tokenizer)
    except TypeError:
        return None


def _read_model_vocab_size(model_path: str | Path) -> int | None:
    """内联读取 <model_path>/config.json 的 vocab_size(含 text_config fallback)。

    与 ``areno/api/tokenizer.py`` 读取 EOS 的模式一致:多模态模型可能把 LM 配置
    嵌套在 ``text_config`` 下。不依赖 ``areno.models.registry`` 以避免拉起 torch/engine。
    """

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    vocab = config.get("vocab_size")
    if vocab is None:
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            vocab = text_config.get("vocab_size")
    return int(vocab) if vocab is not None else None
