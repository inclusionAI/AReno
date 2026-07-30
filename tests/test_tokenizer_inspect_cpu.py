from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from areno.api import tokenizer_inspect as ti
from areno.cli import inspect_tokenizer as inspect_cli
from areno.cli.main import main

# --------------------------------------------------------------------------- #
# FakeTokenizer:暴露 inspector 所需的最小 HF 风格 API,行为完全确定性。
# 普通字符用 ord(c)+1000 编码以避开 special/unk id(0/1/2/99)。
# --------------------------------------------------------------------------- #


class FakeTokenizer:
    """确定性 tokenizer double,用于 CPU 测试,不依赖 transformers。"""

    eos_token_id = 0
    unk_token_id = 99
    chat_template = None  # None -> encode_generation_prompt 走 encode 路径(plain prompt)
    vocab_size = 100000

    # all_special_ids 涵盖 BOS/EOS/PAD/UNK
    all_special_ids = [0, 1, 2, 99]

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, prompt, add_special_tokens: bool = True) -> list[int]:
        # BOS(1) 前缀 + 逐字符映射;U+0001 作为 unknown 指示符 -> 99(unk)。
        ids = [1]
        for ch in prompt:
            ids.append(99 if ch == "" else ord(ch) + 1000)
        return ids

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        # 每条 turn:BOS(仅首条) + role marker + content ids;assistant turn 末尾追加 EOS(0)。
        ids: list[int] = []
        for j, msg in enumerate(messages):
            if j == 0:
                ids.append(1)
            ids.append(self._role_id(msg.get("role")))
            content = msg.get("content") or ""
            for ch in content:
                ids.append(99 if ch == "" else ord(ch) + 1000)
            if msg.get("role") == "assistant":
                ids.append(0)
        if add_generation_prompt:
            # generation prompt 段:用非 special 的 marker(60) 表示 assistant 头部。
            ids.extend([60])
        if tokenize:
            return ids
        return "<" + "|".join(str(m.get("role")) for m in messages) + ">"

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [self._piece(int(tid)) for tid in ids]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        out: list[str] = []
        for tid in ids:
            tid = int(tid)
            # special 与 role/gen marker 在 skip_special_tokens=True 时一并跳过
            if tid in (0, 1, 2, 99) or 50 <= tid <= 54 or tid == 60:
                if not skip_special_tokens:
                    out.append(self._piece(tid))
            else:
                out.append(chr(tid - 1000))
        return "".join(out)

    def _role_id(self, role) -> int:
        # role marker 用 50-53,避开 special id 集合。
        return {"system": 50, "user": 51, "assistant": 52, "tool": 53}.get(role, 54)

    def _piece(self, tid: int) -> str:
        if tid == 0:
            return "</s>"
        if tid == 1:
            return "<s>"
        if tid == 2:
            return "<pad>"
        if tid == 99:
            return "<unk>"
        if tid == 60:
            return "<gen>"
        if 50 <= tid <= 54:
            # role marker token(非 special,仅用于标注 chat turn 边界)
            return f"<r{tid}>"
        return chr(tid - 1000)


class WhitespaceTruncTokenizer(FakeTokenizer):
    """decode 时去除首尾空白,用以触发 non-perfect round trip。"""

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return super().decode(ids, skip_special_tokens=skip_special_tokens).strip()


def _config_dir(vocab_size: int) -> str:
    """创建带 config.json 的临时模型目录,用于 vocab 对齐与 EOS 读取。"""

    tmp = tempfile.mkdtemp()
    Path(tmp, "config.json").write_text(
        json.dumps({"vocab_size": vocab_size, "eos_token_id": 0}),
        encoding="utf-8",
    )
    return tmp


class InspectPromptTest(unittest.TestCase):
    """inspect_prompt 路径:覆盖 segment/EOS/unicode/unknown/truncation/round-trip。"""

    def test_inspect_prompt_segments_and_eos(self):
        """plain prompt:ids/pieces/special 标记正确,role 为 None、loss_mask 为 False。"""
        tok = FakeTokenizer()
        model_path = _config_dir(100000)
        report = ti.inspect_prompt(tok, "Hi", model_path=model_path)

        self.assertEqual(report.kind, "prompt")
        # ids = [BOS(1), 'H'(1072), 'i'(1105)]
        self.assertEqual([s.token_ids[0] for s in report.segments], [1, 1072, 1105])
        self.assertEqual([s.text for s in report.segments], ["<s>", "H", "i"])
        self.assertTrue(report.segments[0].is_special)  # BOS
        self.assertFalse(report.segments[1].is_special)
        self.assertFalse(any(s.is_eos for s in report.segments))  # 无 EOS
        self.assertEqual(report.eos_positions, [])
        self.assertTrue(all(s.role is None for s in report.segments))
        self.assertTrue(all(not s.loss_mask for s in report.segments))

    def test_inspect_prompt_unicode_whitespace(self):
        """Unicode 与首尾空白应正确编码还原。"""
        tok = FakeTokenizer()
        model_path = _config_dir(100000)
        report = ti.inspect_prompt(tok, " 你好 ", model_path=model_path)

        # BOS + ' ' + '你' + '好' + ' '
        self.assertEqual([s.text for s in report.segments], ["<s>", " ", "你", "好", " "])
        self.assertTrue(report.round_trip.ok)  # decode 还原原文(含空白)
        self.assertEqual(report.vocab_alignment.status, ti.ALIGN_OK)

    def test_inspect_prompt_unknown_token(self):
        """包含 U+0001 -> unk(99) 时 has_unknown 为 True。"""
        tok = FakeTokenizer()
        model_path = _config_dir(100000)
        report = ti.inspect_prompt(tok, "ab", model_path=model_path)

        self.assertTrue(report.has_unknown)
        unk_seg = [s for s in report.segments if s.token_ids[0] == 99]
        self.assertEqual(len(unk_seg), 1)
        self.assertTrue(unk_seg[0].is_special)  # unk 属 special

    def test_inspect_prompt_truncation(self):
        """max_length 触发截断,truncated 为 True 且 segments 被裁剪。"""
        tok = FakeTokenizer()
        model_path = _config_dir(100000)
        report = ti.inspect_prompt(tok, "abcdef", model_path=model_path, max_length=3)

        self.assertTrue(report.truncated)
        self.assertEqual(len(report.segments), 3)
        self.assertTrue(any("truncated" in w for w in report.warnings))

    def test_inspect_prompt_round_trip_imperfect(self):
        """decode 丢空白时 round_trip.ok 为 False 且 diff_note 说明空白差异。"""
        tok = WhitespaceTruncTokenizer()
        model_path = _config_dir(100000)
        report = ti.inspect_prompt(tok, " hi ", model_path=model_path)

        self.assertFalse(report.round_trip.ok)
        self.assertIn("whitespace", report.round_trip.diff_note)


class InspectMessagesTest(unittest.TestCase):
    """inspect_messages:差分法标注 role 与 loss_mask,含 generation prompt 段。"""

    @staticmethod
    def _messages():
        return [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]

    def test_roles_and_loss_mask(self):
        """仅 assistant turn 的 token loss_mask 为 True;其它 turn 为 False。"""
        tok = FakeTokenizer()
        model_path = _config_dir(100000)
        report = ti.inspect_messages(tok, self._messages(), model_path=model_path, add_generation_prompt=True)

        self.assertEqual(report.kind, "messages")
        roles = [s.role for s in report.segments]
        self.assertIn("assistant", roles)
        self.assertIn("user", roles)
        # assistant 段计入 loss
        assistant_segs = [s for s in report.segments if s.role == "assistant"]
        self.assertTrue(assistant_segs)
        self.assertTrue(all(s.loss_mask for s in assistant_segs))
        # user/system 段不计 loss
        non_assistant = [s for s in report.segments if s.role in {"user", "system"}]
        self.assertTrue(non_assistant)
        self.assertTrue(all(not s.loss_mask for s in non_assistant))
        # assistant EOS 计入 eos_positions
        self.assertTrue(report.eos_positions)

    def test_add_generation_prompt_segment_is_not_loss(self):
        """add_generation_prompt 追加的头部段 role=None、loss_mask=False。"""
        tok = FakeTokenizer()
        model_path = _config_dir(100000)

        with_gp = ti.inspect_messages(tok, self._messages(), model_path=model_path, add_generation_prompt=True)
        without_gp = ti.inspect_messages(tok, self._messages(), model_path=model_path, add_generation_prompt=False)

        # 开启 generation prompt 时多出头部段(role=None)
        gp_segs = [s for s in with_gp.segments if s.role is None and s.token_ids[0] == 60]
        self.assertEqual(len(gp_segs), 1)
        self.assertFalse(gp_segs[0].loss_mask)
        # 关闭时不存在该段
        self.assertEqual([s for s in without_gp.segments if s.token_ids[0] == 60], [])
        self.assertGreater(len(with_gp.segments), len(without_gp.segments))


class InspectToolCallTest(unittest.TestCase):
    """inspect_tool_call:assistant tool_calls + tool 角色回复标注。"""

    def test_tool_call_roles(self):
        tok = FakeTokenizer()
        model_path = _config_dir(100000)
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "name": "get_weather", "content": "sunny"},
        ]
        report = ti.inspect_tool_call(tok, messages, model_path=model_path)

        self.assertEqual(report.kind, "tool_call")
        roles = [s.role for s in report.segments]
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)
        # tool 段不计 loss,assistant 段计 loss
        self.assertTrue(all(s.loss_mask for s in report.segments if s.role == "assistant"))
        self.assertTrue(all(not s.loss_mask for s in report.segments if s.role == "tool"))


class VocabAlignmentTest(unittest.TestCase):
    """tokenizer vocab_size 与 config.json vocab_size 对齐比较。"""

    def test_match(self):
        tok = FakeTokenizer()
        model_path = _config_dir(100000)
        report = ti.inspect_prompt(tok, "Hi", model_path=model_path)
        self.assertEqual(report.vocab_alignment.status, ti.ALIGN_OK)

    def test_mismatch(self):
        # tokenizer 实际长度(len)超过模型 config vocab_size -> FAIL
        tok = FakeTokenizer()
        tok.vocab_size = 200000  # len(tokenizer)=200000
        model_path = _config_dir(100000)  # 模型词表更小
        report = ti.inspect_prompt(tok, "Hi", model_path=model_path)
        self.assertEqual(report.vocab_alignment.status, ti.ALIGN_FAIL)
        self.assertIn("200000", report.vocab_alignment.note)
        self.assertIn("100000", report.vocab_alignment.note)

    def test_no_config_is_skip_not_fail(self):
        """无 config.json 时对齐为 SKIP,不视为失败。"""
        tok = FakeTokenizer()
        tmp = tempfile.mkdtemp()  # 无 config.json
        report = ti.inspect_prompt(tok, "Hi", model_path=tmp)
        self.assertEqual(report.vocab_alignment.status, ti.ALIGN_SKIP)


class NoMutationTest(unittest.TestCase):
    """inspector 默认不修改 tokenizer 对象(向后兼容)。"""

    def test_inspect_does_not_mutate_tokenizer(self):
        tok = FakeTokenizer()
        before = set(vars(tok).keys())
        ti.inspect_messages(
            tok,
            [{"role": "user", "content": "hi"}],
            model_path=_config_dir(100000),
            enable_thinking=None,  # 默认:不应设置 thinking 属性
        )
        after = set(vars(tok).keys())
        self.assertEqual(before, after)
        self.assertFalse(hasattr(tok, "_areno_chat_template_enable_thinking"))


class InspectTokenizerCliTest(unittest.TestCase):
    """CLI:输入校验、JSON 字段、vocab 失败退出码、命令注册。"""

    @contextlib.contextmanager
    def _patches(self, tok, path):
        with patch.object(inspect_cli, "resolve_model_ref", return_value=path), patch.object(
            inspect_cli, "load_tokenizer", return_value=tok
        ):
            yield

    def test_cli_prompt_json_fields(self):
        tok = FakeTokenizer()
        path = _config_dir(100000)
        with self._patches(tok, path) :
            result = CliRunner().invoke(
                inspect_cli.inspect_tokenizer_command, ["--model", path, "--prompt", "Hi", "--json"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        parsed = json.loads(result.output)
        for key in ("segments", "eos_positions", "round_trip", "vocab_alignment", "kind"):
            self.assertIn(key, parsed)
        self.assertEqual(parsed["kind"], "prompt")
        self.assertEqual(parsed["vocab_alignment"]["status"], ti.ALIGN_OK)

    def test_cli_rejects_missing_input(self):
        path = _config_dir(100000)
        with self._patches(FakeTokenizer(), path) :
            result = CliRunner().invoke(inspect_cli.inspect_tokenizer_command, ["--model", path])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("exactly one", result.output)

    def test_cli_rejects_malformed_json(self):
        path = _config_dir(100000)
        with self._patches(FakeTokenizer(), path) :
            result = CliRunner().invoke(
                inspect_cli.inspect_tokenizer_command, ["--model", path, "--messages", "{bad"]
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("not valid JSON", result.output)

    def test_cli_human_readable_shows_fail_on_vocab_mismatch(self):
        # tokenizer len > config vocab_size -> FAIL -> 非零退出且输出含 FAIL。
        tok = FakeTokenizer()
        tok.vocab_size = 200000
        path = _config_dir(100000)
        with self._patches(tok, path):
            result = CliRunner().invoke(
                inspect_cli.inspect_tokenizer_command, ["--model", path, "--prompt", "Hi"]
            )
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("FAIL", result.output)

    def test_cli_registered_in_main_help(self):
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("inspect-tokenizer", result.output)


if __name__ == "__main__":
    unittest.main()
