from __future__ import annotations

import unittest

from areno.api.tokenizer import assert_compatible_tokenizers, vocab_size


class FakeTokenizer:
    def __init__(self, size=100, ids=(7, 8, 9), eos=1, pad=2, *, expose_size=True):
        if expose_size:
            self.vocab_size = size
        self._size = size
        self._ids = tuple(ids)
        self.eos_token_id = eos
        self.bos_token_id = 0
        self.pad_token_id = pad

    def encode(self, text, add_special_tokens=True):
        del text, add_special_tokens
        return list(self._ids)

    def __len__(self):
        return self._size


class ProbeSensitiveTokenizer(FakeTokenizer):
    """Shift one id only when the probe contains a marker."""

    def __init__(self, marker):
        super().__init__()
        self._marker = marker

    def encode(self, text, add_special_tokens=True):
        ids = super().encode(text, add_special_tokens=add_special_tokens)
        if self._marker in text:
            ids[-1] += 1
        return ids


class TokenizerCompatibilityTest(unittest.TestCase):
    def test_vocab_size_uses_len_as_fallback(self):
        self.assertEqual(vocab_size(FakeTokenizer(size=321, expose_size=False)), 321)
        self.assertEqual(vocab_size(FakeTokenizer(size=100)), 100)

    def test_matching_tokenizers_pass(self):
        assert_compatible_tokenizers(FakeTokenizer(), FakeTokenizer())

    def test_vocab_mismatch_is_rejected_at_startup(self):
        with self.assertRaisesRegex(ValueError, "vocab sizes differ"):
            assert_compatible_tokenizers(FakeTokenizer(size=100), FakeTokenizer(size=128))

    def test_pad_token_mismatch_is_rejected(self):
        # pad_token_id feeds the packed tensors shared with the teacher.
        with self.assertRaisesRegex(ValueError, "pad_token_id differ"):
            assert_compatible_tokenizers(FakeTokenizer(pad=2), FakeTokenizer(pad=3))

    def test_eos_mismatch_allowed_for_non_generating_teacher(self):
        # The teacher never generates, so eos/bos may differ.
        assert_compatible_tokenizers(FakeTokenizer(eos=1), FakeTokenizer(eos=3))

    def test_different_encoding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "encode the same text differently"):
            assert_compatible_tokenizers(FakeTokenizer(ids=(7, 8, 9)), FakeTokenizer(ids=(7, 8, 10)))

    def test_mismatch_on_a_later_probe_is_rejected(self):
        # Only the maths probe encodes differently; a single-probe check would
        # miss this pair.
        with self.assertRaisesRegex(ValueError, "encode the same text differently"):
            assert_compatible_tokenizers(FakeTokenizer(), ProbeSensitiveTokenizer("boxed"))


if __name__ == "__main__":
    unittest.main()
