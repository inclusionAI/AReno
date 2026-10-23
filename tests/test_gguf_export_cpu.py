from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from areno.export.gguf import GGUFPackage


class GGUFExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "source"
        self.model.mkdir()
        # Qwen3 head_dim is independent of hidden_size / head_count.
        self.config = dict(
            model_type="qwen3",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            vocab_size=64,
            max_position_embeddings=1024,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000,
            tie_word_embeddings=True,
        )
        (self.model / "config.json").write_text(json.dumps(self.config))
        tokenizer = {
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]},
            "added_tokens": [{"id": 60, "content": "<|im_end|>", "special": True}],
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {
                            "Regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                        },
                        "behavior": "Isolated",
                        "invert": False,
                    },
                    {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
                ],
            },
        }
        (self.model / "tokenizer.json").write_text(json.dumps(tokenizer))
        (self.model / "tokenizer_config.json").write_text(
            json.dumps({"eos_token": "<|im_end|>", "chat_template": "{{ messages }}"})
        )
        self.shapes = {
            "model.embed_tokens.weight": (64, 32),
            "model.norm.weight": (32,),
            "model.layers.0.input_layernorm.weight": (32,),
            "model.layers.0.post_attention_layernorm.weight": (32,),
            "model.layers.0.self_attn.q_proj.weight": (64, 32),
            "model.layers.0.self_attn.k_proj.weight": (32, 32),
            "model.layers.0.self_attn.v_proj.weight": (32, 32),
            "model.layers.0.self_attn.o_proj.weight": (32, 64),
            "model.layers.0.self_attn.q_norm.weight": (32,),
            "model.layers.0.self_attn.k_norm.weight": (32,),
            "model.layers.0.mlp.gate_proj.weight": (64, 32),
            "model.layers.0.mlp.up_proj.weight": (64, 32),
            "model.layers.0.mlp.down_proj.weight": (32, 64),
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_package(self, file_type="f16"):
        package = GGUFPackage(str(self.model), str(self.root / "output"), self.shapes, file_type=file_type)
        values = {}
        try:
            for name, shape in self.shapes.items():
                values[name] = np.linspace(-0.9, 0.9, np.prod(shape), dtype=np.float32).reshape(shape)
                package.write(name, values[name])
            return Path(package.finish()), values
        finally:
            package.abort()

    def test_f16_readback_and_complete_ollama_package(self):
        import gguf

        path, original = self.write_package()
        reader = gguf.GGUFReader(path)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        self.assertEqual(len(tensors), 13)
        self.assertNotIn("output.weight", tensors)  # tied head is represented by the embedding
        np.testing.assert_array_equal(
            tensors["blk.0.attn_q.weight"].data, original["model.layers.0.self_attn.q_proj.weight"].astype(np.float16)
        )
        np.testing.assert_array_equal(tensors["output_norm.weight"].data, original["model.norm.weight"])
        self.assertEqual(int(reader.fields["qwen3.attention.key_length"].parts[-1][0]), 32)
        template = reader.fields["tokenizer.chat_template"].parts[-1].tobytes().decode()
        self.assertIn("</think>", template)
        self.assertNotIn("tools", template)
        self.assertNotEqual(template, "{{ messages }}")
        manifest = json.loads((path.parent / "export.json").read_text())
        self.assertFalse(manifest["intermediate_checkpoint"])
        self.assertEqual(manifest["tensor_count"], 13)
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertIn("FROM ./model.gguf", (path.parent / "Modelfile").read_text())
        self.assertEqual(list(path.parent.glob("*.safetensors")), [])

    def test_q8_roundtrip_error_and_vector_precision(self):
        import gguf
        from gguf.quants import dequantize

        path, original = self.write_package("q8_0")
        reader = gguf.GGUFReader(path)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        matrix = tensors["blk.0.attn_q.weight"]
        self.assertEqual(matrix.tensor_type, gguf.GGMLQuantizationType.Q8_0)
        restored = dequantize(matrix.data, matrix.tensor_type)
        # Q8 uses nearest integer at a scale max(abs(block))/127, plus FP16 scale rounding.
        np.testing.assert_allclose(
            restored, original["model.layers.0.self_attn.q_proj.weight"], atol=0.9 / 254 + 0.0005, rtol=0
        )
        self.assertEqual(tensors["output_norm.weight"].tensor_type, gguf.GGMLQuantizationType.F32)

    def test_missing_tensor_fails_before_publishing(self):
        shapes = dict(self.shapes)
        shapes.pop("model.layers.0.self_attn.k_norm.weight")
        with self.assertRaisesRegex(ValueError, "tensor contract"):
            GGUFPackage(str(self.model), str(self.root / "output"), shapes)
        self.assertFalse((self.root / "output").exists())

    def test_unrecognized_tokenizer_is_rejected(self):
        path = self.model / "tokenizer.json"
        data = json.loads(path.read_text())
        data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "different"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "pre-tokenizer"):
            GGUFPackage(str(self.model), str(self.root / "output"), self.shapes)
        self.assertEqual(list(self.root.glob(".areno-gguf-*")), [])

    def test_incomplete_and_nonfinite_exports_do_not_publish(self):
        package = GGUFPackage(str(self.model), str(self.root / "output"), self.shapes)
        try:
            with self.assertRaisesRegex(ValueError, "not written"):
                package.finish()
            with self.assertRaisesRegex(ValueError, "non-finite"):
                package.write("model.embed_tokens.weight", np.full((64, 32), np.nan, dtype=np.float32))
        finally:
            package.abort()
        self.assertFalse((self.root / "output").exists())

    def test_existing_output_is_not_replaced(self):
        output = self.root / "output"
        output.mkdir()
        marker = output / "keep"
        marker.write_text("existing")
        with self.assertRaises(FileExistsError):
            GGUFPackage(str(self.model), str(output), self.shapes)
        self.assertEqual(marker.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
