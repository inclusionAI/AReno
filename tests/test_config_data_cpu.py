from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from areno.cli import train as train_cli
from areno.engine.config import EngineConfig, ModelConfig, RuntimeConfig, _parse_dtype
from areno.engine.data import to_cpu, to_device
from areno.api.data import PromptBatch, PromptItem


class ConfigAndDataTest(unittest.TestCase):
    """Config and data utility tests use CPU tensors and tiny configs only."""

    def test_parse_dtype_accepts_common_aliases(self):
        """HF dtype aliases should normalize to torch dtype objects."""
        self.assertIs(_parse_dtype("bf16"), torch.bfloat16)
        self.assertIs(_parse_dtype("fp16"), torch.float16)
        self.assertIs(_parse_dtype("float"), torch.float32)
        with self.assertRaises(ValueError):
            _parse_dtype("int8")

    def test_model_config_rejects_invalid_tp_for_dense_qwen(self):
        """Dense models require KV heads to shard evenly across TP ranks."""
        cfg = ModelConfig(num_attention_heads=8, num_key_value_heads=3, intermediate_size=16, vocab_size=32)

        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            cfg.validate_tp(2)

    def test_model_config_allows_replicated_kv_for_gemma(self):
        """Gemma permits replicated KV heads when TP is a multiple of KV heads."""
        cfg = ModelConfig(
            model_type="gemma4",
            num_attention_heads=8,
            num_key_value_heads=1,
            intermediate_size=16,
            vocab_size=32,
        )

        cfg.validate_tp(4)

    def test_model_config_validates_linear_attention_dims(self):
        """Linear-attention projection dimensions must satisfy TP divisibility."""
        cfg = ModelConfig(
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=16,
            vocab_size=32,
            layer_types=("linear_attention",),
            linear_num_key_heads=3,
        )

        with self.assertRaisesRegex(ValueError, "linear_num_key_heads"):
            cfg.validate_tp(2)

    def test_engine_config_validates_devices_and_kv_block(self):
        """EngineConfig should reject invalid device layouts and KV block sizes."""
        model = ModelConfig(num_attention_heads=4, num_key_value_heads=4, intermediate_size=16, vocab_size=32)

        with self.assertRaisesRegex(ValueError, "len\\(devices\\)"):
            EngineConfig(model=model, tp_size=2, devices=[0, 1, 2])
        with self.assertRaisesRegex(ValueError, "kv_block_size"):
            EngineConfig(model=model, tp_size=1, devices=[0], runtime=RuntimeConfig(kv_block_size=128))

    def test_engine_config_infers_dp_size(self):
        """DP size is inferred from device count divided by TP size."""
        model = ModelConfig(num_attention_heads=4, num_key_value_heads=4, intermediate_size=16, vocab_size=32)

        cfg = EngineConfig(model=model, tp_size=2, devices=[0, 1, 2, 3])

        self.assertEqual(cfg.dp_size, 2)

    def test_to_device_and_to_cpu_walk_nested_containers(self):
        """Device helpers should preserve nested container structure."""
        src = {"x": torch.tensor([1.0]), "items": [torch.tensor([2.0]), (torch.tensor([3.0]), "keep")]}

        moved = to_device(src, torch.device("cpu"))
        out = to_cpu(moved)

        self.assertEqual(out["x"].device.type, "cpu")
        self.assertEqual(out["items"][0].device.type, "cpu")
        self.assertEqual(out["items"][1][1], "keep")

    def test_prompt_batch_prompts_preserves_order(self):
        """PromptBatch.prompts is the rollout-facing order contract."""
        batch = PromptBatch(
            items=[
                PromptItem(prompt="a", solutions=None, input_tokens=[1], record={}),
                PromptItem(prompt="b", solutions=["x"], input_tokens=[2], record={"id": 2}),
            ],
            scanned=2,
            skipped_long=0,
            total_skipped_long=1,
        )

        self.assertEqual(batch.prompts, ["a", "b"])

    def test_cli_dataset_loader_fn_uses_explicit_callable(self):
        """The CLI dataset hook should call only the user-specified loader."""
        with tempfile.TemporaryDirectory() as tmp:
            loader_path = Path(tmp) / "loader.py"
            loader_path.write_text(
                "def normalize(dataset_path, *, default_loader, **kwargs):\n"
                "    raw = default_loader(dataset_path)\n"
                "    return [{'prompt': raw[0]['raw']}]\n",
                encoding="utf-8",
            )

            dataset = train_cli._load_dataset_for_training(
                "ignored",
                dataset_loader_fn=f"{loader_path}:normalize",
                load_dataset=lambda *_args, **_kwargs: [{"raw": "loaded"}],
                load_from_disk=lambda *_args, **_kwargs: None,
            )

        self.assertEqual(dataset, [{"prompt": "loaded"}])


if __name__ == "__main__":
    unittest.main()
