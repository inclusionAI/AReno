from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import torch
from click.testing import CliRunner

from areno.api.data import PromptBatch, PromptItem
from areno.api.trainer_config import RolloutTrainerConfig, TrainerConfig
from areno.cli import train as train_cli
from areno.engine.config import (
    EngineConfig,
    ModelConfig,
    RuntimeConfig,
    _parse_dtype,
    flash_attention_unsupported_gpu_reason,
    flash_attention_unsupported_model_reason,
)
from areno.engine.data import to_cpu, to_device
from areno.engine.layers.attention_backend.common import (
    build_attention_call,
    expand_kv_heads,
    require_flash_attention_supported,
)
from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend, _native_prefill
from areno.engine.runtime.metadata import InferMeta


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

    def test_runtime_config_attn_backend_propagates_to_model_config(self):
        """The runtime attention backend should reach model layer construction."""
        model = ModelConfig(num_attention_heads=4, num_key_value_heads=4, intermediate_size=16, vocab_size=32)

        EngineConfig(model=model, tp_size=1, devices=[0], runtime=RuntimeConfig(attn_backend="native"))

        self.assertEqual(model.attn_backend, "native")

    def test_runtime_config_falls_back_to_native_on_turing_gpu(self):
        """Turing GPUs like T4 should use native attention instead of flash-attn."""
        model = ModelConfig(num_attention_heads=4, num_key_value_heads=4, intermediate_size=16, vocab_size=32)
        runtime = RuntimeConfig(attn_backend="flash")

        with (
            patch("areno.engine.config.torch.cuda.is_available", return_value=True),
            patch("areno.engine.config.torch.cuda.device_count", return_value=1),
            patch("areno.engine.config.torch.cuda.get_device_capability", return_value=(7, 5)),
            patch("areno.engine.config.torch.cuda.get_device_name", return_value="Tesla T4"),
            self.assertWarnsRegex(RuntimeWarning, "falling back to attn_backend='native'.*slower"),
        ):
            cfg = EngineConfig(model=model, tp_size=1, devices=[0], runtime=runtime)

        self.assertEqual(cfg.runtime.attn_backend, "native")
        self.assertEqual(model.attn_backend, "native")

    def test_flash_attention_supported_gpu_keeps_flash_backend(self):
        """Ampere and newer GPUs should keep the explicit flash attention backend."""
        model = ModelConfig(num_attention_heads=4, num_key_value_heads=4, intermediate_size=16, vocab_size=32)
        runtime = RuntimeConfig(attn_backend="flash")

        with (
            patch("areno.engine.config.torch.cuda.is_available", return_value=True),
            patch("areno.engine.config.torch.cuda.device_count", return_value=1),
            patch("areno.engine.config.torch.cuda.get_device_capability", return_value=(8, 0)),
        ):
            cfg = EngineConfig(model=model, tp_size=1, devices=[0], runtime=runtime)

        self.assertEqual(cfg.runtime.attn_backend, "flash")
        self.assertEqual(model.attn_backend, "flash")

    def test_runtime_config_falls_back_to_native_on_large_qk_head_dim(self):
        """Gemma-style qk head dim 512 should use native attention instead of flash-attn."""
        model = ModelConfig(
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=16,
            vocab_size=32,
            head_dim=512,
        )
        runtime = RuntimeConfig(attn_backend="flash")

        with self.assertWarnsRegex(RuntimeWarning, "qk head dim 512.*attn_backend='native'"):
            cfg = EngineConfig(model=model, tp_size=1, devices=[0], runtime=runtime)

        self.assertEqual(cfg.runtime.attn_backend, "native")
        self.assertEqual(model.attn_backend, "native")

    def test_flash_attention_unsupported_model_reason_names_large_head_dim(self):
        """The compatibility warning should identify unsupported model dimensions."""
        model = ModelConfig(
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=16,
            vocab_size=32,
            head_dim=512,
        )

        self.assertEqual(flash_attention_unsupported_model_reason(model), "qk head dim 512")

    def test_flash_attention_unsupported_gpu_reason_names_t4(self):
        """The compatibility warning should identify unsupported visible GPUs."""
        with (
            patch("areno.engine.config.torch.cuda.is_available", return_value=True),
            patch("areno.engine.config.torch.cuda.device_count", return_value=1),
            patch("areno.engine.config.torch.cuda.get_device_capability", return_value=(7, 5)),
            patch("areno.engine.config.torch.cuda.get_device_name", return_value="Tesla T4"),
        ):
            reason = flash_attention_unsupported_gpu_reason([0])

        self.assertEqual(reason, "Tesla T4 cc 7.5")

    def test_runtime_config_rejects_unknown_attn_backend(self):
        """Invalid attention backend names should fail before worker startup."""
        with self.assertRaisesRegex(ValueError, "attn_backend"):
            RuntimeConfig(attn_backend="bogus")

    def test_flash_attention_unsupported_shape_points_to_torch_backend(self):
        """Unsupported flash-attn shapes should not silently fall back to SDPA."""
        call = build_attention_call(
            torch.empty(1, 1, 1, 257),
            torch.empty(1, 1, 1, 257),
            torch.empty(1, 1, 1, 257),
            window_size=None,
            softmax_scale=None,
        )

        with self.assertRaisesRegex(RuntimeError, "--attn-backend native.*slower"):
            require_flash_attention_supported(call, mode="test attention")

    def test_expand_kv_heads_uses_head_axis_for_varlen_layout(self):
        """Native varlen paths pass tensors as [tokens, heads, dim]."""
        kv = torch.arange(3 * 2 * 4).view(3, 2, 4)

        expanded = expand_kv_heads(kv, 8)

        self.assertEqual(tuple(expanded.shape), (3, 8, 4))
        self.assertTrue(torch.equal(expanded[:, 0], kv[:, 0]))
        self.assertTrue(torch.equal(expanded[:, 3], kv[:, 0]))
        self.assertTrue(torch.equal(expanded[:, 4], kv[:, 1]))
        self.assertTrue(torch.equal(expanded[:, 7], kv[:, 1]))

    def test_native_prefill_uses_varlen_gqa_without_expanding_kv_heads(self):
        """Gemma native prefill should leave GQA expansion to the varlen kernel."""
        q = torch.zeros(3, 8, 4)
        k = torch.zeros(3, 2, 4)
        v = torch.zeros(3, 2, 4)
        meta = InferMeta(mode="prefill", cu_seqlens=torch.tensor([0, 3], dtype=torch.int32), max_seqlen=3)
        captured = {}

        def fake_varlen(q_arg, k_arg, v_arg, cu_arg, *, window_left, softmax_scale):
            captured["q_shape"] = tuple(q_arg.shape)
            captured["k_shape"] = tuple(k_arg.shape)
            captured["v_shape"] = tuple(v_arg.shape)
            captured["cu"] = cu_arg.tolist()
            captured["window_left"] = window_left
            captured["softmax_scale"] = softmax_scale
            return q_arg

        with patch("areno.engine.layers.attention_backend.infer.areno_varlen_causal_attention", fake_varlen):
            out = _native_prefill(q, k, v, meta, (-1, -1), None)

        self.assertIs(out, q)
        self.assertEqual(captured["q_shape"], (3, 8, 4))
        self.assertEqual(captured["k_shape"], (3, 2, 4))
        self.assertEqual(captured["v_shape"], (3, 2, 4))
        self.assertEqual(captured["cu"], [0, 3])

    def test_native_prefill_pads_value_dim_and_trims_output(self):
        """Native prefill should match flash/decode behavior when V dim is smaller than QK."""
        backend = FlashAttnInferBackend("native")
        q = torch.zeros(1, 2, 2, 6)
        k = torch.zeros(1, 2, 2, 6)
        v = torch.zeros(1, 2, 2, 4)
        k_cache = torch.zeros(1, 2, 2, 6)
        v_cache = torch.zeros(1, 2, 2, 6)
        meta = InferMeta(
            mode="prefill",
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            max_seqlen=2,
            block_table=torch.zeros(1, 1, dtype=torch.int32),
        )
        captured = {}

        def fake_native_prefill(q_arg, k_arg, v_arg, meta_arg, window_size, softmax_scale):
            captured["q_shape"] = tuple(q_arg.shape)
            captured["k_shape"] = tuple(k_arg.shape)
            captured["v_shape"] = tuple(v_arg.shape)
            captured["value_tail"] = v_arg[..., 4:].clone()
            captured["meta"] = meta_arg
            captured["window_size"] = window_size
            captured["softmax_scale"] = softmax_scale
            return torch.ones_like(v_arg)

        with patch("areno.engine.layers.attention_backend.infer._native_prefill", fake_native_prefill):
            out = backend(q, k, v, k_cache, v_cache, meta, update_cache=False)

        self.assertEqual(captured["q_shape"], (2, 2, 6))
        self.assertEqual(captured["k_shape"], (2, 2, 6))
        self.assertEqual(captured["v_shape"], (2, 2, 6))
        self.assertTrue(torch.equal(captured["value_tail"], torch.zeros(2, 2, 2)))
        self.assertIs(captured["meta"], meta)
        self.assertEqual(tuple(out.shape), (1, 2, 2, 4))

    def test_native_attention_backend_does_not_require_flash_attn_import(self):
        """Native train/infer backends should construct without flash-attn installed."""
        from areno.engine.layers.attention_backend.infer import build_infer_attention_backend
        from areno.engine.layers.attention_backend.train import build_train_attention_backend

        with patch.dict(sys.modules, {"flash_attn": None}):
            build_train_attention_backend("native")
            build_infer_attention_backend("native")

    def test_rollout_config_defaults_max_running_prompts_to_flat_batch(self):
        """Rollout concurrency defaults to batch_size * n_samples, not per-DP."""
        cfg = RolloutTrainerConfig(
            algo="gspo",
            ckpt="unused",
            dataset_path="unused",
            world_size=8,
            tp_size=1,
            batch_size=32,
            n_samples=8,
        )

        self.assertEqual(cfg.resolved_max_running_prompts(), 256)

    def test_rollout_config_respects_explicit_max_running_prompts(self):
        """An explicit max_running_prompts value should pass through unchanged."""
        cfg = RolloutTrainerConfig(
            algo="gspo",
            ckpt="unused",
            dataset_path="unused",
            world_size=8,
            tp_size=1,
            batch_size=32,
            n_samples=8,
            max_running_prompts=64,
        )

        self.assertEqual(cfg.resolved_max_running_prompts(), 64)

    def test_trainer_config_keeps_rollout_state_by_default(self):
        """Runtime defaults should favor rollout speed unless explicitly disabled."""
        cfg = TrainerConfig(algo="sft", ckpt="unused", dataset_path="unused")

        self.assertTrue(cfg.keep_rollout_state)
        self.assertTrue(cfg.areno_config().runtime["keep_rollout_state"])

    def test_train_cli_drop_rollout_state_inverts_runtime_flag(self):
        """The public CLI exposes the memory-saving inverse of keep_rollout_state."""
        args = _train_args(algo="sft", drop_rollout_state=True)

        cfg = train_cli._trainer_config_from_args(args)

        self.assertFalse(cfg.keep_rollout_state)

    def test_train_cli_attn_backend_reaches_backend_runtime_config(self):
        """The train CLI attention backend flag should pass through SDK config."""
        args = _train_args(algo="sft", attn_backend="native")

        cfg = train_cli._trainer_config_from_args(args)

        self.assertEqual(cfg.attn_backend, "native")
        self.assertEqual(cfg.areno_config().runtime["attn_backend"], "native")

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

    def test_train_cli_preflight_rejects_missing_dataset_loader_file(self):
        """Dataset loader path failures should be UsageError before backend init."""
        missing = Path(tempfile.gettempdir()) / "areno_missing_loader.py"

        with self.assertRaisesRegex(
            click.UsageError,
            r"--dataset-loader-fn file does not exist: .*areno_missing_loader.py; expected callable normalize",
        ):
            train_cli._trainer_config_from_options(
                **_train_options(algo="sft", dataset_loader_fn=f"{missing}:normalize")
            )

    def test_train_cli_preflight_rejects_malformed_dataset_loader_spec(self):
        """Malformed dataset loader specs should not escape as raw ValueError."""
        with self.assertRaisesRegex(click.UsageError, r"Invalid --dataset-loader-fn value: :"):
            train_cli._trainer_config_from_options(**_train_options(algo="sft", dataset_loader_fn=":"))

    def test_train_cli_preflight_does_not_execute_hook_modules(self):
        """Static preflight should not trigger module-level side effects."""
        with tempfile.TemporaryDirectory() as tmp:
            loader_path = Path(tmp) / "loader.py"
            loader_path.write_text(
                "def load_training_dataset(*args, **kwargs):\n    return []\n"
                "raise RuntimeError('module executed during preflight')\n",
                encoding="utf-8",
            )

            cfg = train_cli._trainer_config_from_options(
                **_train_options(algo="sft", dataset_loader_fn=str(loader_path))
            )

        self.assertEqual(cfg.dataset_loader_fn, str(loader_path))

    def test_train_cli_preflight_rejects_dataset_loader_missing_function(self):
        """Dataset loader files should name the missing expected symbol."""
        with tempfile.TemporaryDirectory() as tmp:
            loader_path = Path(tmp) / "loader.py"
            loader_path.write_text("def other():\n    return []\n", encoding="utf-8")

            with self.assertRaisesRegex(
                click.UsageError, r"--dataset-loader-fn .*loader.py must define callable normalize\(\.\.\.\)"
            ):
                train_cli._trainer_config_from_options(
                    **_train_options(algo="sft", dataset_loader_fn=f"{loader_path}:normalize")
                )

    def test_train_cli_preflight_rejects_dataset_loader_non_callable(self):
        """Dataset loader symbol must be callable."""
        with tempfile.TemporaryDirectory() as tmp:
            loader_path = Path(tmp) / "loader.py"
            loader_path.write_text("load_training_dataset = 1\n", encoding="utf-8")

            with self.assertRaisesRegex(
                click.UsageError,
                r"--dataset-loader-fn .*loader.py must define callable load_training_dataset\(\.\.\.\)",
            ):
                train_cli._trainer_config_from_options(**_train_options(algo="sft", dataset_loader_fn=str(loader_path)))

    def test_train_cli_preflight_rejects_dataset_loader_without_dataset_path_arg(self):
        """Dataset loader hook should accept at least the dataset path."""
        with tempfile.TemporaryDirectory() as tmp:
            loader_path = Path(tmp) / "loader.py"
            loader_path.write_text("def load_training_dataset():\n    return []\n", encoding="utf-8")

            with self.assertRaisesRegex(
                click.UsageError,
                r"--dataset-loader-fn .*loader.py must define callable load_training_dataset\(\.\.\.\)",
            ):
                train_cli._trainer_config_from_options(**_train_options(algo="sft", dataset_loader_fn=str(loader_path)))

    def test_train_cli_preflight_rejects_missing_reward_file(self):
        """Reward file failures should happen while constructing CLI config."""
        missing = Path(tempfile.gettempdir()) / "areno_missing_reward.py"

        with self.assertRaisesRegex(
            click.UsageError,
            r"--reward-fn-path file does not exist: .*areno_missing_reward.py; expected callable reward_fn\(record\)",
        ):
            train_cli._trainer_config_from_options(**_train_options(algo="gspo", reward_fn_path=str(missing)))

    def test_train_cli_preflight_rejects_reward_file_without_callable_reward_fn(self):
        """Reward files should define callable reward_fn(record)."""
        with tempfile.TemporaryDirectory() as tmp:
            reward_path = Path(tmp) / "reward.py"
            reward_path.write_text("reward_fn = 1\n", encoding="utf-8")

            with self.assertRaisesRegex(
                click.UsageError, r"--reward-fn-path .*reward.py must define callable reward_fn\(record\)"
            ):
                train_cli._trainer_config_from_options(**_train_options(algo="gspo", reward_fn_path=str(reward_path)))

    def test_train_cli_preflight_rejects_reward_fn_without_record_arg(self):
        """Reward hook should accept the training record argument."""
        with tempfile.TemporaryDirectory() as tmp:
            reward_path = Path(tmp) / "reward.py"
            reward_path.write_text("def reward_fn():\n    return 0.0\n", encoding="utf-8")

            with self.assertRaisesRegex(
                click.UsageError, r"--reward-fn-path .*reward.py must define callable reward_fn\(record\)"
            ):
                train_cli._trainer_config_from_options(**_train_options(algo="gspo", reward_fn_path=str(reward_path)))

    def test_train_cli_preflight_skips_unused_reward_file_for_offline_algorithms(self):
        """SFT/DPO should not validate an unused reward hook path."""
        missing = Path(tempfile.gettempdir()) / "areno_missing_unused_reward.py"

        sft_cfg = train_cli._trainer_config_from_options(
            **_train_options(algo="sft", reward_fn_path=str(missing), reward_ckpt=None)
        )
        dpo_cfg = train_cli._trainer_config_from_options(
            **_train_options(algo="dpo", reward_fn_path=str(missing), reward_ckpt=None, ref_ckpt="reference")
        )

        self.assertEqual(sft_cfg.algo, "sft")
        self.assertEqual(dpo_cfg.algo, "dpo")

    def test_train_cli_preflight_rejects_agent_file_without_callable_run_agent(self):
        """Agent hooks should fail before rollout/backend-heavy work."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_path = Path(tmp) / "agent.py"
            agent_path.write_text("def helper():\n    pass\n", encoding="utf-8")

            with self.assertRaisesRegex(
                click.UsageError, r"--agent-fn .*agent.py must define callable run_agent\(ctx, batch\)"
            ):
                train_cli._trainer_config_from_options(
                    **_train_options(algo="gspo", reward_ckpt="reward-model", agent_fn=str(agent_path))
                )

    def test_train_cli_preflight_rejects_agent_fn_without_ctx_and_batch_args(self):
        """Agent hook should accept both ctx and batch arguments."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_path = Path(tmp) / "agent.py"
            agent_path.write_text("def run_agent(ctx):\n    return []\n", encoding="utf-8")

            with self.assertRaisesRegex(
                click.UsageError, r"--agent-fn .*agent.py must define callable run_agent\(ctx, batch\)"
            ):
                train_cli._trainer_config_from_options(
                    **_train_options(algo="gspo", reward_ckpt="reward-model", agent_fn=str(agent_path))
                )

    def test_train_command_reports_hook_usage_error_before_run(self):
        """Malformed hooks should stop the CLI before backend/model setup."""
        with tempfile.TemporaryDirectory() as tmp:
            reward_path = Path(tmp) / "reward.py"
            reward_path.write_text("def other(record):\n    return 0.0\n", encoding="utf-8")

            with patch.object(train_cli, "run") as run_mock:
                result = CliRunner().invoke(
                    train_cli.train_command,
                    [
                        "--algo",
                        "gspo",
                        "--ckpt",
                        "actor",
                        "--dataset-path",
                        "dataset",
                        "--reward-fn-path",
                        str(reward_path),
                    ],
                )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--reward-fn-path", result.output)
        self.assertIn("reward_fn(record)", result.output)
        run_mock.assert_not_called()


def _train_args(**overrides):
    defaults = dict(
        algo="sft",
        ckpt="unused",
        dataset_path="unused",
        dataset_loader_fn=None,
        reward_fn_path=None,
        save_path=None,
        save_interval=100,
        epochs=1,
        tp_size=1,
        world_size=1,
        batch_size=1,
        n_samples=1,
        mini_bs=1,
        gradient_accumulation_steps=None,
        max_prompt_tokens=128,
        max_new_tokens=16,
        max_context_len=None,
        greedy=False,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        max_running_prompts=None,
        lr=1e-6,
        min_lr=1e-7,
        lr_decay_steps=1000,
        lr_decay_style="cosine",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_8bit=False,
        weight_decay=1e-2,
        grad_clip_norm=1.0,
        activation_checkpointing=True,
        drop_rollout_state=False,
        eager_decode=False,
        attn_backend="flash",
        disable_thinking=False,
        metrics_log_dir=None,
        agent_fn=None,
        agent_timeout_s=300.0,
        train_tool_results=False,
        gspo_clip_eps=3.0e-4,
        grpo_clip_eps=0.2,
        ref_ckpt=None,
        dpo_beta=0.1,
        reward_ckpt=None,
        critic_ckpt=None,
        critic_lr=1e-5,
        use_kl_loss=True,
        kl_loss_coef=0.001,
        kl_loss_type="low_var_kl",
        clip_eps=0.2,
        clip_ratio_c=3.0,
        value_clip_eps=0.5,
        value_loss_coef=0.5,
        gamma=1.0,
        lam=1.0,
        critic_warmup_steps=20,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _train_options(**overrides):
    return vars(_train_args(**overrides))


if __name__ == "__main__":
    unittest.main()
