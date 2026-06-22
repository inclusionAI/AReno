"""Command-line entrypoint for SFT/DPO/GSPO/GRPO/PPO training.

The flow is:
    1. `train_command` collects Click options and builds either a
       `TrainerConfig` (sft/gspo/grpo), `DPOTrainerConfig`, or
       `PPOTrainerConfig`.
    2. The algorithm registry selects the default loss function and concrete
       trainer implementation for the requested algorithm.
    3. `run` constructs an `areno.api.Trainer` with the areno backend,
       loads the dataset (optionally through an explicit dataset loader
       function), and runs the trainer to completion.
"""

import ast
import importlib.util
import logging
import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace

import click

from areno.api.algorithms import get_algorithm
from areno.api.defaults import DEFAULT_METRICS_LOG_DIR
from areno.api.trainer_config import (
    DPOTrainerConfig,
    PolicyTrainerConfig,
    PPOTrainerConfig,
    RolloutTrainerConfig,
    TrainerConfig,
)
from areno.cli.model_refs import resolve_model_refs_for_config
from areno.engine.config import (
    ModelConfig,
    flash_attention_unsupported_gpu_reason,
    flash_attention_unsupported_model_reason,
)

# Group `areno train --help` flags by user intent rather than as one flat wall.
# Each entry is (section title, option param names in display order). Every
# declared option must appear in exactly one group; the help renderer keeps any
# unlisted option (e.g. auto-added --help) under a trailing catch-all so the
# help output stays complete. Keep these titles in sync with the section
# headings in docs/cli/training.rst.
TRAIN_OPTION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Grouped by RL-loop phase: Basic (what to run + devices), Rollout
    # (generate + score completions), Train (consume rollouts + update
    # weights), then the produced artifacts (Checkpoint) and logs
    # (Observability). Flags within Train are ordered by sub-area
    # (batching/memory -> optimizer -> reference/critic models -> loss).
    (
        "Basic",
        (
            "algo",
            "ckpt",
            "dataset_path",
            "dataset_loader_fn",
            "epochs",
            "world_size",
            "tp_size",
        ),
    ),
    (
        "Rollout",
        (
            "batch_size",
            "n_samples",
            "max_running_prompts",
            "max_prompt_tokens",
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "greedy",
            "eager_decode",
            "drop_rollout_state",
            "attn_backend",
            "agent_fn",
            "agent_timeout_s",
            "train_tool_results",
            "reward_fn_path",
            "reward_ckpt",
        ),
    ),
    (
        "Train",
        (
            "mini_bs",
            "gradient_accumulation_steps",
            "activation_checkpointing",
            "lr",
            "min_lr",
            "lr_decay_steps",
            "lr_decay_style",
            "adam_beta1",
            "adam_beta2",
            "adam_8bit",
            "weight_decay",
            "grad_clip_norm",
            "ref_ckpt",
            "critic_ckpt",
            "critic_lr",
            "critic_warmup_steps",
            "gspo_clip_eps",
            "grpo_clip_eps",
            "dpo_beta",
            "use_kl_loss",
            "kl_loss_coef",
            "kl_loss_type",
            "clip_eps",
            "clip_ratio_c",
            "value_clip_eps",
            "value_loss_coef",
            "gamma",
            "lam",
        ),
    ),
    ("Checkpoint", ("save_path", "save_interval")),
    ("Observability", ("metrics_log_dir",)),
)


class GroupedOptionsCommand(click.Command):
    """A Click command that renders --help options under intent-based sections."""

    def __init__(self, *args, option_groups: tuple[tuple[str, tuple[str, ...]], ...] = (), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.option_groups = option_groups

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        params = self.get_params(ctx)
        params_by_name = {param.name: param for param in params}
        grouped_names: set[str] = set()
        for title, names in self.option_groups:
            records = []
            for name in names:
                param = params_by_name.get(name)
                if param is None:
                    continue
                record = param.get_help_record(ctx)
                if record is not None:
                    records.append(record)
                    grouped_names.add(name)
            if records:
                with formatter.section(title):
                    formatter.write_dl(records)
        # Keep help complete: any option not placed in a group (notably the
        # auto-added --help) is still shown under a trailing catch-all.
        leftover = [
            record
            for param in params
            if param.name not in grouped_names and (record := param.get_help_record(ctx)) is not None
        ]
        if leftover:
            with formatter.section("Other options"):
                formatter.write_dl(leftover)


def _trainer_config_from_options(**options) -> TrainerConfig:
    """Build a typed trainer config from Click option values."""

    args = SimpleNamespace(**options)
    # Required-argument checks live here so offline trainers can omit reward
    # inputs while RL algorithms still require a reward function or model.
    if args.ckpt is None:
        raise click.UsageError("--ckpt is required")
    if args.dataset_path is None:
        raise click.UsageError("--dataset-path is required")
    algorithm = _algorithm_for_cli(args.algo)
    if algorithm.requires_rollout and args.reward_fn_path is None and args.reward_ckpt is None:
        raise click.UsageError("--reward-fn-path or --reward-ckpt is required")
    if args.save_interval <= 0:
        raise click.UsageError("--save-interval must be positive")
    if args.epochs <= 0:
        raise click.UsageError("--epochs must be positive")
    if args.tp_size <= 0:
        raise click.UsageError("--tp-size must be positive")
    if args.world_size <= 0:
        raise click.UsageError("--world-size must be positive")
    if args.world_size % args.tp_size != 0:
        raise click.UsageError("--world-size must be divisible by --tp-size")
    if args.batch_size <= 0:
        raise click.UsageError("--batch-size must be positive")
    if algorithm.requires_rollout and args.n_samples <= 0:
        raise click.UsageError("--n-samples must be positive")
    if args.mini_bs <= 0:
        raise click.UsageError("--mini-bs must be positive")
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps <= 0:
        raise click.UsageError("--gradient-accumulation-steps must be positive")
    if args.max_prompt_tokens <= 0:
        raise click.UsageError("--max-prompt-tokens must be positive")
    if args.max_new_tokens <= 0:
        raise click.UsageError("--max-new-tokens must be positive")
    if algorithm.requires_rollout and args.max_running_prompts is not None and args.max_running_prompts <= 0:
        raise click.UsageError("--max-running-prompts must be positive")
    if args.agent_timeout_s <= 0:
        raise click.UsageError("--agent-timeout-s must be positive")
    _require_positive_float(args.lr, "--lr")
    if args.min_lr < 0:
        raise click.UsageError("--min-lr must be non-negative")
    if args.lr_decay_steps <= 0:
        raise click.UsageError("--lr-decay-steps must be positive")
    _require_positive_float(args.adam_beta1, "--adam-beta1")
    _require_positive_float(args.adam_beta2, "--adam-beta2")
    if args.weight_decay < 0:
        raise click.UsageError("--weight-decay must be non-negative")
    _require_positive_float(args.grad_clip_norm, "--grad-clip-norm")
    if algorithm.requires_rollout:
        _require_positive_float(args.temperature, "--temperature")
        _require_positive_float(args.top_p, "--top-p")
    if algorithm.name == "gspo":
        _require_positive_float(args.gspo_clip_eps, "--gspo-clip-eps")
    if algorithm.name == "grpo":
        _require_positive_float(args.grpo_clip_eps, "--grpo-clip-eps")
    if algorithm.name == "dpo":
        _require_positive_float(args.dpo_beta, "--dpo-beta")
    if algorithm.name == "ppo":
        _require_positive_float(args.critic_lr, "--critic-lr")
        _require_positive_float(args.kl_loss_coef, "--kl-loss-coef")
        _require_positive_float(args.clip_eps, "--clip-eps")
        _require_positive_float(args.clip_ratio_c, "--clip-ratio-c")
        _require_positive_float(args.value_clip_eps, "--value-clip-eps")
        _require_positive_float(args.value_loss_coef, "--value-loss-coef")
        _require_positive_float(args.gamma, "--gamma")
        _require_positive_float(args.lam, "--lam")
    if args.critic_warmup_steps < 0:
        raise click.UsageError("--critic-warmup-steps must be non-negative")
    _preflight_task_hooks(args, algorithm)
    return _trainer_config_from_args(args)


def _require_positive_float(value: float, option_name: str) -> None:
    if value <= 0:
        raise click.UsageError(f"{option_name} must be positive")


def _format_training_config_summary(
    config: TrainerConfig,
    *,
    reward_ckpt: str | None = None,
    model_config: ModelConfig | None = None,
    color: bool = False,
) -> str:
    """Return a concise user-facing summary of the resolved train config."""

    algorithm = get_algorithm(config.algo)
    attn_backend, attn_warning = _resolved_attn_backend_for_summary(config, model_config=model_config)
    sections = [
        (
            "Algorithm",
            [
                ("name", algorithm.name),
                ("default_loss", _callable_name(algorithm.default_loss_fn)),
                ("requires_rollout", _format_bool(algorithm.requires_rollout)),
            ],
        ),
        (
            "Inputs",
            [
                ("ckpt", config.ckpt),
                ("dataset_path", config.dataset_path),
                ("dataset_loader", _format_optional(config.dataset_loader_fn)),
                (
                    "reward_fn",
                    _format_optional(config.reward_fn_path) if isinstance(config, PolicyTrainerConfig) else "n/a",
                ),
                ("reward_ckpt", _format_optional(_reward_ckpt_for_summary(config, reward_ckpt))),
                ("agent_fn", _format_optional(config.agent_fn)),
            ],
        ),
        (
            "Runtime",
            [
                ("world_size", str(config.world_size)),
                ("tp_size", str(config.tp_size)),
                ("dp_size", _resolved_dp_size_for_summary(config)),
                ("attn_backend", attn_backend),
            ],
        ),
        ("Rollout", _rollout_summary_rows(config)),
        (
            "Training",
            [
                ("mini_bs", str(config.mini_bs)),
                ("gradient_accumulation_steps", _format_optional(config.gradient_accumulation_steps, default="auto")),
                (
                    "optimizer",
                    (
                        f"lr={config.optimizer_lr}, min_lr={config.optimizer_min_lr}, "
                        f"decay={config.lr_decay_style}/{config.lr_decay_steps}, "
                        f"betas=({config.optimizer_beta1}, {config.optimizer_beta2}), "
                        f"weight_decay={config.weight_decay}, adam_8bit={_format_bool(config.adam_8bit)}"
                    ),
                ),
                ("grad_clip_norm", str(config.grad_clip_norm)),
            ],
        ),
        (
            "Outputs",
            [
                ("save_path", _format_optional(config.save_path)),
                ("save_interval", str(config.save_interval)),
                ("metrics_log_dir", _format_optional(config.metrics_log_dir)),
            ],
        ),
    ]
    lines = [_style("AReno training config", fg="bright_white", bold=True, color=color)]
    if attn_warning is not None:
        lines.append(_style("WARNING", fg="yellow", bold=True, color=color) + f": {attn_warning}")
    for section, rows in sections:
        lines.extend(_format_summary_section(section, rows, color=color))
    if config.save_path is None:
        lines.append(
            _style("WARNING", fg="yellow", bold=True, color=color)
            + ": no checkpoint output path configured (--save-path); checkpoints will not be saved."
        )
    return "\n".join(lines)


def _print_training_config_summary(
    config: TrainerConfig, *, reward_ckpt: str | None = None, model_config: ModelConfig | None = None
) -> None:
    click.echo(
        _format_training_config_summary(config, reward_ckpt=reward_ckpt, model_config=model_config, color=True),
        color=True,
    )


def _format_summary_section(section: str, rows: list[tuple[str, str]], *, color: bool) -> list[str]:
    field_width = max([len("Field"), *(len(field) for field, _ in rows)])
    terminal_width = shutil.get_terminal_size(fallback=(100, 24)).columns
    value_width = max(20, terminal_width - field_width - 4)
    label = _style(section, fg="bright_blue", bold=True, color=color)
    rule = _style("-" * len(section), fg="bright_black", color=color)
    lines = ["", label, rule]
    for field, value in rows:
        wrapped = textwrap.wrap(value, width=value_width, break_long_words=True, break_on_hyphens=False) or [""]
        lines.append("  " + _style(field.ljust(field_width), fg="bright_black", color=color) + "  " + wrapped[0])
        padding = "  " + " " * field_width + "  "
        lines.extend(padding + line for line in wrapped[1:])
    return lines


def _rollout_summary_rows(config: TrainerConfig) -> list[tuple[str, str]]:
    base = [
        ("batch_size", str(config.batch_size)),
        ("max_prompt_tokens", str(config.max_prompt_tokens)),
        ("max_new_tokens", str(config.max_new_tokens)),
    ]
    if not isinstance(config, RolloutTrainerConfig):
        return [
            *base,
            ("n_samples", "n/a"),
            ("max_running_prompts", "n/a"),
            ("sampling", "n/a"),
        ]
    return [
        *base,
        ("n_samples", str(config.n_samples)),
        ("max_running_prompts", str(config.resolved_max_running_prompts())),
        (
            "sampling",
            (
                f"greedy={_format_bool(config.greedy)}, "
                f"temperature={config.temperature}, top_k={config.top_k}, top_p={config.top_p}"
            ),
        ),
    ]


def _reward_ckpt_for_summary(config: TrainerConfig, reward_ckpt: str | None) -> str | None:
    if isinstance(config, PPOTrainerConfig):
        return config.reward_ckpt
    return reward_ckpt


def _resolved_attn_backend_for_summary(
    config: TrainerConfig, *, model_config: ModelConfig | None = None
) -> tuple[str, str | None]:
    if config.attn_backend != "flash":
        return config.attn_backend, None
    reasons = [
        reason
        for reason in (
            flash_attention_unsupported_gpu_reason(list(range(config.world_size))),
            flash_attention_unsupported_model_reason(model_config) if model_config is not None else None,
        )
        if reason is not None
    ]
    if not reasons:
        return config.attn_backend, None
    reason = "; ".join(reasons)
    backend = f"native (auto fallback from flash: {reason})"
    warning = (
        f"flash-attn does not support the detected runtime configuration ({reason}); "
        "AReno will use attn_backend='native'. Native attention is a compatibility path and may be slower."
    )
    return backend, warning


def _model_config_for_summary(config: TrainerConfig) -> ModelConfig | None:
    if not Path(config.ckpt).exists():
        return None
    try:
        from areno.models.registry import config_from_hf

        return config_from_hf(config.ckpt)
    except Exception:
        return None


def _callable_name(fn) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


def _resolved_dp_size_for_summary(config: TrainerConfig) -> str:
    return str(config.world_size // config.tp_size) if config.tp_size else "n/a"


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _format_optional(value, *, default: str = "none") -> str:
    return str(value) if value is not None else default


def _style(text: str, *, color: bool, fg: str | None = None, bold: bool = False) -> str:
    return click.style(text, fg=fg, bold=bold) if color else text


def _preflight_task_hooks(args, algorithm) -> None:
    """Validate user task hook files before backend/model initialization."""

    if args.dataset_loader_fn is not None:
        try:
            loader_path, fn_name = _split_loader_fn_spec(args.dataset_loader_fn)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        _validate_python_callable(
            loader_path,
            fn_name,
            option_name="--dataset-loader-fn",
            expected=f"{fn_name}(...)",
            positional_args=1,
        )
    if algorithm.requires_rollout and args.reward_fn_path is not None:
        _validate_python_callable(
            Path(args.reward_fn_path).expanduser().resolve(),
            "reward_fn",
            option_name="--reward-fn-path",
            expected="reward_fn(record)",
            positional_args=1,
        )
    if args.agent_fn is not None:
        _validate_python_callable(
            Path(args.agent_fn).expanduser().resolve(),
            "run_agent",
            option_name="--agent-fn",
            expected="run_agent(ctx, batch)",
            positional_args=2,
        )


def _validate_python_callable(
    path: Path,
    symbol_name: str,
    *,
    option_name: str,
    expected: str,
    positional_args: int | None = None,
) -> None:
    if not path.exists():
        raise click.UsageError(f"{option_name} file does not exist: {path}; expected callable {expected}")
    if not path.is_file():
        raise click.UsageError(f"{option_name} path is not a file: {path}; expected callable {expected}")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise click.UsageError(
            f"{option_name} cannot parse Python file: {path}; expected callable {expected}; {type(exc).__name__}: {exc}"
        ) from exc
    function = _find_top_level_function(tree, symbol_name)
    if function is None:
        raise click.UsageError(f"{option_name} {path} must define callable {expected}")
    if positional_args is not None and not _function_accepts_positional_args(function, positional_args):
        raise click.UsageError(f"{option_name} {path} must define callable {expected}")


def _find_top_level_function(tree: ast.Module, symbol_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == symbol_name:
            return node
    return None


def _function_accepts_positional_args(function: ast.FunctionDef | ast.AsyncFunctionDef, count: int) -> bool:
    args = function.args
    positional_count = len(args.posonlyargs) + len(args.args)
    required_count = positional_count - len(args.defaults)
    if args.vararg is not None:
        return required_count <= count
    return required_count <= count <= positional_count


def _trainer_config_from_args(args) -> TrainerConfig:
    # Each algorithm gets the narrowest config dataclass it needs; offline
    # trainers do not receive rollout/reward/GSPO fields by construction.
    algorithm = get_algorithm(args.algo)
    if algorithm.name == "dpo":
        return DPOTrainerConfig(
            algo=algorithm.name,
            ckpt=args.ckpt,
            dataset_path=args.dataset_path,
            dataset_loader_fn=args.dataset_loader_fn,
            save_path=args.save_path,
            save_interval=args.save_interval,
            epochs=args.epochs,
            tp_size=args.tp_size,
            world_size=args.world_size,
            batch_size=args.batch_size,
            mini_bs=args.mini_bs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            optimizer_lr=args.lr,
            optimizer_min_lr=args.min_lr,
            lr_decay_steps=args.lr_decay_steps,
            lr_decay_style=args.lr_decay_style,
            optimizer_beta1=args.adam_beta1,
            optimizer_beta2=args.adam_beta2,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            adam_8bit=args.adam_8bit,
            activation_checkpointing=args.activation_checkpointing,
            keep_rollout_state=not args.drop_rollout_state,
            eager_decode=args.eager_decode,
            attn_backend=args.attn_backend,
            metrics_log_dir=args.metrics_log_dir,
            agent_fn=args.agent_fn,
            agent_timeout_s=args.agent_timeout_s,
            train_tool_results=args.train_tool_results,
            ref_ckpt=args.ref_ckpt,
            dpo_beta=args.dpo_beta,
        )
    if algorithm.name == "sft":
        return TrainerConfig(
            algo=algorithm.name,
            ckpt=args.ckpt,
            dataset_path=args.dataset_path,
            dataset_loader_fn=args.dataset_loader_fn,
            save_path=args.save_path,
            save_interval=args.save_interval,
            epochs=args.epochs,
            tp_size=args.tp_size,
            world_size=args.world_size,
            batch_size=args.batch_size,
            mini_bs=args.mini_bs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            optimizer_lr=args.lr,
            optimizer_min_lr=args.min_lr,
            lr_decay_steps=args.lr_decay_steps,
            lr_decay_style=args.lr_decay_style,
            optimizer_beta1=args.adam_beta1,
            optimizer_beta2=args.adam_beta2,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            adam_8bit=args.adam_8bit,
            activation_checkpointing=args.activation_checkpointing,
            keep_rollout_state=not args.drop_rollout_state,
            eager_decode=args.eager_decode,
            attn_backend=args.attn_backend,
            metrics_log_dir=args.metrics_log_dir,
            agent_fn=args.agent_fn,
            agent_timeout_s=args.agent_timeout_s,
            train_tool_results=args.train_tool_results,
        )
    if algorithm.name != "ppo":
        return PolicyTrainerConfig(
            algo=algorithm.name,
            ckpt=args.ckpt,
            dataset_path=args.dataset_path,
            dataset_loader_fn=args.dataset_loader_fn,
            reward_fn_path=args.reward_fn_path,
            save_path=args.save_path,
            save_interval=args.save_interval,
            epochs=args.epochs,
            tp_size=args.tp_size,
            world_size=args.world_size,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            mini_bs=args.mini_bs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            greedy=args.greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_running_prompts=args.max_running_prompts,
            optimizer_lr=args.lr,
            optimizer_min_lr=args.min_lr,
            lr_decay_steps=args.lr_decay_steps,
            lr_decay_style=args.lr_decay_style,
            optimizer_beta1=args.adam_beta1,
            optimizer_beta2=args.adam_beta2,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            adam_8bit=args.adam_8bit,
            activation_checkpointing=args.activation_checkpointing,
            keep_rollout_state=not args.drop_rollout_state,
            eager_decode=args.eager_decode,
            attn_backend=args.attn_backend,
            gspo_clip_eps=args.gspo_clip_eps,
            grpo_clip_eps=args.grpo_clip_eps,
            metrics_log_dir=args.metrics_log_dir,
            agent_fn=args.agent_fn,
            agent_timeout_s=args.agent_timeout_s,
            train_tool_results=args.train_tool_results,
        )
    return PPOTrainerConfig(
        algo=algorithm.name,
        ckpt=args.ckpt,
        dataset_path=args.dataset_path,
        dataset_loader_fn=args.dataset_loader_fn,
        reward_fn_path=args.reward_fn_path,
        save_path=args.save_path,
        save_interval=args.save_interval,
        epochs=args.epochs,
        tp_size=args.tp_size,
        world_size=args.world_size,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        mini_bs=args.mini_bs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_running_prompts=args.max_running_prompts,
        optimizer_lr=args.lr,
        optimizer_min_lr=args.min_lr,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        optimizer_beta1=args.adam_beta1,
        optimizer_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        adam_8bit=args.adam_8bit,
        activation_checkpointing=args.activation_checkpointing,
        keep_rollout_state=not args.drop_rollout_state,
        eager_decode=args.eager_decode,
        attn_backend=args.attn_backend,
        gspo_clip_eps=args.gspo_clip_eps,
        grpo_clip_eps=args.grpo_clip_eps,
        metrics_log_dir=args.metrics_log_dir,
        ref_ckpt=args.ref_ckpt,
        reward_ckpt=args.reward_ckpt,
        critic_ckpt=args.critic_ckpt,
        critic_lr=args.critic_lr,
        use_kl_loss=args.use_kl_loss,
        kl_loss_coef=args.kl_loss_coef,
        kl_loss_type=args.kl_loss_type,
        clip_eps=args.clip_eps,
        clip_ratio_c=args.clip_ratio_c,
        value_clip_eps=args.value_clip_eps,
        value_loss_coef=args.value_loss_coef,
        gamma=args.gamma,
        lam=args.lam,
        critic_warmup_steps=args.critic_warmup_steps,
        agent_fn=args.agent_fn,
        agent_timeout_s=args.agent_timeout_s,
        train_tool_results=args.train_tool_results,
    )


def run(trainer_config: TrainerConfig):
    """Build the trainer chosen by `--algo` and run `.fit()` to completion."""

    # Heavy dependencies are imported lazily so `python train.py --help`
    # does not pay the cost of importing torch/areno.
    from datasets import load_dataset, load_from_disk

    import areno.api
    from areno.api.rewards import load_reward_fn
    from areno.api.trainer_factory import build_trainer

    trainer_config = resolve_model_refs_for_config(trainer_config)
    loss_fn = _loss_fn_for_config(trainer_config)
    reward_fn_path = _reward_fn_path_for_config(trainer_config)
    reward_fn = load_reward_fn(reward_fn_path) if reward_fn_path else None

    api_trainer = areno.api.Trainer(
        trainer_config.world_size,
        trainer_config.ckpt,
        backend_type=areno.api.Areno,
        metrics_log_dir=trainer_config.metrics_log_dir,
        custom_config=trainer_config.areno_config(),
    )
    dataset = _load_dataset_for_training(
        trainer_config.dataset_path,
        dataset_loader_fn=trainer_config.dataset_loader_fn,
        load_dataset=load_dataset,
        load_from_disk=load_from_disk,
    )
    trainer = build_trainer(trainer_config, instance=api_trainer, dataset=dataset, reward_fn=reward_fn, loss_fn=loss_fn)
    trainer.fit()


def _loss_fn_for_config(config: TrainerConfig):
    """Return the registered default loss, with algorithm-specific knobs bound."""

    return get_algorithm(config.algo).make_loss_fn(config)


def _algorithm_for_cli(name: str):
    """Convert registry errors into Click usage errors for command-line users."""

    try:
        return get_algorithm(name)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _reward_fn_path_for_config(config: TrainerConfig) -> str | None:
    if isinstance(config, PolicyTrainerConfig):
        return config.reward_fn_path
    return None


def _load_dataset_for_training(dataset_path: str, *, dataset_loader_fn: str | None, load_dataset, load_from_disk):
    def default_loader(path):
        return _load_dataset_from_path(path, load_dataset=load_dataset, load_from_disk=load_from_disk)

    if dataset_loader_fn is not None:
        loader_fn = _load_dataset_loader_fn(dataset_loader_fn)
        return loader_fn(
            dataset_path,
            default_loader=default_loader,
            load_dataset=load_dataset,
            load_from_disk=load_from_disk,
        )
    return default_loader(dataset_path)


def _load_dataset_loader_fn(spec_text: str):
    loader_path, fn_name = _split_loader_fn_spec(spec_text)
    spec = importlib.util.spec_from_file_location(f"areno_example_dataset_loader_{abs(hash(loader_path))}", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load dataset loader from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, fn_name):
        raise AttributeError(f"{loader_path} must define {fn_name}(...)")
    loader_fn = getattr(module, fn_name)
    if not callable(loader_fn):
        raise TypeError(f"{loader_path}:{fn_name} is not callable")
    return loader_fn


def _split_loader_fn_spec(spec_text: str) -> tuple[Path, str]:
    if ":" in spec_text:
        path_text, fn_name = spec_text.rsplit(":", 1)
        if not path_text or not fn_name:
            raise ValueError(f"Invalid --dataset-loader-fn value: {spec_text}")
        return Path(path_text).resolve(), fn_name
    return Path(spec_text).resolve(), "load_training_dataset"


def _load_dataset_from_path(dataset_path: str, *, load_dataset, load_from_disk):
    # Existing local paths may be either HF `save_to_disk` outputs or raw
    # files. Non-existing values without a known suffix are treated as
    # Hugging Face dataset IDs such as `gsm8k:main` or `AI-MO/NuminaMath-TIR`.
    path = Path(dataset_path)
    if path.is_dir():
        try:
            return _select_train_split(load_from_disk(str(path)))
        except Exception:
            data_files = _directory_data_files(path)
            if not data_files:
                raise
            builder = _dataset_builder_for_suffix(Path(data_files[0]).suffix)
            return _load_raw_dataset_files(builder, data_files, load_dataset=load_dataset)
    if path.exists() or path.suffix.lower() in _SUPPORTED_DATASET_SUFFIXES:
        builder = _dataset_builder_for_suffix(path.suffix)
        return _load_raw_dataset_files(builder, str(path), load_dataset=load_dataset)
    return _load_hf_dataset_ref(dataset_path, load_dataset=load_dataset)


_SUPPORTED_DATASET_SUFFIXES = {".json", ".jsonl", ".parquet", ".csv", ".tsv", ".arrow"}


def _load_hf_dataset_ref(dataset_ref: str, *, load_dataset):
    # Keep one CLI arg while supporting common HF forms:
    #   repo/name
    #   repo/name:config
    #   repo/name:config:split
    parts = dataset_ref.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid Hugging Face dataset reference for --dataset-path: {dataset_ref}")
    name = parts[0]
    config = parts[1] if len(parts) >= 2 and parts[1] else None
    split = parts[2] if len(parts) == 3 and parts[2] else "train"
    if not name:
        raise ValueError(f"Invalid Hugging Face dataset reference for --dataset-path: {dataset_ref}")
    if config is None:
        try:
            return load_dataset(name, split=split)
        except ValueError as exc:
            # Some HF datasets (notably `gsm8k`) require a config but expose
            # the common default as `main`; keep `--dataset-path gsm8k` usable.
            if "Config name is missing" not in str(exc) or "'main'" not in str(exc):
                raise
            return load_dataset(name, "main", split=split)
    return load_dataset(name, config, split=split)


def _load_raw_dataset_files(builder: str, data_files, *, load_dataset):
    try:
        return load_dataset(builder, data_files=data_files, split="train")
    except TypeError:
        if builder != "parquet":
            raise
        return _load_parquet_without_hf_metadata(data_files)


def _load_parquet_without_hf_metadata(data_files):
    # Some HF parquet exports carry feature metadata that older/newer datasets
    # versions fail to deserialize. Reading through pandas ignores that metadata
    # and preserves the actual columns (prompt/chosen/rejected/etc.).
    import pandas as pd
    from datasets import Dataset

    files = data_files if isinstance(data_files, list) else [data_files]
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return Dataset.from_pandas(df, preserve_index=False)


def _select_train_split(dataset):
    # `load_from_disk` may return a single split or a DatasetDict; we always
    # want the train shard for RL training loops.
    if hasattr(dataset, "keys") and "train" in dataset:
        return dataset["train"]
    return dataset


def _directory_data_files(path: Path) -> list[str]:
    return sorted(
        str(item) for item in path.iterdir() if item.is_file() and item.suffix.lower() in _SUPPORTED_DATASET_SUFFIXES
    )


def _dataset_builder_for_suffix(suffix: str) -> str:
    # Maps file extensions to the HF `datasets` builder strings.
    suffix = suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix == ".arrow":
        return "arrow"
    raise ValueError(f"Unsupported dataset file suffix for --dataset-path: {suffix}")


@click.command(
    name="train",
    cls=GroupedOptionsCommand,
    option_groups=TRAIN_OPTION_GROUPS,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run SFT, DPO, GSPO, GRPO, or PPO training with the areno backend.",
)
@click.option("--algo", type=str, default="gspo", show_default=True, help="Training algorithm registered in areno.api.")
@click.option("--ckpt", default=None, help="Actor model/tokenizer checkpoint path or Hugging Face repo ID.")
@click.option(
    "--dataset-path", default=None, help="Training dataset path, HF save_to_disk directory, or HF dataset ref."
)
@click.option(
    "--dataset-loader-fn", default=None, help="Optional Python dataset loader function as file.py or file.py:function."
)
@click.option("--reward-fn-path", default=None, help="Python file defining reward_fn(record).")
@click.option(
    "--ref-ckpt", default=None, help="Optional PPO/DPO reference model checkpoint path or Hugging Face repo ID."
)
@click.option("--reward-ckpt", default=None, help="Optional PPO reward model checkpoint path or Hugging Face repo ID.")
@click.option("--critic-ckpt", default=None, help="Optional PPO critic model checkpoint path or Hugging Face repo ID.")
@click.option("--save-path", default=None, help="Optional checkpoint output directory.")
@click.option("--save-interval", type=int, default=100, show_default=True, help="Save checkpoint every N train steps.")
@click.option(
    "--metrics-log-dir", default=DEFAULT_METRICS_LOG_DIR, show_default=True, help="TensorBoard metrics log directory."
)
@click.option("--epochs", type=int, default=10, show_default=True, help="Number of dataset epochs to train.")
@click.option("--tp-size", type=int, default=4, show_default=True, help="Tensor parallel size for the backend.")
@click.option("--world-size", type=int, default=8, show_default=True, help="Total device count for the backend.")
@click.option("--batch-size", type=int, default=32, show_default=True, help="Prompt/pair batch size.")
@click.option(
    "--n-samples", type=int, default=8, show_default=True, help="Rollout samples per prompt for RL algorithms."
)
@click.option("--mini-bs", type=int, default=16, show_default=True, help="Backend training microbatch size.")
@click.option(
    "--gradient-accumulation-steps",
    type=int,
    default=None,
    help="Optimizer step interval in microbatches; defaults to accumulating all mini-batches in one train call.",
)
@click.option("--max-prompt-tokens", type=int, default=1024, show_default=True, help="Maximum tokenized prompt length.")
@click.option(
    "--max-new-tokens",
    type=int,
    default=3071,
    show_default=True,
    help="Maximum generated or supervised response tokens.",
)
@click.option("--temperature", type=float, default=1.0, show_default=True, help="Rollout sampling temperature.")
@click.option("--top-k", type=int, default=-1, show_default=True, help="Rollout top-k; -1 disables top-k filtering.")
@click.option("--top-p", type=float, default=1.0, show_default=True, help="Rollout top-p.")
@click.option("--greedy", is_flag=True, help="Use greedy rollout decoding.")
@click.option(
    "--max-running-prompts",
    type=int,
    default=None,
    help="Override global concurrent rollout prompts; defaults to batch-size * n-samples.",
)
@click.option("--lr", type=float, default=1.0e-6, show_default=True, help="Policy optimizer learning rate.")
@click.option("--min-lr", type=float, default=1.0e-7, show_default=True, help="Policy optimizer minimum learning rate.")
@click.option("--lr-decay-steps", type=int, default=1000, show_default=True, help="Policy LR decay steps.")
@click.option("--lr-decay-style", default="cosine", show_default=True, help="Policy LR decay style.")
@click.option("--adam-beta1", type=float, default=0.9, show_default=True, help="Policy optimizer Adam beta1.")
@click.option("--adam-beta2", type=float, default=0.999, show_default=True, help="Policy optimizer Adam beta2.")
@click.option("--adam-8bit", is_flag=True, help="Use 8-bit Adam moment states instead of FP32 Adam states.")
@click.option("--weight-decay", type=float, default=1.0e-2, show_default=True, help="Policy optimizer weight decay.")
@click.option("--grad-clip-norm", type=float, default=1.0, show_default=True, help="Policy gradient clipping norm.")
@click.option(
    "--activation-checkpointing/--no-activation-checkpointing",
    default=True,
    show_default=True,
    help="Enable decoder-layer activation recompute during training.",
)
@click.option(
    "--drop-rollout-state",
    is_flag=True,
    help="Drop rollout state after each step to save GPU memory.",
)
@click.option("--eager-decode", is_flag=True, help="Disable decode CUDA graph and run rollout decode eagerly.")
@click.option(
    "--attn-backend",
    type=click.Choice(["flash", "native"]),
    default="flash",
    show_default=True,
    help="Attention backend. Use native for slower areno_accel attention consistency diagnostics.",
)
@click.option("--agent-fn", default=None, help="Python file defining async run_agent(ctx, batch) for agentic rollout.")
@click.option(
    "--agent-timeout-s", type=float, default=300.0, show_default=True, help="Agentic rollout proxy request timeout."
)
@click.option("--train-tool-results", is_flag=True, help="Include tool-result spans in agentic policy loss.")
@click.option(
    "--gspo-clip-eps", type=float, default=3.0e-4, show_default=True, help="GSPO sequence-ratio clipping epsilon."
)
@click.option("--grpo-clip-eps", type=float, default=0.2, show_default=True, help="GRPO token-ratio clipping epsilon.")
@click.option("--dpo-beta", type=float, default=0.1, show_default=True, help="DPO preference margin temperature.")
@click.option(
    "--critic-warmup-steps",
    type=int,
    default=20,
    show_default=True,
    help="PPO critic-only warmup steps before actor updates.",
)
@click.option("--critic-lr", type=float, default=1.0e-5, show_default=True, help="PPO critic optimizer learning rate.")
@click.option("--use-kl-loss/--no-use-kl-loss", default=True, show_default=True, help="Enable PPO actor KL loss.")
@click.option("--kl-loss-coef", type=float, default=0.001, show_default=True, help="PPO actor KL loss coefficient.")
@click.option("--kl-loss-type", default="low_var_kl", show_default=True, help="PPO actor KL loss type.")
@click.option("--clip-eps", type=float, default=0.2, show_default=True, help="PPO policy clipping epsilon.")
@click.option(
    "--clip-ratio-c", type=float, default=3.0, show_default=True, help="PPO lower policy clipping bound multiplier."
)
@click.option("--value-clip-eps", type=float, default=0.5, show_default=True, help="PPO value clipping epsilon.")
@click.option("--value-loss-coef", type=float, default=0.5, show_default=True, help="PPO value loss coefficient.")
@click.option("--gamma", type=float, default=1.0, show_default=True, help="PPO GAE discount.")
@click.option("--lam", type=float, default=0.95, show_default=True, help="PPO GAE lambda.")
def train_command(**options) -> None:
    """Click entrypoint for training."""

    trainer_config = _trainer_config_from_options(**options)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _print_training_config_summary(
        trainer_config,
        reward_ckpt=options.get("reward_ckpt"),
        model_config=_model_config_for_summary(trainer_config),
    )
    run(trainer_config)


def main() -> None:
    """Console-script entrypoint for `areno train`."""

    train_command.main(prog_name="areno train")


if __name__ == "__main__":
    main()
