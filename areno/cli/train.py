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

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import click
from areno.api.algorithms import get_algorithm
from areno.api.defaults import DEFAULT_METRICS_LOG_DIR
from areno.api.trainer_config import DPOTrainerConfig, PPOTrainerConfig, PolicyTrainerConfig, TrainerConfig
from areno.cli.model_refs import resolve_model_refs_for_config


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
    if args.lr_decay_steps <= 0:
        raise click.UsageError("--lr-decay-steps must be positive")
    if args.critic_warmup_steps < 0:
        raise click.UsageError("--critic-warmup-steps must be non-negative")
    return _trainer_config_from_args(args)


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
    import areno.api
    from datasets import load_dataset, load_from_disk

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
    default_loader = lambda path: _load_dataset_from_path(path, load_dataset=load_dataset, load_from_disk=load_from_disk)
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
    from datasets import Dataset
    import pandas as pd

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
    return sorted(str(item) for item in path.iterdir() if item.is_file() and item.suffix.lower() in _SUPPORTED_DATASET_SUFFIXES)


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
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run SFT, DPO, GSPO, GRPO, or PPO training with the areno backend.",
)
@click.option("--algo", type=str, default="gspo", show_default=True, help="Training algorithm registered in areno.api.")
@click.option("--ckpt", default=None, help="Actor model/tokenizer checkpoint path or Hugging Face repo ID.")
@click.option("--dataset-path", default=None, help="Training dataset path, HF save_to_disk directory, or HF dataset ref.")
@click.option("--dataset-loader-fn", default=None, help="Optional Python dataset loader function as file.py or file.py:function.")
@click.option("--reward-fn-path", default=None, help="Python file defining reward_fn(record).")
@click.option("--ref-ckpt", default=None, help="Optional PPO/DPO reference model checkpoint path or Hugging Face repo ID.")
@click.option("--reward-ckpt", default=None, help="Optional PPO reward model checkpoint path or Hugging Face repo ID.")
@click.option("--critic-ckpt", default=None, help="Optional PPO critic model checkpoint path or Hugging Face repo ID.")
@click.option("--save-path", default=None, help="Optional checkpoint output directory.")
@click.option("--save-interval", type=int, default=100, show_default=True, help="Save checkpoint every N train steps.")
@click.option("--metrics-log-dir", default=DEFAULT_METRICS_LOG_DIR, show_default=True, help="TensorBoard metrics log directory.")
@click.option("--epochs", type=int, default=10, show_default=True, help="Number of dataset epochs to train.")
@click.option("--tp-size", type=int, default=4, show_default=True, help="Tensor parallel size for the backend.")
@click.option("--world-size", type=int, default=8, show_default=True, help="Total device count for the backend.")
@click.option("--batch-size", type=int, default=32, show_default=True, help="Prompt/pair batch size.")
@click.option("--n-samples", type=int, default=8, show_default=True, help="Rollout samples per prompt for RL algorithms.")
@click.option("--mini-bs", type=int, default=16, show_default=True, help="Backend training microbatch size.")
@click.option(
    "--gradient-accumulation-steps",
    type=int,
    default=None,
    help="Optimizer step interval in microbatches; defaults to accumulating all mini-batches in one train call.",
)
@click.option("--max-prompt-tokens", type=int, default=1024, show_default=True, help="Maximum tokenized prompt length.")
@click.option("--max-new-tokens", type=int, default=3071, show_default=True, help="Maximum generated or supervised response tokens.")
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
@click.option("--agent-fn", default=None, help="Python file defining async run_agent(ctx, batch) for agentic rollout.")
@click.option("--agent-timeout-s", type=float, default=300.0, show_default=True, help="Agentic rollout proxy request timeout.")
@click.option("--train-tool-results", is_flag=True, help="Include tool-result spans in agentic policy loss.")
@click.option("--gspo-clip-eps", type=float, default=3.0e-4, show_default=True, help="GSPO sequence-ratio clipping epsilon.")
@click.option("--grpo-clip-eps", type=float, default=0.2, show_default=True, help="GRPO token-ratio clipping epsilon.")
@click.option("--dpo-beta", type=float, default=0.1, show_default=True, help="DPO preference margin temperature.")
@click.option("--critic-warmup-steps", type=int, default=20, show_default=True, help="PPO critic-only warmup steps before actor updates.")
@click.option("--critic-lr", type=float, default=1.0e-5, show_default=True, help="PPO critic optimizer learning rate.")
@click.option("--use-kl-loss/--no-use-kl-loss", default=True, show_default=True, help="Enable PPO actor KL loss.")
@click.option("--kl-loss-coef", type=float, default=0.001, show_default=True, help="PPO actor KL loss coefficient.")
@click.option("--kl-loss-type", default="low_var_kl", show_default=True, help="PPO actor KL loss type.")
@click.option("--clip-eps", type=float, default=0.2, show_default=True, help="PPO policy clipping epsilon.")
@click.option("--clip-ratio-c", type=float, default=3.0, show_default=True, help="PPO lower policy clipping bound multiplier.")
@click.option("--value-clip-eps", type=float, default=0.5, show_default=True, help="PPO value clipping epsilon.")
@click.option("--value-loss-coef", type=float, default=0.5, show_default=True, help="PPO value loss coefficient.")
@click.option("--gamma", type=float, default=1.0, show_default=True, help="PPO GAE discount.")
@click.option("--lam", type=float, default=0.95, show_default=True, help="PPO GAE lambda.")
def train_command(**options) -> None:
    """Click entrypoint for training."""

    trainer_config = _trainer_config_from_options(**options)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(trainer_config)


def main() -> None:
    """Console-script entrypoint for `areno train`."""

    train_command.main(prog_name="areno train")


if __name__ == "__main__":
    main()
