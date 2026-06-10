"""TensorBoard metric recording for the trainer loop.

`Trainer.train` calls `MetricsRecorder.record_train_step` after each backend
step. The recorder summarises the per-sequence rollout signals
(rewards/advantages/lengths/logprobs) and writes both rollout-side and
backend-reported training scalars to TensorBoard. Keeping the recording in a
single small module makes it trivial to turn off (just pass
`metrics_log_dir=None`) and easy to swap for another backend.
"""

from __future__ import annotations

import numpy as np


class MetricsRecorder:
    """Small TensorBoard facade used by `Trainer.train`."""

    def __init__(self, log_dir: str):
        """Create a writer under `log_dir`."""

        self._writer = create_tensorboard_writer(log_dir)
        self._closed = False

    def record_train_step(self, *, step: int, train_result, train_batch, timings: dict[str, float] | None = None):
        """Record rollout, training, and timing metrics for one step."""

        # `collect_train_batch_stats` extracts only the response-side numbers
        # (prompt tokens are masked out) so reported means match what the loss
        # actually trained on.
        stats = collect_train_batch_stats(train_batch)
        record_training_stats(self._writer, stats, step, train_result, train_batch, timings=timings)

    def close(self) -> None:
        """Flush and close the underlying TensorBoard writer."""

        if self._closed:
            return
        self._closed = True
        self._writer.close()

    def __enter__(self) -> "MetricsRecorder":
        """Return the recorder for context-manager usage."""

        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        """Close the writer on context-manager exit."""

        self.close()

    def __del__(self):
        """Best-effort writer cleanup when callers forget `close()`."""

        try:
            self.close()
        except Exception:
            pass


def create_tensorboard_writer(log_dir: str):
    """Create a TensorBoard writer from torch or tensorboardX.

    `torch.utils.tensorboard` is preferred when torch is installed; otherwise
    the older `tensorboardX` package is used. An explicit error is raised when
    neither is available so misconfigured environments fail loudly.
    """

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError as exc:
            raise ImportError("install TensorBoard: pip install tensorboard or pip install tensorboardX") from exc
    return SummaryWriter(log_dir=log_dir)


def collect_train_batch_stats(train_batch) -> dict:
    """Summarize rewards, advantages, lengths, and rollout logprobs.

    Walks every `TrainSequence` and isolates response-only slices using
    `prompt_mask`. Prompt positions carry zeroed advantage/logprob entries by
    construction (see trainer code), so dropping them gives an honest mean
    over the tokens that actually participated in the loss.
    """

    stats = init_rollout_stats()
    for seq in train_batch:
        prompt_mask = list(seq.prompt_mask)
        # `prompt_mask[i] == True` marks a prompt token; rollout signals only
        # exist on response positions so we filter the prompt prefix out.
        response_logprobs = [lp for lp, is_prompt in zip(seq.logprobs, prompt_mask) if not is_prompt]
        response_advantages = [adv for adv, is_prompt in zip(seq.advantages, prompt_mask) if not is_prompt]
        prefix_len = sum(1 for is_prompt in prompt_mask if is_prompt)
        response_len = len(response_logprobs)
        stats["rewards"].append(seq.reward)
        stats["advantages"].extend(response_advantages)
        record_rollout_sequence_stats(
            stats,
            prefix_len=prefix_len,
            response_logprobs=response_logprobs,
            response_len=response_len,
        )
    return stats


def init_rollout_stats(skipped_long: int = 0, total_skipped_long: int = 0) -> dict:
    """Create the mutable stats accumulator used by metric helpers."""

    return {
        "rewards": [],
        "logprobs": [],
        "advantages": [],
        "seq_len": [],
        "prompt_len": [],
        "response_len": [],
        "skipped_long": skipped_long,
        "total_skipped_long": total_skipped_long,
    }


def record_rollout_sequence_stats(stats, *, prefix_len: int, response_logprobs, response_len: int):
    """Append per-sequence rollout statistics into an accumulator."""

    stats["logprobs"].extend(response_logprobs)
    stats["seq_len"].append(prefix_len + response_len)
    stats["prompt_len"].append(prefix_len)
    stats["response_len"].append(response_len)


def record_training_stats(writer, stats, step, train_res, train_batch, timings: dict[str, float] | None = None):
    """Write scalar metrics for one training step to TensorBoard.

    Output channels:
        rollout/* - sample-side statistics computed from the train batch.
        train/*   - whatever scalars the backend returned in `train_res`.
        time/*    - per-stage wall times (rollout, reward, advantage, train).
    """

    rewards = np.asarray(stats.get("rewards", []), dtype=np.float32)
    advantages = np.asarray(stats.get("advantages", []), dtype=np.float32)
    logprobs = np.asarray(stats.get("logprobs", []), dtype=np.float32)

    if rewards.size:
        writer.add_scalar("rollout/rewards_mean", rewards.mean(), step)
        writer.add_scalar("rollout/rewards_std", rewards.std(), step)
        writer.add_scalar("rollout/rewards_max", rewards.max(), step)
        writer.add_scalar("rollout/rewards_min", rewards.min(), step)
        # Binary verifier rewards conventionally use {0,1}; recording the
        # fraction of strictly-positive rewards approximates pass-rate.
        writer.add_scalar("rollout/accuracy", (rewards > 0).mean(), step)
    if advantages.size:
        writer.add_scalar("rollout/advantages_mean", advantages.mean(), step)
        writer.add_scalar("rollout/advantages_std", advantages.std(), step)
    if logprobs.size:
        writer.add_scalar("rollout/logprobs_mean", logprobs.mean(), step)
    for key in ("seq_len", "prompt_len", "response_len"):
        values = stats.get(key, [])
        if values:
            writer.add_scalar(f"rollout/{key}_mean", np.mean(values), step)
    writer.add_scalar("rollout/num_sequences", len(train_batch), step)
    for key in ("skipped_long", "total_skipped_long"):
        if key in stats:
            writer.add_scalar(f"rollout/{key}", stats[key], step)

    # Backend-supplied training metrics (loss, policy_loss, ratio_mean, ...).
    for key, value in train_res.items():
        writer.add_scalar(f"train/{key}", value, step)
    metric_timings = timings or stats
    for key in ("rollout", "reward", "advantage", "train"):
        if key in metric_timings:
            writer.add_scalar(f"time/{key}", metric_timings[key], step)
    writer.flush()
