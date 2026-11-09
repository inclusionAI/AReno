"""Experimental asynchronous rollout/train policy trainer.

A background producer thread generates rollouts, scores rewards, and pushes
group-complete :class:`GroupBatch` units into a bounded queue while the main
thread consumes groups and runs optimizer steps. Rollout generation on the
rollout engine therefore overlaps training on the train engine.

Weight freshness follows the backend's policy-sync path: the producer opens a
rollout session per rollout batch, which synchronizes whenever the train
policy version advanced, and every queued group is stamped with the rollout
policy version it was generated from. The consumer drops (or rejects) groups
that lag the current policy by more than ``max_staleness``.

Note: step-accounting metrics (wall-clock step timers, global step counters)
are approximate under overlap because the producer and consumer share the
Trainer facade; gradient and weight semantics are unaffected.
"""

from __future__ import annotations

import asyncio
import threading

from areno.api.data import PromptBatch
from areno.api.tokenizer import configure_chat_template_enable_thinking
from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.experimental.async_policy.queueing import BoundedGroupQueue, GroupBatch, QueueClosed


class AsyncPolicyTrainer(PolicyOnlyTrainer):
    """Overlapped rollout/train loop for GRPO/GSPO-style policy RL."""

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        super().__init__(config, instance=instance, dataset=dataset, reward_fn=reward_fn, loss_fn=loss_fn)
        self._current_policy_version = 0
        self._stale_dropped = 0

    def _fit_initialized(self) -> None:
        import areno.api

        tokenizer = self.areno.get_tokenizer()
        processor = self.areno.get_processor()
        configure_chat_template_enable_thinking(tokenizer, getattr(self.config, "chat_template_enable_thinking", None))
        configure_chat_template_enable_thinking(processor, getattr(self.config, "chat_template_enable_thinking", None))
        sampling_params = areno.api.SamplingParams(
            greedy=self.config.greedy,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            max_context_len=getattr(self.config, "max_context_len", None),
            max_prompt_len=self.config.max_prompt_tokens,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )
        self.areno.enable_async_overlap()
        queue = BoundedGroupQueue(self._queue_max_groups())
        stop = threading.Event()
        producer_errors: list[BaseException] = []
        producer = threading.Thread(
            target=self._produce,
            args=(tokenizer, sampling_params, queue, stop, producer_errors),
            name="async-policy-rollout",
            daemon=True,
        )
        producer.start()
        try:
            self._consume(queue, stop)
        finally:
            stop.set()
            queue.close()
            producer.join()
        if producer_errors:
            raise producer_errors[0]

    # ------------------------------------------------------------------
    # Producer: rollout -> reward -> group batches -> bounded queue
    # ------------------------------------------------------------------

    def _produce(self, tokenizer, sampling_params, queue, stop, errors) -> None:
        try:
            for epoch in range(self.config.epochs):
                if stop.is_set():
                    return
                for prompt_batch in self.areno.load_prompt_batches(
                    self.dataset,
                    batch_size=self.config.batch_size,
                    max_prompt_tokens=self.config.max_prompt_tokens,
                ):
                    if stop.is_set():
                        return
                    rollout_results = asyncio.run(self._run_prompt_rollout(sampling_params, prompt_batch))
                    groups = self._materialize_group_batches(tokenizer, epoch, prompt_batch, rollout_results)
                    rewards = [reward for group in groups for reward in group.rewards]
                    if rewards:
                        self.logger.info(
                            "epoch=%d stage=rollout_end groups=%d metric=reward_mean value=%.6f",
                            epoch,
                            len(groups),
                            float(sum(rewards) / len(rewards)),
                        )
                    for group in groups:
                        queue.put(group, stop=stop)
        except QueueClosed:
            return
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer thread
            errors.append(exc)
        finally:
            queue.close()

    def _materialize_group_batches(self, tokenizer, epoch, prompt_batch, rollout_results) -> list[GroupBatch]:
        """Materialize one GroupBatch per prompt group of one rollout batch.

        Each group is materialized on its own so group-relative advantages
        stay within one prompt's `n_samples` completions and every group can
        carry the rollout policy version that produced it.
        """

        groups: list[GroupBatch] = []
        for prompt_index, (item, result) in enumerate(zip(prompt_batch.items, rollout_results, strict=True)):
            single = PromptBatch(items=[item], scanned=1, skipped_long=0, total_skipped_long=0)
            rows, rewards, _ = self._materialize_train_batch(tokenizer, single, [result])
            if not rows:
                continue
            # Stamp with the policy weight version the rollout engine had
            # synced when the batch was generated; `adapter_version` is the
            # adapter registry counter and only kept as a legacy fallback.
            stamped = result.policy_version if result.policy_version is not None else result.adapter_version
            groups.append(
                GroupBatch(
                    epoch=epoch,
                    prompt_index=prompt_index,
                    policy_version=int(stamped or 0),
                    sequences=rows,
                    rewards=rewards,
                )
            )
        return groups

    # ------------------------------------------------------------------
    # Consumer: bounded queue -> staleness filter -> train step
    # ------------------------------------------------------------------

    def _consume(self, queue: BoundedGroupQueue, stop: threading.Event) -> None:
        step = 0
        while True:
            groups = queue.get_many(self.config.batch_size, timeout=self._queue_get_timeout())
            if not groups:
                if queue.closed:
                    break
                continue
            kept = self._filter_stale(groups)
            if not kept:
                continue
            rows = [seq for group in kept for seq in group.sequences]
            epoch = kept[0].epoch
            self.logger.info(
                "epoch=%d step=%d role=%s stage=train_start groups=%d rows=%d queue_depth=%d stale_dropped=%d",
                epoch,
                step,
                self._policy_role_name(),
                len(kept),
                len(rows),
                queue.depth,
                self._stale_dropped,
            )
            result = self.areno.train(
                rows,
                self.loss_fn,
                mini_bs=self.config.mini_bs,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            )
            self._advance_policy_version(result)
            self.logger.info(
                "epoch=%d step=%d role=%s stage=train_end policy_version=%d train_stats=%s",
                epoch,
                step,
                self._policy_role_name(),
                self._current_policy_version,
                result,
            )
            self._maybe_save(epoch, step)
            step += 1
            if self.config.max_steps is not None and step >= self.config.max_steps:
                self.logger.info("epoch=%d step=%d stage=max_steps_reached", epoch, step)
                stop.set()
                return

    def _filter_stale(self, groups: list[GroupBatch]) -> list[GroupBatch]:
        """Drop or reject groups that lag the current policy version."""

        max_staleness = getattr(self.config, "max_staleness", None)
        if max_staleness is None:
            return groups
        kept: list[GroupBatch] = []
        for group in groups:
            staleness = self._current_policy_version - group.policy_version
            if staleness <= max_staleness:
                kept.append(group)
                continue
            self._stale_dropped += 1
            if self._stale_policy() == "error":
                raise RuntimeError(
                    f"group epoch={group.epoch} prompt={group.prompt_index} has staleness {staleness} "
                    f"(policy_version={group.policy_version}, current={self._current_policy_version}), "
                    f"exceeding max_staleness={max_staleness}"
                )
            self.logger.warning(
                "dropping stale group epoch=%d prompt=%d policy_version=%d current=%d staleness=%d",
                group.epoch,
                group.prompt_index,
                group.policy_version,
                self._current_policy_version,
                staleness,
            )
        return kept

    def _advance_policy_version(self, result) -> None:
        version = result.get("adapter_version") if isinstance(result, dict) else None
        self._current_policy_version = int(version) if version is not None else self._current_policy_version + 1

    # ------------------------------------------------------------------
    # Config accessors with safe defaults for plain PolicyTrainerConfig
    # ------------------------------------------------------------------

    def _queue_max_groups(self) -> int:
        return int(getattr(self.config, "queue_max_groups", 8))

    def _queue_get_timeout(self) -> float:
        return float(getattr(self.config, "queue_get_timeout_s", 1.0))

    def _stale_policy(self) -> str:
        return str(getattr(self.config, "stale_policy", "drop"))


__all__ = ["AsyncPolicyTrainer"]
