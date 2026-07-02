"""Reference, critic, and reward role management for `ArenoWorker`."""

from __future__ import annotations

import torch

from areno.engine.checkpoints.io import SafetensorsIndex
from areno.engine.config import EngineConfig
from areno.engine.data import to_device
from areno.engine.modeling import build_model_on_device, build_optimizer, canonical_model_path, param_grad, unwrap_model
from areno.engine.optim import AdamW8bit, AdamWFP32Master
from areno.engine.parallel.context import get_tp_context
from areno.engine.protocol import EnsureRolesPayload, ScorePayload, TrainValuesPayload
from areno.engine.runtime.logprobs import next_token_logprobs
from areno.engine.runtime.train_step import _dense_train_meta
from areno.models.registry import config_from_hf, load_model_weights

_REWARD_HEAD_WEIGHT_KEYS = (
    "score.weight",
    "v_head.weight",
    "classifier.weight",
    "reward_head.weight",
    "value_head.weight",
    "model.score.weight",
    "model.v_head.weight",
    "model.classifier.weight",
    "model.reward_head.weight",
    "model.value_head.weight",
)


def _reward_head_bias_key(weight_key: str) -> str:
    """Map a `*.weight` head key to its matching `*.bias` key."""

    return weight_key.removesuffix(".weight") + ".bias"


def _reward_head_out_features(path: str) -> int:
    """Return the head's output dimension; raises if no reward head is found."""

    out_features = _maybe_reward_head_out_features(path)
    if out_features is not None:
        return out_features
    raise KeyError("reward model checkpoint must contain one of: " + ", ".join(_REWARD_HEAD_WEIGHT_KEYS))


def _maybe_reward_head_out_features(path: str) -> int | None:
    """Probe a safetensors checkpoint for a reward head and return its out dim."""

    index = SafetensorsIndex(path, progress=False)
    try:
        for key in _REWARD_HEAD_WEIGHT_KEYS:
            if key not in index:
                continue
            shape = index.get_shape(key)
            if len(shape) == 1:
                return 1
            if len(shape) != 2:
                raise ValueError(f"reward head {key} must be 1-D or 2-D, got shape {shape}")
            return int(shape[0])
    finally:
        index.close()
    return None


def _reward_head_uses_bias(path: str) -> bool:
    """Return True if the checkpoint stores a bias for the selected head."""

    index = SafetensorsIndex(path, progress=False)
    try:
        for key in _REWARD_HEAD_WEIGHT_KEYS:
            if key in index:
                return _reward_head_bias_key(key) in index
    finally:
        index.close()
    return False


@torch.no_grad()
def _load_reward_head(head: torch.nn.Linear | None, path: str) -> None:
    """Load a reward/value head from a safetensors checkpoint."""

    if not _try_load_reward_head(head, path):
        raise KeyError("reward model checkpoint must contain one of: " + ", ".join(_REWARD_HEAD_WEIGHT_KEYS))


@torch.no_grad()
def _try_load_reward_head(head: torch.nn.Linear | None, path: str) -> bool:
    """Best-effort load of a reward / value head from a safetensors checkpoint."""

    if head is None:
        raise RuntimeError("reward role requires a scalar reward head")
    index = SafetensorsIndex(path, progress=False)
    try:
        for key in _REWARD_HEAD_WEIGHT_KEYS:
            if key not in index:
                continue
            weight = index.get_tensor(key).to(device=head.weight.device, dtype=head.weight.dtype)
            if weight.ndim == 1:
                weight = weight.reshape(1, -1)
            if weight.shape == head.weight.shape:
                head.weight.copy_(weight)
            elif weight.T.shape == head.weight.shape:
                head.weight.copy_(weight.T.contiguous())
            else:
                raise ValueError(
                    f"reward head {key} has shape {tuple(weight.shape)}, expected {tuple(head.weight.shape)}"
                )
            bias_key = _reward_head_bias_key(key)
            if head.bias is not None:
                if bias_key not in index:
                    raise KeyError(f"reward head checkpoint is missing {bias_key}")
                bias = index.get_tensor(bias_key).to(device=head.bias.device, dtype=head.bias.dtype).flatten()
                if bias.shape != head.bias.shape:
                    raise ValueError(
                        f"reward head {bias_key} has shape {tuple(bias.shape)}, expected {tuple(head.bias.shape)}"
                    )
                head.bias.copy_(bias)
            return True
    finally:
        index.close()
    return False


@torch.no_grad()
def _zero_init_value_head(value_head: torch.nn.Module | None) -> None:
    """Zero out the value head when no checkpoint provides one."""

    if value_head is None:
        return
    for param in value_head.parameters():
        param.zero_()


def accumulate_role_main_gradients(role: WorkerRole) -> None:
    """Fold `param.grad` into `param.main_grad` (FP32) for a role's params."""

    for param in role.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        main_grad = getattr(param, "main_grad", None)
        if isinstance(main_grad, torch.Tensor):
            main_grad.add_(grad.to(dtype=main_grad.dtype))
        else:
            param.main_grad = grad.to(dtype=torch.float32)
        param.grad = None


class WorkerRole:
    """A swap-in non-actor model (reference / critic / reward) on this rank."""

    def __init__(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: AdamW8bit | AdamWFP32Master | None,
        value_head: torch.nn.Module | None,
    ):
        self.path = path
        self.model = model
        self.optimizer = optimizer
        self.value_head = value_head

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        device: torch.device,
        trainable: bool,
        critic: bool,
        reward: bool,
        optimizer_config,
        runtime_config,
        tp_size: int,
        dp_size: int,
        devices: list[int] | None,
        optimizer_lr: float | None = None,
        source_model: torch.nn.Module | None = None,
    ) -> WorkerRole:
        """Construct a role from a HF-style checkpoint at `path`."""

        model_config = config_from_hf(path)
        role_config = EngineConfig(
            model=model_config,
            model_path=path,
            train_loss_fn=lambda _pack, logprobs: logprobs.sum() * 0.0,
            optimizer=optimizer_config,
            runtime=runtime_config,
            tp_size=tp_size,
            dp_size=dp_size,
            devices=devices,
            dummy_load=False,
        )
        model = build_model_on_device(role_config, device)
        if source_model is None:
            load_model_weights(model, model_config, path)
        else:
            model.load_state_dict(unwrap_model(source_model).state_dict())
        checkpoint_head_out = _maybe_reward_head_out_features(path) if critic else None
        head_out = _reward_head_out_features(path) if reward else checkpoint_head_out or 1
        head_bias = _reward_head_uses_bias(path) if reward or checkpoint_head_out is not None else False
        value_head = (
            torch.nn.Linear(model_config.hidden_size, head_out, bias=head_bias, dtype=model_config.dtype, device=device)
            if critic or reward
            else None
        )
        if value_head is not None:
            for param in value_head.parameters():
                param.role_tp_average = True
        if reward:
            _load_reward_head(value_head, path)
        elif critic and not _try_load_reward_head(value_head, path):
            _zero_init_value_head(value_head)
        params = list(model.parameters()) + ([] if value_head is None else list(value_head.parameters()))
        optimizer = None
        if trainable:
            ctx = get_tp_context()
            optimizer = build_optimizer(params, optimizer_config, ctx, lr=optimizer_lr)
        return cls(path, model, optimizer, value_head)

    def parameters(self):
        """Iterate over model params then value-head params when present."""

        yield from self.model.parameters()
        if self.value_head is not None:
            yield from self.value_head.parameters()

    def onload(self, device: torch.device) -> None:
        """Move this role's model, value head, and optimizer state to `device`."""

        self.model.to(device)
        self.model.onload_train_weights(device)
        if self.value_head is not None:
            self.value_head.to(device)
        if self.optimizer is not None:
            self.optimizer.onload_state(device)

    def offload(self) -> None:
        """Free all HBM held by this role."""

        self.model.clear_infer_weights()
        self.model.clear_kv_caches()
        self.model.offload_train_weights()
        self.model.to("cpu")
        if self.value_head is not None:
            self.value_head.to("cpu")
        if self.optimizer is not None:
            self.optimizer.offload_state()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RoleManager:
    """Own reference, critic, and reward roles for a worker rank."""

    def __init__(self, worker):
        self.worker = worker
        self.roles: dict[str, WorkerRole] = {}

    def ensure_roles(self, payload: EnsureRolesPayload) -> None:
        """Lazily instantiate non-actor roles."""

        worker = self.worker
        worker._prepare_actor_offloaded()
        model_sources: dict[str, torch.nn.Module] = {}
        actor_path = canonical_model_path(worker.config.model_path)
        if actor_path is not None:
            model_sources[actor_path] = unwrap_model(worker.model)
        for role in self.roles.values():
            role_path = canonical_model_path(role.path)
            if role_path is not None:
                model_sources.setdefault(role_path, role.model)
        for name, spec in payload.roles.items():
            if name == "actor" or name in self.roles:
                continue
            path = spec.path
            cache_key = canonical_model_path(path)
            trainable = bool(spec.trainable)
            role = WorkerRole.from_pretrained(
                path,
                device=worker.device,
                trainable=trainable,
                critic=name == "critic",
                reward=name == "reward",
                optimizer_config=worker.config.optimizer,
                runtime_config=worker.config.runtime,
                tp_size=worker.config.tp_size,
                dp_size=int(worker.config.dp_size),
                devices=worker.config.devices,
                optimizer_lr=spec.optimizer_lr,
                source_model=model_sources.get(cache_key) if cache_key is not None else None,
            )
            role.offload()
            self.roles[name] = role
            if cache_key is not None:
                model_sources.setdefault(cache_key, role.model)

    @torch.inference_mode()
    def score_logprobs(self, payload: ScorePayload) -> list[list[float]] | None:
        """Compute per-token logprobs for either the actor or a swap-in role."""

        worker = self.worker
        ctx = get_tp_context()
        role_name = payload.role
        if role_name == "actor":
            worker._prepare_actor_onloaded()
            model = worker.model
            offload_role = None
        else:
            offload_role = self.roles[role_name]
            worker._prepare_actor_offloaded()
            offload_role.onload(worker.device)
            model = offload_role.model
        model.eval()
        try:
            token_rows = payload.token_rows_by_dp[ctx.dp_rank]
            local = [] if not token_rows else self._score_logprob_rows(model, token_rows, payload)
            return local if ctx.rank == 0 else None
        finally:
            if offload_role is not None:
                offload_role.offload()

    def _score_logprob_rows(self, model, token_rows: list[list[int]], payload: ScorePayload) -> list[list[float]]:
        """Score token logprobs in bounded microbatches."""

        local = []
        microbatch_size = _score_microbatch_size(payload.microbatch_size)
        for start in range(0, len(token_rows), microbatch_size):
            rows = token_rows[start : start + microbatch_size]
            tokens, lengths = _pad_token_rows(rows, self.worker.device, int(payload.pad_token_id))
            out = model(input_ids=tokens, train_meta=_dense_train_meta(tokens, sequence_parallel_enabled=False))
            logprobs = next_token_logprobs(out.logits_shard, tokens)
            local.extend(_unpad_action_rows(logprobs, lengths))
        return local

    @torch.inference_mode()
    def score_values(self, payload: ScorePayload) -> list[list[float]] | None:
        """Compute per-token critic values; always runs on a non-actor role."""

        worker = self.worker
        ctx = get_tp_context()
        role = self.roles[payload.role]
        worker._prepare_actor_offloaded()
        role.onload(worker.device)
        role.model.eval()
        role.value_head.eval()
        try:
            token_rows = payload.token_rows_by_dp[ctx.dp_rank]
            local = [] if not token_rows else self._score_value_rows(role, token_rows, payload)
            return local if ctx.rank == 0 else None
        finally:
            role.offload()

    def _score_value_rows(
        self, role: WorkerRole, token_rows: list[list[int]], payload: ScorePayload
    ) -> list[list[float]]:
        """Score critic values in bounded microbatches."""

        local = []
        microbatch_size = _score_microbatch_size(payload.microbatch_size)
        for start in range(0, len(token_rows), microbatch_size):
            rows = token_rows[start : start + microbatch_size]
            tokens, lengths = _pad_token_rows(rows, self.worker.device, int(payload.pad_token_id))
            out = role.model(input_ids=tokens, train_meta=_dense_train_meta(tokens, sequence_parallel_enabled=False))
            if out.hidden_states is None:
                raise RuntimeError("critic model output must include hidden_states for value scoring")
            values = role.value_head(out.hidden_states).squeeze(-1).float()
            local.extend(_unpad_token_rows(values, lengths))
        return local

    @torch.inference_mode()
    def score_rewards(self, payload: ScorePayload) -> list[float] | None:
        """Compute a single scalar reward per sequence using a reward role."""

        worker = self.worker
        ctx = get_tp_context()
        role = self.roles[payload.role]
        if role.value_head is None:
            raise RuntimeError("reward role must have a scalar reward head")
        worker._prepare_actor_offloaded()
        role.onload(worker.device)
        role.model.eval()
        role.value_head.eval()
        try:
            token_rows = payload.token_rows_by_dp[ctx.dp_rank]
            local = [] if not token_rows else self._score_reward_rows(role, token_rows, payload)
            return local if ctx.rank == 0 else None
        finally:
            role.offload()

    def _score_reward_rows(self, role: WorkerRole, token_rows: list[list[int]], payload: ScorePayload) -> list[float]:
        """Score scalar rewards in bounded microbatches."""

        local = []
        microbatch_size = _score_microbatch_size(payload.microbatch_size)
        for start in range(0, len(token_rows), microbatch_size):
            rows = token_rows[start : start + microbatch_size]
            tokens, lengths = _pad_token_rows(rows, self.worker.device, int(payload.pad_token_id))
            out = role.model(input_ids=tokens, train_meta=_dense_train_meta(tokens, sequence_parallel_enabled=False))
            if out.hidden_states is None:
                raise RuntimeError("reward model output must include hidden_states for reward scoring")
            values = role.value_head(out.hidden_states).float()
            values = values.squeeze(-1) if values.shape[-1] == 1 else values[..., -1]
            indices = torch.tensor([max(length - 1, 0) for length in lengths], device=values.device, dtype=torch.long)
            rewards = values[torch.arange(values.shape[0], device=values.device), indices]
            local.extend(float(value) for value in rewards.detach().cpu().tolist())
        return local

    def train_values(self, payload: TrainValuesPayload) -> dict | None:
        """One critic value-function training pass."""

        worker = self.worker
        ctx = get_tp_context()
        role = self.roles[payload.role]
        worker._prepare_actor_offloaded()
        role.onload(worker.device)
        role.model.train()
        role.value_head.train()
        rows = payload.data_packs_by_dp[ctx.dp_rank]
        accumulation_steps = payload.gradient_accumulation_steps
        accumulation_steps = len(rows) if accumulation_steps is None else max(int(accumulation_steps), 1)
        stats = _CriticStats()
        try:
            role.optimizer.zero_grad(set_to_none=True)
            for index, data_pack_obj in enumerate(rows):
                group_start = (index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(rows) - group_start)
                allow_step = (index + 1) % accumulation_steps == 0 or index == len(rows) - 1
                self._train_value_microbatch(role, data_pack_obj, payload, group_size, allow_step, stats)
            return stats.to_dict() if ctx.is_rank0 else None
        finally:
            role.offload()

    def _train_value_microbatch(
        self,
        role: WorkerRole,
        data_pack_obj: dict,
        payload: TrainValuesPayload,
        group_size: int,
        allow_step: bool,
        stats: _CriticStats,
    ) -> None:
        """Run one critic value microbatch and maybe step optimizer."""

        worker = self.worker
        data_pack = to_device(data_pack_obj, worker.device)
        tokens = data_pack["input_ids"].long()
        prompt_mask = data_pack["prompt_mask"].bool()
        returns = data_pack["returns"].float()
        old_values = data_pack["values"].float()
        out = role.model(input_ids=tokens)
        if out.hidden_states is None:
            raise RuntimeError("critic model output must include hidden_states for value training")
        values = role.value_head(out.hidden_states).squeeze(-1).float()
        value_mask = _critic_value_mask(prompt_mask)
        valid = value_mask & (tokens >= 0)
        if not bool(valid.any()):
            if allow_step:
                self._maybe_step_role(role)
            return
        target = torch.zeros_like(returns)
        target[:, :-1] = returns[:, 1:]
        baseline = torch.zeros_like(old_values)
        baseline[:, :-1] = old_values[:, 1:]
        cliprange_value = float(payload.cliprange_value)
        value_loss_coef = float(payload.value_loss_coef)
        clipped = baseline + (values - baseline).clamp(min=-cliprange_value, max=cliprange_value)
        loss_unclipped = (values - target).pow(2)
        loss_clipped = (clipped - target).pow(2)
        loss = value_loss_coef * 0.5 * torch.maximum(loss_unclipped[valid], loss_clipped[valid]).mean()
        (loss / max(group_size, 1)).backward()
        accumulate_role_main_gradients(role)
        stats.add(loss, loss_unclipped, loss_clipped, values, target, baseline, valid)
        if allow_step:
            self._maybe_step_role(role)

    def _maybe_step_role(self, role: WorkerRole) -> None:
        """Sync role gradients and step if this accumulation window has grads."""

        has_grad = any(param_grad(param) is not None for param in role.parameters())
        if has_grad:
            self.worker._sync_role_grads(role)
            role.optimizer.step()
        role.optimizer.zero_grad(set_to_none=True)


class _CriticStats:
    """Accumulate scalar critic-training diagnostics."""

    def __init__(self):
        self.losses = []
        self.clipfracs = []
        self.mse_values = []
        self.pred_means = []
        self.target_means = []
        self.old_value_means = []
        self.abs_error_means = []

    def add(self, loss, loss_unclipped, loss_clipped, values, target, baseline, valid) -> None:
        """Append scalar metrics for one value microbatch."""

        self.losses.append(float(loss.detach().cpu()))
        self.clipfracs.append(float((loss_clipped[valid] > loss_unclipped[valid]).float().mean().detach().cpu()))
        with torch.no_grad():
            pred_valid = values[valid]
            target_valid = target[valid]
            baseline_valid = baseline[valid]
            self.mse_values.append(float(loss_unclipped[valid].mean().detach().cpu()))
            self.pred_means.append(float(pred_valid.mean().detach().cpu()))
            self.target_means.append(float(target_valid.mean().detach().cpu()))
            self.old_value_means.append(float(baseline_valid.mean().detach().cpu()))
            self.abs_error_means.append(float((pred_valid - target_valid).abs().mean().detach().cpu()))

    def to_dict(self) -> dict[str, float]:
        """Return averaged critic diagnostics."""

        return {
            "critic_value_loss": _mean(self.losses),
            "critic_value_clipfrac": _mean(self.clipfracs),
            "critic_value_mse": _mean(self.mse_values),
            "critic_value_pred_mean": _mean(self.pred_means),
            "critic_value_target_mean": _mean(self.target_means),
            "critic_old_value_mean": _mean(self.old_value_means),
            "critic_value_abs_error_mean": _mean(self.abs_error_means),
        }


def _mean(values: list[float]) -> float:
    """Mean with empty-list fallback used by metric reducers."""

    return sum(values) / max(len(values), 1)


def _pad_token_rows(
    token_rows: list[list[int]], device: torch.device, pad_token_id: int
) -> tuple[torch.Tensor, list[int]]:
    """Pad variable-length token rows into a `(B, max_len)` tensor."""

    if not token_rows:
        return torch.empty(0, 0, dtype=torch.long, device=device), []
    lengths = [len(row) for row in token_rows]
    max_len = max(lengths)
    tokens = torch.full((len(token_rows), max_len), pad_token_id, dtype=torch.long, device=device)
    for row_idx, row in enumerate(token_rows):
        tokens[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return tokens, lengths


def _unpad_action_rows(logprobs: torch.Tensor, lengths: list[int]) -> list[list[float]]:
    """Slice action-token logprobs out of a padded `(B, T)` tensor."""

    rows = []
    for row_idx, length in enumerate(lengths):
        values = [0.0]
        if length > 1:
            values.extend(logprobs[row_idx, : length - 1].detach().cpu().tolist())
        rows.append(values)
    return rows


def _unpad_token_rows(values: torch.Tensor, lengths: list[int]) -> list[list[float]]:
    """Slice per-token scalar values out of a padded `(B, T)` tensor."""

    return [values[row_idx, :length].detach().cpu().tolist() for row_idx, length in enumerate(lengths)]


def _score_microbatch_size(value: int) -> int:
    """Resolve the score-time microbatch size from a payload."""

    return max(int(value), 1)


def _critic_value_mask(prompt_mask: torch.Tensor) -> torch.Tensor:
    """Build a `(B, T)` bool mask marking response positions for value loss."""

    mask = torch.zeros_like(prompt_mask, dtype=torch.bool)
    if prompt_mask.shape[1] > 1:
        response_mask = ~prompt_mask[:, 1:].bool()
        mask[:, :-1] = response_mask
    return mask
