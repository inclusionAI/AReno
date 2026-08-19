"""MLX optimizer construction, parameter groups, and 8-bit moment states."""

from __future__ import annotations

from typing import Any

from areno.api.backend.mlx.provider import parameter_group


def build_optimizer(config: dict[str, Any]):
    """Build AdamW groups matching CUDA policy/tower/projector controls."""

    import mlx.optimizers as optim

    groups: list[tuple[str, Any, dict[str, Any]]] = []
    filters = []
    if config.get("unfreeze_multimodal_tower"):
        tower_config = _group_config(config, "tower")
        groups.append(("tower", _adamw(tower_config), tower_config))
        filters.append(lambda path, _: parameter_group(path) == "tower")
    if config.get("unfreeze_multimodal_projector"):
        projector_config = _group_config(config, "projector")
        groups.append(("projector", _adamw(projector_config), projector_config))
        filters.append(lambda path, _: parameter_group(path) == "projector")
    model_config = dict(config)
    groups.append(("model", _adamw(model_config), model_config))
    if len(groups) == 1:
        return groups[0][1], groups
    optimizer_type = _streaming_multi_optimizer_class() if config.get("adam_8bit") else optim.MultiOptimizer
    return optimizer_type([item[1] for item in groups], filters=filters), groups


def set_group_learning_rates(groups: list[tuple[str, Any, dict[str, Any]]], step: int) -> dict[str, float]:
    """Advance every optimizer group with its own schedule."""

    from areno.api.backend.mlx.training import learning_rate_for_step

    rates = {}
    for name, optimizer, config in groups:
        rate = learning_rate_for_step(config, step)
        optimizer.learning_rate = rate
        rates[name] = rate
    return rates


def materialize_optimizer_update(model: Any, optimizer: Any, *, chunk_size: int = 8) -> None:
    """Evaluate a lazy optimizer update without realizing every parameter at once."""

    import mlx.core as mx
    from mlx.utils import tree_flatten

    size = max(int(chunk_size), 1)
    arrays = [value for _, value in tree_flatten(model.trainable_parameters())]
    arrays.extend(value for _, value in tree_flatten(optimizer.state) if isinstance(value, mx.array))
    for start in range(0, len(arrays), size):
        mx.eval(*arrays[start : start + size])
        mx.clear_cache()


def apply_optimizer_update(model: Any, optimizer: Any, gradients: dict) -> None:
    """Apply an optimizer update, streaming custom 8-bit states when available."""

    update_streaming = getattr(optimizer, "update_streaming", None)
    if update_streaming is not None:
        update_streaming(model, gradients)
        return
    optimizer.update(model, gradients)
    materialize_optimizer_update(model, optimizer)


def _group_config(config: dict[str, Any], group: str) -> dict[str, Any]:
    result = dict(config)
    prefix = f"multimodal_{group}_"
    for key in ("lr", "min_lr", "lr_decay_steps", "lr_decay_style"):
        value = config.get(prefix + key)
        if value is not None:
            result[key] = value
    return result


def _adamw(config: dict[str, Any]):
    import mlx.optimizers as optim

    kwargs = {
        "learning_rate": float(config.get("lr", 1e-6)),
        "betas": tuple(config.get("betas", (0.9, 0.999))),
        "weight_decay": float(config.get("weight_decay", 1e-2)),
    }
    if not config.get("adam_8bit"):
        return optim.AdamW(**kwargs, bias_correction=True)
    return _quantized_adamw_class()(**kwargs)


def _quantized_adamw_class():
    import mlx.core as mx
    from mlx.optimizers import Optimizer

    class AdamW8bit(Optimizer):
        """AdamW with blockwise uint8 first-moment and root-second-moment storage."""

        def __init__(
            self,
            learning_rate: float,
            betas=(0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
            block_size: int = 256,
            update_blocks: int = 8192,
        ) -> None:
            super().__init__()
            self._maybe_schedule("learning_rate", learning_rate)
            self.betas = tuple(float(value) for value in betas)
            self.eps = float(eps)
            self.weight_decay = float(weight_decay)
            self.block_size = int(block_size)
            self.update_blocks = int(update_blocks)

        def init_single(self, parameter, state: dict) -> None:
            size = int(parameter.size)
            state["size"] = size
            state["initialized"] = False

        def apply_single(self, gradient, parameter, state: dict):
            beta1, beta2 = self.betas
            lr = self.learning_rate.astype(mx.float32)
            step = self.step.astype(mx.float32)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2_sqrt = mx.sqrt(1.0 - beta2**step)
            size = int(state["size"])
            initialized = bool(state["initialized"])
            block_count = (size + self.block_size - 1) // self.block_size
            grad = gradient.reshape(-1)
            values = parameter.reshape(-1)
            outputs = []
            m_quantized = []
            m_scales = []
            v_quantized = []
            v_scales = []
            for block_start in range(0, block_count, self.update_blocks):
                block_end = min(block_start + self.update_blocks, block_count)
                value_start = block_start * self.block_size
                value_end = min(block_end * self.block_size, size)
                padded_end = block_end * self.block_size
                actual = value_end - value_start
                grad_chunk = grad[value_start:value_end].astype(mx.float32)
                value_chunk = values[value_start:value_end].astype(mx.float32)
                padded_size = padded_end - value_start
                if actual < padded_size:
                    grad_chunk = mx.pad(grad_chunk, (0, padded_size - actual))
                    value_chunk = mx.pad(value_chunk, (0, padded_size - actual))
                if initialized:
                    m = _dequant_blocks(
                        state["m_q"][value_start:padded_end],
                        state["m_scale"][block_start:block_end],
                        signed=True,
                    )
                    v_root = _dequant_blocks(
                        state["v_q"][value_start:padded_end],
                        state["v_scale"][block_start:block_end],
                        signed=False,
                    )
                    v = mx.square(v_root)
                    m = beta1 * m + (1.0 - beta1) * grad_chunk
                    v = beta2 * v + (1.0 - beta2) * mx.square(grad_chunk)
                else:
                    m = (1.0 - beta1) * grad_chunk
                    v = (1.0 - beta2) * mx.square(grad_chunk)
                next_m_q, next_m_scale = _quantize_signed(m, self.block_size)
                next_v_q, next_v_scale = _quantize_unsigned(mx.sqrt(v), self.block_size)
                denom = mx.sqrt(v) / bias_correction2_sqrt + self.eps
                updated = (value_chunk * (1.0 - lr * self.weight_decay) - (lr / bias_correction1) * m / denom)[
                    :actual
                ].astype(parameter.dtype)
                mx.eval(updated, next_m_q, next_m_scale, next_v_q, next_v_scale)
                mx.clear_cache()
                outputs.append(updated)
                m_quantized.append(next_m_q)
                m_scales.append(next_m_scale)
                v_quantized.append(next_v_q)
                v_scales.append(next_v_scale)
            state["m_q"] = mx.concatenate(m_quantized)
            state["m_scale"] = mx.concatenate(m_scales)
            state["v_q"] = mx.concatenate(v_quantized)
            state["v_scale"] = mx.concatenate(v_scales)
            state["initialized"] = True
            updated_parameter = mx.concatenate(outputs).reshape(parameter.shape)
            mx.eval(updated_parameter, state["m_q"], state["m_scale"], state["v_q"], state["v_scale"])
            mx.clear_cache()
            return updated_parameter

        def update_streaming(self, model, gradients: dict) -> None:
            if not self._initialized:
                self.init(gradients)
            self._begin_streaming_step()
            _apply_streaming_leaves(model, gradients, lambda _: self)

        def _begin_streaming_step(self) -> None:
            for name, scheduler in self._schedulers.items():
                self.state[name] = scheduler(self.step)
            self.state["step"] = self.step + 1

    return AdamW8bit


def _streaming_multi_optimizer_class():
    from mlx.optimizers import MultiOptimizer

    class StreamingMultiOptimizer(MultiOptimizer):
        """Route each parameter to its 8-bit optimizer and commit it immediately."""

        def update_streaming(self, model, gradients: dict) -> None:
            if not all(optimizer._initialized for optimizer in self.optimizers):
                for optimizer, parameters in zip(self.optimizers, self._split_dictionary(gradients)):
                    if not optimizer._initialized:
                        optimizer.init(parameters)
            for optimizer in self.optimizers:
                optimizer._begin_streaming_step()

            def select(path: str):
                for optimizer, predicate in zip(self.optimizers, self.filters):
                    if predicate(path, _tree_get(gradients, path)):
                        return optimizer
                raise RuntimeError(f"no optimizer group accepted parameter {path!r}")

            _apply_streaming_leaves(model, gradients, select)

    return StreamingMultiOptimizer


def _apply_streaming_leaves(model: Any, gradients: dict, optimizer_for_path) -> None:
    from mlx.utils import tree_flatten

    leaves = tree_flatten(gradients)
    for index, (path, gradient) in enumerate(leaves):
        optimizer = optimizer_for_path(path)
        state = _tree_get(optimizer.state, path)
        parameter = _model_parameter(model, path)
        updated = optimizer.apply_single(gradient, parameter, state)
        _set_model_parameter(model, path, updated)
        _tree_set(gradients, path, None)
        leaves[index] = (path, None)
        del gradient, parameter, updated


def _tree_get(tree: Any, path: str) -> Any:
    current = tree
    for part in path.split("."):
        current = current[int(part)] if isinstance(current, (list, tuple)) else current[part]
    return current


def _tree_set(tree: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    current = tree
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, (list, tuple)) else current[part]
    final = parts[-1]
    if isinstance(current, list):
        current[int(final)] = value
    else:
        current[final] = value


def _model_parameter(model: Any, path: str):
    current = model
    for part in path.split("."):
        current = current[int(part)] if isinstance(current, (list, tuple)) else getattr(current, part)
    return current


def _set_model_parameter(model: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    current = model
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, (list, tuple)) else getattr(current, part)
    final = parts[-1]
    if isinstance(current, list):
        current[int(final)] = value
    else:
        setattr(current, final, value)


def _blocked(value, block_size: int):
    import mlx.core as mx

    flat = value.reshape(-1).astype(mx.float32)
    padded = ((int(flat.size) + block_size - 1) // block_size) * block_size
    if padded != flat.size:
        flat = mx.pad(flat, (0, padded - int(flat.size)))
    return flat.reshape(-1, block_size)


def _quantize_signed(value, block_size: int):
    import mlx.core as mx

    blocks = _blocked(value, block_size)
    scale = mx.maximum(mx.max(mx.abs(blocks), axis=1, keepdims=True) / 127.0, mx.array(1e-12))
    quantized = mx.clip(mx.round(blocks / scale), -127, 127).astype(mx.int16) + 128
    return quantized.astype(mx.uint8).reshape(-1), scale


def _quantize_unsigned(value, block_size: int):
    import mlx.core as mx

    blocks = _blocked(value, block_size)
    scale = mx.maximum(mx.max(blocks, axis=1, keepdims=True) / 255.0, mx.array(1e-12))
    quantized = mx.clip(mx.round(blocks / scale), 0, 255).astype(mx.uint8)
    return quantized.reshape(-1), scale


def _dequant_blocks(quantized, scale, *, signed: bool):
    values = quantized.reshape(scale.shape[0], -1).astype(scale.dtype)
    if signed:
        values = values - 128.0
    return (values * scale).reshape(-1)


__all__ = ["apply_optimizer_update", "build_optimizer", "materialize_optimizer_update", "set_group_learning_rates"]
