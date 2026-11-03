"""Model-independent affine QAT with floating master weights and STE gradients."""

import copy

from areno.api.backend.mlx.quantization import validate_quantization


def fake_quantize(weight, *, bits, group_size, mode="affine"):
    """Use quantized values in forward and the identity Jacobian in backward."""
    import mlx.core as mx

    packed = mx.quantize(weight, bits=bits, group_size=group_size, mode=mode)
    rounded = mx.dequantize(*packed, bits=bits, group_size=group_size, mode=mode)
    return mx.stop_gradient(rounded) + (weight - mx.stop_gradient(weight))


def configure_qat(model, config, *, bits, group_size):
    """Preserve parameter names and each leaf's native forward, including MoE.

    Fake quantization uses floating arithmetic on round-tripped Q4 weights;
    it does not claim QLoRA's packed-weight memory usage or integer GEMMs.
    Model-protected and incompatible leaf weights remain floating and trainable.
    """
    from mlx.utils import tree_unflatten

    validate_quantization(bits, group_size)
    if config.get("quantization") is not None or config.get("quantization_config") is not None:
        raise ValueError("MLX full-parameter QAT requires floating source/master weights")
    predicate = getattr(model, "quant_predicate", None)
    replacements = []
    for path, module in model.named_modules():
        if not path or not callable(getattr(module, "to_quantized", None)):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.shape[-1] % group_size:
            continue
        decision = predicate(path, module) if predicate is not None else True
        if not decision:
            continue
        parameters = {"bits": bits, "group_size": group_size, "mode": "affine"}
        if isinstance(decision, dict):
            parameters.update(decision)
        if parameters["mode"] != "affine":
            raise ValueError(f"MLX QAT requires affine quantization: {path}")
        replacements.append((path, _qat_module(module, parameters)))
    if not replacements:
        raise ValueError("checkpoint contains no quantizable modules for full-parameter QAT")
    model.update_modules(tree_unflatten(replacements))
    return {**config, "areno_qat": {"format": "mlx-affine-ste-v1", "bits": bits, "group_size": group_size}}


def _qat_module(module, parameters):
    original_type = type(module)

    def forward_copy(self):
        # Clone only this leaf. Never mutate the live model during autograd or
        # retain a graph across candidate scoring/replay microbatches.
        leaf = copy.copy(self)
        leaf.__class__ = original_type
        leaf.weight = fake_quantize(self.weight, **parameters)
        return leaf

    def forward(self, *args, **kwargs):
        return forward_copy(self)(*args, **kwargs)

    methods = {"__call__": forward}
    if callable(getattr(module, "as_linear", None)):
        methods["as_linear"] = lambda self, *args, **kwargs: forward_copy(self).as_linear(*args, **kwargs)
    wrapped = copy.copy(module)
    wrapped.__class__ = type("Qat" + original_type.__name__, (original_type,), methods)
    return wrapped
