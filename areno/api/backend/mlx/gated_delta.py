"""Model-independent parallel blocks for differentiable gated-delta recurrence."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _block_operator():
    import mlx.core as mx

    @mx.compile
    def block(q, k, v, gate, beta, state, mask):
        # Work in FP32, matching the reference recurrence's state arithmetic.
        # A masked token still has a hypothetical output in MLX-LM's training
        # reference, while its state update is discarded.
        key_query = (k * q).sum(-1, keepdims=True)
        query = mx.where(mask, q, gate * (q - beta * key_query * k))
        bias = mx.where(mask, mx.zeros_like(v), beta * key_query * v)
        gate = mx.where(mask, gate, 1.0)
        beta = mx.where(mask, beta, 0.0)
        length = q.shape[-2]
        positions = mx.arange(length)
        strict = positions[:, None] > positions[None, :]
        causal = positions[:, None] >= positions[None, :]
        # decay[t,i] = product(g[i+1:t+1]). Direct products handle zero gates
        # without division, logarithms, overflow, or epsilon approximations.
        decay = mx.cumprod(mx.where(strict[None, None, :, :, None], gate[..., :, None, :], 1.0), axis=2)
        prefix = mx.cumprod(gate, axis=2)
        key_pairs = (k[..., :, None, :] * k[..., None, :, :] * decay).sum(-1)
        lower = mx.where(strict, beta * key_pairs, 0.0)
        # Unit-lower triangular block inversion. Each doubling step joins two
        # already inverted diagonal blocks via -D^-1 B A^-1. Unlike powers of
        # the full lower triangle, intermediates remain well conditioned even
        # for gates near one and beta near two.
        inverse = mx.broadcast_to(mx.eye(length), lower.shape)
        span = 1
        while span < length:
            rows, columns = positions[:, None], positions[None, :]
            same_block = rows // (2 * span) == columns // (2 * span)
            cross_block = same_block & (rows % (2 * span) >= span) & (columns % (2 * span) < span)
            cross = mx.where(cross_block, lower, 0.0)
            inverse = inverse - (inverse @ cross) @ inverse
            span *= 2
        initial_keys = (state @ (k * prefix).swapaxes(-1, -2)).swapaxes(-1, -2)
        residual = inverse @ (beta * (v - initial_keys))
        readout = (query[..., :, None, :] * k[..., None, :, :] * decay).sum(-1)
        readout = mx.where(causal, readout, 0.0)
        initial_query = (state @ (query * prefix).swapaxes(-1, -2)).swapaxes(-1, -2)
        output = initial_query + readout @ residual + bias
        final_keys = k * decay[..., -1, :, :]
        state = state * prefix[..., -1:, :] + residual.swapaxes(-1, -2) @ final_keys
        return output, state

    return block


def gated_delta_parallel(q, k, v, g, beta, state=None, mask=None, *, chunk_size=16):
    """Exact chunked recurrence for scalar/vector gates, GQA and token masks."""
    import mlx.core as mx

    batch, length, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if length < 1 or chunk_size < 1 or value_heads % key_heads:
        raise ValueError("gated-delta requires nonempty time, positive chunk size and grouped heads")
    dtype = q.dtype
    repeats = value_heads // key_heads
    q = mx.repeat(q, repeats, axis=2).transpose(0, 2, 1, 3).astype(mx.float32)
    k = mx.repeat(k, repeats, axis=2).transpose(0, 2, 1, 3).astype(mx.float32)
    v = v.transpose(0, 2, 1, 3).astype(mx.float32)
    g = (g[..., None] if g.ndim == 3 else g).transpose(0, 2, 1, 3).astype(mx.float32)
    beta = beta.transpose(0, 2, 1)[..., None].astype(mx.float32)
    mask = mx.ones((batch, 1, length, 1), dtype=mx.bool_) if mask is None else mask[:, None, :, None]
    state = mx.zeros((batch, value_heads, value_dim, key_dim), dtype=mx.float32) if state is None else state
    padded_length = ((length + chunk_size - 1) // chunk_size) * chunk_size
    padding = padded_length - length
    if padding:
        pads = [(0, 0), (0, 0), (0, padding), (0, 0)]
        q, k, v = (mx.pad(value, pads) for value in (q, k, v))
        g = mx.pad(g, pads, constant_values=1)
        beta = mx.pad(beta, pads)
        mask = mx.pad(mask, pads, constant_values=False)
    outputs = []
    block = _block_operator()
    for start in range(0, padded_length, chunk_size):
        end = start + chunk_size
        output, state = block(
            q[:, :, start:end],
            k[:, :, start:end],
            v[:, :, start:end],
            g[:, :, start:end],
            beta[:, :, start:end],
            state,
            mask[:, :, start:end],
        )
        outputs.append(output)
    return mx.concatenate(outputs, axis=2)[:, :, :length].transpose(0, 2, 1, 3).astype(dtype), state


def install_parallel_gated_delta():
    """Replace the shared training primitive in memory; never edit model files."""
    import inspect

    import mlx.core as mx

    try:
        from mlx_lm.models import gated_delta
    except ImportError:
        return

    original = gated_delta.gated_delta_ops
    if getattr(original, "_areno_parallel", False):
        return
    if tuple(inspect.signature(original).parameters) != ("q", "k", "v", "g", "beta", "state", "mask"):
        return

    def dispatch(q, k, v, g, beta, state=None, mask=None):
        if mx.default_device() != mx.gpu or not mx.metal.is_available() or q.shape[1] < 2:
            return original(q, k, v, g, beta, state, mask)
        return gated_delta_parallel(q, k, v, g, beta, state, mask)

    dispatch._areno_reference = original
    dispatch._areno_parallel = True
    gated_delta.gated_delta_ops = dispatch
