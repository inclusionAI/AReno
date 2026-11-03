"""Expert-major bias and interleaved limited-SwiGLU, with owned CUDA backward."""

import torch


def _bias(x, bias, counts):
    ends = counts.cumsum(0)
    ids = torch.searchsorted(ends, torch.arange(x.shape[0], device=x.device), right=True)
    valid = ids < bias.shape[0]
    return (x + bias[ids.clamp(max=bias.shape[0] - 1)]) * valid[:, None]


def _activation(x, alpha, limit):
    gate = x[..., ::2].clamp(max=limit)
    up = x[..., 1::2].clamp(min=-limit, max=limit)
    return (up + 1) * (gate * torch.sigmoid(gate * alpha))


class _Bias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, counts):
        from areno.accel.kernels.grouped_bias import bias_forward

        out, ends = bias_forward(x, bias, counts)
        ctx.save_for_backward(ends)
        ctx.shape = bias.shape
        return out

    @staticmethod
    def backward(ctx, grad):
        from areno.accel.kernels.grouped_bias import bias_backward

        dx, db = bias_backward(grad, ctx.saved_tensors[0], ctx.shape)
        return dx, db, None


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, limit):
        from areno.accel.kernels.grouped_bias import activation_forward

        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.limit = limit
        return activation_forward(x, alpha, limit)

    @staticmethod
    def backward(ctx, grad):
        from areno.accel.kernels.grouped_bias import activation_backward

        return activation_backward(ctx.saved_tensors[0], grad, ctx.alpha, ctx.limit), None, None


def grouped_bias(x, bias, counts):
    return _Bias.apply(x, bias, counts) if x.is_cuda else _bias(x, bias, counts)


def limited_swiglu(x, alpha=1.702, limit=7.0):
    return _Activation.apply(x, alpha, limit) if x.is_cuda else _activation(x, alpha, limit)
