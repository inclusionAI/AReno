"""Packed delta-rule forward using the same recurrence as paged inference."""

import torch
import torch.nn.functional as F

from areno.accel.kernels.kda_fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update


def _scan(q, k, v, a, b, decay, bias, cu, state, slots, bound):
    kda = a.ndim == 4
    return fused_sigmoid_gating_delta_rule_update(
        A_log=decay,
        a=a.flatten(2) if kda else a,
        dt_bias=bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=slots,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        is_kda=kda,
        lower_bound=bound,
        value_tile=4 if kda else 8,
    )


def _differentiable_block(q, k, v, a, b, decay, bias, state, bound):
    """Recompute one block for BPTT in fp32, with state laid out as (H,V,K)."""
    heads = v.shape[-2]
    group = heads // k.shape[-2]

    def normalize(x):
        x = x.float()
        return (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(group, dim=-2)

    q = normalize(q) * q.shape[-1] ** -0.5
    k = normalize(k)
    x = a.float() + bias.reshape(a.shape[1:])
    rate = decay.float().exp().reshape(heads, -1)
    x = x.reshape(x.shape[0], heads, -1)
    gate = (-rate * F.softplus(x) if bound is None else bound * torch.sigmoid(rate * x)).exp()
    beta = b.float().sigmoid()
    outputs = []
    for t in range(q.shape[0]):
        state = state * gate[t, :, None, :]
        delta = (v[t].float() - (state * k[t, :, None, :]).sum(-1)) * beta[t, :, None]
        state = state + delta[..., None] * k[t, :, None, :]
        outputs.append((state * q[t, :, None, :]).sum(-1))
    return torch.stack(outputs), state


class _TrainDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, a, b, decay, bias, cu, bound):
        ctx.save_for_backward(q, k, v, a, b, decay, bias, cu)
        ctx.bound = bound
        state = v.new_zeros((cu.numel() - 1, v.shape[2], v.shape[3], k.shape[3]), dtype=torch.float32)
        slots = torch.arange(cu.numel() - 1, device=q.device)
        return _scan(q, k, v, a, b, decay, bias, cu, state, slots, bound)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, a, b, decay, bias, cu = ctx.saved_tensors
        if q.dtype in (torch.bfloat16, torch.float16):
            # Differentiate the block factorization of the same linear recurrence.
            # This avoids materializing a full H x V x K state at every token.
            with torch.enable_grad():
                args = [x.detach().requires_grad_() for x in (q, k, v, a, b, decay, bias)]
                query, key, value, raw_gate, beta, rate, offset = args
                if raw_gate.ndim == 4:
                    from areno.accel.kda import areno_kda_chunk

                    out, _ = areno_kda_chunk(
                        query,
                        key,
                        value,
                        raw_gate,
                        beta.float().sigmoid(),
                        scale=q.shape[-1] ** -0.5,
                        cu_seqlens=cu.long(),
                        a_log=rate,
                        dt_bias=offset,
                        lower_bound=ctx.bound,
                    )
                else:
                    from areno.models._shared.dynamo_wrappers import _fla_chunk_gated_delta_rule_no_compile

                    gate = -rate.float().exp() * F.softplus(raw_gate.float() + offset.float())
                    out, _ = _fla_chunk_gated_delta_rule_no_compile(
                        query,
                        key,
                        value,
                        g=gate,
                        beta=beta.float().sigmoid(),
                        cu_seqlens=cu.long(),
                        use_qk_l2norm_in_kernel=True,
                    )
                grads = torch.autograd.grad(out, args, dout)
            return (*grads, None, None)
        inputs = [x.flatten(0, 1) for x in (q, k, v, a, b)]
        grads = [torch.zeros_like(x) for x in inputs]
        params = [p.detach().float().requires_grad_() for p in (decay, bias)]
        param_grads = [torch.zeros_like(p) for p in params]
        positions = cu.tolist()
        for start, end in zip(positions[:-1], positions[1:], strict=True):
            state = v.new_zeros((v.shape[2], v.shape[3], k.shape[3]), dtype=torch.float32)
            checkpoints = []
            # Store only block boundaries. Gradients across adjacent blocks use
            # the final-state adjoint from the following block.
            for left in range(start, end, 32):
                right = min(left + 32, end)
                checkpoints.append((left, right, state))
                _, state = _differentiable_block(*(x[left:right] for x in inputs), *params, state, ctx.bound)
            adjoint = torch.zeros_like(state)
            for left, right, state in reversed(checkpoints):
                with torch.enable_grad():
                    block = [x[left:right].detach().requires_grad_() for x in inputs]
                    initial = state.detach().requires_grad_()
                    out, final = _differentiable_block(*block, *params, initial, ctx.bound)
                    derivatives = torch.autograd.grad(
                        (out, final), (*block, *params, initial), (dout.flatten(0, 1)[left:right].float(), adjoint)
                    )
                for dst, dx in zip(grads, derivatives[:5], strict=True):
                    dst[left:right] = dx
                for dst, dx in zip(param_grads, derivatives[5:7], strict=True):
                    dst.add_(dx)
                adjoint = derivatives[-1]
        return (*(g.reshape_as(x) for g, x in zip(grads, (q, k, v, a, b), strict=True)), *param_grads, None, None)


def native_delta(q, k, v, a, b, decay, bias, train_meta, infer_meta, state_cache, lower_bound=None):
    if infer_meta is None:
        cu = train_meta.cu_seqlens if train_meta is not None else None
        if cu is None:
            batch, seq = q.shape[:2]
            cu = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * seq
        return _TrainDelta.apply(q, k, v, a, b, decay, bias, cu, lower_bound)
    slots = infer_meta.recurrent_slots if infer_meta.recurrent_slots is not None else infer_meta.block_table[:, 0]
    cu = infer_meta.cu_seqlens
    if cu is None:
        cu = torch.arange(slots.numel() + 1, device=q.device, dtype=torch.int32) * infer_meta.tokens_per_seq
    return _scan(q, k, v, a, b, decay, bias, cu, state_cache, slots, lower_bound)
