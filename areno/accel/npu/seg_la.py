"""Map AReno's linear-attention state pool to FLA's sequence-state API."""

import torch


def chunk_lightning_attn(
    q,
    k,
    v,
    layer_idx,
    num_layers,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    *,
    g_gamma=None,
    head_first=False,
    **kwargs,
):
    """Preserve Bailing's TP-local decay with the current FLA interface."""
    if head_first:
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if g_gamma is None:
        from fla.ops.lightning_attn import chunk_lightning_attn as implementation

        kwargs.update(layer_idx=layer_idx, num_layers=num_layers)
    else:
        # Upstream Lightning computes its own decay from the local head count.
        # Bailing already supplies the correct layer-scaled, TP-sharded slopes.
        from fla.ops.simple_gla import chunk_simple_gla as implementation

        kwargs["g_gamma"] = g_gamma
    out, final = implementation(
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        **kwargs,
    )
    return (out.transpose(1, 2) if head_first else out), final


def seg_la_fwd(q, k, v, state, decay_scales, meta, caches=None, softmax_scale=None):
    from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla

    if caches is not None or meta.mask is not None:
        raise NotImplementedError("Ascend FLA state snapshots and tree-mask seg-LA are not integrated")
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("seg_la_fwd requires matching [tokens, heads, dim] q/k/v without GQA")
    if meta.batch_size <= 0 or meta.s_offsets.numel() != meta.batch_size:
        raise ValueError("seg_la_fwd requires one state slot per sequence")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, state, decay_scales)):
        raise RuntimeError("seg_la_fwd mutates inference state; use chunk_lightning_attn for training")
    slots = meta.s_offsets.to(dtype=torch.long)
    # The scheduler supplies allocated (nonnegative) slots, including padding
    # slots. index_select also rejects invalid indices before the library call.
    initial = state.index_select(0, slots)
    if q.shape[0] <= meta.batch_size:
        if q.shape[0] != meta.batch_size:
            raise ValueError("seg-LA decode requires one token per sequence")
        kernel = fused_recurrent_simple_gla
        inputs = [x.unsqueeze(1) for x in (q, k, v)]
        cu = None
    else:
        kernel = chunk_simple_gla
        inputs = [x.unsqueeze(0) for x in (q, k, v)]
        initial = torch.where((meta.s_scales > 0).view(-1, 1, 1, 1), initial, 0)
        # Preserve the metadata's last sequence length, including trailing
        # empty sequences; no device-to-host length reads or Python token loop.
        starts = meta.q_offsets[: meta.batch_size]
        cu = torch.cat((starts, starts[-1:] + meta.q_lengths[-1:])).to(dtype=torch.long)
    out, final = kernel(
        *inputs,
        g_gamma=-decay_scales.float(),
        scale=softmax_scale,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
    )
    state.index_copy_(0, slots, final.to(state.dtype))
    return out.reshape_as(q)
