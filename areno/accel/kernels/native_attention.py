"""Tensor-core forward shared by packed causal attention and single-token decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    Q,
    K,
    V,
    UPDATE_K,
    UPDATE_V,
    BOUNDS,
    TABLE,
    OUT,
    CACHE_K,
    CACHE_V,
    BLOCK_IDS,
    OFFSETS,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DV: tl.constexpr,
    BV: tl.constexpr,
    CACHE_D: tl.constexpr,
    STORE_PAGE: tl.constexpr,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    SEQUENCES: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    LSE=None,
    PREFIX=None,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    kv = head // (HQ // HK)
    lane = tl.arange(0, M)
    d = tl.arange(0, BLOCK_D)
    e = tl.arange(0, BV)
    if PAGE and PREFIX is None:
        seq = tile
        pos = tl.load(BOUNDS + seq)
        qpos = pos + lane
        row = seq + lane
        active = lane == 0
        end = pos + 1
        current_page = tl.load(TABLE + seq * WIDTH + pos // PAGE)
        dest = ((current_page * PAGE + pos % PAGE) * HK + kv) * D + d
        new_k = tl.load(UPDATE_K + (seq * HK + kv) * D + d, d < D, 0)
        new_v = tl.load(UPDATE_V + (seq * HK + kv) * DV + e, e < DV, 0)
        if head % (HQ // HK) == 0:
            tl.store(K + dest, new_k, d < D)
            tl.store(
                V + ((current_page * PAGE + pos % PAGE) * HK + kv) * CACHE_D + d,
                tl.load(UPDATE_V + (seq * HK + kv) * DV + d, d < DV, 0),
                d < CACHE_D,
            )
        first = pos
    else:
        # Packed sequence s owns tiles starting at floor(cu[s] / M) + s.
        lo = 0
        hi = SEQUENCES
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if tl.load(BOUNDS + mid) // M + mid <= tile:
                lo = mid
            else:
                hi = mid
        start = tl.load(BOUNDS + lo)
        end = tl.load(BOUNDS + lo + 1) - start
        first = (tile - start // M - lo) * M
        if first >= end:
            return
        qpos = first + lane
        active = qpos < end
        row = start + qpos
        if PREFIX is not None:
            seq = lo
            prefix = tl.load(PREFIX + seq)
            qpos += prefix
            first += prefix
            end += prefix

    if STORE_PAGE and head % (HQ // HK) == 0:
        page_id = tl.load(BLOCK_IDS + row, active, 0)
        off = tl.load(OFFSETS + row, active, 0)
        dst = (page_id * STORE_PAGE + off) * HK + kv
        src = row * HK + kv
        kk = tl.load(K + src[:, None] * D + d[None, :], active[:, None] & (d[None, :] < D), 0)
        vv = tl.load(V + src[:, None] * DV + d[None, :], active[:, None] & (d[None, :] < DV), 0)
        tl.store(CACHE_K + dst[:, None] * D + d[None, :], kk, active[:, None] & (d[None, :] < D))
        tl.store(CACHE_V + dst[:, None] * CACHE_D + d[None, :], vv, active[:, None] & (d[None, :] < CACHE_D))
    query = tl.load(Q + (row[:, None] * HQ + head) * D + d[None, :], active[:, None] & (d[None, :] < D), 0)
    acc = tl.full((M, BV), 0, tl.float32)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denom = tl.full((M,), 0, tl.float32)
    begin = 0
    if WINDOW >= 0:
        begin = tl.maximum(first - WINDOW, 0) // N * N
    stop = tl.minimum(first + M, end)
    for split in range(begin // 256 * 256, stop, 256):
        block_max = tl.full((M,), -float("inf"), tl.float32)
        block_sum = tl.full((M,), 0, tl.float32)
        partial = tl.full((M, BV), 0, tl.float32)
        for base in range(split, tl.minimum(stop, split + 256), N):
            col = base + tl.arange(0, N)
            if PAGE:
                page = tl.load(TABLE + seq * WIDTH + col // PAGE, col < end, 0)
                loc = page * PAGE + col % PAGE
                read = col < (end if PREFIX is not None else pos)
            else:
                loc = start + col
                read = col < end
            offset = (loc[:, None] * HK + kv) * D + d[None, :]
            keys = tl.load(K + offset, read[:, None] & (d[None, :] < D), 0)
            vals = tl.load(
                V + (loc[:, None] * HK + kv) * (CACHE_D if PAGE else DV) + e[None, :],
                read[:, None] & (e[None, :] < DV),
                0,
            )
            if PAGE and PREFIX is None:
                keys = tl.where(col[:, None] == pos, new_k[None, :], keys)
                vals = tl.where(col[:, None] == pos, new_v[None, :], vals)
            scores = tl.dot(query, tl.trans(keys)) * (SCALE * 1.4426950408889634)
            mask = (col[None, :] <= qpos[:, None]) & (col[None, :] < end)
            if WINDOW >= 0:
                mask &= col[None, :] >= qpos[:, None] - WINDOW
            scores = tl.where(mask, scores, -float("inf"))
            peak = tl.maximum(block_max, tl.max(scores, 1))
            safe_peak = tl.where(peak == -float("inf"), 0.0, peak)
            prob = tl.exp2(scores - safe_peak[:, None])
            alpha = tl.exp2(block_max - safe_peak)
            partial = tl.dot(prob.to(vals.dtype), vals, partial * alpha[:, None])
            block_sum = block_sum * alpha + tl.sum(prob, 1)
            block_max = peak
        peak = tl.maximum(maximum, block_max)
        safe_peak = tl.where(peak == -float("inf"), 0.0, peak)
        alpha = tl.exp2(maximum - safe_peak)
        beta = tl.exp2(block_max - safe_peak)
        acc = acc * alpha[:, None] + partial * beta[:, None]
        denom = denom * alpha + block_sum * beta
        maximum = peak
    if LSE is not None:
        tl.store(LSE + row * HQ + head, maximum + tl.log2(denom), active)
    tl.store(
        OUT + (row[:, None] * HQ + head) * DV + e[None, :], acc / denom[:, None], active[:, None] & (e[None, :] < DV)
    )


def tiled_forward(
    q,
    k,
    v,
    bounds,
    window,
    scale,
    *,
    table=None,
    update_k=None,
    update_v=None,
    store_cache=None,
    lse=None,
    prefix=None,
    value_dim=None,
):
    tokens, heads, dim = q.shape
    dv = value_dim if value_dim is not None else (v.shape[-1] if table is None else update_v.shape[-1])
    if table is not None and prefix is None and heads // k.shape[-2] <= 16 and table.shape[1] * k.shape[1] >= 512:
        return split_decode(q, k, v, bounds, table, update_k, update_v, window, scale)
    out = q.new_empty((tokens, heads, dv))
    packed = table is None or prefix is not None
    sequences = bounds.numel() - 1 if packed else tokens
    # On Hopper a 64-row packed tile can select WGMMA while decode uses
    # warp-level MMA. Keep the same accumulation order across both paths.
    block_m = 16
    # A 512-wide head needs smaller query and KV tiles to fit shared memory.
    if packed and dim < 512:
        block_m = 32 if torch.cuda.get_device_capability(q.device)[0] == 9 else 64
    tiles = triton.cdiv(tokens, block_m) + sequences - 1 if packed else tokens
    if tokens:
        _forward[(tiles, heads)](
            q,
            k,
            v,
            k if packed else update_k,
            v if packed else update_v,
            bounds,
            bounds if table is None else table,
            out,
            *(store_cache if store_cache else (q, q, q, q)),
            heads,
            k.shape[-2],
            dim,
            triton.next_power_of_2(dim),
            dv,
            triton.next_power_of_2(dv),
            store_cache[1].shape[-1] if store_cache else v.shape[-1],
            store_cache[0].shape[1] if store_cache else 0,
            0 if table is None else k.shape[1],
            0 if table is None else table.shape[1],
            sequences,
            window,
            scale,
            block_m,
            32 if dim == 512 else 64,
            LSE=lse,
            PREFIX=prefix,
            num_warps=4,
            num_stages=1,
        )
    return out


@triton.jit
def _delta(O, DO, DELTA, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, B)
    o = tl.load(O + row * D + d, d < D, 0).to(tl.float32)
    do = tl.load(DO + row * D + d, d < D, 0).to(tl.float32)
    tl.store(DELTA + row, tl.sum(o * do, 0))


@triton.jit
def _backward(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    BOUNDS,
    DQ,
    DK,
    DVOUT,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BD: tl.constexpr,
    BE: tl.constexpr,
    SEQ: tl.constexpr,
    SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    B: tl.constexpr,
    QUERY: tl.constexpr,
):
    tile, head = tl.program_id(0), tl.program_id(1)
    lo = 0
    hi = SEQ
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if tl.load(BOUNDS + mid) // B + mid <= tile:
            lo = mid
        else:
            hi = mid
    start = tl.load(BOUNDS + lo)
    length = tl.load(BOUNDS + lo + 1) - start
    first = (tile - start // B - lo) * B
    if first >= length:
        return
    r = first + tl.arange(0, B)
    d = tl.arange(0, BD)
    e = tl.arange(0, BE)
    if QUERY:
        q = tl.load(Q + ((start + r[:, None]) * H + head) * D + d[None, :], (r[:, None] < length) & (d[None, :] < D), 0)
        do = tl.load(
            DO + ((start + r[:, None]) * H + head) * E + e[None, :], (r[:, None] < length) & (e[None, :] < E), 0
        )
        normal = tl.load(LSE + (start + r) * H + head, r < length, 0)
        delta = tl.load(DELTA + (start + r) * H + head, r < length, 0)
        kv = head // (H // HK)
        dq = tl.full((B, BD), 0, tl.float32)
        begin = 0
        if WINDOW >= 0:
            begin = tl.maximum(first - WINDOW, 0) // B * B
        for b in range(begin, first + B, B):
            c = b + tl.arange(0, B)
            k = tl.load(
                K + ((start + c[:, None]) * HK + kv) * D + d[None, :], (c[:, None] < length) & (d[None, :] < D), 0
            )
            v = tl.load(
                V + ((start + c[:, None]) * HK + kv) * E + e[None, :], (c[:, None] < length) & (e[None, :] < E), 0
            )
            mask = (c[None, :] <= r[:, None]) & (c[None, :] < length)
            if WINDOW >= 0:
                mask &= c[None, :] >= r[:, None] - WINDOW
            p = tl.where(mask, tl.exp2(tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634) - normal[:, None]), 0.0)
            ds = (tl.dot(do, tl.trans(v)) - delta[:, None]) * p * SCALE
            dq = tl.dot(ds.to(q.dtype), k, dq)
        tl.store(DQ + ((start + r[:, None]) * H + head) * D + d[None, :], dq, (r[:, None] < length) & (d[None, :] < D))
    else:
        k = tl.load(
            K + ((start + r[:, None]) * HK + head) * D + d[None, :], (r[:, None] < length) & (d[None, :] < D), 0
        )
        v = tl.load(
            V + ((start + r[:, None]) * HK + head) * E + e[None, :], (r[:, None] < length) & (e[None, :] < E), 0
        )
        dk = tl.full((B, BD), 0, tl.float32)
        dv = tl.full((B, BE), 0, tl.float32)
        stop = length
        if WINDOW >= 0:
            stop = tl.minimum(stop, first + B + WINDOW)
        for h in range(head * (H // HK), (head + 1) * (H // HK)):
            for b in range(first, stop, B):
                c = b + tl.arange(0, B)
                q = tl.load(
                    Q + ((start + c[:, None]) * H + h) * D + d[None, :], (c[:, None] < length) & (d[None, :] < D), 0
                )
                do = tl.load(
                    DO + ((start + c[:, None]) * H + h) * E + e[None, :], (c[:, None] < length) & (e[None, :] < E), 0
                )
                normal = tl.load(LSE + (start + c) * H + h, c < length, 0)
                delta = tl.load(DELTA + (start + c) * H + h, c < length, 0)
                mask = (r[:, None] <= c[None, :]) & (c[None, :] < length)
                if WINDOW >= 0:
                    mask &= r[:, None] >= c[None, :] - WINDOW
                pt = tl.where(
                    mask, tl.exp2(tl.dot(k, tl.trans(q)) * (SCALE * 1.4426950408889634) - normal[None, :]), 0.0
                )
                dst = (tl.dot(v, tl.trans(do)) - delta[None, :]) * pt * SCALE
                dk = tl.dot(dst.to(k.dtype), q, dk)
                dv = tl.dot(pt.to(k.dtype), do, dv)
        tl.store(DK + ((start + r[:, None]) * HK + head) * D + d[None, :], dk, (r[:, None] < length) & (d[None, :] < D))
        tl.store(
            DVOUT + ((start + r[:, None]) * HK + head) * E + e[None, :], dv, (r[:, None] < length) & (e[None, :] < E)
        )


class PackedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bounds, window, scale):
        lse = q.new_empty(q.shape[:2], dtype=torch.float32)
        out = tiled_forward(q, k, v, bounds, window, scale, lse=lse)
        ctx.save_for_backward(q, k, v, out, lse, bounds)
        ctx.window, ctx.scale = window, scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, bounds = ctx.saved_tensors
        do = do.contiguous()
        t, h, d = q.shape
        kv, e = v.shape[1:]
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        delta = torch.empty_like(lse)
        if t:
            _delta[(t * h,)](o, do, delta, e, triton.next_power_of_2(e))
            seq = bounds.numel() - 1
            block = 16 if d == 512 else 32
            args = (
                q,
                k,
                v,
                do,
                lse,
                delta,
                bounds,
                dq,
                dk,
                dv,
                h,
                kv,
                d,
                e,
                triton.next_power_of_2(d),
                triton.next_power_of_2(e),
                seq,
                ctx.scale,
                ctx.window,
                block,
            )
            for heads, query in [(h, True), (kv, False)]:
                _backward[(triton.cdiv(t, block) + seq - 1, heads)](*args, query, num_warps=4, num_stages=1)
        return dq, dk, dv, None, None, None


@triton.jit
def _decode_partial(
    Q,
    K,
    V,
    UK,
    UV,
    TABLE,
    LEN,
    SCRATCH,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BD: tl.constexpr,
    BE: tl.constexpr,
    PAGE: tl.constexpr,
    CACHE_E: tl.constexpr,
    WIDTH: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
):
    s, kv, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pos = tl.load(LEN + seq)
    start = s * 256
    if start > pos:
        return
    m = tl.arange(0, 16)
    h = kv * (H // HK) + m
    active = m < (H // HK)
    d = tl.arange(0, BD)
    e = tl.arange(0, BE)
    query = tl.load(Q + (seq * H + h[:, None]) * D + d[None, :], active[:, None] & (d[None, :] < D), 0)
    new_k = tl.load(UK + (seq * HK + kv) * D + d, d < D, 0)
    new_v = tl.load(UV + (seq * HK + kv) * E + e, e < E, 0)
    if start + 256 > pos:
        page = tl.load(TABLE + seq * WIDTH + pos // PAGE)
        slot = (page * PAGE + pos % PAGE) * HK + kv
        tl.store(K + slot * D + d, new_k, d < D)
        tl.store(V + slot * CACHE_E + d, tl.load(UV + (seq * HK + kv) * E + d, d < E, 0), d < CACHE_E)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    total = tl.full((16,), 0, tl.float32)
    acc = tl.full((16, BE), 0, tl.float32)
    N: tl.constexpr = 32 if D == 512 else 64
    for base in range(start, tl.minimum(start + 256, pos + 1), N):
        c = base + tl.arange(0, N)
        page = tl.load(TABLE + seq * WIDTH + c // PAGE, c < pos + 1, 0)
        slot = (page * PAGE + c % PAGE) * HK + kv
        keys = tl.load(K + slot[:, None] * D + d[None, :], (c[:, None] < pos) & (d[None, :] < D), 0)
        vals = tl.load(V + slot[:, None] * CACHE_E + e[None, :], (c[:, None] < pos) & (e[None, :] < E), 0)
        keys = tl.where(c[:, None] == pos, new_k[None, :], keys)
        vals = tl.where(c[:, None] == pos, new_v[None, :], vals)
        score = tl.dot(query, tl.trans(keys)) * (SCALE * 1.4426950408889634)
        valid = c <= pos
        if WINDOW >= 0:
            valid &= c >= pos - WINDOW
        score = tl.where(valid[None, :], score, -float("inf"))
        peak = tl.maximum(maximum, tl.max(score, 1))
        safe = tl.where(peak == -float("inf"), 0.0, peak)
        p = tl.exp2(score - safe[:, None])
        alpha = tl.exp2(maximum - safe)
        acc = tl.dot(p.to(vals.dtype), vals, acc * alpha[:, None])
        total = total * alpha + tl.sum(p, 1)
        maximum = peak
    off = ((seq * SPLITS + s) * H + h) * (E + 2)
    tl.store(SCRATCH + off[:, None] + e[None, :], acc, active[:, None] & (e[None, :] < E))
    tl.store(SCRATCH + off + E, maximum, active)
    tl.store(SCRATCH + off + E + 1, total, active)


@triton.jit
def _decode_reduce(
    S, O, LEN, H: tl.constexpr, E: tl.constexpr, BE: tl.constexpr, SPLITS: tl.constexpr, WINDOW: tl.constexpr
):
    h, seq = tl.program_id(0), tl.program_id(1)
    pos = tl.load(LEN + seq)
    e = tl.arange(0, BE)
    acc = tl.full((BE,), 0, tl.float32)
    maximum = -float("inf")
    total = 0.0
    start = 0
    if WINDOW >= 0:
        start = tl.maximum(0, pos - WINDOW) // 256
    for s in range(start, pos // 256 + 1):
        off = ((seq * SPLITS + s) * H + h) * (E + 2)
        block = tl.load(S + off + e, e < E, 0)
        peak = tl.load(S + off + E)
        mass = tl.load(S + off + E + 1)
        new_max = tl.maximum(maximum, peak)
        alpha = tl.exp2(maximum - new_max)
        beta = tl.exp2(peak - new_max)
        acc = acc * alpha + block * beta
        total = total * alpha + mass * beta
        maximum = new_max
    tl.store(O + (seq * H + h) * E + e, acc / total, e < E)


def split_decode(q, k, v, bounds, table, uk, uv, window, scale):
    batch, heads, dim = q.shape
    val = uv.shape[-1]
    splits = triton.cdiv(table.shape[1] * k.shape[1], 256)
    scratch = q.new_empty((batch, splits, heads, val + 2), dtype=torch.float32)
    out = q.new_empty((batch, heads, val))
    _decode_partial[(splits, k.shape[2], batch)](
        q,
        k,
        v,
        uk,
        uv,
        table,
        bounds,
        scratch,
        heads,
        k.shape[2],
        dim,
        val,
        triton.next_power_of_2(dim),
        triton.next_power_of_2(val),
        k.shape[1],
        v.shape[-1],
        table.shape[1],
        splits,
        scale,
        window,
        num_warps=4,
        num_stages=1,
    )
    _decode_reduce[(heads, batch)](
        scratch, out, bounds, heads, val, triton.next_power_of_2(val), splits, window, num_warps=4
    )
    return out
