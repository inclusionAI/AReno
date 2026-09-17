// Compatibility kernels for shapes/dtypes outside flash-attn-npu's range.
#include "kernel_operator.h"
#include <math.h>
#include "attention_launch.h"

namespace areno_npu {
using namespace AscendC;

class AttentionIO {
    TQue<QuePosition::VECIN, 1> inQueue, indexQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECCALC> scratch;
    LocalTensor<float> scalar, work;
public:
    TPipe pipe;
    __aicore__ inline void Init() {
        pipe.InitBuffer(inQueue, 1, kAttentionTile * sizeof(float));
        pipe.InitBuffer(outQueue, 1, kAttentionTile * sizeof(float));
        pipe.InitBuffer(indexQueue, 1, 32);
        pipe.InitBuffer(scratch, 2 * kAttentionTile * sizeof(float));
        scalar = scratch.Get<float>();
        work = scalar[kAttentionTile];
    }
    template <HardEvent Event>
    __aicore__ inline void Sync() {
        auto id = pipe.FetchEventID(Event);
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }
    template <typename T>
    __aicore__ inline void Read(GlobalTensor<T>& gm, LocalTensor<float> dst, int64_t offset, uint32_t n) {
        auto local = inQueue.template AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, pad);
        inQueue.EnQue(local);
        local = inQueue.template DeQue<T>();
        if constexpr (sizeof(T) == sizeof(float)) Adds(dst, local, 0.0f, n);
        else Cast(dst, local, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(local);
    }
    template <typename T, bool Atomic = false>
    __aicore__ inline void Write(GlobalTensor<T>& gm, LocalTensor<float> src, int64_t offset, uint32_t n) {
        auto local = outQueue.template AllocTensor<T>();
        if constexpr (sizeof(T) == sizeof(float)) Adds(local, src, 0.0f, n);
        else Cast(local, src, RoundMode::CAST_RINT, n);
        outQueue.EnQue(local);
        local = outQueue.template DeQue<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        if constexpr (Atomic) SetAtomicAdd<float>();
        DataCopyPad(gm[offset], local, copy);
        if constexpr (Atomic) DisableDmaAtomic();
        outQueue.FreeTensor(local);
    }
    __aicore__ inline float Scalar() {
        Sync<HardEvent::V_S>();
        float result = scalar.GetValue(0);
        Sync<HardEvent::S_V>();
        return result;
    }
    __aicore__ inline float Sum(LocalTensor<float> src, uint32_t n) {
        ReduceSum(scalar, src, work, n);
        return Scalar();
    }
    __aicore__ inline float ExpScalar(float value) {
        Duplicate(scalar, value, 1);
        PipeBarrier<PIPE_V>();
        Exp(scalar, scalar, 1);
        return Scalar();
    }
    __aicore__ inline float ReadScalar(GlobalTensor<float>& gm, int64_t offset) {
        Read(gm, scalar, offset, 1);
        return Scalar();
    }
    __aicore__ inline void WriteScalar(GlobalTensor<float>& gm, int64_t offset, float value) {
        Duplicate(scalar, value, 1);
        PipeBarrier<PIPE_V>();
        Write(gm, scalar, offset, 1);
    }
    __aicore__ inline int64_t Index(GlobalTensor<int32_t>& gm, int64_t offset) {
        auto local = indexQueue.AllocTensor<int32_t>();
        DataCopyExtParams copy{1, sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, pad);
        indexQueue.EnQue(local);
        local = indexQueue.DeQue<int32_t>();
        Sync<HardEvent::MTE2_S>();
        int64_t result = local.GetValue(0);
        Sync<HardEvent::S_MTE2>();
        indexQueue.FreeTensor(local);
        return result;
    }
    template <typename T>
    __aicore__ inline void Copy(GlobalTensor<T>& src, GlobalTensor<T>& dst, int64_t from, int64_t to, uint32_t n) {
        auto local = inQueue.template AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, src[from], copy, pad);
        inQueue.EnQue(local);
        local = inQueue.template DeQue<T>();
        Sync<HardEvent::MTE2_MTE3>();
        DataCopyPad(dst[to], local, copy);
        Sync<HardEvent::MTE3_MTE2>();
        inQueue.FreeTensor(local);
    }
};

template <typename T, AttentionLayout Layout, bool Backward>
class AttentionKernel {
    AttentionIO io;
    TBuf<QuePosition::VECCALC> scratch;
    GlobalTensor<T> q, k, v, grad, saved, output;
    GlobalTensor<int32_t> boundaries, table, lengths;
    GlobalTensor<float> dq, dk, dv, stats, splitAcc;
    LocalTensor<float> qpart, gpart, opart, acc, qt, kt, vt, gt, ot, product;
    AttentionShape s;
    int64_t row, col, batch, kvHead, kvBase, first, end, split;
    uint32_t n;
    float scale;

    __aicore__ inline int64_t KeyBase(int64_t token) {
        if constexpr (Layout == DenseAttention) return kvBase + token * s.dim;
        if constexpr (Layout == PackedAttention) return (token * s.kv_heads + kvHead) * s.dim;
        int64_t block = io.Index(table, batch * s.max_blocks + token / s.block_size);
        return ((block * s.block_size + token % s.block_size) * s.kv_heads + kvHead) * s.dim;
    }
    __aicore__ inline void SetRow() {
        if constexpr (Layout == DenseAttention) {
            int64_t bh = row / s.q_len, pos = s.query_start + row % s.q_len;
            kvBase = bh * s.k_len * s.dim;
            first = s.window_left >= 0 && pos > s.window_left ? pos - s.window_left : 0;
            end = pos + 1;
        } else if constexpr (Layout == PackedAttention) {
            int64_t token = row / s.q_heads;
            kvHead = row % s.q_heads / (s.q_heads / s.kv_heads);
            int64_t lo = 0, hi = s.sequences;
            while (lo + 1 < hi) {
                int64_t mid = (lo + hi) / 2;
                if (io.Index(boundaries, mid) <= token) lo = mid;
                else hi = mid;
            }
            first = io.Index(boundaries, lo);
            if (s.window_left >= 0 && token - first > s.window_left) first = token - s.window_left;
            end = token + 1;
        } else {
            batch = row / s.q_heads;
            kvHead = row % s.q_heads / (s.q_heads / s.kv_heads);
            int64_t length = io.Index(lengths, batch) + 1;
            int64_t allowed = s.window_left >= 0 && length - 1 > s.window_left ? length - 1 - s.window_left : 0;
            int64_t size = (length + s.splits - 1) / s.splits;
            first = split * size > allowed ? split * size : allowed;
            end = (split + 1) * size < length ? (split + 1) * size : length;
        }
        io.Read(q, qpart, row * s.dim + col, n);
        if constexpr (Backward) {
            io.Read(grad, gpart, row * s.dim + col, n);
            io.Read(saved, opart, row * s.dim + col, n);
        }
        Duplicate(acc, 0.0f, n);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline float Dot(int64_t base) {
        float sum = 0.0f;
        // Typical heads fit in one UB tile and reuse the resident Q row.
        // Larger heads stream bounded tiles; no head-size ceiling or QK
        // matrix allocation is introduced.
        for (int64_t d = 0; d < s.dim; d += kAttentionTile) {
            uint32_t width = s.dim - d < kAttentionTile ? s.dim - d : kAttentionTile;
            auto query = qpart;
            if (s.dim > kAttentionTile) { io.Read(q, qt, row * s.dim + d, width); query = qt; }
            io.Read(k, kt, base + d, width);
            Mul(product, query, kt, width);
            PipeBarrier<PIPE_V>();
            sum += io.Sum(product, width);
        }
        return sum * scale;
    }
    __aicore__ inline float Delta(int64_t base) {
        float sum = 0.0f;
        for (int64_t d = 0; d < s.dim; d += kAttentionTile) {
            uint32_t width = s.dim - d < kAttentionTile ? s.dim - d : kAttentionTile;
            auto go = gpart, out = opart;
            if (s.dim > kAttentionTile) {
                io.Read(grad, gt, row * s.dim + d, width);
                io.Read(saved, ot, row * s.dim + d, width);
                go = gt; out = ot;
            }
            io.Read(v, vt, base + d, width);
            Sub(vt, vt, out, width);
            PipeBarrier<PIPE_V>();
            Mul(product, go, vt, width);
            PipeBarrier<PIPE_V>();
            sum += io.Sum(product, width);
        }
        return sum;
    }
    __aicore__ inline void Forward() {
        float maximum = -INFINITY, denom = 0.0f;
        for (int64_t token = first; token < end; ++token) {
            int64_t base = KeyBase(token);
            float score = Dot(base), next = maximum > score ? maximum : score;
            float alpha = denom == 0.0f ? 0.0f : io.ExpScalar(maximum - next);
            float beta = io.ExpScalar(score - next);
            denom = denom * alpha + beta;
            maximum = next;
            io.Read(v, vt, base + col, n);
            Muls(acc, acc, alpha, n);
            Muls(vt, vt, beta, n);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, vt, n);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (Layout == PagedAttention) {
            int64_t offset = row * s.splits + split;
            if (col == 0) {
                io.WriteScalar(stats, 2 * offset, maximum);
                io.WriteScalar(stats, 2 * offset + 1, denom);
            }
            io.Write(splitAcc, acc, offset * s.dim + col, n);
        } else {
            Muls(acc, acc, 1.0f / denom, n);
            PipeBarrier<PIPE_V>();
            io.Write(output, acc, row * s.dim + col, n);
        }
    }
    __aicore__ inline void Backpropagate() {
        float maximum = -INFINITY;
        for (int64_t token = first; token < end; ++token) {
            float score = Dot(KeyBase(token));
            maximum = maximum > score ? maximum : score;
        }
        float denom = 0.0f;
        for (int64_t token = first; token < end; ++token) denom += io.ExpScalar(Dot(KeyBase(token)) - maximum);
        for (int64_t token = first; token < end; ++token) {
            int64_t base = KeyBase(token);
            float p = io.ExpScalar(Dot(base) - maximum) / denom;
            // Match CUDA's gradient using the saved, storage-rounded output.
            float ds = p * Delta(base) * scale;
            io.Read(k, kt, base + col, n);
            Muls(kt, kt, ds, n);
            Muls(product, qpart, ds, n);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, kt, n);
            io.Write<float, true>(dk, product, base + col, n);
            Muls(product, gpart, p, n);
            PipeBarrier<PIPE_V>();
            io.Write<float, true>(dv, product, base + col, n);
        }
        io.Write(dq, acc, row * s.dim + col, n);
    }
public:
    __aicore__ inline AttentionKernel() {}
    __aicore__ inline void Init(GM_ADDR qIn, GM_ADDR kIn, GM_ADDR vIn, GM_ADDR g, GM_ADDR o,
        GM_ADDR cu, GM_ADDR pages, GM_ADDR lens, GM_ADDR result, GM_ADDR gradK, GM_ADDR gradV,
        GM_ADDR splitStats, GM_ADDR splitAccumulator) {
        io.Init();
        io.pipe.InitBuffer(scratch, 10 * kAttentionTile * sizeof(float));
        qpart = scratch.Get<float>(); gpart = qpart[kAttentionTile]; opart = qpart[2 * kAttentionTile];
        acc = qpart[3 * kAttentionTile]; qt = qpart[4 * kAttentionTile]; kt = qpart[5 * kAttentionTile];
        vt = qpart[6 * kAttentionTile]; gt = qpart[7 * kAttentionTile]; ot = qpart[8 * kAttentionTile];
        product = qpart[9 * kAttentionTile];
        q.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(qIn));
        k.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(kIn));
        v.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(vIn));
        if constexpr (Layout == PackedAttention) boundaries.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cu));
        if constexpr (Layout == PagedAttention) {
            table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pages));
            lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lens));
            stats.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(splitStats));
            splitAcc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(splitAccumulator));
        }
        if constexpr (Backward) {
            grad.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(g));
            saved.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(o));
            dq.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(result));
            dk.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradK));
            dv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradV));
        } else output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(result));
    }
    __aicore__ inline void Process(AttentionShape shape, float softmaxScale) {
        s = shape; scale = softmaxScale;
        int64_t tiles = (s.dim - 1) / kAttentionTile + 1;
        for (int64_t task = GetBlockIdx(); task < s.rows * s.splits * tiles; task += GetBlockNum()) {
            col = task % tiles * kAttentionTile;
            split = task / tiles % s.splits;
            row = task / tiles / s.splits;
            n = s.dim - col < kAttentionTile ? s.dim - col : kAttentionTile;
            SetRow();
            if constexpr (Backward) Backpropagate();
            else Forward();
        }
    }
};

} // namespace areno_npu

template<typename T>
__global__ __aicore__ void attention_cache_update_kernel(GM_ADDR ku, GM_ADDR vu, GM_ADDR kc, GM_ADDR vc,
    GM_ADDR pages, GM_ADDR lens, int64_t batch, int64_t heads, int64_t dim, int64_t blockSize, int64_t maxBlocks) {
    using namespace AscendC;
    using namespace areno_npu;
    AttentionIO io;
    io.Init();
    GlobalTensor<T> k, v, kCache, vCache;
    GlobalTensor<int32_t> table, lengths;
    k.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ku));
    v.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(vu));
    kCache.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(kc));
    vCache.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(vc));
    table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pages));
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lens));
    int64_t tiles = (dim - 1) / kAttentionTile + 1;
    for (int64_t task = GetBlockIdx(); task < batch * heads * tiles; task += GetBlockNum()) {
        int64_t row = task / tiles, col = task % tiles * kAttentionTile;
        int64_t b = row / heads, h = row % heads;
        int64_t pos = io.Index(lengths, b), block = io.Index(table, b * maxBlocks + pos / blockSize);
        int64_t dest = ((block * blockSize + pos % blockSize) * heads + h) * dim + col;
        uint32_t n = dim - col < kAttentionTile ? dim - col : kAttentionTile;
        // Cache updates are copies, with no arithmetic or storage conversion.
        io.Copy(k, kCache, row * dim + col, dest, n);
        io.Copy(v, vCache, row * dim + col, dest, n);
    }
}

template<typename T>
__global__ __aicore__ void attention_split_reduce_kernel(GM_ADDR statistics, GM_ADDR accumulator, GM_ADDR result,
    int64_t rows, int64_t dim, int64_t splits) {
    using namespace AscendC;
    using namespace areno_npu;
    AttentionIO io;
    io.Init();
    TBuf<QuePosition::VECCALC> scratch;
    io.pipe.InitBuffer(scratch, 2 * kAttentionTile * sizeof(float));
    auto acc = scratch.Get<float>(), part = acc[kAttentionTile];
    GlobalTensor<float> stats, splitAcc;
    GlobalTensor<T> output;
    stats.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(statistics));
    splitAcc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(accumulator));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(result));
    int64_t tiles = (dim - 1) / kAttentionTile + 1;
    for (int64_t task = GetBlockIdx(); task < rows * tiles; task += GetBlockNum()) {
        int64_t row = task / tiles, col = task % tiles * kAttentionTile;
        uint32_t n = dim - col < kAttentionTile ? dim - col : kAttentionTile;
        Duplicate(acc, 0.0f, n);
        PipeBarrier<PIPE_V>();
        float maximum = -INFINITY, denom = 0.0f;
        for (int64_t split = 0; split < splits; ++split) {
            int64_t offset = row * splits + split;
            float partDenom = io.ReadScalar(stats, 2 * offset + 1);
            if (partDenom == 0.0f) continue;
            float partMax = io.ReadScalar(stats, 2 * offset);
            float next = maximum > partMax ? maximum : partMax;
            float alpha = denom == 0.0f ? 0.0f : io.ExpScalar(maximum - next);
            float beta = io.ExpScalar(partMax - next);
            denom = denom * alpha + partDenom * beta;
            maximum = next;
            io.Read(splitAcc, part, offset * dim + col, n);
            Muls(acc, acc, alpha, n);
            Muls(part, part, beta, n);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, part, n);
            PipeBarrier<PIPE_V>();
        }
        Muls(acc, acc, 1.0f / denom, n);
        PipeBarrier<PIPE_V>();
        io.Write(output, acc, row * dim + col, n);
    }
}

template<typename T, uint32_t Layout, bool Backward>
__global__ __aicore__ void attention_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR grad, GM_ADDR saved,
    GM_ADDR cu, GM_ADDR table, GM_ADDR lengths, GM_ADDR output, GM_ADDR dk, GM_ADDR dv, GM_ADDR stats, GM_ADDR acc,
    int64_t rows, int64_t qHeads, int64_t kvHeads, int64_t dim, int64_t qLen, int64_t kLen, int64_t sequences,
    int64_t start, int64_t window, int64_t blockSize, int64_t maxBlocks, int64_t splits, float scale) {
    using namespace AscendC;
    using namespace areno_npu;
    AttentionKernel<T, static_cast<AttentionLayout>(Layout), Backward> kernel;
    kernel.Init(q, k, v, grad, saved, cu, table, lengths, output, dk, dv, stats, acc);
    kernel.Process({rows, qHeads, kvHeads, dim, qLen, kLen, sequences, start, window, blockSize, maxBlocks, splits}, scale);
}

// Materialize device entries before CANN extracts host launcher specializations.
#define ARENO_ATTN_INSTANCE(T, L, B) \
    template void attention_kernel<T, areno_npu::L, B>( \
        GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, \
        GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, \
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, \
        int64_t, int64_t, int64_t, int64_t, float);
#define ARENO_ATTN_INSTANCES(T) \
    ARENO_ATTN_INSTANCE(T, DenseAttention, false) ARENO_ATTN_INSTANCE(T, DenseAttention, true) \
    ARENO_ATTN_INSTANCE(T, PackedAttention, false) ARENO_ATTN_INSTANCE(T, PackedAttention, true) \
    ARENO_ATTN_INSTANCE(T, PagedAttention, false)
ARENO_ATTN_INSTANCES(float)
ARENO_ATTN_INSTANCES(half)
ARENO_ATTN_INSTANCES(bfloat16_t)
#undef ARENO_ATTN_INSTANCES
#undef ARENO_ATTN_INSTANCE

namespace areno_npu {

template <typename T>
void launch_attention_typed(uint32_t blocks, void* stream, AttentionLayout layout, bool backward,
    const void* q, const void* k, const void* v, const void* grad, const void* saved,
    const int32_t* boundaries, const int32_t* table, const int32_t* lengths,
    void* output, float* dk, float* dv, float* stats, float* acc, AttentionShape s, float scale) {
#define ARENO_ATTN(L, B) attention_kernel<T, L, B><<<blocks, nullptr, stream>>>( \
    (uint8_t*)q, (uint8_t*)k, (uint8_t*)v, (uint8_t*)grad, (uint8_t*)saved, (uint8_t*)boundaries, \
    (uint8_t*)table, (uint8_t*)lengths, (uint8_t*)output, (uint8_t*)dk, (uint8_t*)dv, (uint8_t*)stats, (uint8_t*)acc, \
    s.rows, s.q_heads, s.kv_heads, s.dim, s.q_len, s.k_len, s.sequences, s.query_start, s.window_left, \
    s.block_size, s.max_blocks, s.splits, scale)
    if (layout == PagedAttention) { ARENO_ATTN(PagedAttention, false); }
    else if (layout == PackedAttention) {
        if (backward) { ARENO_ATTN(PackedAttention, true); } else { ARENO_ATTN(PackedAttention, false); }
    } else {
        if (backward) { ARENO_ATTN(DenseAttention, true); } else { ARENO_ATTN(DenseAttention, false); }
    }
#undef ARENO_ATTN
}

void launch_attention(uint32_t blocks, void* stream, uint32_t storage, AttentionLayout layout, bool backward,
    const void* q, const void* k, const void* v, const void* grad, const void* saved,
    const int32_t* boundaries, const int32_t* table, const int32_t* lengths,
    void* output, float* dk, float* dv, float* stats, float* acc, AttentionShape shape, float scale) {
#define ARENO_ATTN_TYPE(T) launch_attention_typed<T>(blocks, stream, layout, backward, q, k, v, grad, saved, \
    boundaries, table, lengths, output, dk, dv, stats, acc, shape, scale)
    switch (storage) {
        case 0: ARENO_ATTN_TYPE(float); break;
        case 1: ARENO_ATTN_TYPE(half); break;
        case 2: ARENO_ATTN_TYPE(bfloat16_t); break;
    }
#undef ARENO_ATTN_TYPE
}
void launch_attention_cache_update(uint32_t blocks, void* stream, uint32_t storage,
    const void* k, const void* v, void* kCache, void* vCache, const int32_t* table, const int32_t* lengths,
    int64_t batch, AttentionShape s) {
#define ARENO_CACHE(T) attention_cache_update_kernel<T><<<blocks, nullptr, stream>>>( \
    (uint8_t*)k, (uint8_t*)v, (uint8_t*)kCache, (uint8_t*)vCache, (uint8_t*)table, (uint8_t*)lengths, \
    batch, s.kv_heads, s.dim, s.block_size, s.max_blocks)
    switch (storage) {
        case 0: ARENO_CACHE(float); break;
        case 1: ARENO_CACHE(half); break;
        case 2: ARENO_CACHE(bfloat16_t); break;
    }
#undef ARENO_CACHE
}
void launch_attention_split_reduce(uint32_t blocks, void* stream, uint32_t storage,
    const float* stats, const float* acc, void* output, AttentionShape s) {
#define ARENO_REDUCE(T) attention_split_reduce_kernel<T><<<blocks, nullptr, stream>>>( \
    (uint8_t*)stats, (uint8_t*)acc, (uint8_t*)output, s.rows, s.dim, s.splits)
    switch (storage) {
        case 0: ARENO_REDUCE(float); break;
        case 1: ARENO_REDUCE(half); break;
        case 2: ARENO_REDUCE(bfloat16_t); break;
    }
#undef ARENO_REDUCE
}
} // namespace areno_npu
