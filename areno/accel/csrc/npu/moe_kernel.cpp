#include "kernel_operator.h"
#include "moe_launch.h"

namespace areno_npu {
using namespace AscendC;

class RouteIO {
public:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> keyQueue, weightQueue, prefixQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;

    __aicore__ inline void Init() {
        pipe.InitBuffer(keyQueue, 1, kRouteTile * sizeof(int64_t));
        pipe.InitBuffer(weightQueue, 1, kRouteTile * sizeof(float));
        pipe.InitBuffer(prefixQueue, 1, kExpertTile * sizeof(int64_t));
        pipe.InitBuffer(outQueue, 1, kExpertTile * sizeof(int64_t));
    }

    template <HardEvent Event>
    __aicore__ inline void Sync() {
        auto id = pipe.FetchEventID(Event);
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> Load(TQue<QuePosition::VECIN, 1>& queue,
        GlobalTensor<T>& gm, int64_t offset, uint32_t n) {
        auto local = queue.template AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, pad);
        queue.EnQue(local);
        local = queue.template DeQue<T>();
        Sync<HardEvent::MTE2_S>();
        return local;
    }

    template <typename T>
    __aicore__ inline void Free(TQue<QuePosition::VECIN, 1>& queue, LocalTensor<T> local) {
        Sync<HardEvent::S_MTE2>();
        queue.FreeTensor(local);
    }

    template <typename T>
    __aicore__ inline void Scalar(GlobalTensor<T>& gm, int64_t offset, T value) {
        auto local = outQueue.template AllocTensor<T>();
        Sync<HardEvent::MTE3_S>();
        local.SetValue(0, value);
        Sync<HardEvent::S_MTE3>();
        outQueue.EnQue(local);
        local = outQueue.template DeQue<T>();
        DataCopyExtParams copy{1, sizeof(T), 0, 0, 0};
        DataCopyPad(gm[offset], local, copy);
        outQueue.FreeTensor(local);
    }

    __aicore__ inline void Vector(GlobalTensor<int32_t>& gm, int64_t offset, LocalTensor<int32_t> src, uint32_t n) {
        auto local = outQueue.template AllocTensor<int32_t>();
        Adds(local, src, int32_t{0}, n);
        outQueue.EnQue(local);
        local = outQueue.template DeQue<int32_t>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(gm[offset], local, copy);
        outQueue.FreeTensor(local);
    }

    __aicore__ inline void Fill(GlobalTensor<int32_t>& gm, int64_t begin, int64_t end, int32_t value) {
        for (int64_t offset = begin; offset < end; offset += kExpertTile) {
            uint32_t n = end - offset < kExpertTile ? end - offset : kExpertTile;
            auto local = outQueue.template AllocTensor<int32_t>();
            Duplicate(local, value, n);
            outQueue.EnQue(local);
            local = outQueue.template DeQue<int32_t>();
            DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(gm[offset], local, copy);
            outQueue.FreeTensor(local);
        }
    }
};

template <typename Id, RouteKind Kind, bool Metadata>
class RouteKernel {
    RouteIO io;
    TBuf<QuePosition::VECCALC> histogramBuffer;
    GlobalTensor<Id> keys;
    GlobalTensor<float> weights, routeWeight;
    GlobalTensor<int32_t> partial, counts, positions, aligned;
    GlobalTensor<int64_t> tokens;

    __aicore__ inline int64_t Expert(LocalTensor<Id> ids, LocalTensor<float> w, uint32_t i,
        int64_t route, int64_t columns, int64_t start, int64_t experts) {
        if constexpr (Kind == DenseRoutes) return ids.GetValue(i) ? route % columns : -1;
        else {
            int64_t expert = static_cast<int64_t>(ids.GetValue(i));
            if (expert < start || expert >= start + experts) return -1;
            if constexpr (Kind == TopKRoutes) if (w.GetValue(i) == 0.0f) return -1;
            return expert - start;
        }
    }

public:
    __aicore__ inline RouteKernel() {}
    __aicore__ inline void Init(GM_ADDR ids, GM_ADDR w, GM_ADDR p, GM_ADDR c,
        GM_ADDR rw, GM_ADDR ti, GM_ADDR pos, GM_ADDR aligned_ids) {
        io.Init();
        keys.SetGlobalBuffer(reinterpret_cast<__gm__ Id*>(ids));
        partial.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(p));
        if constexpr (Kind != AlignRoutes) weights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
        if constexpr (!Metadata) {
            counts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(c));
            io.pipe.InitBuffer(histogramBuffer, kExpertTile * sizeof(int32_t));
        } else if constexpr (Kind == AlignRoutes) aligned.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(aligned_ids));
        else {
            routeWeight.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rw));
            tokens.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ti));
            if constexpr (Kind == TopKRoutes) positions.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pos));
        }
    }

    __aicore__ inline void Process(int64_t routes, int64_t columns, int64_t start, int64_t experts, int64_t capacity) {
        int64_t routeTiles = (routes + kRouteTile - 1) / kRouteTile;
        int64_t expertTiles = (experts + kExpertTile - 1) / kExpertTile;
        for (int64_t task = GetBlockIdx(); task < routeTiles * expertTiles; task += GetBlockNum()) {
            int64_t tile = task / expertTiles, expertBase = task % expertTiles * kExpertTile;
            int64_t routeBase = tile * kRouteTile;
            uint32_t n = routes - routeBase < kRouteTile ? routes - routeBase : kRouteTile;
            uint32_t ne = experts - expertBase < kExpertTile ? experts - expertBase : kExpertTile;
            auto ids = io.Load(io.keyQueue, keys, routeBase, n);
            LocalTensor<float> w;
            if constexpr (Kind == TopKRoutes || (Metadata && Kind == DenseRoutes))
                w = io.Load(io.weightQueue, weights, routeBase, n);
            LocalTensor<int32_t> counters;
            if constexpr (Metadata) counters = io.Load(io.prefixQueue, partial, tile * experts + expertBase, ne);
            else {
                counters = histogramBuffer.Get<int32_t>();
                Duplicate(counters, int32_t{0}, ne);
                io.Sync<HardEvent::V_S>();
            }
            for (uint32_t i = 0; i < n; ++i) {
                int64_t expert = Expert(ids, w, i, routeBase + i, columns, start, experts);
                if (expert < expertBase || expert >= expertBase + ne) continue;
                uint32_t local = expert - expertBase;
                int32_t row = counters.GetValue(local);
                counters.SetValue(local, row + 1);
                if constexpr (Metadata) {
                    // Valid callers provide the exact dense size or sufficient
                    // alignment capacity. Do not write outside a bad buffer.
                    if (row < 0 || row >= capacity) continue;
                    int64_t route = routeBase + i;
                    if constexpr (Kind == AlignRoutes) io.Scalar(aligned, row, static_cast<int32_t>(route));
                    else {
                        io.Scalar(tokens, row, route / columns);
                        io.Scalar(routeWeight, row, w.GetValue(i));
                        if constexpr (Kind == TopKRoutes) io.Scalar(positions, row, static_cast<int32_t>(route % columns));
                    }
                }
            }
            if constexpr (Metadata) io.Free(io.prefixQueue, counters);
            else {
                io.Sync<HardEvent::S_V>();
                auto local = io.outQueue.AllocTensor<int32_t>();
                Adds(local, counters, int32_t{0}, ne);
                io.outQueue.EnQue(local);
                local = io.outQueue.DeQue<int32_t>();
                DataCopyExtParams copy{1, static_cast<uint32_t>(ne * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(partial[tile * experts + expertBase], local, copy);
                SetAtomicAdd<int32_t>();
                DataCopyPad(counts[expertBase], local, copy);
                DisableDmaAtomic();
                io.outQueue.FreeTensor(local);
            }
            if constexpr (Kind == TopKRoutes || (Metadata && Kind == DenseRoutes)) io.Free(io.weightQueue, w);
            io.Free(io.keyQueue, ids);
        }
    }
};

class RoutePrefixKernel {
    RouteIO io;
    TBuf<QuePosition::VECCALC> prefixBuffer;
    GlobalTensor<int32_t> partial;
    GlobalTensor<int64_t> offsets;
public:
    __aicore__ inline RoutePrefixKernel() {}
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR off) {
        io.Init();
        io.pipe.InitBuffer(prefixBuffer, kExpertTile * sizeof(int32_t));
        partial.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(p));
        offsets.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(off));
    }
    __aicore__ inline void Process(int64_t tiles, int64_t experts) {
        auto prefix = prefixBuffer.Get<int32_t>();
        for (int64_t base = GetBlockIdx() * kExpertTile; base < experts; base += GetBlockNum() * kExpertTile) {
            uint32_t n = experts - base < kExpertTile ? experts - base : kExpertTile;
            auto initial = io.Load(io.prefixQueue, offsets, base, n);
            io.Sync<HardEvent::V_S>();
            for (uint32_t i = 0; i < n; ++i) prefix.SetValue(i, static_cast<int32_t>(initial.GetValue(i)));
            io.Free(io.prefixQueue, initial);
            io.Sync<HardEvent::S_V>();
            for (int64_t tile = 0; tile < tiles; ++tile) {
                auto count = io.Load(io.prefixQueue, partial, tile * experts + base, n);
                io.Vector(partial, tile * experts + base, prefix, n);
                Add(prefix, prefix, count, n);
                PipeBarrier<PIPE_V>();
                io.Free(io.prefixQueue, count);
            }
        }
    }
};

__global__ __aicore__ void route_offsets_kernel(GM_ADDR c, GM_ADDR off, int64_t experts, int64_t blockSize,
    GM_ADDR blockIds, GM_ADDR paddedTotal, GM_ADDR scratch, int64_t blockCapacity) {
    RouteIO io;
    io.Init();
    GlobalTensor<int32_t> counts, blocks, total, cumsum;
    GlobalTensor<int64_t> offsets;
    counts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(c));
    offsets.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(off));
    bool align = paddedTotal != nullptr;
    if (align) {
        blocks.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockIds));
        total.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(paddedTotal));
        cumsum.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch));
    }
    int64_t offset = 0;
    for (int64_t base = 0; base < experts; base += kExpertTile) {
        uint32_t n = experts - base < kExpertTile ? experts - base : kExpertTile;
        auto local = io.Load(io.prefixQueue, counts, base, n);
        for (uint32_t i = 0; i < n; ++i) {
            int64_t count = local.GetValue(i);
            int64_t next = offset + (count + blockSize - 1) / blockSize * blockSize;
            io.Scalar(offsets, base + i, offset);
            if (align) {
                io.Scalar(cumsum, base + i, static_cast<int32_t>(offset));
                int64_t end = next / blockSize < blockCapacity ? next / blockSize : blockCapacity;
                io.Fill(blocks, offset / blockSize, end, static_cast<int32_t>(base + i - 1));
            }
            offset = next;
        }
        io.Free(io.prefixQueue, local);
    }
    io.Scalar(offsets, experts, offset);
    if (align) {
        io.Scalar(cumsum, experts, static_cast<int32_t>(offset));
        io.Scalar(total, 0, static_cast<int32_t>(offset));
    }
}

__global__ __aicore__ void route_prefix_kernel(GM_ADDR p, GM_ADDR offsets, int64_t tiles, int64_t experts) {
    RoutePrefixKernel kernel;
    kernel.Init(p, offsets);
    kernel.Process(tiles, experts);
}

__global__ __aicore__ void route_weight_backward_kernel(GM_ADDR g, GM_ADDR ti, GM_ADDR pos, GM_ADDR out,
    int64_t rows, int64_t top_k) {
    RouteIO io;
    io.Init();
    GlobalTensor<float> grad, output;
    GlobalTensor<int64_t> tokens;
    GlobalTensor<int32_t> positions;
    grad.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
    tokens.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ti));
    positions.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pos));
    for (int64_t base = GetBlockIdx() * kRouteTile; base < rows; base += GetBlockNum() * kRouteTile) {
        uint32_t n = rows - base < kRouteTile ? rows - base : kRouteTile;
        auto t = io.Load(io.keyQueue, tokens, base, n);
        auto p = io.Load(io.prefixQueue, positions, base, n);
        auto dy = io.Load(io.weightQueue, grad, base, n);
        for (uint32_t i = 0; i < n; ++i) io.Scalar(output, t.GetValue(i) * top_k + p.GetValue(i), dy.GetValue(i));
        io.Free(io.keyQueue, t);
        io.Free(io.prefixQueue, p);
        io.Free(io.weightQueue, dy);
    }
}

__global__ __aicore__ void route_fill_kernel(GM_ADDR out, int64_t elements, int32_t value) {
    RouteIO io;
    io.Init();
    GlobalTensor<int32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(out));
    for (int64_t base = GetBlockIdx() * kRouteTile; base < elements; base += GetBlockNum() * kRouteTile) {
        int64_t end = elements - base < kRouteTile ? elements : base + kRouteTile;
        io.Fill(output, base, end, value);
    }
}

template <typename Id, RouteKind Kind, bool Metadata>
__global__ __aicore__ void route_kernel(GM_ADDR keys, GM_ADDR weights, GM_ADDR partial, GM_ADDR counts,
    GM_ADDR rw, GM_ADDR ti, GM_ADDR pos, GM_ADDR aligned, int64_t routes, int64_t columns,
    int64_t start, int64_t experts, int64_t capacity) {
    RouteKernel<Id, Kind, Metadata> kernel;
    kernel.Init(keys, weights, partial, counts, rw, ti, pos, aligned);
    kernel.Process(routes, columns, start, experts, capacity);
}

template <bool Metadata>
void launch_routes(uint32_t blocks, void* stream, RouteKind kind, uint32_t storage, const void* keys,
    const float* weights, int32_t* partial, int32_t* counts, float* rw, int64_t* ti, int32_t* pos, int32_t* aligned,
    int64_t routes, int64_t columns, int64_t start, int64_t experts, int64_t capacity) {
#define ARENO_ROUTES(T, K) route_kernel<T, K, Metadata><<<blocks, nullptr, stream>>>( \
    (uint8_t*)keys, (uint8_t*)weights, (uint8_t*)partial, (uint8_t*)counts, (uint8_t*)rw, (uint8_t*)ti, \
    (uint8_t*)pos, (uint8_t*)aligned, routes, columns, start, experts, capacity)
    if (kind == DenseRoutes) { ARENO_ROUTES(uint8_t, DenseRoutes); }
    else if (kind == TopKRoutes) { ARENO_ROUTES(int64_t, TopKRoutes); }
    else switch (storage) {
        case 0: { ARENO_ROUTES(int64_t, AlignRoutes); } break;
        case 1: { ARENO_ROUTES(int32_t, AlignRoutes); } break;
        case 2: { ARENO_ROUTES(int16_t, AlignRoutes); } break;
        case 3: { ARENO_ROUTES(int8_t, AlignRoutes); } break;
        case 4: { ARENO_ROUTES(uint8_t, AlignRoutes); } break;
    }
#undef ARENO_ROUTES
}

void launch_route_count(uint32_t blocks, void* stream, RouteKind kind, uint32_t storage,
    const void* keys, const float* weights, int32_t* partial, int32_t* counts,
    int64_t routes, int64_t columns, int64_t start, int64_t experts) {
    launch_routes<false>(blocks, stream, kind, storage, keys, weights, partial, counts,
        nullptr, nullptr, nullptr, nullptr, routes, columns, start, experts, 0);
}

void launch_route_metadata(uint32_t blocks, void* stream, RouteKind kind, uint32_t storage,
    const void* keys, const float* weights, const int32_t* partial, float* rw, int64_t* ti,
    int32_t* pos, int32_t* aligned, int64_t routes, int64_t columns, int64_t start, int64_t experts, int64_t capacity) {
    launch_routes<true>(blocks, stream, kind, storage, keys, weights, const_cast<int32_t*>(partial), nullptr,
        rw, ti, pos, aligned, routes, columns, start, experts, capacity);
}

void launch_route_offsets(void* stream, const int32_t* counts, int64_t* offsets, int64_t experts,
    int64_t blockSize, int32_t* blockIds, int32_t* total, int32_t* scratch, int64_t blockCapacity) {
    route_offsets_kernel<<<1, nullptr, stream>>>((uint8_t*)counts, (uint8_t*)offsets, experts, blockSize,
        (uint8_t*)blockIds, (uint8_t*)total, (uint8_t*)scratch, blockCapacity);
}

void launch_route_prefix(uint32_t blocks, void* stream, int32_t* partial, const int64_t* offsets,
    int64_t tiles, int64_t experts) {
    route_prefix_kernel<<<blocks, nullptr, stream>>>((uint8_t*)partial, (uint8_t*)offsets, tiles, experts);
}

void launch_route_weight_backward(uint32_t blocks, void* stream, const float* grad, const int64_t* tokens,
    const int32_t* positions, float* output, int64_t rows, int64_t top_k) {
    route_weight_backward_kernel<<<blocks, nullptr, stream>>>((uint8_t*)grad, (uint8_t*)tokens,
        (uint8_t*)positions, (uint8_t*)output, rows, top_k);
}

void launch_route_fill(uint32_t blocks, void* stream, int32_t* output, int64_t elements, int32_t value) {
    route_fill_kernel<<<blocks, nullptr, stream>>>((uint8_t*)output, elements, value);
}
} // namespace areno_npu
