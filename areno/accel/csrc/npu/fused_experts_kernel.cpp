#include "kernel_operator.h"
#include "fused_experts_launch.h"

namespace areno_npu {
using namespace AscendC;

class ExpertIO {
    TQue<QuePosition::VECIN, 1> inputQueue;
    TBuf<QuePosition::VECCALC> scratch;
public:
    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    LocalTensor<float> value, sum;

    __aicore__ inline void Init() {
        pipe.InitBuffer(inputQueue, 1, kExpertVectorTile * sizeof(float));
        pipe.InitBuffer(outputQueue, 1, kExpertVectorTile * sizeof(int64_t));
        pipe.InitBuffer(scratch, 2 * kExpertVectorTile * sizeof(float));
        value = scratch.Get<float>();
        sum = value[kExpertVectorTile];
    }
    template <HardEvent Event>
    __aicore__ inline void Sync() {
        auto id = pipe.FetchEventID(Event);
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }
    template <typename T>
    __aicore__ inline T Scalar(GlobalTensor<T>& gm, int64_t offset) {
        auto source = gm[offset];
        DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(source);
        return source.GetValue(0);
    }
    template <typename T>
    __aicore__ inline void Read(GlobalTensor<T>& gm, int64_t offset, uint32_t n) {
        auto local = inputQueue.template AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, pad);
        inputQueue.EnQue(local);
        local = inputQueue.template DeQue<T>();
        if constexpr (sizeof(T) == sizeof(float)) Adds(value, local, 0.0f, n);
        else Cast(value, local, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        inputQueue.FreeTensor(local);
    }
    template <typename T>
    __aicore__ inline void Write(GlobalTensor<T>& gm, LocalTensor<float> src, int64_t offset, uint32_t n) {
        auto local = outputQueue.template AllocTensor<T>();
        Cast(local, src, RoundMode::CAST_RINT, n);
        outputQueue.EnQue(local);
        local = outputQueue.template DeQue<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPad(gm[offset], local, copy);
        outputQueue.FreeTensor(local);
    }
};

} // namespace areno_npu

// CANN's generated non-template wrapper calls this entry in global scope.
__global__ __aicore__ void expert_tokens_kernel(GM_ADDR sorted, GM_ADDR expertIds, GM_ADDR paddedTotal,
    GM_ADDR out, int64_t capacity, int64_t routes, int64_t topK) {
    using namespace AscendC;
    using namespace areno_npu;
    ExpertIO io;
    io.Init();
    GlobalTensor<int32_t> aligned, experts, total;
    GlobalTensor<int64_t> tokens;
    aligned.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sorted));
    experts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertIds));
    total.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(paddedTotal));
    tokens.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(out));
    int64_t used = io.Scalar(total, 0);
    for (int64_t begin = GetBlockIdx() * kExpertVectorTile; begin < capacity; begin += GetBlockNum() * kExpertVectorTile) {
        uint32_t n = capacity - begin < kExpertVectorTile ? capacity - begin : kExpertVectorTile;
        auto local = io.outputQueue.AllocTensor<int64_t>();
        io.Sync<HardEvent::MTE3_S>();
        for (uint32_t i = 0; i < n; ++i) {
            int64_t row = begin + i, token = -1;
            if (row < used && io.Scalar(experts, row / kExpertM) >= 0) {
                int64_t route = io.Scalar(aligned, row);
                if (route >= 0 && route < routes) token = route / topK;
            }
            local.SetValue(i, token);
        }
        io.Sync<HardEvent::S_MTE3>();
        io.outputQueue.EnQue(local);
        local = io.outputQueue.DeQue<int64_t>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(tokens[begin], local, copy);
        io.outputQueue.FreeTensor(local);
    }
}


template<typename T>
__global__ __aicore__ void expert_cast_kernel(GM_ADDR in, GM_ADDR out, int64_t rows, int64_t width, int64_t inputStride) {
    using namespace AscendC;
    using namespace areno_npu;
    ExpertIO io;
    io.Init();
    GlobalTensor<float> input;
    GlobalTensor<T> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(in));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    int64_t tiles = (width + kExpertVectorTile - 1) / kExpertVectorTile;
    for (int64_t task = GetBlockIdx(); task < rows * tiles; task += GetBlockNum()) {
        int64_t row = task / tiles, column = task % tiles * kExpertVectorTile;
        uint32_t n = width - column < kExpertVectorTile ? width - column : kExpertVectorTile;
        io.Read(input, row * inputStride + column, n);
        io.Write(output, io.value, row * width + column, n);
    }
}


template<typename T>
__global__ __aicore__ void expert_weight_scatter_kernel(GM_ADDR in, GM_ADDR w, GM_ADDR sorted,
    GM_ADDR expertIds, GM_ADDR paddedTotal, GM_ADDR out, int64_t capacity, int64_t routes, int64_t hidden, int64_t inputStride) {
    using namespace AscendC;
    using namespace areno_npu;
    ExpertIO io;
    io.Init();
    GlobalTensor<float> input, weights;
    GlobalTensor<int32_t> aligned, experts, total;
    GlobalTensor<T> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(in));
    weights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
    aligned.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sorted));
    experts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertIds));
    total.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(paddedTotal));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    int64_t used = io.Scalar(total, 0), tiles = (hidden + kExpertVectorTile - 1) / kExpertVectorTile;
    for (int64_t task = GetBlockIdx(); task < capacity * tiles; task += GetBlockNum()) {
        int64_t row = task / tiles, column = task % tiles * kExpertVectorTile;
        if (row >= used) continue;
        int64_t route = io.Scalar(aligned, row);
        if (route < 0 || route >= routes) continue;
        uint32_t n = hidden - column < kExpertVectorTile ? hidden - column : kExpertVectorTile;
        if (io.Scalar(experts, row / kExpertM) < 0) {
            // CUDA writes zero for -1 directly, even if its route weight is NaN.
            Duplicate(io.value, 0.0f, n);
        } else {
            io.Read(input, row * inputStride + column, n);
            // Multiplication precedes the only storage cast of down-projection
            // output. In particular, a tiny weight can rescue FP16 overflow.
            Muls(io.value, io.value, io.Scalar(weights, route), n);
        }
        PipeBarrier<PIPE_V>();
        io.Write(output, io.value, route * hidden + column, n);
    }
}


template<typename T>
__global__ __aicore__ void expert_reduce_kernel(GM_ADDR in, GM_ADDR out,
    int64_t tokens, int64_t hidden, int64_t topK, float scale) {
    using namespace AscendC;
    using namespace areno_npu;
    ExpertIO io;
    io.Init();
    GlobalTensor<T> input, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(in));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    int64_t tiles = (hidden + kExpertVectorTile - 1) / kExpertVectorTile;
    for (int64_t task = GetBlockIdx(); task < tokens * tiles; task += GetBlockNum()) {
        int64_t token = task / tiles, column = task % tiles * kExpertVectorTile;
        uint32_t n = hidden - column < kExpertVectorTile ? hidden - column : kExpertVectorTile;
        Duplicate(io.sum, 0.0f, n);
        PipeBarrier<PIPE_V>();
        // Preserve CUDA's top-k slot order, independent of sorted expert order.
        for (int64_t slot = 0; slot < topK; ++slot) {
            io.Read(input, (token * topK + slot) * hidden + column, n);
            Add(io.sum, io.sum, io.value, n);
            PipeBarrier<PIPE_V>();
        }
        Muls(io.sum, io.sum, scale, n);
        PipeBarrier<PIPE_V>();
        io.Write(output, io.sum, token * hidden + column, n);
    }
}

namespace areno_npu {

void launch_expert_tokens(uint32_t blocks, void* stream, const int32_t* aligned,
    const int32_t* experts, const int32_t* total, int64_t* tokens, int64_t capacity, int64_t routes, int64_t top_k) {
    expert_tokens_kernel<<<blocks, nullptr, stream>>>((uint8_t*)aligned, (uint8_t*)experts,
        (uint8_t*)total, (uint8_t*)tokens, capacity, routes, top_k);
}
void launch_expert_cast(uint32_t blocks, void* stream, uint32_t storage,
    const float* input, void* output, int64_t rows, int64_t width, int64_t input_stride) {
#define ARENO_EXPERT_CAST(T) expert_cast_kernel<T><<<blocks, nullptr, stream>>>( \
    (uint8_t*)input, (uint8_t*)output, rows, width, input_stride)
    if (storage == 1) { ARENO_EXPERT_CAST(half); }
    else { ARENO_EXPERT_CAST(bfloat16_t); }
#undef ARENO_EXPERT_CAST
}
void launch_expert_weight_scatter(uint32_t blocks, void* stream, uint32_t storage,
    const float* input, const float* weights, const int32_t* aligned, const int32_t* experts, const int32_t* total,
    void* output, int64_t capacity, int64_t routes, int64_t hidden, int64_t input_stride) {
#define ARENO_EXPERT_SCATTER(T) expert_weight_scatter_kernel<T><<<blocks, nullptr, stream>>>( \
    (uint8_t*)input, (uint8_t*)weights, (uint8_t*)aligned, (uint8_t*)experts, (uint8_t*)total, \
    (uint8_t*)output, capacity, routes, hidden, input_stride)
    if (storage == 1) { ARENO_EXPERT_SCATTER(half); }
    else { ARENO_EXPERT_SCATTER(bfloat16_t); }
#undef ARENO_EXPERT_SCATTER
}
void launch_expert_reduce(uint32_t blocks, void* stream, uint32_t storage,
    const void* input, void* output, int64_t tokens, int64_t hidden, int64_t top_k, float scale) {
#define ARENO_EXPERT_REDUCE(T) expert_reduce_kernel<T><<<blocks, nullptr, stream>>>( \
    (uint8_t*)input, (uint8_t*)output, tokens, hidden, top_k, scale)
    if (storage == 1) { ARENO_EXPERT_REDUCE(half); }
    else { ARENO_EXPERT_REDUCE(bfloat16_t); }
#undef ARENO_EXPERT_REDUCE
}
} // namespace areno_npu
