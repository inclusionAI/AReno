#include "kernel_operator.h"
#include "kernel_dtype.h"
#include "linear_launch.h"

namespace areno_npu {
using namespace AscendC;

template <typename T, bool Backward>
class LinearBiasKernel {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECCALC> scratch;
    GlobalTensor<T> input, bias, output;
    LocalTensor<float> value, accumulator;

    __aicore__ inline void Read(GlobalTensor<T>& source, LocalTensor<float> dst, int64_t offset, uint32_t n) {
        auto local = inQueue.template AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, source[offset], copy, pad);
        inQueue.EnQue(local);
        local = inQueue.template DeQue<T>();
        if constexpr (sizeof(T) == sizeof(float)) Adds(dst, local, 0.0f, n);
        else Cast(dst, local, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(local);
    }

    __aicore__ inline void Store(int64_t offset, uint32_t n) {
        auto local = outQueue.template AllocTensor<T>();
        if constexpr (sizeof(T) == sizeof(float)) Adds(local, accumulator, 0.0f, n);
        else Cast(local, accumulator, RoundMode::CAST_RINT, n);
        outQueue.EnQue(local);
        local = outQueue.template DeQue<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPad(output[offset], local, copy);
        outQueue.FreeTensor(local);
    }

public:
    __aicore__ inline LinearBiasKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR b, GM_ADDR y) {
        input.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if constexpr (!Backward) bias.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));
        pipe.InitBuffer(inQueue, 1, kLinearTile * sizeof(T));
        pipe.InitBuffer(outQueue, 1, kLinearTile * sizeof(T));
        pipe.InitBuffer(scratch, 2 * kLinearTile * sizeof(float));
        value = scratch.Get<float>();
        accumulator = value[kLinearTile];
    }

    __aicore__ inline void Process(int64_t rows, int64_t columns) {
        int64_t tiles = (columns - 1) / kLinearTile + 1;
        int64_t tasks = Backward ? tiles : rows * tiles;
        for (int64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
            int64_t col = task % tiles * kLinearTile;
            uint32_t n = columns - col < kLinearTile ? columns - col : kLinearTile;
            if constexpr (Backward) {
                // One owner per feature tile, FP32 accumulation and one final
                // cast. No storage-dtype atomic rounding between row groups.
                Duplicate(accumulator, 0.0f, n);
                PipeBarrier<PIPE_V>();
                for (int64_t row = 0; row < rows; ++row) {
                    Read(input, value, row * columns + col, n);
                    Add(accumulator, accumulator, value, n);
                    PipeBarrier<PIPE_V>();
                }
                Store(col, n);
            } else {
                int64_t offset = task / tiles * columns + col;
                Read(input, accumulator, offset, n);
                Read(bias, value, col, n);
                Add(accumulator, accumulator, value, n);
                PipeBarrier<PIPE_V>();
                Store(offset, n);
            }
        }
    }
};

} // namespace areno_npu

template<uint32_t Storage, bool Backward>
__global__ __aicore__ void linear_bias_kernel(GM_ADDR x, GM_ADDR bias, GM_ADDR y, int64_t rows, int64_t columns) {
    using T = typename areno_npu::KernelDtype<Storage>::type;
    using namespace AscendC;
    using namespace areno_npu;
    LinearBiasKernel<T, Backward> kernel;
    kernel.Init(x, bias, y);
    kernel.Process(rows, columns);
}

namespace areno_npu {

void launch_linear_bias(uint32_t blocks, void* stream, uint32_t storage, bool backward,
                        const void* input, const void* bias, void* output, int64_t rows, int64_t columns) {
#define ARENO_LINEAR_BIAS(T, B) linear_bias_kernel<KernelDtypeId<T>::value, B><<<blocks, nullptr, stream>>>( \
    (uint8_t*)input, (uint8_t*)bias, (uint8_t*)output, rows, columns)
#define ARENO_LINEAR_TYPE(T) \
    if (backward) { ARENO_LINEAR_BIAS(T, true); } else { ARENO_LINEAR_BIAS(T, false); }
    switch (storage) {
        case 0: { ARENO_LINEAR_TYPE(float); } break;
        case 1: { ARENO_LINEAR_TYPE(half); } break;
        case 2: { ARENO_LINEAR_TYPE(bfloat16_t); } break;
    }
#undef ARENO_LINEAR_TYPE
#undef ARENO_LINEAR_BIAS
}
} // namespace areno_npu
