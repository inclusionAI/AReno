#include "kernel_operator.h"
#include "normalization_launch.h"

namespace areno_npu {
using namespace AscendC;
constexpr uint32_t kNormTile = 1024;

// A block reduces and emits complete rows. Large feature widths stream through
// fixed-size UB tiles; no sequence-sized scratch or host reduction is required.
template <typename T, typename W, bool Backward, bool Scale, uint32_t Gate>
class NormalizationKernel {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> xQueue, gateQueue, weightQueue, gradQueue, invInQueue;
    TQue<QuePosition::VECOUT, 1> outQueue, gateOutQueue, weightOutQueue, invOutQueue;
    TBuf<QuePosition::VECCALC> scratch;
    GlobalTensor<T> input, gate, grad, output, gradGate;
    GlobalTensor<W> weight;
    GlobalTensor<float> invGlobal, gradWeight;
    LocalTensor<float> x, g, dy, w, sigmoid, silu, weighted, y, a, b, c, reduced, work;

    template <typename Storage>
    __aicore__ inline void Read(TQue<QuePosition::VECIN, 1>& queue, GlobalTensor<Storage>& gm,
                                LocalTensor<float> dst, int64_t offset, uint32_t n) {
        auto local = queue.template AllocTensor<Storage>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(Storage)), 0, 0, 0};
        DataCopyPadExtParams<Storage> padding{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, padding);
        queue.EnQue(local);
        local = queue.template DeQue<Storage>();
        if constexpr (sizeof(Storage) == sizeof(float)) Adds(dst, local, 0.0f, n);
        else Cast(dst, local, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        queue.FreeTensor(local);
    }

    template <typename Storage, bool Atomic = false>
    __aicore__ inline void Write(TQue<QuePosition::VECOUT, 1>& queue, GlobalTensor<Storage>& gm,
                                 LocalTensor<float> src, int64_t offset, uint32_t n) {
        auto local = queue.template AllocTensor<Storage>();
        if constexpr (sizeof(Storage) == sizeof(float)) Adds(local, src, 0.0f, n);
        else Cast(local, src, RoundMode::CAST_RINT, n);
        queue.EnQue(local);
        local = queue.template DeQue<Storage>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(Storage)), 0, 0, 0};
        if constexpr (Atomic) SetAtomicAdd<float>();
        DataCopyPad(gm[offset], local, copy);
        if constexpr (Atomic) DisableDmaAtomic();
        queue.FreeTensor(local);
    }

    __aicore__ inline float Scalar(LocalTensor<float> tensor) {
        auto readEvent = pipe.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(readEvent);
        WaitFlag<HardEvent::V_S>(readEvent);
        float result = tensor.GetValue(0);
        // Protect the scalar read before this UB slot is reused by vector ops.
        auto reuseEvent = pipe.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(reuseEvent);
        WaitFlag<HardEvent::S_V>(reuseEvent);
        return result;
    }

    __aicore__ inline float Sum(LocalTensor<float> src, uint32_t n) {
        ReduceSum(reduced, src, work, n);
        return Scalar(reduced);
    }

    __aicore__ inline void Load(int64_t row, int64_t col, int64_t width, int64_t groups, uint32_t n) {
        Read(xQueue, input, x, row * width + col, n);
        if constexpr (Scale) Read(weightQueue, weight, w, (row % groups) * width + col, n);
        if constexpr (Backward) Read(gradQueue, grad, dy, row * width + col, n);
        if constexpr (Gate != 0) {
            Read(gateQueue, gate, g, row * width + col, n);
            Muls(sigmoid, g, -1.0f, n);
            PipeBarrier<PIPE_V>();
            Exp(sigmoid, sigmoid, n);
            PipeBarrier<PIPE_V>();
            Adds(sigmoid, sigmoid, 1.0f, n);
            PipeBarrier<PIPE_V>();
            Reciprocal(sigmoid, sigmoid, n);
            PipeBarrier<PIPE_V>();
            if constexpr (Gate == 1) Mul(silu, g, sigmoid, n);
            else Adds(silu, sigmoid, 0.0f, n);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (Backward) {
            if constexpr (Scale) Mul(weighted, dy, w, n);
            else Adds(weighted, dy, 0.0f, n);
            PipeBarrier<PIPE_V>();
            if constexpr (Gate != 0) {
                Mul(weighted, weighted, silu, n);
                PipeBarrier<PIPE_V>();
            }
        }
    }

public:
    __aicore__ inline NormalizationKernel() {}

    __aicore__ inline void Init(GM_ADDR in, GM_ADDR gateIn, GM_ADDR weightIn, GM_ADDR gradIn,
                                GM_ADDR out, GM_ADDR gateOut, GM_ADDR inv, GM_ADDR weightOut) {
        input.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(in));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        invGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inv));
        if constexpr (Scale) weight.SetGlobalBuffer(reinterpret_cast<__gm__ W*>(weightIn));
        if constexpr (Gate != 0) gate.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gateIn));
        if constexpr (Backward) {
            grad.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradIn));
            if constexpr (Scale) gradWeight.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightOut));
            if constexpr (Gate != 0) gradGate.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gateOut));
        }
        pipe.InitBuffer(xQueue, 1, kNormTile * sizeof(T));
        pipe.InitBuffer(outQueue, 1, kNormTile * sizeof(T));
        if constexpr (Scale) pipe.InitBuffer(weightQueue, 1, kNormTile * sizeof(W));
        if constexpr (Gate != 0) pipe.InitBuffer(gateQueue, 1, kNormTile * sizeof(T));
        if constexpr (Backward) {
            pipe.InitBuffer(gradQueue, 1, kNormTile * sizeof(T));
            pipe.InitBuffer(invInQueue, 1, 32);
            if constexpr (Scale) pipe.InitBuffer(weightOutQueue, 1, kNormTile * sizeof(float));
            if constexpr (Gate != 0) pipe.InitBuffer(gateOutQueue, 1, kNormTile * sizeof(T));
        } else pipe.InitBuffer(invOutQueue, 1, 32);
        pipe.InitBuffer(scratch, 13 * kNormTile * sizeof(float));
        x = scratch.Get<float>();
        g = x[kNormTile]; dy = x[2 * kNormTile]; w = x[3 * kNormTile];
        sigmoid = x[4 * kNormTile]; silu = x[5 * kNormTile]; weighted = x[6 * kNormTile];
        y = x[7 * kNormTile]; a = x[8 * kNormTile]; b = x[9 * kNormTile]; c = x[10 * kNormTile];
        reduced = x[11 * kNormTile]; work = x[12 * kNormTile];
    }

    __aicore__ inline void Process(int64_t rows, int64_t width, int64_t groups, float eps) {
        for (int64_t row = GetBlockIdx(); row < rows; row += GetBlockNum()) {
            float aggregate = 0.0f;
            for (int64_t col = 0; col < width; col += kNormTile) {
                uint32_t n = width - col < kNormTile ? width - col : kNormTile;
                if constexpr (Backward) {
                    Load(row, col, width, groups, n);
                    Mul(a, x, weighted, n);
                } else {
                    Read(xQueue, input, x, row * width + col, n);
                    Mul(a, x, x, n);
                }
                PipeBarrier<PIPE_V>();
                aggregate += Sum(a, n);
            }
            float inv;
            if constexpr (Backward) {
                Read(invInQueue, invGlobal, reduced, row, 1);
                inv = Scalar(reduced);
            } else {
                Duplicate(reduced, aggregate / static_cast<float>(width) + eps, 1);
                PipeBarrier<PIPE_V>();
                Rsqrt(reduced, reduced, 1);
                inv = Scalar(reduced);
                Write(invOutQueue, invGlobal, reduced, row, 1);
            }
            const float correction = aggregate * inv * inv / static_cast<float>(width);
            for (int64_t col = 0; col < width; col += kNormTile) {
                uint32_t n = width - col < kNormTile ? width - col : kNormTile;
                Load(row, col, width, groups, n);
                if constexpr (!Backward) {
                    Muls(y, x, inv, n);
                    PipeBarrier<PIPE_V>();
                    if constexpr (Scale) {
                        Mul(y, y, w, n);
                        PipeBarrier<PIPE_V>();
                    }
                    if constexpr (Gate != 0) {
                        Mul(y, y, silu, n);
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    Muls(a, x, correction, n);
                    PipeBarrier<PIPE_V>();
                    Sub(y, weighted, a, n);
                    PipeBarrier<PIPE_V>();
                    Muls(y, y, inv, n);
                    PipeBarrier<PIPE_V>();
                    // Shared base for dw and dg: dy * normalized_x.
                    Mul(b, dy, x, n);
                    PipeBarrier<PIPE_V>();
                    Muls(b, b, inv, n);
                    PipeBarrier<PIPE_V>();
                    if constexpr (Scale) {
                        if constexpr (Gate != 0) Mul(c, b, silu, n);
                        else Adds(c, b, 0.0f, n);
                        PipeBarrier<PIPE_V>();
                        Write<float, true>(weightOutQueue, gradWeight, c, (row % groups) * width + col, n);
                    }
                    if constexpr (Gate == 1) {
                        Muls(a, sigmoid, -1.0f, n);
                        PipeBarrier<PIPE_V>();
                        Adds(a, a, 1.0f, n);
                        PipeBarrier<PIPE_V>();
                        Mul(a, a, g, n);
                        PipeBarrier<PIPE_V>();
                        Adds(a, a, 1.0f, n);
                        PipeBarrier<PIPE_V>();
                        Mul(a, a, sigmoid, n);
                        PipeBarrier<PIPE_V>();
                        Mul(a, a, b, n);
                        PipeBarrier<PIPE_V>();
                        Mul(a, a, w, n);
                        PipeBarrier<PIPE_V>();
                        Write(gateOutQueue, gradGate, a, row * width + col, n);
                    }
                }
                Write(outQueue, output, y, row * width + col, n);
            }
        }
    }
};

} // namespace areno_npu

template <typename T, typename W, bool Backward, bool Scale, uint32_t Gate>
__global__ __aicore__ void normalization_kernel(GM_ADDR input, GM_ADDR gate, GM_ADDR weight, GM_ADDR grad,
                                               GM_ADDR output, GM_ADDR gradGate, GM_ADDR inv, GM_ADDR gradWeight,
                                               int64_t rows, int64_t width, int64_t groups, float eps) {
    using namespace AscendC;
    using namespace areno_npu;
    NormalizationKernel<T, W, Backward, Scale, Gate> kernel;
    kernel.Init(input, gate, weight, grad, output, gradGate, inv, gradWeight);
    kernel.Process(rows, width, groups, eps);
}

namespace areno_npu {

template <typename T>
void launch_norm_typed(uint32_t blocks, void* stream, uint32_t weightStorage,
                       bool backward, bool scale, uint32_t gateKind,
                       const void* input, const void* gate, const void* weight, const void* grad,
                       void* output, void* gradGate, float* inv, float* gradWeight,
                       int64_t rows, int64_t width, int64_t groups, float eps) {
#define ARENO_NORM_LAUNCH(W, BACKWARD, SCALE, GATE) \
    normalization_kernel<T, W, BACKWARD, SCALE, GATE><<<blocks, nullptr, stream>>>( \
        (uint8_t*)input, (uint8_t*)gate, (uint8_t*)weight, (uint8_t*)grad, \
        (uint8_t*)output, (uint8_t*)gradGate, (uint8_t*)inv, (uint8_t*)gradWeight, rows, width, groups, eps)
    if (gateKind == 2) {
        switch (weightStorage) {
            case 0: ARENO_NORM_LAUNCH(float, false, true, 2); break;
            case 1: ARENO_NORM_LAUNCH(half, false, true, 2); break;
            case 2: ARENO_NORM_LAUNCH(bfloat16_t, false, true, 2); break;
        }
    } else if (backward) {
        if (gateKind == 1) { ARENO_NORM_LAUNCH(float, true, true, 1); }
        else if (scale) { ARENO_NORM_LAUNCH(float, true, true, 0); }
        else { ARENO_NORM_LAUNCH(float, true, false, 0); }
    } else {
        if (gateKind == 1) { ARENO_NORM_LAUNCH(float, false, true, 1); }
        else if (scale) { ARENO_NORM_LAUNCH(float, false, true, 0); }
        else { ARENO_NORM_LAUNCH(float, false, false, 0); }
    }
#undef ARENO_NORM_LAUNCH
}

void launch_normalization(uint32_t blocks, void* stream, uint32_t storage, uint32_t weightStorage,
                          bool backward, bool scale, uint32_t gateKind,
                          const void* input, const void* gate, const void* weight, const void* grad,
                          void* output, void* gradGate, float* inv, float* gradWeight,
                          int64_t rows, int64_t width, int64_t groups, float eps) {
#define ARENO_NORM_TYPE(T) launch_norm_typed<T>(blocks, stream, weightStorage, backward, scale, gateKind, \
    input, gate, weight, grad, output, gradGate, inv, gradWeight, rows, width, groups, eps)
    switch (storage) {
        case 0: ARENO_NORM_TYPE(float); break;
        case 1: ARENO_NORM_TYPE(half); break;
        case 2: ARENO_NORM_TYPE(bfloat16_t); break;
    }
#undef ARENO_NORM_TYPE
}
} // namespace areno_npu
