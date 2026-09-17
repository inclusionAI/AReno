#include "kernel_operator.h"
#include "activation_launch.h"

namespace areno_npu {
using namespace AscendC;

// Every intermediate stays in FP32. DataCopyPad transfers the exact tail byte
// count, including width=1. A block owns complete, disjoint output tiles.
template <typename T, uint32_t Op>
class ActivationKernel {
    static constexpr bool backward = (Op & 1) != 0;
    static constexpr uint32_t kind = Op / 2;
    static constexpr bool gated = kind >= 3;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> xQueue, upQueue, gradQueue;
    TQue<QuePosition::VECOUT, 1> outQueue, dupQueue;
    TBuf<QuePosition::VECCALC> scratch;
    GlobalTensor<T> input, output, grad;
    LocalTensor<float> x, up, dy, y, a, b, c, dup;

    __aicore__ inline void Read(TQue<QuePosition::VECIN, 1>& queue,
                                GlobalTensor<T>& gm, int64_t offset, uint32_t count) {
        auto local = queue.template AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, padding);
        queue.EnQue(local);
    }

    __aicore__ inline void ToFloat(TQue<QuePosition::VECIN, 1>& queue,
                                   LocalTensor<float> dst, uint32_t count) {
        auto src = queue.template DeQue<T>();
        if constexpr (sizeof(T) == sizeof(float)) Adds(dst, src, 0.0f, count);
        else Cast(dst, src, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        queue.FreeTensor(src);
    }

    __aicore__ inline void Write(TQue<QuePosition::VECOUT, 1>& queue,
                                 LocalTensor<float> src, int64_t offset, uint32_t count) {
        auto dst = queue.template AllocTensor<T>();
        if constexpr (sizeof(T) == sizeof(float)) Adds(dst, src, 0.0f, count);
        else Cast(dst, src, RoundMode::CAST_RINT, count);
        queue.EnQue(dst);
        dst = queue.template DeQue<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(output[offset], dst, copy);
        queue.FreeTensor(dst);
    }

    __aicore__ inline void SigmoidInto(LocalTensor<float> dst, LocalTensor<float> src, uint32_t n) {
        Muls(dst, src, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Exp(dst, dst, n);
        PipeBarrier<PIPE_V>();
        Adds(dst, dst, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Reciprocal(dst, dst, n);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t n) {
        if constexpr (kind == 0 || kind == 3) {
            SigmoidInto(y, x, n);
            if constexpr (backward) {
                if constexpr (gated) {
                    Mul(dup, x, y, n);
                    PipeBarrier<PIPE_V>();
                }
                Muls(a, y, -1.0f, n);
                PipeBarrier<PIPE_V>();
                Adds(a, a, 1.0f, n);
                PipeBarrier<PIPE_V>();
                Mul(a, x, a, n);
                PipeBarrier<PIPE_V>();
                Adds(a, a, 1.0f, n);
                PipeBarrier<PIPE_V>();
                Mul(y, y, a, n);
            } else Mul(y, x, y, n);
        } else if constexpr (kind == 1) {
            if constexpr (backward) {
                // Backward consumes saved sigmoid output in storage dtype.
                Muls(a, x, -1.0f, n);
                PipeBarrier<PIPE_V>();
                Adds(a, a, 1.0f, n);
                PipeBarrier<PIPE_V>();
                Mul(y, x, a, n);
            } else SigmoidInto(y, x, n);
        } else if constexpr (kind == 2) {
            if constexpr (backward) SigmoidInto(y, x, n);
            else {
                // softplus(x) = max(x,0) + log1p(exp(-abs(x))). Evaluate
                // log1p(z) as 2*atanh(z/(2+z)); |z/(2+z)| <= 1/3. The
                // degree-19 series avoids overflow and FP32 cancellation
                // when exp(x) is too small to add to 1 in the negative tail.
                Abs(a, x, n);
                PipeBarrier<PIPE_V>();
                Muls(a, a, -1.0f, n);
                PipeBarrier<PIPE_V>();
                Exp(a, a, n);
                PipeBarrier<PIPE_V>();
                Adds(b, a, 2.0f, n);
                PipeBarrier<PIPE_V>();
                Div(a, a, b, n);
                PipeBarrier<PIPE_V>();
                Mul(b, a, a, n);
                Duplicate(y, 1.0f / 19.0f, n);
                PipeBarrier<PIPE_V>();
                for (int k = 8; k >= 0; --k) {
                    Mul(y, y, b, n);
                    PipeBarrier<PIPE_V>();
                    Adds(y, y, 1.0f / (2 * k + 1), n);
                    PipeBarrier<PIPE_V>();
                }
                Mul(y, y, a, n);
                PipeBarrier<PIPE_V>();
                Muls(y, y, 2.0f, n);
                Maxs(c, x, 0.0f, n);
                PipeBarrier<PIPE_V>();
                Add(y, y, c, n);
            }
        } else {
            // tanh-GELU: cdf = sigmoid(2*sqrt(2/pi)*(x + .044715*x^3)).
            Mul(a, x, x, n);
            PipeBarrier<PIPE_V>();
            Mul(b, a, x, n);
            PipeBarrier<PIPE_V>();
            Muls(b, b, 0.044715f, n);
            PipeBarrier<PIPE_V>();
            Add(b, b, x, n);
            PipeBarrier<PIPE_V>();
            Muls(b, b, 1.5957691216057308f, n);
            PipeBarrier<PIPE_V>();
            SigmoidInto(y, b, n);
            if constexpr (backward) {
                Mul(dup, x, y, n);
                Muls(a, a, 3.0f * 0.044715f, n);
                PipeBarrier<PIPE_V>();
                Adds(a, a, 1.0f, n);
                PipeBarrier<PIPE_V>();
                Muls(a, a, 1.5957691216057308f, n);
                Muls(b, y, -1.0f, n);
                PipeBarrier<PIPE_V>();
                Adds(b, b, 1.0f, n);
                Mul(a, a, x, n);
                PipeBarrier<PIPE_V>();
                Mul(a, a, b, n);
                PipeBarrier<PIPE_V>();
                Mul(a, a, y, n);
                PipeBarrier<PIPE_V>();
                Add(y, y, a, n);
            } else Mul(y, x, y, n);
        }
        PipeBarrier<PIPE_V>();
        if constexpr (gated) {
            Mul(y, y, up, n);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (backward) {
            Mul(y, y, dy, n);
            if constexpr (gated) Mul(dup, dup, dy, n);
            PipeBarrier<PIPE_V>();
        }
    }

public:
    __aicore__ inline ActivationKernel() {}

    __aicore__ inline void Init(GM_ADDR out, GM_ADDR in, GM_ADDR gradient) {
        input.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(in));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        if constexpr (backward) grad.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradient));
        pipe.InitBuffer(xQueue, 1, kActivationTile * sizeof(T));
        pipe.InitBuffer(outQueue, 1, kActivationTile * sizeof(T));
        if constexpr (gated) pipe.InitBuffer(upQueue, 1, kActivationTile * sizeof(T));
        if constexpr (backward) pipe.InitBuffer(gradQueue, 1, kActivationTile * sizeof(T));
        if constexpr (gated && backward) pipe.InitBuffer(dupQueue, 1, kActivationTile * sizeof(T));
        pipe.InitBuffer(scratch, 8 * kActivationTile * sizeof(float));
        x = scratch.Get<float>();
        up = x[kActivationTile];
        dy = x[2 * kActivationTile];
        y = x[3 * kActivationTile];
        a = x[4 * kActivationTile];
        b = x[5 * kActivationTile];
        c = x[6 * kActivationTile];
        dup = x[7 * kActivationTile];
    }

    __aicore__ inline void Process(int64_t rows, int64_t width) {
        const int64_t perRow = (width + kActivationTile - 1) / kActivationTile;
        for (int64_t tile = GetBlockIdx(); tile < rows * perRow; tile += GetBlockNum()) {
            const int64_t row = tile / perRow;
            const int64_t col = tile % perRow * kActivationTile;
            const uint32_t count = width - col < kActivationTile ? width - col : kActivationTile;
            const int64_t inOffset = row * width * (gated ? 2 : 1) + col;
            const int64_t outOffset = row * width + col;
            Read(xQueue, input, inOffset, count);
            if constexpr (gated) Read(upQueue, input, inOffset + width, count);
            if constexpr (backward) Read(gradQueue, grad, outOffset, count);
            ToFloat(xQueue, x, count);
            if constexpr (gated) ToFloat(upQueue, up, count);
            if constexpr (backward) ToFloat(gradQueue, dy, count);
            Compute(count);
            Write(outQueue, y, backward ? inOffset : outOffset, count);
            if constexpr (gated && backward) Write(dupQueue, dup, inOffset + width, count);
        }
    }
};

} // namespace areno_npu

template<typename T, uint32_t Op>
__global__ __aicore__ void activation_kernel(GM_ADDR out, GM_ADDR in, GM_ADDR grad,
                                            int64_t rows, int64_t width) {
    using namespace AscendC;
    using namespace areno_npu;
    ActivationKernel<T, Op> kernel;
    kernel.Init(out, in, grad);
    kernel.Process(rows, width);
}

// CANN discovers launcher specializations from the device object. Calls
// inside host templates alone do not reliably instantiate that object.
// Attributes are inherited from the definition. Repeating them here makes
// CANN's source scanner mistake declarations for kernel definitions.
#define ARENO_ACTIVATION_INSTANCE(T, OP) \
    template void activation_kernel<T, areno_npu::OP>( \
        GM_ADDR, GM_ADDR, GM_ADDR, int64_t, int64_t);
#define ARENO_ACTIVATION_INSTANCES(T) \
    ARENO_ACTIVATION_INSTANCE(T, Silu) ARENO_ACTIVATION_INSTANCE(T, DSilu) \
    ARENO_ACTIVATION_INSTANCE(T, Sigmoid) ARENO_ACTIVATION_INSTANCE(T, DSigmoid) \
    ARENO_ACTIVATION_INSTANCE(T, Softplus) ARENO_ACTIVATION_INSTANCE(T, DSoftplus) \
    ARENO_ACTIVATION_INSTANCE(T, SiluMul) ARENO_ACTIVATION_INSTANCE(T, DSiluMul) \
    ARENO_ACTIVATION_INSTANCE(T, GeluTanhMul) ARENO_ACTIVATION_INSTANCE(T, DGeluTanhMul)
ARENO_ACTIVATION_INSTANCES(float)
ARENO_ACTIVATION_INSTANCES(half)
ARENO_ACTIVATION_INSTANCES(bfloat16_t)
#undef ARENO_ACTIVATION_INSTANCES
#undef ARENO_ACTIVATION_INSTANCE

namespace areno_npu {

template <typename T>
void launch_typed(uint32_t blocks, void* stream, Activation op, void* output,
                  const void* input, const void* grad, int64_t rows, int64_t width) {
#define ARENO_LAUNCH(OP) case OP: \
    activation_kernel<T, OP><<<blocks, nullptr, stream>>>( \
        static_cast<uint8_t*>(output), static_cast<uint8_t*>(const_cast<void*>(input)), \
        static_cast<uint8_t*>(const_cast<void*>(grad)), rows, width); break
    switch (op) {
        ARENO_LAUNCH(Silu); ARENO_LAUNCH(DSilu);
        ARENO_LAUNCH(Sigmoid); ARENO_LAUNCH(DSigmoid);
        ARENO_LAUNCH(Softplus); ARENO_LAUNCH(DSoftplus);
        ARENO_LAUNCH(SiluMul); ARENO_LAUNCH(DSiluMul);
        ARENO_LAUNCH(GeluTanhMul); ARENO_LAUNCH(DGeluTanhMul);
    }
#undef ARENO_LAUNCH
}

void launch_activation(uint32_t blocks, void* stream, uint32_t storage, Activation op,
                       void* output, const void* input, const void* grad, int64_t rows, int64_t width) {
    switch (storage) {
        case 0: launch_typed<float>(blocks, stream, op, output, input, grad, rows, width); break;
        case 1: launch_typed<half>(blocks, stream, op, output, input, grad, rows, width); break;
        case 2: launch_typed<bfloat16_t>(blocks, stream, op, output, input, grad, rows, width); break;
    }
}
} // namespace areno_npu
