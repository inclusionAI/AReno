#include "kernel_operator.h"
#include "conv_launch.h"

namespace areno_npu {
using namespace AscendC;

// Each task owns a channel tile. Accumulation and saved preactivations are
// FP32, as in conv.cu. Weight gradients have one owner per (tap, channel tile)
// and need neither storage-dtype atomics nor a full-size intermediate tensor.
template <typename T, ConvOp Op, bool Packed>
class ConvKernel {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue, segmentQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECCALC> scratch, accumulatorBuffer, packedScratch, indexBuffer, expandedIndexBuffer;
    GlobalTensor<T> input, grad, history, output;
    GlobalTensor<float> weight, preact, weightGrad;
    GlobalTensor<int32_t> cu;
    LocalTensor<float> value, coefficient, accumulator, z, sigmoid, temporary;
    LocalTensor<uint32_t> indices, expandedIndices;

    template <HardEvent Event>
    __aicore__ inline void Sync() {
        auto id = pipe.FetchEventID(Event);
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }

    template <typename S>
    __aicore__ inline void Read(GlobalTensor<S>& gm, LocalTensor<float> dst, int64_t offset,
                                uint32_t n, int64_t stride = 1) {
        auto local = inQueue.template AllocTensor<S>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(S)), 0, 0, 0};
        if (stride != 1) {
            copy.blockCount = static_cast<uint16_t>(n);
            copy.blockLen = sizeof(S);
            copy.srcStride = static_cast<uint32_t>((stride - 1) * sizeof(S));
        }
        DataCopyPadExtParams<S> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, pad);
        inQueue.EnQue(local);
        local = inQueue.template DeQue<S>();
        if (stride != 1) {
            // DMA puts every scalar in a separate 32-byte UB block. Gather
            // repacks it before vector math, including BF16 history values.
            auto dense = packedScratch.Get<S>();
            Gather(dense, local, indices, uint32_t{0}, n);
            PipeBarrier<PIPE_V>();
            if constexpr (sizeof(S) == sizeof(float)) Adds(dst, dense, 0.0f, n);
            else Cast(dst, dense, RoundMode::CAST_NONE, n);
        } else {
            if constexpr (sizeof(S) == sizeof(float)) Adds(dst, local, 0.0f, n);
            else Cast(dst, local, RoundMode::CAST_NONE, n);
        }
        PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(local);
    }

    template <typename S>
    __aicore__ inline void Write(GlobalTensor<S>& gm, LocalTensor<float> src, int64_t offset, uint32_t n) {
        auto local = outQueue.template AllocTensor<S>();
        if constexpr (sizeof(S) == sizeof(float)) Adds(local, src, 0.0f, n);
        else Cast(local, src, RoundMode::CAST_RINT, n);
        outQueue.EnQue(local);
        local = outQueue.template DeQue<S>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(S)), 0, 0, 0};
        DataCopyPad(gm[offset], local, copy);
        outQueue.FreeTensor(local);
    }

    __aicore__ inline void WriteWeight(int64_t offset, uint32_t n, int64_t kernel) {
        if (kernel == 1) { Write(weightGrad, accumulator, offset, n); return; }
        auto local = outQueue.template AllocTensor<float>();
        // Expand one channel value to each UB block. Only its first float is
        // transferred to GM, with a kernel-sized channel stride.
        Gather(local, accumulator, expandedIndices, uint32_t{0}, n * 8);
        outQueue.EnQue(local);
        local = outQueue.template DeQue<float>();
        DataCopyExtParams copy{static_cast<uint16_t>(n), sizeof(float), 0,
                              static_cast<uint32_t>((kernel - 1) * sizeof(float)), 0};
        DataCopyPad(weightGrad[offset], local, copy);
        outQueue.FreeTensor(local);
    }

    __aicore__ inline int64_t Boundary(int64_t position) {
        auto local = segmentQueue.template AllocTensor<int32_t>();
        DataCopyExtParams copy{1, sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(local, cu[position], copy, pad);
        segmentQueue.EnQue(local);
        local = segmentQueue.template DeQue<int32_t>();
        Sync<HardEvent::MTE2_S>();
        int64_t result = local.GetValue(0);
        Sync<HardEvent::S_MTE2>();
        segmentQueue.FreeTensor(local);
        return result;
    }

    __aicore__ inline void Bounds(int64_t row, int64_t seqlen, int64_t segments,
                                 int64_t& begin, int64_t& end) {
        if constexpr (Packed) {
            // upper_bound handles repeated offsets (empty packed sequences).
            int64_t lo = 0, hi = segments;
            while (lo + 1 < hi) {
                int64_t mid = lo + (hi - lo) / 2;
                if (Boundary(mid) <= row) lo = mid;
                else hi = mid;
            }
            begin = Boundary(lo);
            end = Boundary(lo + 1);
        } else {
            begin = row / seqlen * seqlen;
            end = begin + seqlen;
        }
    }

    __aicore__ inline void Sigmoid(uint32_t n) {
        Muls(sigmoid, z, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Exp(sigmoid, sigmoid, n);
        PipeBarrier<PIPE_V>();
        Adds(sigmoid, sigmoid, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Reciprocal(sigmoid, sigmoid, n);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Gradient(int64_t offset, uint32_t n) {
        Read(preact, z, offset, n);
        Read(grad, value, offset, n);
        Sigmoid(n);
        Muls(temporary, sigmoid, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Adds(temporary, temporary, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Mul(temporary, z, temporary, n);
        PipeBarrier<PIPE_V>();
        Adds(temporary, temporary, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Mul(temporary, sigmoid, temporary, n);
        PipeBarrier<PIPE_V>();
        Mul(value, value, temporary, n);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Accumulate(uint32_t n) {
        Mul(value, value, coefficient, n);
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, value, n);
        PipeBarrier<PIPE_V>();
    }

public:
    __aicore__ inline ConvKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR g, GM_ADDR p, GM_ADDR out, GM_ADDR h, GM_ADDR offsets) {
        input.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        weight.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
        preact.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p));
        if constexpr (Op == ConvWeightGrad) weightGrad.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        else output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        if constexpr (Op == ConvInputGrad || Op == ConvWeightGrad) grad.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(g));
        if constexpr (Op == ConvDecode) history.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h));
        if constexpr (Packed) {
            cu.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets));
            pipe.InitBuffer(segmentQueue, 1, 32);
        }
        pipe.InitBuffer(inQueue, 1, kConvTile * 32);
        pipe.InitBuffer(outQueue, 1, kConvTile * 32);
        pipe.InitBuffer(packedScratch, kConvTile * sizeof(float));
        pipe.InitBuffer(scratch, 5 * kConvTile * sizeof(float));
        // Gather's expanded output count must also fit its source descriptor;
        // only the first kConvTile accumulator values are ever indexed.
        pipe.InitBuffer(accumulatorBuffer, (Op == ConvWeightGrad ? 8 : 1) * kConvTile * sizeof(float));
        pipe.InitBuffer(indexBuffer, kConvTile * sizeof(uint32_t));
        indices = indexBuffer.Get<uint32_t>();
        for (uint32_t i = 0; i < kConvTile; ++i) indices.SetValue(i, i * 32);
        if constexpr (Op == ConvWeightGrad) {
            pipe.InitBuffer(expandedIndexBuffer, kConvTile * 8 * sizeof(uint32_t));
            expandedIndices = expandedIndexBuffer.Get<uint32_t>();
            for (uint32_t i = 0; i < kConvTile * 8; ++i) expandedIndices.SetValue(i, i / 8 * sizeof(float));
        }
        Sync<HardEvent::S_V>();
        value = scratch.Get<float>();
        coefficient = value[kConvTile];
        accumulator = accumulatorBuffer.Get<float>();
        z = value[2 * kConvTile];
        sigmoid = value[3 * kConvTile];
        temporary = value[4 * kConvTile];
    }

    __aicore__ inline void Process(int64_t batch, int64_t seqlen, int64_t channels, int64_t kernel, int64_t segments) {
        int64_t tiles = (channels - 1) / kConvTile + 1;
        int64_t tasks = Op == ConvWeightGrad ? kernel * tiles : batch * seqlen * tiles;
        for (int64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
            int64_t col = task % tiles * kConvTile, row = task / tiles;
            uint32_t n = channels - col < kConvTile ? channels - col : kConvTile;
            Duplicate(accumulator, 0.0f, n);
            PipeBarrier<PIPE_V>();
            if constexpr (Op == ConvWeightGrad) {
                int64_t tap = row;
                for (int64_t seq = 0; seq < segments; ++seq) {
                    int64_t begin, end;
                    if constexpr (Packed) { begin = Boundary(seq); end = Boundary(seq + 1); }
                    else { begin = seq * seqlen; end = begin + seqlen; }
                    for (int64_t token = begin + kernel - 1 - tap; token < end; ++token) {
                        Gradient(token * channels + col, n);
                        Read(input, coefficient, (token + tap - (kernel - 1)) * channels + col, n);
                        Accumulate(n);
                    }
                }
                WriteWeight(col * kernel + tap, n, kernel);
            } else if constexpr (Op == ConvDecode) {
                for (int64_t tap = 0; tap < kernel; ++tap) {
                    Read(weight, coefficient, col * kernel + tap, n, kernel);
                    if (tap == kernel - 1) Read(input, value, row * channels + col, n);
                    else Read(history, value, (row * channels + col) * (kernel - 1) + tap, n, kernel - 1);
                    Accumulate(n);
                }
                Write(preact, accumulator, row * channels + col, n);
                Adds(z, accumulator, 0.0f, n);
                PipeBarrier<PIPE_V>();
                Sigmoid(n);
                Mul(value, accumulator, sigmoid, n);
                PipeBarrier<PIPE_V>();
                Write(output, value, row * channels + col, n);
            } else {
                int64_t begin, end;
                Bounds(row, seqlen, segments, begin, end);
                if constexpr (Op == ConvForward) {
                    int64_t first = kernel - 1 - (row - begin);
                    if (first < 0) first = 0;
                    for (int64_t tap = first; tap < kernel; ++tap) {
                        Read(input, value, (row + tap - (kernel - 1)) * channels + col, n);
                        Read(weight, coefficient, col * kernel + tap, n, kernel);
                        Accumulate(n);
                    }
                    Write(preact, accumulator, row * channels + col, n);
                    Adds(z, accumulator, 0.0f, n);
                    PipeBarrier<PIPE_V>();
                    Sigmoid(n);
                    Mul(value, accumulator, sigmoid, n);
                    PipeBarrier<PIPE_V>();
                    Write(output, value, row * channels + col, n);
                } else {
                    int64_t stop = end - row < kernel ? end : row + kernel;
                    for (int64_t token = row; token < stop; ++token) {
                        Gradient(token * channels + col, n);
                        Read(weight, coefficient, col * kernel + kernel - 1 - (token - row), n, kernel);
                        Accumulate(n);
                    }
                    Write(output, accumulator, row * channels + col, n);
                }
            }
        }
    }
};

template <typename T, ConvOp Op, bool Packed>
__global__ __aicore__ void conv_kernel(GM_ADDR x, GM_ADDR w, GM_ADDR g, GM_ADDR p, GM_ADDR out,
    GM_ADDR h, GM_ADDR cu, int64_t batch, int64_t seqlen, int64_t channels, int64_t kernel_size, int64_t segments) {
    ConvKernel<T, Op, Packed> kernel;
    kernel.Init(x, w, g, p, out, h, cu);
    kernel.Process(batch, seqlen, channels, kernel_size, segments);
}

template <typename T, bool Packed>
void launch_conv_typed(uint32_t blocks, void* stream, ConvOp op, const void* input, const float* weight,
    const void* grad, float* preact, void* output, const void* history, const int32_t* cu,
    int64_t batch, int64_t seqlen, int64_t channels, int64_t kernel_size, int64_t segments) {
#define ARENO_CONV(OP) case OP: conv_kernel<T, OP, Packed><<<blocks, nullptr, stream>>>( \
    (uint8_t*)input, (uint8_t*)weight, (uint8_t*)grad, (uint8_t*)preact, (uint8_t*)output, (uint8_t*)history, \
    (uint8_t*)cu, batch, seqlen, channels, kernel_size, segments); break
    switch (op) {
        ARENO_CONV(ConvForward); ARENO_CONV(ConvInputGrad); ARENO_CONV(ConvWeightGrad); ARENO_CONV(ConvDecode);
    }
#undef ARENO_CONV
}

void launch_conv(uint32_t blocks, void* stream, uint32_t storage, ConvOp op, bool packed,
    const void* input, const float* weight, const void* grad, float* preact, void* output,
    const void* history, const int32_t* cu, int64_t batch, int64_t seqlen,
    int64_t channels, int64_t kernel_size, int64_t segments) {
#define ARENO_CONV_TYPE(T, P) launch_conv_typed<T, P>(blocks, stream, op, input, weight, grad, preact, output, \
    history, cu, batch, seqlen, channels, kernel_size, segments)
#define ARENO_CONV_PACKED(T) if (packed) { ARENO_CONV_TYPE(T, true); } else { ARENO_CONV_TYPE(T, false); }
    switch (storage) {
        case 0: { ARENO_CONV_PACKED(float); } break;
        case 1: { ARENO_CONV_PACKED(half); } break;
        case 2: { ARENO_CONV_PACKED(bfloat16_t); } break;
    }
#undef ARENO_CONV_PACKED
#undef ARENO_CONV_TYPE
}
} // namespace areno_npu
