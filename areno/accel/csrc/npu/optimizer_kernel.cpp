#include "kernel_operator.h"
#include "optimizer_launch.h"

namespace areno_npu {
using namespace AscendC;

// FP32 math stays in fixed-size UB tiles. Compact master bit packing runs on
// the AI Core scalar pipeline; no expanded master tensor or CPU copy is used.
template <typename Model, typename Grad, bool Compact>
class AdamFp32Kernel {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECCALC> scratch, metadata;
    GlobalTensor<Model> model;
    GlobalTensor<Grad> gradient;
    GlobalTensor<float> moment, variance;
    GlobalTensor<uint16_t> modelBits, lowBits;
    GlobalTensor<uint8_t> carryBits;
    LocalTensor<float> p, g, m, v, a, b;
    LocalTensor<uint16_t> high, low;
    LocalTensor<uint8_t> carries;

    template <HardEvent Event>
    __aicore__ inline void Sync() {
        auto id = pipe.FetchEventID(Event);
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> Load(GlobalTensor<T>& gm, int64_t offset, uint32_t n) {
        auto local = inQueue.template AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, pad);
        inQueue.EnQue(local);
        return inQueue.template DeQue<T>();
    }

    template <typename T>
    __aicore__ inline void ReadFloat(GlobalTensor<T>& gm, LocalTensor<float> dst, int64_t offset, uint32_t n) {
        auto local = Load(gm, offset, n);
        if constexpr (sizeof(T) == sizeof(float)) Adds(dst, local, 0.0f, n);
        else Cast(dst, local, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(local);
    }

    template <typename T>
    __aicore__ inline void ReadBits(GlobalTensor<T>& gm, LocalTensor<T> dst, int64_t offset, uint32_t n) {
        auto local = Load(gm, offset, n);
        Sync<HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, local.GetValue(i));
        Sync<HardEvent::S_MTE2>();
        inQueue.FreeTensor(local);
    }

    template <typename T>
    __aicore__ inline void Store(GlobalTensor<T>& gm, LocalTensor<T> local, int64_t offset, uint32_t n) {
        outQueue.EnQue(local);
        local = outQueue.template DeQue<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPad(gm[offset], local, copy);
        outQueue.FreeTensor(local);
    }

    template <typename T>
    __aicore__ inline void WriteFloat(GlobalTensor<T>& gm, LocalTensor<float> src, int64_t offset, uint32_t n) {
        auto local = outQueue.template AllocTensor<T>();
        if constexpr (sizeof(T) == sizeof(float)) Adds(local, src, 0.0f, n);
        else Cast(local, src, RoundMode::CAST_RINT, n);
        Store(gm, local, offset, n);
    }

    template <typename T>
    __aicore__ inline void WriteBits(GlobalTensor<T>& gm, LocalTensor<T> src, int64_t offset, uint32_t n) {
        auto local = outQueue.template AllocTensor<T>();
        Sync<HardEvent::MTE3_S>();
        for (uint32_t i = 0; i < n; ++i) local.SetValue(i, src.GetValue(i));
        Sync<HardEvent::S_MTE3>();
        Store(gm, local, offset, n);
    }

    __aicore__ inline void Decode(int64_t local, int64_t state, uint32_t n) {
        ReadBits(modelBits, high, local, n);
        ReadBits(lowBits, low, state, n);
        ReadBits(carryBits, carries, state / 8, ((state % 8) + n + 7) / 8);
        Sync<HardEvent::V_S>();
        auto words = p.template ReinterpretCast<uint32_t>();
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t bit = (state % 8) + i;
            uint32_t carry = (carries.GetValue(bit / 8) >> (bit % 8)) & 1u;
            uint32_t original = (static_cast<uint32_t>(high.GetValue(i)) - carry) & 0xffffu;
            words.SetValue(i, (original << 16) | low.GetValue(i));
        }
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void Encode(int64_t local, int64_t state, uint32_t n) {
        // Use the actual BF16 cast result when deriving carries, including
        // ties and nonfinite inputs; do not substitute a truncated high half.
        auto rounded = outQueue.template AllocTensor<bfloat16_t>();
        Cast(rounded, p, RoundMode::CAST_RINT, n);
        Sync<HardEvent::V_S>();
        auto roundedBits = rounded.template ReinterpretCast<uint16_t>();
        auto words = p.template ReinterpretCast<uint32_t>();
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t word = words.GetValue(i), bit = (state % 8) + i;
            low.SetValue(i, static_cast<uint16_t>(word));
            uint8_t mask = static_cast<uint8_t>(1u << (bit % 8));
            uint8_t old = carries.GetValue(bit / 8);
            bool incremented = roundedBits.GetValue(i) != static_cast<uint16_t>(word >> 16);
            carries.SetValue(bit / 8, static_cast<uint8_t>(incremented ? old | mask : old & ~mask));
        }
        Sync<HardEvent::S_V>();
        Store(modelBits, roundedBits, local, n);
        WriteBits(lowBits, low, state, n);
        WriteBits(carryBits, carries, state / 8, ((state % 8) + n + 7) / 8);
    }

    __aicore__ inline void Update(uint32_t n, float beta1, float beta2, float lr, float decay,
                                  float eps, float step, float bias) {
        // Keep the CUDA order: (1-beta2)*g*g, with the first multiply
        // preceding g*g so representable results do not overflow early.
        Muls(m, m, beta1, n);
        Muls(a, g, 1.0f - beta1, n);
        Muls(v, v, beta2, n);
        Muls(b, g, 1.0f - beta2, n);
        if (decay != 0.0f) Muls(p, p, 1.0f - lr * decay, n);
        PipeBarrier<PIPE_V>();
        Add(m, m, a, n);
        Mul(b, b, g, n);
        PipeBarrier<PIPE_V>();
        Add(v, v, b, n);
        PipeBarrier<PIPE_V>();
        Sqrt(a, v, n);
        Duplicate(b, bias, n);
        PipeBarrier<PIPE_V>();
        Div(a, a, b, n);
        PipeBarrier<PIPE_V>();
        Adds(a, a, eps, n);
        Muls(b, m, step, n);
        PipeBarrier<PIPE_V>();
        Div(b, b, a, n);
        PipeBarrier<PIPE_V>();
        Sub(p, p, b, n);
        PipeBarrier<PIPE_V>();
    }

public:
    __aicore__ inline AdamFp32Kernel() {}

    __aicore__ inline void Init(GM_ADDR modelIn, GM_ADDR gradIn, GM_ADDR lowIn, GM_ADDR carryIn,
                                GM_ADDR momentIn, GM_ADDR varianceIn) {
        model.SetGlobalBuffer(reinterpret_cast<__gm__ Model*>(modelIn));
        gradient.SetGlobalBuffer(reinterpret_cast<__gm__ Grad*>(gradIn));
        moment.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(momentIn));
        variance.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(varianceIn));
        pipe.InitBuffer(inQueue, 1, kAdamTile * sizeof(float));
        pipe.InitBuffer(outQueue, 1, kAdamTile * sizeof(float));
        pipe.InitBuffer(scratch, 6 * kAdamTile * sizeof(float));
        p = scratch.Get<float>();
        g = p[kAdamTile]; m = p[2 * kAdamTile]; v = p[3 * kAdamTile];
        a = p[4 * kAdamTile]; b = p[5 * kAdamTile];
        if constexpr (Compact) {
            modelBits.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(modelIn));
            lowBits.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(lowIn));
            carryBits.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(carryIn));
            pipe.InitBuffer(metadata, 2 * kAdamTile * sizeof(uint16_t) + kAdamTile / 8);
            high = metadata.Get<uint16_t>();
            low = high[kAdamTile];
            carries = high[2 * kAdamTile].template ReinterpretCast<uint8_t>();
        }
    }

    __aicore__ inline void Process(int64_t numel, int64_t offset, float beta1, float beta2,
                                   float lr, float decay, float eps, float step, float bias) {
        const int64_t first = offset / kAdamTile, end = offset + numel;
        const int64_t last = (end + kAdamTile - 1) / kAdamTile;
        for (int64_t tile = first + GetBlockIdx(); tile < last; tile += GetBlockNum()) {
            int64_t start = tile * kAdamTile > offset ? tile * kAdamTile : offset;
            int64_t stop = (tile + 1) * kAdamTile < end ? (tile + 1) * kAdamTile : end;
            uint32_t n = stop - start;
            int64_t local = start - offset;
            if constexpr (Compact) Decode(local, start, n);
            else ReadFloat(model, p, local, n);
            ReadFloat(gradient, g, local, n);
            ReadFloat(moment, m, start, n);
            ReadFloat(variance, v, start, n);
            Update(n, beta1, beta2, lr, decay, eps, step, bias);
            WriteFloat(moment, m, start, n);
            WriteFloat(variance, v, start, n);
            if constexpr (Compact) Encode(local, start, n);
            else WriteFloat(model, p, local, n);
        }
    }
};

} // namespace areno_npu

template<typename Model, typename Grad, bool Compact>
__global__ __aicore__ void adam_fp32_kernel(GM_ADDR model, GM_ADDR grad, GM_ADDR low, GM_ADDR carries,
    GM_ADDR moment, GM_ADDR variance, int64_t n, int64_t offset, float b1, float b2,
    float lr, float decay, float eps, float step, float bias) {
    using namespace AscendC;
    using namespace areno_npu;
    AdamFp32Kernel<Model, Grad, Compact> kernel;
    kernel.Init(model, grad, low, carries, moment, variance);
    kernel.Process(n, offset, b1, b2, lr, decay, eps, step, bias);
}

namespace areno_npu {

void launch_adamw_fp32(uint32_t blocks, void* stream, bool model_bf16, bool grad_bf16,
                       bool compact_master, void* model, const void* grad,
                       uint16_t* low_bits, uint8_t* carries, float* moment, float* variance,
                       int64_t n, int64_t offset, float b1, float b2, float lr, float decay,
                       float eps, float step, float bias) {
#define ARENO_ADAM_LAUNCH(M, G, C) adam_fp32_kernel<M, G, C><<<blocks, nullptr, stream>>>( \
    (uint8_t*)model, (uint8_t*)grad, (uint8_t*)low_bits, (uint8_t*)carries, \
    (uint8_t*)moment, (uint8_t*)variance, n, offset, b1, b2, lr, decay, eps, step, bias)
    if (compact_master) {
        if (grad_bf16) { ARENO_ADAM_LAUNCH(bfloat16_t, bfloat16_t, true); }
        else { ARENO_ADAM_LAUNCH(bfloat16_t, float, true); }
    } else if (model_bf16) {
        if (grad_bf16) { ARENO_ADAM_LAUNCH(bfloat16_t, bfloat16_t, false); }
        else { ARENO_ADAM_LAUNCH(bfloat16_t, float, false); }
    } else {
        if (grad_bf16) { ARENO_ADAM_LAUNCH(float, bfloat16_t, false); }
        else { ARENO_ADAM_LAUNCH(float, float, false); }
    }
#undef ARENO_ADAM_LAUNCH
}
} // namespace areno_npu
