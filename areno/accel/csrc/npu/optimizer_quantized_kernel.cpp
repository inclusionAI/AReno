#include "kernel_operator.h"
#include "kernel_dtype.h"
#include "optimizer_launch.h"

namespace areno_npu {
using namespace AscendC;

// A core owns complete quantization blocks. A block's new FP32 values stay
// in UB until all lanes pass the finite check, then codes/scales/model commit
// together. UB usage is bounded (~142 KiB at the largest 4096-element block).
template <typename Model, typename Grad, bool FourBit, bool Factored = false>
class AdamQuantizedKernel {
    static constexpr uint32_t Tile = FourBit ? 1024 : 4096;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECCALC> scratch, metadata;
    GlobalTensor<Model> model;
    GlobalTensor<Grad> gradient;
    GlobalTensor<uint8_t> moment, variance;
    GlobalTensor<float> momentScale, varianceScale, signedMap, unsignedMap;
    GlobalTensor<float> factors, rowMean;
    GlobalTensor<int32_t> invalidFlag;
    LocalTensor<float> p, g, m, v, a, b, signedTable, unsignedTable, scales;
    LocalTensor<uint8_t> mCodes, vCodes;
    float meanDenominator;
    bool invalidParameter;

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
    __aicore__ inline void ReadScalar(GlobalTensor<T>& gm, LocalTensor<T> dst, int64_t offset, uint32_t n) {
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
    __aicore__ inline void WriteScalar(GlobalTensor<T>& gm, LocalTensor<T> src, int64_t offset, uint32_t n) {
        auto local = outQueue.template AllocTensor<T>();
        Sync<HardEvent::MTE3_S>();
        for (uint32_t i = 0; i < n; ++i) local.SetValue(i, src.GetValue(i));
        Sync<HardEvent::S_MTE3>();
        Store(gm, local, offset, n);
    }

    __aicore__ inline void Update(uint32_t n, float beta1, float beta2, float lr, float decay,
                                  float eps, float step, float bias) {
        Muls(m, m, beta1, n);
        Muls(a, g, 1.0f - beta1, n);
        if constexpr (!Factored) {
            Muls(v, v, beta2, n);
            Muls(b, g, 1.0f - beta2, n);
        }
        if (decay != 0.0f) Muls(p, p, 1.0f - lr * decay, n);
        PipeBarrier<PIPE_V>();
        Add(m, m, a, n);
        if constexpr (!Factored) {
            Mul(b, b, g, n);
            PipeBarrier<PIPE_V>();
            Add(v, v, b, n);
        }
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

    __aicore__ inline uint8_t Nearest(float value, LocalTensor<float> table) {
        // CUDA 8-bit uses lower_bound followed by a distance comparison;
        // signed 4-bit's exhaustive nearest search has the same tie rule.
        int lo = 0, hi = FourBit ? 15 : 255;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (table.GetValue(mid) < value) lo = mid + 1;
            else hi = mid;
        }
        if (lo == 0) return 0;
        float left = value - table.GetValue(lo - 1), right = table.GetValue(lo) - value;
        if (left < 0.0f) left = -left;
        if (right < 0.0f) right = -right;
        return static_cast<uint8_t>(left <= right ? lo - 1 : lo);
    }

    __aicore__ inline bool Quantize(uint32_t n) {
        Sync<HardEvent::V_S>();
        auto gp = g.template ReinterpretCast<uint32_t>(), pp = p.template ReinterpretCast<uint32_t>();
        auto mp = m.template ReinterpretCast<uint32_t>(), vp = v.template ReinterpretCast<uint32_t>();
        float mMax = 0.0f, vMax = 0.0f;
        bool invalid = false;
        for (uint32_t i = 0; i < n; ++i) {
            invalid |= (gp.GetValue(i) & 0x7f800000u) == 0x7f800000u;
            invalid |= (pp.GetValue(i) & 0x7f800000u) == 0x7f800000u;
            invalid |= (mp.GetValue(i) & 0x7f800000u) == 0x7f800000u;
            invalid |= (vp.GetValue(i) & 0x7f800000u) == 0x7f800000u;
            float absolute = m.GetValue(i);
            if (absolute < 0.0f) absolute = -absolute;
            if (absolute > mMax) mMax = absolute;
            if constexpr (!Factored) {
                if (v.GetValue(i) > vMax) vMax = v.GetValue(i);
            }
        }
        Sync<HardEvent::S_V>();
        if (invalid) return false;
        scales.SetValue(0, mMax);
        Duplicate(a, mMax > 1e-30f ? mMax : 1e-30f, n);
        if constexpr (!Factored) {
            scales.SetValue(8, vMax);
            Duplicate(b, vMax > 1e-30f ? vMax : 1e-30f, n);
        }
        PipeBarrier<PIPE_V>();
        Div(m, m, a, n);
        if constexpr (!Factored) Div(v, v, b, n);
        PipeBarrier<PIPE_V>();
        if constexpr (FourBit && !Factored) {
            Muls(v, v, 16.0f, n);
            PipeBarrier<PIPE_V>();
            Adds(v, v, -1.0f, n);
            PipeBarrier<PIPE_V>();
            Cast(a.template ReinterpretCast<int32_t>(), v, RoundMode::CAST_RINT, n);
        }
        Sync<HardEvent::V_S>();
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t mc = Nearest(m.GetValue(i), signedTable), vc = 0;
            if constexpr (FourBit) {
                if constexpr (!Factored) {
                    int32_t code = a.template ReinterpretCast<int32_t>().GetValue(i);
                    vc = static_cast<uint8_t>(code < 0 ? 0 : code > 15 ? 15 : code);
                }
                if ((i & 1u) == 0) {
                    // The unused last high nibble is canonical zero state.
                    mCodes.SetValue(i / 2, static_cast<uint8_t>(mc | 0x70u));
                    if constexpr (!Factored) vCodes.SetValue(i / 2, vc);
                } else {
                    mCodes.SetValue(i / 2, static_cast<uint8_t>((mCodes.GetValue(i / 2) & 15u) | (mc << 4)));
                    if constexpr (!Factored)
                        vCodes.SetValue(i / 2, static_cast<uint8_t>(vCodes.GetValue(i / 2) | (vc << 4)));
                }
            } else {
                vc = Nearest(v.GetValue(i), unsignedTable);
                mCodes.SetValue(i, mc);
                vCodes.SetValue(i, vc);
            }
        }
        Sync<HardEvent::S_V>();
        return true;
    }

    __aicore__ inline void LoadFactors(int64_t start, uint32_t n, int64_t rows, int64_t columns) {
        // Load contiguous column-factor runs into aligned UB scratch. A
        // quantization block may start inside a row or span several rows.
        Sync<HardEvent::V_S>();
        uint32_t done = 0;
        while (done < n) {
            int64_t position = start + done;
            int64_t col = position % columns;
            uint32_t count = columns - col < n - done ? columns - col : n - done;
            ReadScalar(factors, a, rows + col, count);
            ReadScalar(factors, scales[8], position / columns, 1);
            float row = scales.GetValue(8);
            for (uint32_t i = 0; i < count; ++i) v.SetValue(done + i, row * a.GetValue(i) / meanDenominator);
            done += count;
        }
        Sync<HardEvent::S_V>();
    }

public:
    __aicore__ inline AdamQuantizedKernel() {}

    __aicore__ inline void Init(GM_ADDR modelIn, GM_ADDR gradIn, GM_ADDR mIn, GM_ADDR msIn,
        GM_ADDR vIn, GM_ADDR vsIn, GM_ADDR signedIn, GM_ADDR unsignedIn) {
        model.SetGlobalBuffer(reinterpret_cast<__gm__ Model*>(modelIn));
        gradient.SetGlobalBuffer(reinterpret_cast<__gm__ Grad*>(gradIn));
        moment.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mIn));
        momentScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(msIn));
        if constexpr (!Factored) {
            variance.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(vIn));
            varianceScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vsIn));
        }
        pipe.InitBuffer(inQueue, 1, Tile * sizeof(float));
        pipe.InitBuffer(outQueue, 1, Tile * sizeof(float));
        pipe.InitBuffer(scratch, 6 * Tile * sizeof(float));
        pipe.InitBuffer(metadata, 2 * Tile + (512 + 16) * sizeof(float));
        p = scratch.Get<float>();
        g = p[Tile]; m = p[2 * Tile]; v = p[3 * Tile]; a = p[4 * Tile]; b = p[5 * Tile];
        mCodes = metadata.Get<uint8_t>();
        vCodes = mCodes[Tile];
        signedTable = mCodes[2 * Tile].template ReinterpretCast<float>();
        unsignedTable = signedTable[256];
        scales = signedTable[512];
        if constexpr (FourBit) {
            const float values[16] = {-0.8875f, -0.6625f, -0.4375f, -0.2125f, -0.0775f, -0.0325f, -0.0055f, 0.0f,
                0.0055f, 0.0325f, 0.0775f, 0.2125f, 0.4375f, 0.6625f, 0.8875f, 1.0f};
            for (uint32_t i = 0; i < 16; ++i) signedTable.SetValue(i, values[i]);
        } else {
            signedMap.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(signedIn));
            unsignedMap.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(unsignedIn));
            ReadScalar(signedMap, signedTable, 0, 256);
            ReadScalar(unsignedMap, unsignedTable, 0, 256);
        }
    }

    __aicore__ inline void InitFactors(GM_ADDR factorIn, GM_ADDR meanIn, GM_ADDR invalidIn) {
        factors.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(factorIn));
        rowMean.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(meanIn));
        invalidFlag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(invalidIn));
        auto flag = scales[8].template ReinterpretCast<int32_t>();
        ReadScalar(invalidFlag, flag, 0, 1);
        invalidParameter = flag.GetValue(0) != 0;
        ReadScalar(rowMean, scales[8], 0, 1);
        float mean = scales.GetValue(8);
        // Match fmaxf(mean, 1e-30), including a NaN mean.
        meanDenominator = mean > 1e-30f ? mean : 1e-30f;
    }

    __aicore__ inline void Process(int64_t numel, int64_t mo, int64_t mso, int64_t vo, int64_t vso,
        uint32_t blockSize, float beta1, float beta2, float lr, float decay, float eps, float step, float bias,
        int64_t shardStart = 0, int64_t rows = 0, int64_t columns = 0) {
        if constexpr (Factored) { if (invalidParameter) return; }
        int64_t count = (numel + blockSize - 1) / blockSize;
        for (int64_t block = GetBlockIdx(); block < count; block += GetBlockNum()) {
            int64_t start = block * blockSize;
            uint32_t n = numel - start < blockSize ? numel - start : blockSize;
            int64_t codeStart = FourBit ? start / 2 : start;
            uint32_t codeCount = FourBit ? (n + 1) / 2 : n;
            ReadFloat(model, p, start, n);
            ReadFloat(gradient, g, start, n);
            ReadScalar(moment, mCodes, mo + codeStart, codeCount);
            ReadScalar(momentScale, scales, mso + block, 1);
            if constexpr (Factored) LoadFactors(shardStart + start, n, rows, columns);
            else {
                ReadScalar(variance, vCodes, vo + codeStart, codeCount);
                ReadScalar(varianceScale, scales[8], vso + block, 1);
            }
            Sync<HardEvent::V_S>();
            float ms = scales.GetValue(0), vs = scales.GetValue(8);
            for (uint32_t i = 0; i < n; ++i) {
                uint8_t mc, vc;
                if constexpr (FourBit) {
                    mc = (mCodes.GetValue(i / 2) >> (4 * (i % 2))) & 15u;
                    if constexpr (!Factored) {
                        vc = (vCodes.GetValue(i / 2) >> (4 * (i % 2))) & 15u;
                        // AI Core scalar conversion requires a signed integer;
                        // the decoded nibble is in [0, 15], so this is exact.
                        const int32_t varianceCode = static_cast<int32_t>(vc);
                        v.SetValue(i, (static_cast<float>(varianceCode) + 1.0f) * vs / 16.0f);
                    }
                } else {
                    mc = mCodes.GetValue(i); vc = vCodes.GetValue(i);
                    v.SetValue(i, unsignedTable.GetValue(vc) * vs);
                }
                m.SetValue(i, signedTable.GetValue(mc) * ms);
            }
            Sync<HardEvent::S_V>();
            Update(n, beta1, beta2, lr, decay, eps, step, bias);
            if (!Quantize(n)) continue;
            WriteScalar(moment, mCodes, mo + codeStart, codeCount);
            WriteScalar(momentScale, scales, mso + block, 1);
            if constexpr (!Factored) {
                WriteScalar(variance, vCodes, vo + codeStart, codeCount);
                WriteScalar(varianceScale, scales[8], vso + block, 1);
            }
            auto out = outQueue.template AllocTensor<Model>();
            if constexpr (sizeof(Model) == sizeof(float)) Adds(out, p, 0.0f, n);
            else Cast(out, p, RoundMode::CAST_RINT, n);
            Store(model, out, start, n);
        }
    }
};

} // namespace areno_npu

template<uint32_t ModelStorage, uint32_t GradStorage, bool FourBit>
__global__ __aicore__ void adam_quantized_kernel(GM_ADDR model, GM_ADDR grad, GM_ADDR m, GM_ADDR ms,
    GM_ADDR v, GM_ADDR vs, GM_ADDR signedMap, GM_ADDR unsignedMap, int64_t n,
    int64_t mo, int64_t mso, int64_t vo, int64_t vso, uint32_t blockSize,
    float beta1, float beta2, float lr, float decay, float eps, float step, float bias) {
    using Model = typename areno_npu::KernelDtype<ModelStorage>::type;
    using Grad = typename areno_npu::KernelDtype<GradStorage>::type;
    using namespace AscendC;
    using namespace areno_npu;
    AdamQuantizedKernel<Model, Grad, FourBit> kernel;
    kernel.Init(model, grad, m, ms, v, vs, signedMap, unsignedMap);
    kernel.Process(n, mo, mso, vo, vso, blockSize, beta1, beta2, lr, decay, eps, step, bias);
}

namespace areno_npu {

void launch_adamw_quantized(uint32_t blocks, void* stream, bool model_bf16, bool grad_bf16, bool four_bit,
    void* model, const void* grad, uint8_t* moment, float* moment_scale,
    uint8_t* variance, float* variance_scale, const float* signed_map, const float* unsigned_map,
    int64_t n, int64_t mo, int64_t mso, int64_t vo, int64_t vso, uint32_t block_size,
    float beta1, float beta2, float lr, float decay, float eps, float step, float bias) {
#define ARENO_QUANT_LAUNCH(M, G, FOUR) adam_quantized_kernel<KernelDtypeId<M>::value, KernelDtypeId<G>::value, FOUR><<<blocks, nullptr, stream>>>( \
    (uint8_t*)model, (uint8_t*)grad, (uint8_t*)moment, (uint8_t*)moment_scale, (uint8_t*)variance, \
    (uint8_t*)variance_scale, (uint8_t*)signed_map, (uint8_t*)unsigned_map, n, mo, mso, vo, vso, \
    block_size, beta1, beta2, lr, decay, eps, step, bias)
#define ARENO_QUANT_TYPE(M, G) \
    if (four_bit) { ARENO_QUANT_LAUNCH(M, G, true); } else { ARENO_QUANT_LAUNCH(M, G, false); }
    if (model_bf16) {
        if (grad_bf16) { ARENO_QUANT_TYPE(bfloat16_t, bfloat16_t); }
        else { ARENO_QUANT_TYPE(bfloat16_t, float); }
    } else {
        if (grad_bf16) { ARENO_QUANT_TYPE(float, bfloat16_t); }
        else { ARENO_QUANT_TYPE(float, float); }
    }
#undef ARENO_QUANT_TYPE
#undef ARENO_QUANT_LAUNCH
}

} // namespace areno_npu

template<uint32_t ModelStorage, uint32_t GradStorage>
__global__ __aicore__ void adam_factored_step_kernel(GM_ADDR model, GM_ADDR grad, GM_ADDR m, GM_ADDR ms,
    GM_ADDR factors, GM_ADDR mean, GM_ADDR invalid, int64_t n, int64_t mo, int64_t mso,
    int64_t start, int64_t rows, int64_t columns, uint32_t blockSize,
    float beta1, float lr, float decay, float eps, float step, float bias) {
    using Model = typename areno_npu::KernelDtype<ModelStorage>::type;
    using Grad = typename areno_npu::KernelDtype<GradStorage>::type;
    using namespace AscendC;
    using namespace areno_npu;
    AdamQuantizedKernel<Model, Grad, true, true> kernel;
    kernel.Init(model, grad, m, ms, nullptr, nullptr, nullptr, nullptr);
    kernel.InitFactors(factors, mean, invalid);
    kernel.Process(n, mo, mso, 0, 0, blockSize, beta1, 0.0f, lr, decay, eps, step, bias, start, rows, columns);
}

namespace areno_npu {

void launch_adamw_factored_step(uint32_t blocks, void* stream, bool model_bf16, bool grad_bf16,
    void* model, const void* grad, uint8_t* moment, float* moment_scale,
    const float* factors, const float* row_mean, const int32_t* invalid,
    int64_t n, int64_t mo, int64_t mso, int64_t start, int64_t rows, int64_t columns, uint32_t block_size,
    float beta1, float lr, float decay, float eps, float step, float bias) {
#define ARENO_FACTORED_STEP(M, G) adam_factored_step_kernel<KernelDtypeId<M>::value, KernelDtypeId<G>::value><<<blocks, nullptr, stream>>>( \
    (uint8_t*)model, (uint8_t*)grad, (uint8_t*)moment, (uint8_t*)moment_scale, (uint8_t*)factors, \
    (uint8_t*)row_mean, (uint8_t*)invalid, n, mo, mso, start, rows, columns, block_size, \
    beta1, lr, decay, eps, step, bias)
    if (model_bf16) {
        if (grad_bf16) { ARENO_FACTORED_STEP(bfloat16_t, bfloat16_t); }
        else { ARENO_FACTORED_STEP(bfloat16_t, float); }
    } else {
        if (grad_bf16) { ARENO_FACTORED_STEP(float, bfloat16_t); }
        else { ARENO_FACTORED_STEP(float, float); }
    }
#undef ARENO_FACTORED_STEP
}
} // namespace areno_npu
