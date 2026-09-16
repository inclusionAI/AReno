#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_experts_launch.h"

namespace areno_npu {
using namespace AscendC;

__aicore__ constexpr MatmulConfig expert_matmul_config() {
    auto config = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(
        MatmulShapeParams{kExpertM, kExpertN, kExpertK, kExpertM, kExpertN, kExpertK});
    config.enableSetBias = false;
    config.enableStaticPadZeros = true;
    return config;
}

template <typename T>
__global__ __aicore__ void expert_matmul_kernel(GM_ADDR in, GM_ADDR w, GM_ADDR out,
    GM_ADDR expertIds, GM_ADDR paddedTotal, int64_t capacity, int64_t n, int64_t k, int64_t outputStride, GM_ADDR workspace) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using A = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using B = MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using C = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    constexpr static auto config = GetMatmulApiTiling<A, B, C, C>(expert_matmul_config());
    Matmul<A, B, C, C, config> mm;
    TPipe pipe;
    SetSysWorkspace(workspace);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, (TCubeTiling*)nullptr);
    GlobalTensor<T> input, weight;
    GlobalTensor<float> output;
    GlobalTensor<int32_t> experts, total;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(in));
    weight.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(w));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
    experts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertIds));
    total.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(paddedTotal));
    // The preceding vector launches wrote this metadata. Scalar GM reads
    // need explicit cache invalidation, including during graph replay.
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(total);
    int64_t used = total.GetValue(0);
    int64_t columns = (n + kExpertN - 1) / kExpertN;
    for (int64_t task = GetBlockIdx(); task < capacity / kExpertM * columns; task += GetBlockNum()) {
        int64_t row = task / columns * kExpertM, col = task % columns * kExpertN;
        if (row >= used) continue;
        auto id = experts[row / kExpertM];
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(id);
        int64_t expert = id.GetValue(0);
        if (expert < 0) continue;
        int32_t tailN = n - col < kExpertN ? n - col : kExpertN;
        for (int64_t inner = 0; inner < k; inner += kExpertK) {
            int32_t tailK = k - inner < kExpertK ? k - inner : kExpertK;
            // Keep FP32 GM rows aligned for Fixpipe, including odd N tails.
            mm.SetOrgShape(kExpertM, n, k, k, outputStride);
            mm.SetTensorA(input[row * k + inner]);
            mm.SetTensorB(weight[(expert * n + col) * k + inner], true);
            mm.SetTail(kExpertM, tailN, tailK);
            // First tile overwrites; later tiles accumulate FP32 through
            // Fixpipe atomic add. No BF16/FP16 rounding between K tiles.
            mm.IterateAll(output[row * outputStride + col], inner == 0 ? 0 : 1);
            mm.End();
        }
    }
}

void launch_expert_matmul(uint32_t blocks, void* stream, uint32_t storage,
    const void* input, const void* weight, float* output, const int32_t* experts,
    const int32_t* total, int64_t capacity, int64_t n, int64_t k, int64_t output_stride, void* workspace) {
#define ARENO_EXPERT_MATMUL(T) expert_matmul_kernel<T><<<blocks, nullptr, stream>>>( \
    (uint8_t*)input, (uint8_t*)weight, (uint8_t*)output, (uint8_t*)experts, (uint8_t*)total, \
    capacity, n, k, output_stride, (uint8_t*)workspace)
    if (storage == 1) { ARENO_EXPERT_MATMUL(half); }
    else { ARENO_EXPERT_MATMUL(bfloat16_t); }
#undef ARENO_EXPERT_MATMUL
}
} // namespace areno_npu
