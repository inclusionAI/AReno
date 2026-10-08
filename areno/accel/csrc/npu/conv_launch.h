#pragma once
#include <cstdint>

namespace areno_npu {
constexpr uint32_t kConvTile = 256;
enum ConvOp : uint32_t { ConvForward, ConvInputGrad, ConvWeightGrad, ConvDecode };
void launch_conv(uint32_t blocks, void* stream, uint32_t storage, ConvOp op, bool packed,
    const void* input, const float* weight, const void* grad, float* preact, void* output,
    const void* history, const int32_t* cu_seqlens, int64_t batch, int64_t seqlen,
    int64_t channels, int64_t kernel_size, int64_t segments);
} // namespace areno_npu
