#pragma once
#include <cstdint>

namespace areno_npu {
constexpr int64_t kExpertM = 16;
constexpr int64_t kExpertN = 64;
constexpr int64_t kExpertK = 128;
constexpr int64_t kExpertVectorTile = 512;

void launch_expert_tokens(uint32_t blocks, void* stream, const int32_t* aligned,
    const int32_t* experts, const int32_t* total, int64_t* tokens,
    int64_t capacity, int64_t routes, int64_t top_k);
// Packed rows are grouped in kExpertM-sized blocks. B is [expert, N, K].
// Both projections keep FP32 output, including across K tiles. The vector
// epilogue controls the two distinct CUDA-compatible storage-rounding points.
void launch_expert_matmul(uint32_t blocks, void* stream, uint32_t storage,
    const void* input, const void* weight, float* output, const int32_t* experts,
    const int32_t* total, int64_t capacity, int64_t n, int64_t k, int64_t output_stride, void* workspace);
void launch_expert_cast(uint32_t blocks, void* stream, uint32_t storage,
    const float* input, void* output, int64_t rows, int64_t width, int64_t input_stride);
void launch_expert_weight_scatter(uint32_t blocks, void* stream, uint32_t storage,
    const float* input, const float* weights, const int32_t* aligned,
    const int32_t* experts, const int32_t* total, void* output,
    int64_t capacity, int64_t routes, int64_t hidden, int64_t input_stride);
void launch_expert_reduce(uint32_t blocks, void* stream, uint32_t storage,
    const void* input, void* output, int64_t tokens, int64_t hidden, int64_t top_k, float scale);
} // namespace areno_npu
