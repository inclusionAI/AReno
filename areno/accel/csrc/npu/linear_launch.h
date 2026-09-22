#pragma once
#include <cstdint>

namespace areno_npu {
constexpr uint32_t kLinearTile = 1024;
void launch_linear_bias(uint32_t blocks, void* stream, uint32_t storage, bool backward,
                        const void* input, const void* bias, void* output, int64_t rows, int64_t columns);
} // namespace areno_npu
