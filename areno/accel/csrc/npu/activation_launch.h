#pragma once
#include <cstdint>

namespace areno_npu {
enum Activation : uint32_t {
    Silu = 0, DSilu, Sigmoid, DSigmoid, Softplus, DSoftplus,
    SiluMul, DSiluMul, GeluTanhMul, DGeluTanhMul,
};
constexpr uint32_t kActivationTile = 1024;

// Storage: 0 = FP32, 1 = FP16, 2 = BF16. Unary tensors are flattened;
// gated tensors consist of rows of [gate(width), up(width)].
void launch_activation(uint32_t blocks, void* stream, uint32_t storage, Activation op,
                       void* output, const void* input, const void* grad,
                       int64_t rows, int64_t width);
} // namespace areno_npu
