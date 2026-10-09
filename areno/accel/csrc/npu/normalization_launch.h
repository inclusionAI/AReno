#pragma once
#include <cstdint>

namespace areno_npu {
// gate_kind: 0 = no gate, 1 = SiLU, 2 = grouped sigmoid (forward only).
// Storage codes match activation_launch.h: FP32=0, FP16=1, BF16=2.
// output is y in forward and dx in backward; inv is saved/loaded per row.
void launch_normalization(uint32_t blocks, void* stream, uint32_t storage, uint32_t weight_storage,
                          bool backward, bool scale, uint32_t gate_kind,
                          const void* input, const void* gate, const void* weight, const void* grad,
                          void* output, void* grad_gate, float* inv, float* grad_weight,
                          int64_t rows, int64_t width, int64_t groups, float eps);
} // namespace areno_npu
