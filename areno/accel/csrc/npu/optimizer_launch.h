#pragma once
#include <cstdint>

namespace areno_npu {
constexpr uint32_t kAdamTile = 1024;

void launch_adamw_fp32(uint32_t blocks, void* stream, bool model_bf16, bool grad_bf16,
                       bool compact_master, void* model, const void* grad,
                       uint16_t* low_bits, uint8_t* carries, float* moment, float* variance,
                       int64_t numel, int64_t state_offset, float beta1, float beta2,
                       float effective_lr, float weight_decay, float eps,
                       float step_size, float bias_correction2_sqrt);

void launch_adamw_quantized(uint32_t blocks, void* stream, bool model_bf16, bool grad_bf16, bool four_bit,
                            void* model, const void* grad, uint8_t* moment, float* moment_scale,
                            uint8_t* variance, float* variance_scale, const float* signed_map,
                            const float* unsigned_map, int64_t numel, int64_t moment_offset,
                            int64_t moment_scale_offset, int64_t variance_offset, int64_t variance_scale_offset,
                            uint32_t block_size, float beta1, float beta2, float effective_lr, float weight_decay,
                            float eps, float step_size, float bias_correction2_sqrt);
} // namespace areno_npu
