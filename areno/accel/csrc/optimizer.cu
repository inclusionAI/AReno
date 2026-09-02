#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

__device__ __constant__ float kSigned4bitDynamicMap[16] = {
    -0.8875f, -0.6625f, -0.4375f, -0.2125f, -0.0775f, -0.0325f, -0.0055f, 0.0f,
    0.0055f, 0.0325f, 0.0775f, 0.2125f, 0.4375f, 0.6625f, 0.8875f, 1.0f};

__device__ __forceinline__ uint8_t load_nibble(const uint8_t* packed, int64_t index) {
  const uint8_t byte = packed[index >> 1];
  return (index & 1) == 0 ? byte & 0x0Fu : byte >> 4;
}

__device__ __forceinline__ uint8_t nearest_signed_dynamic_code(float normalized) {
  uint8_t best = 0;
  float best_distance = fabsf(normalized - kSigned4bitDynamicMap[0]);
#pragma unroll
  for (uint8_t code = 1; code < 16; ++code) {
    const float distance = fabsf(normalized - kSigned4bitDynamicMap[code]);
    if (distance < best_distance) {
      best = code;
      best_distance = distance;
    }
  }
  return best;
}

__device__ __forceinline__ float atomic_max_nonnegative(float* address, float value) {
  int* address_as_int = reinterpret_cast<int*>(address);
  const int old = atomicMax(address_as_int, __float_as_int(value));
  return __int_as_float(old);
}

__device__ __forceinline__ float rank1_scale(
    const float* axis_scales,
    const int64_t* shape,
    const int64_t* strides,
    int64_t ndim,
    int64_t flat_index) {
  float scale = __int_as_float(0x7f800000);
  int64_t axis_offset = 0;
  for (int64_t axis = 0; axis < ndim; ++axis) {
    const int64_t coordinate = (flat_index / strides[axis]) % shape[axis];
    scale = fminf(scale, axis_scales[axis_offset + coordinate]);
    axis_offset += shape[axis];
  }
  return scale;
}

__device__ __forceinline__ uint8_t nearest_dynamic_code(float value, const float* codebook) {
  int lower = 0;
  int upper = 255;
  while (lower < upper) {
    const int middle = (lower + upper) >> 1;
    if (codebook[middle] < value) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  if (lower == 0) {
    return 0;
  }
  const float left_distance = fabsf(value - codebook[lower - 1]);
  const float right_distance = fabsf(codebook[lower] - value);
  return static_cast<uint8_t>(left_distance <= right_distance ? lower - 1 : lower);
}

__device__ __forceinline__ float adamw_update(
    float master,
    float grad,
    float& exp_avg,
    float& exp_avg_sq,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  if (weight_decay != 0.0f) {
    master *= 1.0f - effective_lr * weight_decay;
  }
  exp_avg = beta1 * exp_avg + (1.0f - beta1) * grad;
  exp_avg_sq = beta2 * exp_avg_sq + (1.0f - beta2) * grad * grad;
  const float denom = sqrtf(exp_avg_sq) / bias_correction2_sqrt + eps;
  return master - step_size * exp_avg / denom;
}

template <typename grad_t>
__device__ __forceinline__ float load_grad(const grad_t* grad, int64_t index) {
  return static_cast<float>(grad[index]);
}

template <>
__device__ __forceinline__ float load_grad<at::BFloat16>(const at::BFloat16* grad, int64_t index) {
  const auto* raw = reinterpret_cast<const __nv_bfloat16*>(grad);
  return __bfloat162float(raw[index]);
}

template <typename model_t>
__device__ __forceinline__ float load_model(const model_t* model, int64_t index) {
  return static_cast<float>(model[index]);
}

template <>
__device__ __forceinline__ float load_model<at::BFloat16>(const at::BFloat16* model, int64_t index) {
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(model)[index]);
}

template <typename model_t>
__device__ __forceinline__ void store_model(model_t* model, int64_t index, float value) {
  model[index] = static_cast<model_t>(value);
}

template <>
__device__ __forceinline__ void store_model<at::BFloat16>(at::BFloat16* model, int64_t index, float value) {
  reinterpret_cast<__nv_bfloat16*>(model)[index] = __float2bfloat16_rn(value);
}

template <typename model_t, typename grad_t>
__global__ void adamw_4bit_kernel(
    model_t* model,
    const grad_t* grad,
    uint8_t* exp_avg_q,
    float* exp_avg_scale,
    uint8_t* exp_avg_sq_q,
    float* exp_avg_sq_scale,
    int64_t numel,
    int64_t packed_offset,
    int64_t moment_scale_offset,
    int64_t variance_scale_offset,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  constexpr int warp_size = 32;
  constexpr int max_warps = 32;
  __shared__ float warp_moment_maxima[max_warps];
  __shared__ float warp_variance_maxima[max_warps];
  extern __shared__ uint8_t packed_codes[];
  uint8_t* moment_codes = packed_codes;
  uint8_t* variance_codes = packed_codes + blockDim.x;
  __shared__ float old_moment_scale;
  __shared__ float old_variance_scale;
  __shared__ float new_moment_scale;
  __shared__ float new_variance_scale;
  __shared__ int invalid_block;

  const int tid = threadIdx.x;
  const int64_t block_start = static_cast<int64_t>(blockIdx.x) * blockDim.x;
  const int64_t local_index = block_start + tid;
  const bool active = local_index < numel;
  if (tid == 0) {
    invalid_block = 0;
    old_moment_scale = exp_avg_scale[moment_scale_offset + blockIdx.x];
    old_variance_scale = exp_avg_sq_scale[variance_scale_offset + blockIdx.x];
  }
  __syncthreads();

  float local_moment_max = 0.0f;
  float local_variance_max = 0.0f;
  if (active) {
    const uint8_t moment_code = load_nibble(exp_avg_q + packed_offset, local_index);
    const uint8_t variance_code = load_nibble(exp_avg_sq_q + packed_offset, local_index);
    float moment = kSigned4bitDynamicMap[moment_code] * old_moment_scale;
    float variance = (static_cast<float>(variance_code) + 1.0f) * old_variance_scale / 16.0f;
    const float gradient = load_grad(grad, local_index);
    float updated_weight = load_model(model, local_index);
    if (weight_decay != 0.0f) {
      updated_weight *= 1.0f - effective_lr * weight_decay;
    }
    moment = beta1 * moment + (1.0f - beta1) * gradient;
    variance = beta2 * variance + (1.0f - beta2) * gradient * gradient;
    const float denom = sqrtf(variance) / bias_correction2_sqrt + eps;
    updated_weight -= step_size * moment / denom;
    if (!isfinite(gradient) || !isfinite(moment) || !isfinite(variance) || !isfinite(updated_weight)) {
      atomicExch(&invalid_block, 1);
    }
    local_moment_max = fabsf(moment);
    local_variance_max = variance;
  }
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    local_moment_max = fmaxf(local_moment_max, __shfl_down_sync(0xFFFFFFFFu, local_moment_max, offset));
    local_variance_max =
        fmaxf(local_variance_max, __shfl_down_sync(0xFFFFFFFFu, local_variance_max, offset));
  }
  const int lane = tid & (warp_size - 1);
  const int warp = tid / warp_size;
  if (lane == 0) {
    warp_moment_maxima[warp] = local_moment_max;
    warp_variance_maxima[warp] = local_variance_max;
  }
  __syncthreads();
  if (invalid_block != 0) {
    return;
  }
  if (warp == 0) {
    const int warp_count = blockDim.x / warp_size;
    float block_moment_max = lane < warp_count ? warp_moment_maxima[lane] : 0.0f;
    float block_variance_max = lane < warp_count ? warp_variance_maxima[lane] : 0.0f;
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
      block_moment_max =
          fmaxf(block_moment_max, __shfl_down_sync(0xFFFFFFFFu, block_moment_max, offset));
      block_variance_max =
          fmaxf(block_variance_max, __shfl_down_sync(0xFFFFFFFFu, block_variance_max, offset));
    }
    if (lane == 0) {
      new_moment_scale = block_moment_max;
      new_variance_scale = block_variance_max;
      exp_avg_scale[moment_scale_offset + blockIdx.x] = block_moment_max;
      exp_avg_sq_scale[variance_scale_offset + blockIdx.x] = block_variance_max;
    }
  }
  __syncthreads();

  // Recompute instead of keeping FP32 moment, variance and weight values live
  // across the reduction. This mirrors the 8-bit kernel and avoids register
  // spills into CUDA local memory for the packed 4-bit update.
  if (active) {
    const uint8_t moment_code = load_nibble(exp_avg_q + packed_offset, local_index);
    const uint8_t variance_code = load_nibble(exp_avg_sq_q + packed_offset, local_index);
    float moment = kSigned4bitDynamicMap[moment_code] * old_moment_scale;
    float variance = (static_cast<float>(variance_code) + 1.0f) * old_variance_scale / 16.0f;
    const float gradient = load_grad(grad, local_index);
    float updated_weight = load_model(model, local_index);
    if (weight_decay != 0.0f) {
      updated_weight *= 1.0f - effective_lr * weight_decay;
    }
    moment = beta1 * moment + (1.0f - beta1) * gradient;
    variance = beta2 * variance + (1.0f - beta2) * gradient * gradient;
    const float denom = sqrtf(variance) / bias_correction2_sqrt + eps;
    updated_weight -= step_size * moment / denom;
    const float normalized_moment = moment / fmaxf(new_moment_scale, 1.0e-30f);
    moment_codes[tid] = nearest_signed_dynamic_code(normalized_moment);
    const float normalized_variance = variance / fmaxf(new_variance_scale, 1.0e-30f);
    int updated_variance_code = __float2int_rn(normalized_variance * 16.0f - 1.0f);
    updated_variance_code =
        updated_variance_code < 0 ? 0 : (updated_variance_code > 15 ? 15 : updated_variance_code);
    variance_codes[tid] = static_cast<uint8_t>(updated_variance_code);
    store_model(model, local_index, updated_weight);
  } else {
    moment_codes[tid] = 7;
    variance_codes[tid] = 0;
  }
  __syncthreads();
  if ((tid & 1) == 0 && local_index < numel) {
    const int64_t byte_index = packed_offset + (local_index >> 1);
    exp_avg_q[byte_index] = moment_codes[tid] | static_cast<uint8_t>(moment_codes[tid + 1] << 4);
    exp_avg_sq_q[byte_index] = variance_codes[tid] | static_cast<uint8_t>(variance_codes[tid + 1] << 4);
  }
}

template <typename grad_t>
__global__ void adamw_4bit_rank1_stats_kernel(
    const grad_t* grad,
    const uint8_t* exp_avg_sq_q,
    const float* previous_scales,
    float* updated_scales,
    int* invalid,
    const int64_t* shape,
    const int64_t* strides,
    int64_t ndim,
    int64_t numel,
    int64_t packed_offset,
    int64_t parameter_shard_start,
    float beta2,
    bool has_state) {
  __shared__ int invalid_block;
  const int tid = threadIdx.x;
  const int64_t local_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
  const bool active = local_index < numel;
  if (tid == 0) {
    invalid_block = 0;
  }
  __syncthreads();
  float variance = 0.0f;
  if (active) {
    const int64_t parameter_index = parameter_shard_start + local_index;
    if (has_state) {
      const float old_scale = rank1_scale(previous_scales, shape, strides, ndim, parameter_index);
      const uint8_t code = load_nibble(exp_avg_sq_q + packed_offset, local_index);
      variance = (static_cast<float>(code) + 1.0f) * old_scale / 16.0f;
    }
    const float gradient = load_grad(grad, local_index);
    variance = beta2 * variance + (1.0f - beta2) * gradient * gradient;
    if (!isfinite(gradient) || !isfinite(variance)) {
      atomicExch(&invalid_block, 1);
    }
  }
  __syncthreads();
  if (invalid_block != 0) {
    if (tid == 0) {
      atomicExch(invalid, 1);
    }
    return;
  }
  if (active) {
    const int64_t parameter_index = parameter_shard_start + local_index;
    int64_t axis_offset = 0;
    for (int64_t axis = 0; axis < ndim; ++axis) {
      const int64_t coordinate = (parameter_index / strides[axis]) % shape[axis];
      atomic_max_nonnegative(updated_scales + axis_offset + coordinate, variance);
      axis_offset += shape[axis];
    }
  }
}

template <typename model_t, typename grad_t>
__global__ void adamw_4bit_rank1_step_kernel(
    model_t* model,
    const grad_t* grad,
    uint8_t* exp_avg_q,
    float* exp_avg_scale,
    uint8_t* exp_avg_sq_q,
    const float* previous_scales,
    const float* updated_scales,
    const int* invalid,
    const int64_t* shape,
    const int64_t* strides,
    int64_t ndim,
    int64_t numel,
    int64_t packed_offset,
    int64_t moment_scale_offset,
    int64_t parameter_shard_start,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  constexpr int warp_size = 32;
  constexpr int max_warps = 32;
  __shared__ float warp_moment_maxima[max_warps];
  extern __shared__ uint8_t packed_codes[];
  uint8_t* moment_codes = packed_codes;
  uint8_t* variance_codes = packed_codes + blockDim.x;
  __shared__ float old_moment_scale;
  __shared__ float new_moment_scale;
  __shared__ int invalid_block;
  const int tid = threadIdx.x;
  if (*invalid != 0) {
    return;
  }
  const int64_t local_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
  const bool active = local_index < numel;
  if (tid == 0) {
    invalid_block = 0;
    old_moment_scale = exp_avg_scale[moment_scale_offset + blockIdx.x];
  }
  __syncthreads();
  float local_moment_max = 0.0f;
  if (active) {
    const int64_t parameter_index = parameter_shard_start + local_index;
    const uint8_t moment_code = load_nibble(exp_avg_q + packed_offset, local_index);
    const uint8_t variance_code = load_nibble(exp_avg_sq_q + packed_offset, local_index);
    float moment = kSigned4bitDynamicMap[moment_code] * old_moment_scale;
    const float old_scale = rank1_scale(previous_scales, shape, strides, ndim, parameter_index);
    float variance = (static_cast<float>(variance_code) + 1.0f) * old_scale / 16.0f;
    const float gradient = load_grad(grad, local_index);
    float updated_weight = load_model(model, local_index);
    if (weight_decay != 0.0f) {
      updated_weight *= 1.0f - effective_lr * weight_decay;
    }
    moment = beta1 * moment + (1.0f - beta1) * gradient;
    variance = beta2 * variance + (1.0f - beta2) * gradient * gradient;
    const float denom = sqrtf(variance) / bias_correction2_sqrt + eps;
    updated_weight -= step_size * moment / denom;
    if (!isfinite(gradient) || !isfinite(moment) || !isfinite(variance) || !isfinite(updated_weight)) {
      atomicExch(&invalid_block, 1);
    }
    local_moment_max = fabsf(moment);
  }
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    local_moment_max = fmaxf(local_moment_max, __shfl_down_sync(0xFFFFFFFFu, local_moment_max, offset));
  }
  const int lane = tid & (warp_size - 1);
  const int warp = tid / warp_size;
  if (lane == 0) {
    warp_moment_maxima[warp] = local_moment_max;
  }
  __syncthreads();
  if (invalid_block != 0) {
    return;
  }
  if (warp == 0) {
    const int warp_count = blockDim.x / warp_size;
    float block_moment_max = lane < warp_count ? warp_moment_maxima[lane] : 0.0f;
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
      block_moment_max =
          fmaxf(block_moment_max, __shfl_down_sync(0xFFFFFFFFu, block_moment_max, offset));
    }
    if (lane == 0) {
      new_moment_scale = block_moment_max;
      exp_avg_scale[moment_scale_offset + blockIdx.x] = block_moment_max;
    }
  }
  __syncthreads();
  if (active) {
    const int64_t parameter_index = parameter_shard_start + local_index;
    const uint8_t moment_code = load_nibble(exp_avg_q + packed_offset, local_index);
    const uint8_t variance_code = load_nibble(exp_avg_sq_q + packed_offset, local_index);
    float moment = kSigned4bitDynamicMap[moment_code] * old_moment_scale;
    const float old_scale = rank1_scale(previous_scales, shape, strides, ndim, parameter_index);
    float variance = (static_cast<float>(variance_code) + 1.0f) * old_scale / 16.0f;
    const float gradient = load_grad(grad, local_index);
    float updated_weight = load_model(model, local_index);
    if (weight_decay != 0.0f) {
      updated_weight *= 1.0f - effective_lr * weight_decay;
    }
    moment = beta1 * moment + (1.0f - beta1) * gradient;
    variance = beta2 * variance + (1.0f - beta2) * gradient * gradient;
    const float denom = sqrtf(variance) / bias_correction2_sqrt + eps;
    updated_weight -= step_size * moment / denom;
    moment_codes[tid] = nearest_signed_dynamic_code(moment / fmaxf(new_moment_scale, 1.0e-30f));
    const float new_scale = rank1_scale(updated_scales, shape, strides, ndim, parameter_index);
    int updated_variance_code =
        __float2int_rn(variance / fmaxf(new_scale, 1.0e-30f) * 16.0f - 1.0f);
    updated_variance_code =
        updated_variance_code < 0 ? 0 : (updated_variance_code > 15 ? 15 : updated_variance_code);
    variance_codes[tid] = static_cast<uint8_t>(updated_variance_code);
    store_model(model, local_index, updated_weight);
  } else {
    moment_codes[tid] = 7;
    variance_codes[tid] = 0;
  }
  __syncthreads();
  if ((tid & 1) == 0 && local_index < numel) {
    const int64_t byte_index = packed_offset + (local_index >> 1);
    exp_avg_q[byte_index] = moment_codes[tid] | static_cast<uint8_t>(moment_codes[tid + 1] << 4);
    exp_avg_sq_q[byte_index] = variance_codes[tid] | static_cast<uint8_t>(variance_codes[tid + 1] << 4);
  }
}

template <typename grad_t>
__global__ void adamw_bf16_master_kernel(
    at::BFloat16* model,
    uint16_t* low_bits,
    uint8_t* round_up_bits,
    const grad_t* grad,
    float* exp_avg,
    float* exp_avg_sq,
    int64_t numel,
    int64_t state_offset,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  const int64_t first_byte = state_offset >> 3;
  const int64_t last_byte = (state_offset + numel + 7) >> 3;
  const int64_t byte_index = first_byte + blockIdx.x * blockDim.x + threadIdx.x;
  if (byte_index >= last_byte) {
    return;
  }

  uint8_t carries = round_up_bits[byte_index];
  const int64_t byte_start = byte_index << 3;
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) {
    const int64_t state_index = byte_start + bit;
    if (state_index < state_offset || state_index >= state_offset + numel) {
      continue;
    }
    const int64_t local_index = state_index - state_offset;
    const auto* model_raw = reinterpret_cast<const uint16_t*>(model);
    const uint32_t rounded_high = static_cast<uint32_t>(model_raw[local_index]);
    const uint32_t rounded_up = static_cast<uint32_t>((carries >> bit) & 1u);
    const uint32_t original_high = (rounded_high - rounded_up) & 0xFFFFu;
    const uint32_t master_word = (original_high << 16) | static_cast<uint32_t>(low_bits[state_index]);
    float moment = exp_avg[state_index];
    float variance = exp_avg_sq[state_index];
    const float updated = adamw_update(
        __uint_as_float(master_word),
        load_grad(grad, local_index),
        moment,
        variance,
        beta1,
        beta2,
        effective_lr,
        weight_decay,
        eps,
        step_size,
        bias_correction2_sqrt);
    exp_avg[state_index] = moment;
    exp_avg_sq[state_index] = variance;

    const uint32_t updated_word = __float_as_uint(updated);
    const __nv_bfloat16 rounded = __float2bfloat16_rn(updated);
    const uint16_t rounded_word = reinterpret_cast<const uint16_t&>(rounded);
    reinterpret_cast<uint16_t*>(model)[local_index] = rounded_word;
    low_bits[state_index] = static_cast<uint16_t>(updated_word & 0xFFFFu);
    const uint8_t mask = static_cast<uint8_t>(1u << bit);
    if (rounded_word != static_cast<uint16_t>(updated_word >> 16)) {
      carries = static_cast<uint8_t>(carries | mask);
    } else {
      carries = static_cast<uint8_t>(carries & static_cast<uint8_t>(~mask));
    }
  }
  round_up_bits[byte_index] = carries;
}

template <typename grad_t>
__global__ void adamw_fp32_model_kernel(
    float* model,
    const grad_t* grad,
    float* exp_avg,
    float* exp_avg_sq,
    int64_t numel,
    int64_t state_offset,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  const int64_t local_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_index >= numel) {
    return;
  }
  const int64_t state_index = state_offset + local_index;
  float moment = exp_avg[state_index];
  float variance = exp_avg_sq[state_index];
  model[local_index] = adamw_update(
      model[local_index],
      load_grad(grad, local_index),
      moment,
      variance,
      beta1,
      beta2,
      effective_lr,
      weight_decay,
      eps,
      step_size,
      bias_correction2_sqrt);
  exp_avg[state_index] = moment;
  exp_avg_sq[state_index] = variance;
}

template <typename grad_t>
void launch_adamw(
    torch::Tensor model,
    torch::Tensor low_bits,
    torch::Tensor round_up_bits,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int64_t state_offset,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  constexpr int threads = 256;
  const int64_t numel = model.numel();
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (model.scalar_type() == at::kBFloat16) {
    const int64_t first_byte = state_offset >> 3;
    const int64_t last_byte = (state_offset + numel + 7) >> 3;
    const int blocks = static_cast<int>((last_byte - first_byte + threads - 1) / threads);
    adamw_bf16_master_kernel<<<blocks, threads, 0, stream>>>(
        model.data_ptr<at::BFloat16>(),
        low_bits.data_ptr<uint16_t>(),
        round_up_bits.data_ptr<uint8_t>(),
        grad.data_ptr<grad_t>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        numel,
        state_offset,
        beta1,
        beta2,
        effective_lr,
        weight_decay,
        eps,
        step_size,
        bias_correction2_sqrt);
  } else {
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    adamw_fp32_model_kernel<<<blocks, threads, 0, stream>>>(
        model.data_ptr<float>(),
        grad.data_ptr<grad_t>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        numel,
        state_offset,
        beta1,
        beta2,
        effective_lr,
        weight_decay,
        eps,
        step_size,
        bias_correction2_sqrt);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename model_t, typename grad_t>
void launch_adamw_4bit(
    torch::Tensor model,
    torch::Tensor grad,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scale,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_scale,
    int64_t packed_offset,
    int64_t moment_scale_offset,
    int64_t variance_scale_offset,
    int64_t quant_block_size,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  const int blocks = static_cast<int>((model.numel() + quant_block_size - 1) / quant_block_size);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_bytes = static_cast<size_t>(2 * quant_block_size) * sizeof(uint8_t);
  adamw_4bit_kernel<model_t, grad_t><<<blocks, static_cast<int>(quant_block_size), shared_bytes, stream>>>(
      model.data_ptr<model_t>(),
      grad.data_ptr<grad_t>(),
      exp_avg_q.data_ptr<uint8_t>(),
      exp_avg_scale.data_ptr<float>(),
      exp_avg_sq_q.data_ptr<uint8_t>(),
      exp_avg_sq_scale.data_ptr<float>(),
      model.numel(),
      packed_offset,
      moment_scale_offset,
      variance_scale_offset,
      beta1,
      beta2,
      effective_lr,
      weight_decay,
      eps,
      step_size,
      bias_correction2_sqrt);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename grad_t>
void launch_adamw_4bit_rank1_stats(
    torch::Tensor grad,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor previous_scales,
    torch::Tensor updated_scales,
    torch::Tensor invalid,
    torch::Tensor shape,
    torch::Tensor strides,
    int64_t packed_offset,
    int64_t parameter_shard_start,
    int64_t quant_block_size,
    float beta2,
    bool has_state) {
  const int blocks = static_cast<int>((grad.numel() + quant_block_size - 1) / quant_block_size);
  const auto stream = at::cuda::getCurrentCUDAStream();
  adamw_4bit_rank1_stats_kernel<grad_t><<<blocks, static_cast<int>(quant_block_size), 0, stream>>>(
      grad.data_ptr<grad_t>(),
      exp_avg_sq_q.data_ptr<uint8_t>(),
      previous_scales.data_ptr<float>(),
      updated_scales.data_ptr<float>(),
      invalid.data_ptr<int>(),
      shape.data_ptr<int64_t>(),
      strides.data_ptr<int64_t>(),
      shape.numel(),
      grad.numel(),
      packed_offset,
      parameter_shard_start,
      beta2,
      has_state);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename model_t, typename grad_t>
void launch_adamw_4bit_rank1_step(
    torch::Tensor model,
    torch::Tensor grad,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scale,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor previous_scales,
    torch::Tensor updated_scales,
    torch::Tensor invalid,
    torch::Tensor shape,
    torch::Tensor strides,
    int64_t packed_offset,
    int64_t moment_scale_offset,
    int64_t parameter_shard_start,
    int64_t quant_block_size,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  const int blocks = static_cast<int>((model.numel() + quant_block_size - 1) / quant_block_size);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_bytes = static_cast<size_t>(2 * quant_block_size) * sizeof(uint8_t);
  adamw_4bit_rank1_step_kernel<model_t, grad_t>
      <<<blocks, static_cast<int>(quant_block_size), shared_bytes, stream>>>(
      model.data_ptr<model_t>(),
      grad.data_ptr<grad_t>(),
      exp_avg_q.data_ptr<uint8_t>(),
      exp_avg_scale.data_ptr<float>(),
      exp_avg_sq_q.data_ptr<uint8_t>(),
      previous_scales.data_ptr<float>(),
      updated_scales.data_ptr<float>(),
      invalid.data_ptr<int>(),
      shape.data_ptr<int64_t>(),
      strides.data_ptr<int64_t>(),
      shape.numel(),
      model.numel(),
      packed_offset,
      moment_scale_offset,
      parameter_shard_start,
      beta1,
      beta2,
      effective_lr,
      weight_decay,
      eps,
      step_size,
      bias_correction2_sqrt);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename model_t, typename grad_t>
__global__ void adamw_8bit_blockwise_kernel(
    model_t* model,
    const grad_t* grad,
    uint8_t* exp_avg_q,
    float* exp_avg_scale,
    uint8_t* exp_avg_sq_q,
    float* exp_avg_sq_scale,
    const float* signed_codebook,
    const float* unsigned_codebook,
    int64_t numel,
    int64_t quant_block_size,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  constexpr int warp_size = 32;
  constexpr int max_warps = 8;
  __shared__ float warp_moment_maxima[max_warps];
  __shared__ float warp_variance_maxima[max_warps];
  __shared__ int invalid_block;

  const int64_t block_start = static_cast<int64_t>(blockIdx.x) * quant_block_size;
  const int64_t remaining = numel - block_start;
  const int64_t block_numel = quant_block_size < remaining ? quant_block_size : remaining;
  const float old_moment_scale = exp_avg_scale[blockIdx.x];
  const float old_variance_scale = exp_avg_sq_scale[blockIdx.x];
  float local_moment_max = 0.0f;
  float local_variance_max = 0.0f;
  if (threadIdx.x == 0) {
    invalid_block = 0;
  }
  __syncthreads();

  for (int64_t offset = threadIdx.x; offset < block_numel; offset += blockDim.x) {
    const int64_t index = block_start + offset;
    const float gradient = load_grad(grad, index);
    float moment = signed_codebook[exp_avg_q[index]] * old_moment_scale;
    float variance = unsigned_codebook[exp_avg_sq_q[index]] * old_variance_scale;
    float weight = load_model(model, index);
    weight = adamw_update(
        weight,
        gradient,
        moment,
        variance,
        beta1,
        beta2,
        effective_lr,
        weight_decay,
        eps,
        step_size,
        bias_correction2_sqrt);
    if (!isfinite(gradient) || !isfinite(moment) || !isfinite(variance) || !isfinite(weight)) {
      atomicExch(&invalid_block, 1);
    }
    local_moment_max = fmaxf(local_moment_max, fabsf(moment));
    local_variance_max = fmaxf(local_variance_max, variance);
  }

  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    local_moment_max = fmaxf(local_moment_max, __shfl_down_sync(0xFFFFFFFFu, local_moment_max, offset));
    local_variance_max =
        fmaxf(local_variance_max, __shfl_down_sync(0xFFFFFFFFu, local_variance_max, offset));
  }
  const int lane = threadIdx.x & (warp_size - 1);
  const int warp = threadIdx.x / warp_size;
  if (lane == 0) {
    warp_moment_maxima[warp] = local_moment_max;
    warp_variance_maxima[warp] = local_variance_max;
  }
  __syncthreads();
  if (invalid_block != 0) {
    return;
  }

  if (warp == 0) {
    const int warp_count = blockDim.x / warp_size;
    float block_moment_max = lane < warp_count ? warp_moment_maxima[lane] : 0.0f;
    float block_variance_max = lane < warp_count ? warp_variance_maxima[lane] : 0.0f;
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
      block_moment_max =
          fmaxf(block_moment_max, __shfl_down_sync(0xFFFFFFFFu, block_moment_max, offset));
      block_variance_max =
          fmaxf(block_variance_max, __shfl_down_sync(0xFFFFFFFFu, block_variance_max, offset));
    }
    if (lane == 0) {
      exp_avg_scale[blockIdx.x] = block_moment_max;
      exp_avg_sq_scale[blockIdx.x] = block_variance_max;
    }
  }
  __syncthreads();
  const float new_moment_scale = exp_avg_scale[blockIdx.x];
  const float new_variance_scale = exp_avg_sq_scale[blockIdx.x];
  for (int64_t offset = threadIdx.x; offset < block_numel; offset += blockDim.x) {
    const int64_t index = block_start + offset;
    const float gradient = load_grad(grad, index);
    const float previous_moment = signed_codebook[exp_avg_q[index]] * old_moment_scale;
    const float previous_variance = unsigned_codebook[exp_avg_sq_q[index]] * old_variance_scale;
    const float moment = beta1 * previous_moment + (1.0f - beta1) * gradient;
    const float variance = beta2 * previous_variance + (1.0f - beta2) * gradient * gradient;
    float weight = load_model(model, index);
    if (weight_decay != 0.0f) {
      weight *= 1.0f - effective_lr * weight_decay;
    }
    const float denom = sqrtf(variance) / bias_correction2_sqrt + eps;
    weight -= step_size * moment / denom;
    const float normalized_moment = moment / fmaxf(new_moment_scale, 1.0e-30f);
    const float normalized_variance = variance / fmaxf(new_variance_scale, 1.0e-30f);
    exp_avg_q[index] = nearest_dynamic_code(normalized_moment, signed_codebook);
    exp_avg_sq_q[index] = nearest_dynamic_code(normalized_variance, unsigned_codebook);
    store_model(model, index, weight);
  }
}

template <typename model_t, typename grad_t>
__global__ void adamw_fp32_state_kernel(
    model_t* model,
    const grad_t* grad,
    float* exp_avg,
    float* exp_avg_sq,
    int64_t numel,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < numel;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float moment = exp_avg[index];
    float variance = exp_avg_sq[index];
    const float weight = adamw_update(
        load_model(model, index), load_grad(grad, index), moment, variance, beta1, beta2,
        effective_lr, weight_decay, eps, step_size, bias_correction2_sqrt);
    exp_avg[index] = moment;
    exp_avg_sq[index] = variance;
    store_model(model, index, weight);
  }
}

template <typename model_t, typename grad_t>
void launch_adamw_8bit(
    torch::Tensor model,
    torch::Tensor grad,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scale,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_scale,
    torch::Tensor signed_codebook,
    torch::Tensor unsigned_codebook,
    int64_t quant_block_size,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((model.numel() + quant_block_size - 1) / quant_block_size);
  const auto stream = at::cuda::getCurrentCUDAStream();
  adamw_8bit_blockwise_kernel<model_t, grad_t><<<blocks, threads, 0, stream>>>(
      model.data_ptr<model_t>(),
      grad.data_ptr<grad_t>(),
      exp_avg_q.data_ptr<uint8_t>(),
      exp_avg_scale.data_ptr<float>(),
      exp_avg_sq_q.data_ptr<uint8_t>(),
      exp_avg_sq_scale.data_ptr<float>(),
      signed_codebook.data_ptr<float>(),
      unsigned_codebook.data_ptr<float>(),
      model.numel(),
      quant_block_size,
      beta1,
      beta2,
      effective_lr,
      weight_decay,
      eps,
      step_size,
      bias_correction2_sqrt);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename model_t, typename grad_t>
void launch_adamw_fp32_state(
    torch::Tensor model,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    float beta1,
    float beta2,
    float effective_lr,
    float weight_decay,
    float eps,
    float step_size,
    float bias_correction2_sqrt) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((model.numel() + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream();
  adamw_fp32_state_kernel<model_t, grad_t><<<blocks, threads, 0, stream>>>(
      model.data_ptr<model_t>(), grad.data_ptr<grad_t>(), exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
      model.numel(), beta1, beta2, effective_lr, weight_decay, eps, step_size, bias_correction2_sqrt);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void areno_adamw_fp32_master_step_cuda(
    torch::Tensor model,
    torch::Tensor low_bits,
    torch::Tensor round_up_bits,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int64_t state_offset,
    double beta1,
    double beta2,
    double effective_lr,
    double weight_decay,
    double eps,
    double step_size,
    double bias_correction2_sqrt) {
  c10::cuda::CUDAGuard guard(model.device());
  TORCH_CHECK(model.is_cuda() && grad.is_cuda(), "model and gradient must be CUDA tensors");
  TORCH_CHECK(model.is_contiguous() && grad.is_contiguous(), "model and gradient must be contiguous");
  TORCH_CHECK(model.numel() == grad.numel(), "model and gradient sizes must match");
  if (grad.scalar_type() == at::kBFloat16) {
    launch_adamw<at::BFloat16>(
        model, low_bits, round_up_bits, grad, exp_avg, exp_avg_sq, state_offset, beta1, beta2,
        effective_lr, weight_decay, eps, step_size, bias_correction2_sqrt);
  } else if (grad.scalar_type() == at::kFloat) {
    launch_adamw<float>(
        model, low_bits, round_up_bits, grad, exp_avg, exp_avg_sq, state_offset, beta1, beta2,
        effective_lr, weight_decay, eps, step_size, bias_correction2_sqrt);
  } else {
    TORCH_CHECK(false, "gradient must be bfloat16 or float32");
  }
}

void areno_adamw_4bit_step_cuda(
    torch::Tensor model,
    torch::Tensor grad,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scale,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_scale,
    int64_t packed_offset,
    int64_t moment_scale_offset,
    int64_t variance_scale_offset,
    int64_t quant_block_size,
    double beta1,
    double beta2,
    double effective_lr,
    double weight_decay,
    double eps,
    double step_size,
    double bias_correction2_sqrt) {
  c10::cuda::CUDAGuard guard(model.device());
  TORCH_CHECK(
      model.is_cuda() && grad.is_cuda() && exp_avg_q.is_cuda() && exp_avg_scale.is_cuda() &&
          exp_avg_sq_q.is_cuda() && exp_avg_sq_scale.is_cuda(),
      "AdamW4bit tensors must be CUDA tensors");
  TORCH_CHECK(model.is_contiguous() && grad.is_contiguous(), "model and gradient must be contiguous");
  TORCH_CHECK(model.numel() == grad.numel(), "model and gradient sizes must match");
  TORCH_CHECK(
      quant_block_size >= 32 && quant_block_size <= 1024 &&
          (quant_block_size & (quant_block_size - 1)) == 0,
      "AdamW4bit block size must be a power of two between 32 and 1024");

#define LAUNCH_ADAMW4(MODEL_T, GRAD_T)                                                                    \
  launch_adamw_4bit<MODEL_T, GRAD_T>(                                                                     \
      model, grad, exp_avg_q, exp_avg_scale, exp_avg_sq_q, exp_avg_sq_scale, packed_offset,               \
      moment_scale_offset, variance_scale_offset,                                                          \
      quant_block_size, beta1, beta2, effective_lr, weight_decay, eps, step_size, bias_correction2_sqrt)

  if (model.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kBFloat16) {
    LAUNCH_ADAMW4(at::BFloat16, at::BFloat16);
  } else if (model.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kFloat) {
    LAUNCH_ADAMW4(at::BFloat16, float);
  } else if (model.scalar_type() == at::kFloat && grad.scalar_type() == at::kBFloat16) {
    LAUNCH_ADAMW4(float, at::BFloat16);
  } else if (model.scalar_type() == at::kFloat && grad.scalar_type() == at::kFloat) {
    LAUNCH_ADAMW4(float, float);
  } else {
    TORCH_CHECK(false, "model and gradient must be bfloat16 or float32");
  }
#undef LAUNCH_ADAMW4
}

void areno_adamw_4bit_rank1_stats_cuda(
    torch::Tensor grad,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor previous_scales,
    torch::Tensor updated_scales,
    torch::Tensor invalid,
    torch::Tensor shape,
    torch::Tensor strides,
    int64_t packed_offset,
    int64_t parameter_shard_start,
    int64_t quant_block_size,
    double beta2,
    bool has_state) {
  c10::cuda::CUDAGuard guard(grad.device());
  TORCH_CHECK(
      grad.is_cuda() && exp_avg_sq_q.is_cuda() && previous_scales.is_cuda() && updated_scales.is_cuda() &&
          invalid.is_cuda() && shape.is_cuda() && strides.is_cuda(),
      "AdamW4bit rank-1 statistics tensors must be CUDA tensors");
  TORCH_CHECK(
      quant_block_size >= 32 && quant_block_size <= 1024 &&
          (quant_block_size & (quant_block_size - 1)) == 0,
      "AdamW4bit block size must be a power of two between 32 and 1024");
  if (grad.scalar_type() == at::kBFloat16) {
    launch_adamw_4bit_rank1_stats<at::BFloat16>(
        grad, exp_avg_sq_q, previous_scales, updated_scales, invalid, shape, strides, packed_offset,
        parameter_shard_start, quant_block_size, beta2, has_state);
  } else if (grad.scalar_type() == at::kFloat) {
    launch_adamw_4bit_rank1_stats<float>(
        grad, exp_avg_sq_q, previous_scales, updated_scales, invalid, shape, strides, packed_offset,
        parameter_shard_start, quant_block_size, beta2, has_state);
  } else {
    TORCH_CHECK(false, "AdamW4bit rank-1 gradient must be bfloat16 or float32");
  }
}

void areno_adamw_4bit_rank1_step_cuda(
    torch::Tensor model,
    torch::Tensor grad,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scale,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor previous_scales,
    torch::Tensor updated_scales,
    torch::Tensor invalid,
    torch::Tensor shape,
    torch::Tensor strides,
    int64_t packed_offset,
    int64_t moment_scale_offset,
    int64_t parameter_shard_start,
    int64_t quant_block_size,
    double beta1,
    double beta2,
    double effective_lr,
    double weight_decay,
    double eps,
    double step_size,
    double bias_correction2_sqrt) {
  c10::cuda::CUDAGuard guard(model.device());
#define LAUNCH_ADAMW4_RANK1(MODEL_T, GRAD_T)                                                              \
  launch_adamw_4bit_rank1_step<MODEL_T, GRAD_T>(                                                          \
      model, grad, exp_avg_q, exp_avg_scale, exp_avg_sq_q, previous_scales, updated_scales, invalid,      \
      shape, strides, packed_offset, moment_scale_offset, parameter_shard_start, quant_block_size, beta1, \
      beta2, effective_lr, weight_decay, eps, step_size, bias_correction2_sqrt)
  if (model.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kBFloat16) {
    LAUNCH_ADAMW4_RANK1(at::BFloat16, at::BFloat16);
  } else if (model.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kFloat) {
    LAUNCH_ADAMW4_RANK1(at::BFloat16, float);
  } else if (model.scalar_type() == at::kFloat && grad.scalar_type() == at::kBFloat16) {
    LAUNCH_ADAMW4_RANK1(float, at::BFloat16);
  } else if (model.scalar_type() == at::kFloat && grad.scalar_type() == at::kFloat) {
    LAUNCH_ADAMW4_RANK1(float, float);
  } else {
    TORCH_CHECK(false, "AdamW4bit rank-1 model and gradient must be bfloat16 or float32");
  }
#undef LAUNCH_ADAMW4_RANK1
}

void areno_adamw_8bit_step_cuda(
    torch::Tensor model,
    torch::Tensor grad,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_scale,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_scale,
    torch::Tensor signed_codebook,
    torch::Tensor unsigned_codebook,
    int64_t quant_block_size,
    double beta1,
    double beta2,
    double effective_lr,
    double weight_decay,
    double eps,
    double step_size,
    double bias_correction2_sqrt) {
  c10::cuda::CUDAGuard guard(model.device());
  TORCH_CHECK(
      model.is_cuda() && grad.is_cuda() && exp_avg_q.is_cuda() && exp_avg_scale.is_cuda() &&
          exp_avg_sq_q.is_cuda() && exp_avg_sq_scale.is_cuda() && signed_codebook.is_cuda() &&
          unsigned_codebook.is_cuda(),
      "all 8-bit AdamW inputs must be CUDA tensors");
  TORCH_CHECK(
      model.is_contiguous() && grad.is_contiguous() && exp_avg_q.is_contiguous() && exp_avg_scale.is_contiguous() &&
          exp_avg_sq_q.is_contiguous() && exp_avg_sq_scale.is_contiguous() && signed_codebook.is_contiguous() &&
          unsigned_codebook.is_contiguous(),
      "all 8-bit AdamW inputs must be contiguous");
  TORCH_CHECK(model.numel() == grad.numel(), "model and gradient sizes must match");
  TORCH_CHECK(model.numel() == exp_avg_q.numel(), "model and first-moment sizes must match");
  TORCH_CHECK(model.numel() == exp_avg_sq_q.numel(), "model and second-moment sizes must match");
  TORCH_CHECK(quant_block_size >= 1 && quant_block_size <= 4096, "quantization block size must be in [1, 4096]");
  const int64_t block_count = (model.numel() + quant_block_size - 1) / quant_block_size;
  TORCH_CHECK(exp_avg_scale.numel() == block_count, "first-moment scale count must match quantization blocks");
  TORCH_CHECK(exp_avg_sq_scale.numel() == block_count, "second-moment scale count must match quantization blocks");
  TORCH_CHECK(signed_codebook.numel() == 256, "signed dynamic codebook must have 256 entries");
  TORCH_CHECK(unsigned_codebook.numel() == 256, "unsigned dynamic codebook must have 256 entries");

#define LAUNCH_ADAMW8(MODEL_T, GRAD_T)                                                              \
  launch_adamw_8bit<MODEL_T, GRAD_T>(                                                               \
      model, grad, exp_avg_q, exp_avg_scale, exp_avg_sq_q, exp_avg_sq_scale, signed_codebook,        \
      unsigned_codebook, quant_block_size, beta1, beta2, effective_lr, weight_decay, eps, step_size, \
      bias_correction2_sqrt)

  if (model.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kBFloat16) {
    LAUNCH_ADAMW8(at::BFloat16, at::BFloat16);
  } else if (model.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kFloat) {
    LAUNCH_ADAMW8(at::BFloat16, float);
  } else if (model.scalar_type() == at::kFloat && grad.scalar_type() == at::kBFloat16) {
    LAUNCH_ADAMW8(float, at::BFloat16);
  } else if (model.scalar_type() == at::kFloat && grad.scalar_type() == at::kFloat) {
    LAUNCH_ADAMW8(float, float);
  } else {
    TORCH_CHECK(false, "model and gradient must be bfloat16 or float32");
  }
#undef LAUNCH_ADAMW8
}

void areno_adamw_fp32_state_step_cuda(
    torch::Tensor model,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    double beta1,
    double beta2,
    double effective_lr,
    double weight_decay,
    double eps,
    double step_size,
    double bias_correction2_sqrt) {
  c10::cuda::CUDAGuard guard(model.device());
  TORCH_CHECK(
      model.is_cuda() && grad.is_cuda() && exp_avg.is_cuda() && exp_avg_sq.is_cuda(),
      "all FP32-state AdamW inputs must be CUDA tensors");
  TORCH_CHECK(
      model.is_contiguous() && grad.is_contiguous() && exp_avg.is_contiguous() && exp_avg_sq.is_contiguous(),
      "all FP32-state AdamW inputs must be contiguous");
  TORCH_CHECK(model.numel() == grad.numel(), "model and gradient sizes must match");
  TORCH_CHECK(model.numel() == exp_avg.numel(), "model and first-moment sizes must match");
  TORCH_CHECK(model.numel() == exp_avg_sq.numel(), "model and second-moment sizes must match");

#define LAUNCH_ADAMW_FP32_STATE(MODEL_T, GRAD_T)                                                     \
  launch_adamw_fp32_state<MODEL_T, GRAD_T>(                                                         \
      model, grad, exp_avg, exp_avg_sq, beta1, beta2, effective_lr, weight_decay, eps, step_size,    \
      bias_correction2_sqrt)

  if (model.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kBFloat16) {
    LAUNCH_ADAMW_FP32_STATE(at::BFloat16, at::BFloat16);
  } else if (model.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kFloat) {
    LAUNCH_ADAMW_FP32_STATE(at::BFloat16, float);
  } else if (model.scalar_type() == at::kFloat && grad.scalar_type() == at::kBFloat16) {
    LAUNCH_ADAMW_FP32_STATE(float, at::BFloat16);
  } else if (model.scalar_type() == at::kFloat && grad.scalar_type() == at::kFloat) {
    LAUNCH_ADAMW_FP32_STATE(float, float);
  } else {
    TORCH_CHECK(false, "model and gradient must be bfloat16 or float32");
  }
#undef LAUNCH_ADAMW_FP32_STATE
}
