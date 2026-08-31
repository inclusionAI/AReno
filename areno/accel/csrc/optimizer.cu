#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

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
