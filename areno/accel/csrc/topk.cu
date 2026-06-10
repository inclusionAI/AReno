#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace areno_accel {

constexpr int kTopKThreads = 128;
constexpr int kMaxTopKExperts = 512;
constexpr int kMaxTopK = 16;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ void insert_topk(float value, int index, float* values, int* indices, int k) {
  for (int pos = 0; pos < k; ++pos) {
    if (value > values[pos] || (value == values[pos] && index < indices[pos])) {
      for (int move = k - 1; move > pos; --move) {
        values[move] = values[move - 1];
        indices[move] = indices[move - 1];
      }
      values[pos] = value;
      indices[pos] = index;
      break;
    }
  }
}

template <typename scalar_t>
__global__ void topk_softmax_forward_kernel(
    const scalar_t* __restrict__ logits,
    int64_t* __restrict__ topk_idx,
    float* __restrict__ topk_weight,
    int tokens,
    int experts,
    int top_k,
    bool renormalize) {
  extern __shared__ float smem[];
  float* probs = smem;
  int token = blockIdx.x;
  if (token >= tokens) {
    return;
  }
  __shared__ float reductions[kTopKThreads];

  float local_max = -INFINITY;
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    float value = to_float(logits[static_cast<int64_t>(token) * experts + expert]);
    local_max = fmaxf(local_max, value);
  }
  reductions[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float block_max = reductions[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    float prob = expf(to_float(logits[static_cast<int64_t>(token) * experts + expert]) - block_max);
    probs[expert] = prob;
    local_sum += prob;
  }
  reductions[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float block_sum = reductions[0];
  __syncthreads();

  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    probs[expert] = probs[expert] / fmaxf(block_sum, 1.0e-20f);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float values[kMaxTopK];
    int indices[kMaxTopK];
    for (int pos = 0; pos < top_k; ++pos) {
      values[pos] = -INFINITY;
      indices[pos] = 0;
    }
    for (int expert = 0; expert < experts; ++expert) {
      insert_topk(probs[expert], expert, values, indices, top_k);
    }
    float denom = 0.0f;
    for (int pos = 0; pos < top_k; ++pos) {
      denom += values[pos];
    }
    denom = denom > 1.0e-20f ? denom : 1.0e-20f;
    for (int pos = 0; pos < top_k; ++pos) {
      topk_idx[static_cast<int64_t>(token) * top_k + pos] = static_cast<int64_t>(indices[pos]);
      topk_weight[static_cast<int64_t>(token) * top_k + pos] = renormalize ? values[pos] / denom : values[pos];
    }
  }
}

template <typename scalar_t>
__global__ void topk_softmax_backward_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ topk_idx,
    const float* __restrict__ grad_topk_weight,
    scalar_t* __restrict__ grad_logits,
    int tokens,
    int experts,
    int top_k,
    bool renormalize) {
  extern __shared__ float smem[];
  float* probs = smem;
  float* dprob = probs + experts;
  int token = blockIdx.x;
  if (token >= tokens) {
    return;
  }
  __shared__ float reductions[kTopKThreads];

  float local_max = -INFINITY;
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    float value = to_float(logits[static_cast<int64_t>(token) * experts + expert]);
    local_max = fmaxf(local_max, value);
  }
  reductions[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float block_max = reductions[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    float prob = expf(to_float(logits[static_cast<int64_t>(token) * experts + expert]) - block_max);
    probs[expert] = prob;
    dprob[expert] = 0.0f;
    local_sum += prob;
  }
  reductions[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float block_sum = reductions[0];
  __syncthreads();
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    probs[expert] = probs[expert] / fmaxf(block_sum, 1.0e-20f);
  }
  __syncthreads();

  __shared__ float softmax_dot;
  if (threadIdx.x == 0) {
    float selected_sum = 0.0f;
    float weighted_grad_sum = 0.0f;
    for (int pos = 0; pos < top_k; ++pos) {
      int expert = static_cast<int>(topk_idx[static_cast<int64_t>(token) * top_k + pos]);
      float prob = probs[expert];
      float grad = grad_topk_weight[static_cast<int64_t>(token) * top_k + pos];
      selected_sum += prob;
      weighted_grad_sum += grad * prob;
    }
    selected_sum = selected_sum > 1.0e-20f ? selected_sum : 1.0e-20f;
    float dot = 0.0f;
    for (int pos = 0; pos < top_k; ++pos) {
      int expert = static_cast<int>(topk_idx[static_cast<int64_t>(token) * top_k + pos]);
      float prob = probs[expert];
      float grad = grad_topk_weight[static_cast<int64_t>(token) * top_k + pos];
      float dp = renormalize ? (grad * selected_sum - weighted_grad_sum) / (selected_sum * selected_sum) : grad;
      dprob[expert] += dp;
      dot += dp * prob;
    }
    softmax_dot = dot;
  }
  __syncthreads();

  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    float grad = probs[expert] * (dprob[expert] - softmax_dot);
    grad_logits[static_cast<int64_t>(token) * experts + expert] = static_cast<scalar_t>(grad);
  }
}

}  // namespace areno_accel

std::vector<torch::Tensor> areno_topk_softmax_forward_cuda(torch::Tensor logits, int64_t top_k, bool renormalize) {
  TORCH_CHECK(logits.is_cuda(), "areno_topk_softmax logits must be CUDA");
  TORCH_CHECK(logits.dim() == 2, "areno_topk_softmax logits must be 2D");
  int tokens = static_cast<int>(logits.size(0));
  int experts = static_cast<int>(logits.size(1));
  TORCH_CHECK(top_k > 0 && top_k <= areno_accel::kMaxTopK, "areno_topk_softmax unsupported top_k");
  TORCH_CHECK(experts > 0 && experts <= areno_accel::kMaxTopKExperts, "areno_topk_softmax unsupported expert count");
  auto idx = torch::empty({tokens, static_cast<int>(top_k)}, logits.options().dtype(torch::kInt64));
  auto weight = torch::empty({tokens, static_cast<int>(top_k)}, logits.options().dtype(torch::kFloat32));
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  size_t smem_bytes = static_cast<size_t>(experts) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(), "areno_topk_softmax_forward", [&] {
    areno_accel::topk_softmax_forward_kernel<scalar_t><<<tokens, areno_accel::kTopKThreads, smem_bytes, stream>>>(
        logits.data_ptr<scalar_t>(),
        idx.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        tokens,
        experts,
        static_cast<int>(top_k),
        renormalize);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {idx, weight};
}

torch::Tensor areno_topk_softmax_backward_cuda(
    torch::Tensor grad_topk_weight,
    torch::Tensor logits,
    torch::Tensor topk_idx,
    bool renormalize) {
  TORCH_CHECK(grad_topk_weight.is_cuda(), "areno_topk_softmax backward grad_topk_weight must be CUDA");
  TORCH_CHECK(logits.is_cuda(), "areno_topk_softmax backward logits must be CUDA");
  TORCH_CHECK(topk_idx.is_cuda(), "areno_topk_softmax backward topk_idx must be CUDA");
  TORCH_CHECK(logits.dim() == 2, "areno_topk_softmax backward logits must be 2D");
  TORCH_CHECK(topk_idx.scalar_type() == at::kLong, "areno_topk_softmax backward topk_idx must be int64");
  int tokens = static_cast<int>(logits.size(0));
  int experts = static_cast<int>(logits.size(1));
  int top_k = static_cast<int>(topk_idx.size(1));
  TORCH_CHECK(topk_idx.size(0) == tokens, "areno_topk_softmax backward topk_idx token mismatch");
  TORCH_CHECK(grad_topk_weight.sizes() == topk_idx.sizes(), "areno_topk_softmax backward grad shape mismatch");
  auto grad_logits = torch::empty_like(logits);
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  size_t smem_bytes = static_cast<size_t>(experts) * 2 * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(), "areno_topk_softmax_backward", [&] {
    areno_accel::topk_softmax_backward_kernel<scalar_t><<<tokens, areno_accel::kTopKThreads, smem_bytes, stream>>>(
        logits.data_ptr<scalar_t>(),
        topk_idx.data_ptr<int64_t>(),
        grad_topk_weight.data_ptr<float>(),
        grad_logits.data_ptr<scalar_t>(),
        tokens,
        experts,
        top_k,
        renormalize);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_logits;
}
