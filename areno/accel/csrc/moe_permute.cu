#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstring>
#include <vector>

#include "atomic_utils.cuh"

namespace areno_accel {

template <typename scalar_t>
__global__ void moe_permute_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ probs,
    const bool* __restrict__ routing_map,
    scalar_t* __restrict__ output,
    float* __restrict__ route_weight,
    int64_t* __restrict__ token_index,
    int tokens,
    int experts,
    int hidden) {
  int expert = blockIdx.x;
  int64_t out_row = 0;
  for (int prev = 0; prev < expert; ++prev) {
    for (int token = 0; token < tokens; ++token) {
      out_row += routing_map[token * experts + prev] ? 1 : 0;
    }
  }
  for (int token = 0; token < tokens; ++token) {
    if (!routing_map[token * experts + expert]) {
      continue;
    }
    if (threadIdx.x == 0) {
      route_weight[out_row] = probs[token * experts + expert];
      token_index[out_row] = token;
    }
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
      output[out_row * hidden + col] = input[token * hidden + col];
    }
    ++out_row;
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void moe_unpermute_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ token_index,
    scalar_t* __restrict__ output,
    int rows,
    int hidden) {
  int row = blockIdx.x;
  int64_t token = token_index[row];
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    atomic_add(output + token * hidden + col, input[row * hidden + col]);
  }
}

template <typename scalar_t>
__global__ void moe_gather_by_token_index_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ token_index,
    scalar_t* __restrict__ output,
    int rows,
    int hidden) {
  int row = blockIdx.x;
  int64_t token = token_index[row];
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    output[row * hidden + col] = input[token * hidden + col];
  }
}

__global__ void moe_topk_route_count_kernel(
    const int64_t* __restrict__ topk_idx,
    const float* __restrict__ topk_weight,
    int32_t* __restrict__ tokens_per_expert,
    int tokens,
    int top_k,
    int local_expert_start,
    int local_num_experts) {
  int64_t total = static_cast<int64_t>(tokens) * top_k;
  for (int64_t route = blockIdx.x * blockDim.x + threadIdx.x; route < total; route += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int expert = static_cast<int>(topk_idx[route]);
    int local_expert = expert - local_expert_start;
    if (local_expert >= 0 && local_expert < local_num_experts && topk_weight[route] != 0.0f) {
      atomicAdd(tokens_per_expert + local_expert, 1);
    }
  }
}

template <typename scalar_t>
__global__ void moe_topk_permute_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ topk_idx,
    const float* __restrict__ topk_weight,
    const int64_t* __restrict__ offsets,
    int32_t* __restrict__ counters,
    scalar_t* __restrict__ output,
    float* __restrict__ route_weight,
    int64_t* __restrict__ token_index,
    int32_t* __restrict__ topk_position,
    int tokens,
    int top_k,
    int hidden,
    int local_expert_start,
    int local_num_experts) {
  __shared__ int64_t row_s;
  __shared__ int valid_s;
  __shared__ float weight_s;
  __shared__ int token_s;
  __shared__ int topk_pos_s;
  int64_t total = static_cast<int64_t>(tokens) * top_k;
  for (int64_t route = blockIdx.x; route < total; route += gridDim.x) {
    if (threadIdx.x == 0) {
      int token = static_cast<int>(route / top_k);
      int topk_pos = static_cast<int>(route - static_cast<int64_t>(token) * top_k);
      int expert = static_cast<int>(topk_idx[route]);
      int local_expert = expert - local_expert_start;
      float weight = topk_weight[route];
      valid_s = local_expert >= 0 && local_expert < local_num_experts && weight != 0.0f;
      if (valid_s) {
        row_s = offsets[local_expert] + atomicAdd(counters + local_expert, 1);
        weight_s = weight;
        token_s = token;
        topk_pos_s = topk_pos;
        route_weight[row_s] = weight_s;
        token_index[row_s] = token_s;
        topk_position[row_s] = topk_pos_s;
      }
    }
    __syncthreads();
    if (!valid_s) {
      __syncthreads();
      continue;
    }
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
      output[row_s * hidden + col] = input[token_s * hidden + col];
    }
    __syncthreads();
  }
}

__global__ void moe_topk_weight_backward_kernel(
    const float* __restrict__ grad_route_weight,
    const int64_t* __restrict__ token_index,
    const int32_t* __restrict__ topk_position,
    float* __restrict__ grad_topk_weight,
    int rows,
    int top_k) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += blockDim.x * gridDim.x) {
    int64_t token = token_index[row];
    int pos = static_cast<int>(topk_position[row]);
    grad_topk_weight[token * top_k + pos] = grad_route_weight[row];
  }
}

}  // namespace areno_accel

std::vector<torch::Tensor> areno_moe_permute_forward_cuda(torch::Tensor input, torch::Tensor probs, torch::Tensor routing_map, int64_t num_out_tokens) {
  TORCH_CHECK(input.is_cuda(), "areno_moe_permute input must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "areno_moe_permute probs must be CUDA");
  TORCH_CHECK(routing_map.is_cuda(), "areno_moe_permute routing_map must be CUDA");
  TORCH_CHECK(input.dim() == 2, "areno_moe_permute input must be 2D");
  TORCH_CHECK(probs.dim() == 2 && routing_map.dim() == 2, "areno_moe_permute probs/routing_map must be 2D");
  TORCH_CHECK(routing_map.scalar_type() == at::kBool, "areno_moe_permute routing_map must be bool");
  TORCH_CHECK(input.size(0) == probs.size(0) && probs.sizes() == routing_map.sizes(), "areno_moe_permute shape mismatch");
  auto output = torch::empty({num_out_tokens, input.size(1)}, input.options());
  auto route_weight = torch::empty({num_out_tokens}, input.options().dtype(torch::kFloat32));
  auto token_index = torch::empty({num_out_tokens}, input.options().dtype(torch::kInt64));
  int tokens = static_cast<int>(input.size(0));
  int experts = static_cast<int>(probs.size(1));
  int hidden = static_cast<int>(input.size(1));
  int threads = 256;
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_moe_permute_forward", [&] {
    areno_accel::moe_permute_kernel<scalar_t><<<experts, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        probs.data_ptr<float>(),
        routing_map.data_ptr<bool>(),
        output.data_ptr<scalar_t>(),
        route_weight.data_ptr<float>(),
        token_index.data_ptr<int64_t>(),
        tokens,
        experts,
        hidden);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, route_weight, token_index};
}

torch::Tensor areno_moe_unpermute_forward_cuda(torch::Tensor input, torch::Tensor token_index, int64_t tokens, int64_t hidden) {
  TORCH_CHECK(input.is_cuda(), "areno_moe_unpermute input must be CUDA");
  TORCH_CHECK(token_index.is_cuda(), "areno_moe_unpermute token_index must be CUDA");
  TORCH_CHECK(token_index.scalar_type() == at::kLong, "areno_moe_unpermute token_index must be int64");
  auto output = torch::zeros({tokens, hidden}, input.options());
  int rows = static_cast<int>(input.size(0));
  int threads = 256;
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_moe_unpermute_forward", [&] {
    areno_accel::moe_unpermute_kernel<scalar_t><<<rows, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        token_index.data_ptr<int64_t>(),
        output.data_ptr<scalar_t>(),
        rows,
        static_cast<int>(hidden));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor areno_moe_gather_by_token_index_cuda(torch::Tensor input, torch::Tensor token_index) {
  TORCH_CHECK(input.is_cuda(), "areno_moe_gather_by_token_index input must be CUDA");
  TORCH_CHECK(token_index.is_cuda(), "areno_moe_gather_by_token_index token_index must be CUDA");
  auto output = torch::empty({token_index.numel(), input.size(1)}, input.options());
  int rows = static_cast<int>(token_index.numel());
  int hidden = static_cast<int>(input.size(1));
  int threads = 256;
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_moe_gather_by_token_index", [&] {
    areno_accel::moe_gather_by_token_index_kernel<scalar_t><<<rows, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        token_index.data_ptr<int64_t>(),
        output.data_ptr<scalar_t>(),
        rows,
        hidden);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> areno_moe_topk_permute_forward_cuda(
    torch::Tensor input,
    torch::Tensor topk_idx,
    torch::Tensor topk_weight,
    int64_t local_expert_start,
    int64_t local_num_experts) {
  TORCH_CHECK(input.is_cuda(), "areno_moe_topk_permute input must be CUDA");
  TORCH_CHECK(topk_idx.is_cuda(), "areno_moe_topk_permute topk_idx must be CUDA");
  TORCH_CHECK(topk_weight.is_cuda(), "areno_moe_topk_permute topk_weight must be CUDA");
  TORCH_CHECK(input.dim() == 2, "areno_moe_topk_permute input must be 2D");
  TORCH_CHECK(topk_idx.dim() == 2 && topk_weight.dim() == 2, "areno_moe_topk_permute topk tensors must be 2D");
  TORCH_CHECK(topk_idx.sizes() == topk_weight.sizes(), "areno_moe_topk_permute topk shape mismatch");
  TORCH_CHECK(topk_idx.scalar_type() == at::kLong, "areno_moe_topk_permute topk_idx must be int64");
  TORCH_CHECK(topk_weight.scalar_type() == at::kFloat, "areno_moe_topk_permute topk_weight must be float32");

  const int tokens = static_cast<int>(input.size(0));
  const int hidden = static_cast<int>(input.size(1));
  const int top_k = static_cast<int>(topk_idx.size(1));
  auto tokens_per_expert_i32 = torch::zeros({local_num_experts}, input.options().dtype(torch::kInt32));

  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t route_count = static_cast<int64_t>(tokens) * top_k;
  if (route_count == 0) {
    auto empty_output = torch::empty({0, hidden}, input.options());
    auto empty_weight = torch::empty({0}, input.options().dtype(torch::kFloat32));
    auto empty_token_index = torch::empty({0}, input.options().dtype(torch::kInt64));
    auto empty_topk_position = torch::empty({0}, input.options().dtype(torch::kInt32));
    auto empty_counts = torch::zeros({local_num_experts}, input.options().dtype(torch::kInt64));
    return {empty_output, empty_weight, empty_token_index, empty_topk_position, empty_counts};
  }
  int blocks = static_cast<int>(std::min<int64_t>((route_count + threads - 1) / threads, 4096));
  areno_accel::moe_topk_route_count_kernel<<<blocks, threads, 0, stream>>>(
      topk_idx.data_ptr<int64_t>(),
      topk_weight.data_ptr<float>(),
      tokens_per_expert_i32.data_ptr<int32_t>(),
      tokens,
      top_k,
      static_cast<int>(local_expert_start),
      static_cast<int>(local_num_experts));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto counts_cpu = tokens_per_expert_i32.to(torch::kCPU);
  std::vector<int64_t> offsets_host(static_cast<size_t>(local_num_experts) + 1, 0);
  const int32_t* counts_ptr = counts_cpu.data_ptr<int32_t>();
  for (int64_t expert = 0; expert < local_num_experts; ++expert) {
    offsets_host[static_cast<size_t>(expert) + 1] = offsets_host[static_cast<size_t>(expert)] + counts_ptr[expert];
  }
  const int64_t num_out_tokens = offsets_host.back();
  auto offsets_cpu = torch::empty({local_num_experts + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  std::memcpy(offsets_cpu.data_ptr<int64_t>(), offsets_host.data(), offsets_host.size() * sizeof(int64_t));
  auto offsets = offsets_cpu.to(input.device());
  auto tokens_per_expert = counts_cpu.to(torch::kInt64).to(input.device());
  auto counters = torch::zeros({local_num_experts}, input.options().dtype(torch::kInt32));
  auto output = torch::empty({num_out_tokens, hidden}, input.options());
  auto route_weight = torch::empty({num_out_tokens}, input.options().dtype(torch::kFloat32));
  auto token_index = torch::empty({num_out_tokens}, input.options().dtype(torch::kInt64));
  auto topk_position = torch::empty({num_out_tokens}, input.options().dtype(torch::kInt32));
  if (num_out_tokens == 0) {
    return {output, route_weight, token_index, topk_position, tokens_per_expert};
  }
  blocks = static_cast<int>(std::min<int64_t>(route_count, 65535));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_moe_topk_permute_forward", [&] {
    areno_accel::moe_topk_permute_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        topk_idx.data_ptr<int64_t>(),
        topk_weight.data_ptr<float>(),
        offsets.data_ptr<int64_t>(),
        counters.data_ptr<int32_t>(),
        output.data_ptr<scalar_t>(),
        route_weight.data_ptr<float>(),
        token_index.data_ptr<int64_t>(),
        topk_position.data_ptr<int32_t>(),
        tokens,
        top_k,
        hidden,
        static_cast<int>(local_expert_start),
        static_cast<int>(local_num_experts));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, route_weight, token_index, topk_position, tokens_per_expert};
}

torch::Tensor areno_moe_topk_weight_backward_cuda(
    torch::Tensor grad_route_weight,
    torch::Tensor token_index,
    torch::Tensor topk_position,
    int64_t tokens,
    int64_t top_k) {
  TORCH_CHECK(grad_route_weight.is_cuda(), "areno_moe_topk_weight_backward grad_route_weight must be CUDA");
  TORCH_CHECK(token_index.is_cuda(), "areno_moe_topk_weight_backward token_index must be CUDA");
  TORCH_CHECK(topk_position.is_cuda(), "areno_moe_topk_weight_backward topk_position must be CUDA");
  grad_route_weight = grad_route_weight.contiguous();
  auto grad_topk_weight = torch::zeros({tokens, top_k}, grad_route_weight.options().dtype(torch::kFloat32));
  const int rows = static_cast<int>(grad_route_weight.numel());
  if (rows == 0) {
    return grad_topk_weight;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(grad_route_weight));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((rows + threads - 1) / threads, 4096));
  areno_accel::moe_topk_weight_backward_kernel<<<blocks, threads, 0, stream>>>(
      grad_route_weight.data_ptr<float>(),
      token_index.data_ptr<int64_t>(),
      topk_position.data_ptr<int32_t>(),
      grad_topk_weight.data_ptr<float>(),
      rows,
      static_cast<int>(top_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_topk_weight;
}
