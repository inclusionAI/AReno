#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <vector>

#include "atomic_utils.cuh"
#include "moe_permute_common.h"

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

constexpr int kTopkTile = 512;

template <int ExpertTile>
__global__ void moe_topk_tile_rank_kernel(
    const int64_t* __restrict__ topk_idx,
    const float* __restrict__ topk_weight,
    int32_t* __restrict__ route_rank,
    int32_t* __restrict__ tile_offsets,
    int32_t* __restrict__ tokens_per_expert,
    int64_t routes,
    int tiles,
    int local_expert_start,
    int local_num_experts) {
  constexpr int warps = kTopkTile / 32;
  __shared__ int32_t warp_counts[warps * ExpertTile];
  for (int index = threadIdx.x; index < warps * ExpertTile; index += blockDim.x) {
    warp_counts[index] = 0;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  const int expert_begin = blockIdx.y * ExpertTile;
  const int64_t route = static_cast<int64_t>(blockIdx.x) * kTopkTile + threadIdx.x;
  int expert = ExpertTile;  // One shared sentinel for all invalid lanes.
  if (route < routes) {
    int local_expert = static_cast<int>(topk_idx[route]) - local_expert_start;
    if (local_expert >= expert_begin && local_expert < expert_begin + ExpertTile &&
        local_expert < local_num_experts && topk_weight[route] != 0.0f) {
      expert = local_expert - expert_begin;
    }
  }
  unsigned peers;
#if __CUDA_ARCH__ >= 700
  peers = __match_any_sync(0xffffffffu, expert);
#else
  // Same peer groups on older CUDA architectures without match_any.
  peers = 0xffffffffu;
  for (int bit = 1; bit <= ExpertTile; bit <<= 1) {
    unsigned mask = __ballot_sync(0xffffffffu, (expert & bit) != 0);
    peers &= (expert & bit) ? mask : ~mask;
  }
#endif
  if (expert < ExpertTile && lane == __ffs(peers) - 1) {
    warp_counts[warp * ExpertTile + expert] = __popc(peers);
  }
  __syncthreads();
  if (expert < ExpertTile) {
    int rank = __popc(peers & ((1u << lane) - 1));
    for (int previous = 0; previous < warp; ++previous) {
      rank += warp_counts[previous * ExpertTile + expert];
    }
    route_rank[route] = rank;
  }
  if (threadIdx.x < ExpertTile && expert_begin + threadIdx.x < local_num_experts) {
    int count = 0;
#pragma unroll
    for (int previous = 0; previous < warps; ++previous) {
      count += warp_counts[previous * ExpertTile + threadIdx.x];
    }
    int local_expert = expert_begin + threadIdx.x;
    tile_offsets[static_cast<int64_t>(local_expert) * tiles + blockIdx.x] = tiles == 1 ? 0 : count;
    if (tiles == 1) {
      tokens_per_expert[local_expert] = count;
    }
  }
}

__global__ void moe_topk_tile_prefix_kernel(
    int32_t* __restrict__ tile_offsets,
    int32_t* __restrict__ tokens_per_expert,
    int tiles) {
  const int lane = threadIdx.x;
  int total = 0;
  for (int begin = 0; begin < tiles; begin += 32) {
    int tile = begin + lane;
    int64_t index = static_cast<int64_t>(blockIdx.x) * tiles + tile;
    int count = tile < tiles ? tile_offsets[index] : 0;
    int inclusive = count;
#pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
      int previous = __shfl_up_sync(0xffffffffu, inclusive, delta);
      if (lane >= delta) {
        inclusive += previous;
      }
    }
    if (tile < tiles) {
      tile_offsets[index] = total + inclusive - count;
    }
    total += __shfl_sync(0xffffffffu, inclusive, 31);
  }
  if (lane == 0) {
    tokens_per_expert[blockIdx.x] = total;
  }
}

template <typename scalar_t>
__global__ void moe_topk_permute_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ topk_idx,
    const float* __restrict__ topk_weight,
    const int32_t* __restrict__ route_rank,
    const int32_t* __restrict__ tile_offsets,
    const int64_t* __restrict__ offsets,
    scalar_t* __restrict__ output,
    float* __restrict__ route_weight,
    int64_t* __restrict__ token_index,
    int32_t* __restrict__ topk_position,
    int64_t route_count,
    int tiles,
    int top_k,
    int hidden,
    int local_expert_start,
    int local_num_experts) {
  for (int64_t route = blockIdx.x; route < route_count; route += gridDim.x) {
    int expert = static_cast<int>(topk_idx[route]) - local_expert_start;
    float weight = topk_weight[route];
    if (expert < 0 || expert >= local_num_experts || weight == 0.0f) {
      continue;
    }
    int64_t row = offsets[expert] + tile_offsets[static_cast<int64_t>(expert) * tiles + route / kTopkTile] + route_rank[route];
    int64_t token = route / top_k;
    if (threadIdx.x == 0) {
      route_weight[row] = weight;
      token_index[row] = token;
      topk_position[row] = static_cast<int32_t>(route % top_k);
    }
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
      output[row * hidden + col] = input[token * hidden + col];
    }
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
  auto tokens_per_expert_i32 = torch::empty({local_num_experts}, input.options().dtype(torch::kInt32));

  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t route_count = static_cast<int64_t>(tokens) * top_k;
  if (route_count == 0 || local_num_experts == 0) {
    auto empty_output = torch::empty({0, hidden}, input.options());
    auto empty_weight = torch::empty({0}, input.options().dtype(torch::kFloat32));
    auto empty_token_index = torch::empty({0}, input.options().dtype(torch::kInt64));
    auto empty_topk_position = torch::empty({0}, input.options().dtype(torch::kInt32));
    auto empty_counts = torch::zeros({local_num_experts}, input.options().dtype(torch::kInt64));
    return {empty_output, empty_weight, empty_token_index, empty_topk_position, empty_counts};
  }
  const int tiles = static_cast<int>((route_count + areno_accel::kTopkTile - 1) / areno_accel::kTopkTile);
  auto route_rank = torch::empty({route_count}, input.options().dtype(torch::kInt32));
  auto tile_offsets = torch::empty({local_num_experts, tiles}, input.options().dtype(torch::kInt32));
  if (local_num_experts <= 32) {
    areno_accel::moe_topk_tile_rank_kernel<32><<<dim3(tiles, 1), areno_accel::kTopkTile, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(), topk_weight.data_ptr<float>(), route_rank.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(), tokens_per_expert_i32.data_ptr<int32_t>(), route_count, tiles,
        static_cast<int>(local_expert_start), static_cast<int>(local_num_experts));
  } else {
    areno_accel::moe_topk_tile_rank_kernel<128><<<dim3(tiles, (local_num_experts + 127) / 128), areno_accel::kTopkTile, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(), topk_weight.data_ptr<float>(), route_rank.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(), tokens_per_expert_i32.data_ptr<int32_t>(), route_count, tiles,
        static_cast<int>(local_expert_start), static_cast<int>(local_num_experts));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (tiles > 1) {
    areno_accel::moe_topk_tile_prefix_kernel<<<local_num_experts, 32, 0, stream>>>(
        tile_offsets.data_ptr<int32_t>(), tokens_per_expert_i32.data_ptr<int32_t>(), tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto buffers = areno_accel::moe::allocate_topk(input, tokens_per_expert_i32);
  auto& output = buffers.output;
  auto& route_weight = buffers.weight;
  auto& token_index = buffers.token_index;
  auto& topk_position = buffers.position;
  if (output.size(0) == 0) {
    return buffers.result();
  }
  int blocks = static_cast<int>(std::min<int64_t>(route_count, 65535));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_moe_topk_permute_forward", [&] {
    areno_accel::moe_topk_permute_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        topk_idx.data_ptr<int64_t>(),
        topk_weight.data_ptr<float>(),
        route_rank.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        buffers.offsets.data_ptr<int64_t>(),
        output.data_ptr<scalar_t>(),
        route_weight.data_ptr<float>(),
        token_index.data_ptr<int64_t>(),
        topk_position.data_ptr<int32_t>(),
        route_count,
        tiles,
        top_k,
        hidden,
        static_cast<int>(local_expert_start),
        static_cast<int>(local_num_experts));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return buffers.result();
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
