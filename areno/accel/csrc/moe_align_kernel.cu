#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace areno_accel {

constexpr int kWarp = 32;
constexpr int kRouteThreads = 1024;
constexpr int kFillThreads = 256;

__host__ __device__ __forceinline__ int div_up(int x, int y) {
  return (x + y - 1) / y;
}

int next_power_of_two(int value) {
  int out = 1;
  while (out < value) {
    out <<= 1;
  }
  return out;
}

__device__ __forceinline__ int warp_exclusive_sum(int value) {
  int original = value;
#pragma unroll
  for (int offset = 1; offset < kWarp; offset <<= 1) {
    int peer = __shfl_up_sync(0xffffffffu, value, offset);
    if ((threadIdx.x & (kWarp - 1)) >= offset) {
      value += peer;
    }
  }
  return value - original;
}

template <typename id_t>
__global__ void route_scatter_kernel(
    const id_t* __restrict__ expert_for_token,
    int32_t* __restrict__ routed_token_ids,
    int32_t* __restrict__ padded_offsets,
    int total_routes) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int route = lane; route < total_routes; route += stride) {
    int expert = static_cast<int>(expert_for_token[route]) + 1;
    int slot = atomicAdd(padded_offsets + expert, 1);
    routed_token_ids[slot] = route;
  }
}

template <typename id_t>
__global__ void general_route_plan_kernel(
    const id_t* __restrict__ expert_for_token,
    int32_t* __restrict__ routed_token_ids,
    int32_t* __restrict__ block_expert_ids,
    int32_t* __restrict__ total_padded_routes,
    int32_t* __restrict__ padded_offsets,
    int expert_slots,
    int block_size,
    int total_routes,
    int scan_width,
    int routed_capacity,
    bool initialize_routed_ids) {
  if (blockIdx.x == 1) {
    if (initialize_routed_ids) {
      for (int idx = threadIdx.x; idx < routed_capacity; idx += blockDim.x) {
        routed_token_ids[idx] = total_routes;
      }
    }
    return;
  }

  extern __shared__ int32_t shared[];
  int32_t* route_counts = shared;
  int32_t* exclusive_offsets = route_counts + expert_slots;
  int32_t* scan = exclusive_offsets + expert_slots + 1;
  int32_t* warp_totals = scan + scan_width;
  __shared__ int32_t padded_total;

  const int tid = threadIdx.x;
  if (tid < expert_slots) {
    route_counts[tid] = 0;
  }
  __syncthreads();

  for (int route = tid; route < total_routes; route += blockDim.x) {
    int expert = static_cast<int>(expert_for_token[route]) + 1;
    atomicAdd(route_counts + expert, 1);
  }
  __syncthreads();

  int padded_count = 0;
  if (tid < expert_slots) {
    padded_count = div_up(route_counts[tid], block_size) * block_size;
    scan[tid] = padded_count;
  }

  const int warp_id = tid / kWarp;
  const int lane = tid & (kWarp - 1);
  const int scan_warps = div_up(scan_width, kWarp);

  int inclusive = warp_exclusive_sum(padded_count) + padded_count;
  if (lane == kWarp - 1) {
    warp_totals[warp_id] = inclusive;
  }
  __syncthreads();

  if (tid < kWarp) {
    int value = tid < scan_warps ? warp_totals[tid] : 0;
    warp_totals[tid] = warp_exclusive_sum(value) + value;
  }
  __syncthreads();

  if (tid == 0) {
    exclusive_offsets[expert_slots] = warp_totals[scan_warps - 1];
    padded_total = exclusive_offsets[expert_slots];
    *total_padded_routes = padded_total;
  }
  __syncthreads();

  if (tid >= expert_slots && tid < scan_width) {
    scan[tid] = 0;
  }
  __syncthreads();

  int value = tid < scan_width ? scan[tid] : 0;
  int local_prefix = warp_exclusive_sum(value);
  if (lane == kWarp - 1) {
    warp_totals[warp_id] = local_prefix + value;
  }
  __syncthreads();

  if (warp_id == 0) {
    int total = lane < scan_warps ? warp_totals[lane] : 0;
    warp_totals[lane] = warp_exclusive_sum(total);
  }
  __syncthreads();

  if (tid < scan_width) {
    scan[tid] = local_prefix + warp_totals[warp_id];
  }
  __syncthreads();

  if (tid < expert_slots) {
    exclusive_offsets[tid] = scan[tid];
  }
  if (tid <= expert_slots) {
    padded_offsets[tid] = exclusive_offsets[tid];
  }

  int blocks = padded_total / block_size;
  for (int block = tid; block < blocks; block += blockDim.x) {
    int route_start = block * block_size;
    int lo = 0;
    int hi = expert_slots;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (exclusive_offsets[mid] <= route_start) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    block_expert_ids[block] = lo - 2;
  }
}

template <typename id_t>
__global__ void small_route_plan_kernel(
    const id_t* __restrict__ expert_for_token,
    int32_t* __restrict__ routed_token_ids,
    int32_t* __restrict__ block_expert_ids,
    int32_t* __restrict__ total_padded_routes,
    int expert_slots,
    int block_size,
    int total_routes,
    bool initialize_routed_ids,
    int routed_capacity) {
  if (threadIdx.x < kFillThreads) {
    if (initialize_routed_ids) {
      for (int idx = threadIdx.x; idx < routed_capacity; idx += kFillThreads) {
        routed_token_ids[idx] = total_routes;
      }
    }
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const int worker = threadIdx.x - kFillThreads;
  const int workers = blockDim.x - kFillThreads;
  extern __shared__ int32_t scratch[];
  int32_t* route_offsets = scratch;
  int32_t* lane_counts = route_offsets + expert_slots + 1;

  for (int expert = 0; expert < expert_slots; ++expert) {
    lane_counts[(worker + 1) * expert_slots + expert] = 0;
  }
  for (int route = worker; route < total_routes; route += workers) {
    int expert = static_cast<int>(expert_for_token[route]) + 1;
    lane_counts[(worker + 1) * expert_slots + expert] += 1;
  }
  __syncthreads();

  if (worker < expert_slots) {
    lane_counts[worker] = 0;
    for (int lane = 1; lane <= workers; ++lane) {
      lane_counts[lane * expert_slots + worker] += lane_counts[(lane - 1) * expert_slots + worker];
    }
  }
  __syncthreads();

  if (worker == 0) {
    route_offsets[0] = 0;
    for (int expert = 1; expert <= expert_slots; ++expert) {
      int raw_count = lane_counts[workers * expert_slots + expert - 1];
      route_offsets[expert] = route_offsets[expert - 1] + div_up(raw_count, block_size) * block_size;
    }
    *total_padded_routes = route_offsets[expert_slots];
  }
  __syncthreads();

  if (worker < expert_slots) {
    for (int slot = route_offsets[worker]; slot < route_offsets[worker + 1]; slot += block_size) {
      block_expert_ids[slot / block_size] = worker - 1;
    }
  }

  for (int route = worker; route < total_routes; route += workers) {
    int expert = static_cast<int>(expert_for_token[route]) + 1;
    int local_index = lane_counts[worker * expert_slots + expert]++;
    routed_token_ids[route_offsets[expert] + local_index] = route;
  }
}

template <typename id_t>
void launch_route_planner(
    torch::Tensor topk_ids,
    int expert_slots,
    int block_size,
    torch::Tensor routed_token_ids,
    torch::Tensor block_expert_ids,
    torch::Tensor total_padded_routes,
    torch::Tensor padded_offsets,
    bool initialize_routed_ids) {
  const int total_routes = static_cast<int>(topk_ids.numel());
  const int routed_capacity = static_cast<int>(routed_token_ids.size(0));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (total_routes < 1024 && expert_slots <= 64) {
    int workers = std::max(expert_slots, kWarp);
    int scratch_bytes = ((workers + 1) * expert_slots + (expert_slots + 1)) * sizeof(int32_t);
    small_route_plan_kernel<id_t><<<1, kFillThreads + workers, scratch_bytes, stream>>>(
        topk_ids.data_ptr<id_t>(),
        routed_token_ids.data_ptr<int32_t>(),
        block_expert_ids.data_ptr<int32_t>(),
        total_padded_routes.data_ptr<int32_t>(),
        expert_slots,
        block_size,
        total_routes,
        initialize_routed_ids,
        routed_capacity);
    return;
  }

  int scan_width = next_power_of_two(expert_slots);
  int shared_bytes = (expert_slots + (expert_slots + 1) + scan_width + kWarp) * sizeof(int32_t);
  general_route_plan_kernel<id_t><<<2, kRouteThreads, shared_bytes, stream>>>(
      topk_ids.data_ptr<id_t>(),
      routed_token_ids.data_ptr<int32_t>(),
      block_expert_ids.data_ptr<int32_t>(),
      total_padded_routes.data_ptr<int32_t>(),
      padded_offsets.data_ptr<int32_t>(),
      expert_slots,
      block_size,
      total_routes,
      scan_width,
      routed_capacity,
      initialize_routed_ids);

  int scatter_threads = 256;
  int scatter_blocks = std::min(div_up(total_routes, scatter_threads), 65535);
  route_scatter_kernel<id_t><<<scatter_blocks, scatter_threads, 0, stream>>>(
      topk_ids.data_ptr<id_t>(),
      routed_token_ids.data_ptr<int32_t>(),
      padded_offsets.data_ptr<int32_t>(),
      total_routes);
}

}  // namespace areno_accel

void areno_moe_align_cuda(
    torch::Tensor topk_ids,
    int64_t expert_slots,
    int64_t block_size,
    torch::Tensor routed_token_ids,
    torch::Tensor block_expert_ids,
    torch::Tensor total_padded_routes,
    torch::Tensor padded_offsets,
    bool initialize_routed_ids) {
  TORCH_CHECK(topk_ids.is_cuda(), "areno_moe_align topk_ids must be CUDA");
  TORCH_CHECK(routed_token_ids.scalar_type() == at::kInt, "areno_moe_align routed_token_ids must be int32");
  TORCH_CHECK(block_expert_ids.scalar_type() == at::kInt, "areno_moe_align block_expert_ids must be int32");
  TORCH_CHECK(total_padded_routes.scalar_type() == at::kInt, "areno_moe_align total_padded_routes must be int32");
  TORCH_CHECK(padded_offsets.scalar_type() == at::kInt, "areno_moe_align padded_offsets must be int32");

  const at::cuda::OptionalCUDAGuard guard(device_of(topk_ids));
  AT_DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "areno_moe_align", [&] {
    areno_accel::launch_route_planner<scalar_t>(
        topk_ids,
        static_cast<int>(expert_slots),
        static_cast<int>(block_size),
        routed_token_ids,
        block_expert_ids,
        total_padded_routes,
        padded_offsets,
        initialize_routed_ids);
  });
}
