#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace areno_accel {

constexpr int kRouterThreads = 128;
constexpr int kMaxRouterExperts = 512;
constexpr int kMaxRouterGroups = 64;
constexpr int kMaxRouterTopK = 16;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
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
__global__ void grouped_topk_router_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ expert_bias,
    int64_t* __restrict__ topk_idx,
    float* __restrict__ topk_weight,
    int tokens,
    int experts,
    int top_k,
    int num_groups,
    int topk_group,
    int group_score_k) {
  extern __shared__ unsigned char smem[];
  float* scores = reinterpret_cast<float*>(smem);
  float* route_scores = scores + experts;
  float* group_scores = route_scores + experts;
  int* selected_groups = reinterpret_cast<int*>(group_scores + num_groups);

  int token = blockIdx.x;
  if (token >= tokens) {
    return;
  }

  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    float score = sigmoid(to_float(logits[static_cast<int64_t>(token) * experts + expert]));
    scores[expert] = score;
    route_scores[expert] = score + expert_bias[expert];
  }
  for (int group = threadIdx.x; group < num_groups; group += blockDim.x) {
    selected_groups[group] = 0;
  }
  __syncthreads();

  int experts_per_group = experts / num_groups;
  for (int group = threadIdx.x; group < num_groups; group += blockDim.x) {
    float local_values[kMaxRouterTopK];
    int local_indices[kMaxRouterTopK];
    for (int pos = 0; pos < group_score_k; ++pos) {
      local_values[pos] = -INFINITY;
      local_indices[pos] = experts;
    }
    int start = group * experts_per_group;
    for (int offset = 0; offset < experts_per_group; ++offset) {
      int expert = start + offset;
      insert_topk(route_scores[expert], expert, local_values, local_indices, group_score_k);
    }
    float group_score = 0.0f;
    for (int pos = 0; pos < group_score_k; ++pos) {
      group_score += local_values[pos];
    }
    group_scores[group] = group_score;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float group_values[kMaxRouterGroups];
    int group_indices[kMaxRouterGroups];
    for (int group = 0; group < topk_group; ++group) {
      group_values[group] = -INFINITY;
      group_indices[group] = num_groups;
    }
    for (int group = 0; group < num_groups; ++group) {
      insert_topk(group_scores[group], group, group_values, group_indices, topk_group);
    }
    for (int pos = 0; pos < topk_group; ++pos) {
      selected_groups[group_indices[pos]] = 1;
    }

    float selected_values[kMaxRouterTopK];
    int selected_indices[kMaxRouterTopK];
    for (int pos = 0; pos < top_k; ++pos) {
      selected_values[pos] = -INFINITY;
      selected_indices[pos] = 0;
    }
    for (int expert = 0; expert < experts; ++expert) {
      int group = expert / experts_per_group;
      if (selected_groups[group]) {
        insert_topk(route_scores[expert], expert, selected_values, selected_indices, top_k);
      }
    }

    float denom = 0.0f;
    for (int pos = 0; pos < top_k; ++pos) {
      denom += scores[selected_indices[pos]];
    }
    denom = denom > 1.0e-20f ? denom : 1.0e-20f;
    for (int pos = 0; pos < top_k; ++pos) {
      int expert = selected_indices[pos];
      topk_idx[static_cast<int64_t>(token) * top_k + pos] = static_cast<int64_t>(expert);
      topk_weight[static_cast<int64_t>(token) * top_k + pos] = scores[expert] / denom;
    }
  }
}

}  // namespace areno_accel

std::vector<torch::Tensor> areno_grouped_topk_router_cuda(
    torch::Tensor logits,
    torch::Tensor expert_bias,
    int64_t top_k,
    int64_t num_groups,
    int64_t topk_group) {
  TORCH_CHECK(logits.is_cuda(), "areno_grouped_topk_router logits must be CUDA");
  TORCH_CHECK(expert_bias.is_cuda(), "areno_grouped_topk_router expert_bias must be CUDA");
  TORCH_CHECK(logits.dim() == 2, "areno_grouped_topk_router logits must be 2D");
  TORCH_CHECK(expert_bias.dim() == 1, "areno_grouped_topk_router expert_bias must be 1D");
  TORCH_CHECK(expert_bias.scalar_type() == at::kFloat, "areno_grouped_topk_router expert_bias must be float32");
  int tokens = static_cast<int>(logits.size(0));
  int experts = static_cast<int>(logits.size(1));
  TORCH_CHECK(expert_bias.numel() == experts, "areno_grouped_topk_router expert_bias size mismatch");
  TORCH_CHECK(top_k > 0 && top_k <= areno_accel::kMaxRouterTopK, "areno_grouped_topk_router unsupported top_k");
  TORCH_CHECK(num_groups > 0 && num_groups <= areno_accel::kMaxRouterGroups, "areno_grouped_topk_router unsupported num_groups");
  TORCH_CHECK(topk_group > 0 && topk_group <= num_groups, "areno_grouped_topk_router unsupported topk_group");
  TORCH_CHECK(experts > 0 && experts <= areno_accel::kMaxRouterExperts, "areno_grouped_topk_router unsupported expert count");
  TORCH_CHECK(experts % num_groups == 0, "areno_grouped_topk_router experts must be divisible by num_groups");
  int group_score_k = static_cast<int>(top_k / topk_group);
  TORCH_CHECK(group_score_k > 0 && group_score_k <= areno_accel::kMaxRouterTopK, "areno_grouped_topk_router unsupported group score k");

  auto idx = torch::empty({tokens, static_cast<int>(top_k)}, logits.options().dtype(torch::kInt64));
  auto weight = torch::empty({tokens, static_cast<int>(top_k)}, logits.options().dtype(torch::kFloat32));
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  size_t smem_bytes =
      (static_cast<size_t>(experts) * 2 + static_cast<size_t>(num_groups)) * sizeof(float) +
      static_cast<size_t>(num_groups) * sizeof(int);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(), "areno_grouped_topk_router", [&] {
    areno_accel::grouped_topk_router_kernel<scalar_t><<<tokens, areno_accel::kRouterThreads, smem_bytes, stream>>>(
        logits.data_ptr<scalar_t>(),
        expert_bias.data_ptr<float>(),
        idx.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        tokens,
        experts,
        static_cast<int>(top_k),
        static_cast<int>(num_groups),
        static_cast<int>(topk_group),
        group_score_k);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {idx, weight};
}
