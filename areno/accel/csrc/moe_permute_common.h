#pragma once
#include <ATen/ATen.h>
#include <limits>
#include <vector>

namespace areno_accel {
namespace moe {
struct TopKBuffers {
  at::Tensor output, weight, token_index, position, counts, offsets;
  std::vector<at::Tensor> result() const { return {output, weight, token_index, position, counts}; }
};

// Keep CUDA's existing dynamic-shape contract: native route counting is
// followed by a CPU metadata copy, prefix scan and exact-size allocation.
// Both device implementations use this same host workflow.
inline TopKBuffers allocate_topk(const at::Tensor& input, const at::Tensor& counts_i32) {
  TORCH_CHECK(input.dim() == 2 && counts_i32.dim() == 1 && counts_i32.scalar_type() == at::kInt,
              "MoE top-k allocation requires a matrix and int32 counts");
  auto counts_cpu = counts_i32.to(at::kCPU).contiguous();
  auto offsets_cpu = at::empty({counts_cpu.numel() + 1}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  auto offsets = offsets_cpu.data_ptr<int64_t>();
  auto counts = counts_cpu.const_data_ptr<int32_t>();
  offsets[0] = 0;
  for (int64_t expert = 0; expert < counts_cpu.numel(); ++expert) {
    TORCH_CHECK(counts[expert] >= 0 && offsets[expert] <= std::numeric_limits<int64_t>::max() - counts[expert],
                "MoE route counts must be non-negative and fit int64");
    offsets[expert + 1] = offsets[expert] + counts[expert];
  }
  int64_t rows = offsets[counts_cpu.numel()];
  return {
      at::empty({rows, input.size(1)}, input.options()),
      at::empty({rows}, input.options().dtype(at::kFloat)),
      at::empty({rows}, input.options().dtype(at::kLong)),
      at::empty({rows}, input.options().dtype(at::kInt)),
      counts_cpu.to(at::kLong).to(input.device()),
      offsets_cpu.to(input.device()),
  };
}
} // namespace moe
} // namespace areno_accel
