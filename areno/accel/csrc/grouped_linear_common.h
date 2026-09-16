#pragma once
#include <ATen/ATen.h>
#include <vector>

// Shared host-side grouping from linear.cu. Device adapters supply validation
// and GEMM; allocation, expert slices, empty groups and gradient selection are
// identical on every device. No device kernels or backend imports live here.
namespace areno_accel {
namespace grouped_linear {

inline void check_layout(const at::Tensor& input, const at::Tensor& weight, const std::vector<int64_t>& counts) {
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
              "areno_grouped_linear supports FP32, FP16 or BF16");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "areno_grouped_linear input and weight dtype must match");
  TORCH_CHECK(input.dim() == 2, "areno_grouped_linear input must be 2D");
  TORCH_CHECK(weight.dim() == 3, "areno_grouped_linear weight must be 3D");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "areno_grouped_linear native inputs must be contiguous");
  TORCH_CHECK(input.size(1) == weight.size(2), "areno_grouped_linear input and weight shape mismatch");
  TORCH_CHECK(static_cast<int64_t>(counts.size()) == weight.size(0), "tokens_per_expert must match expert count");
  int64_t total = 0;
  for (int64_t count : counts) {
    TORCH_CHECK(count >= 0, "tokens_per_expert values must be non-negative");
    // Bound before adding, so hostile or corrupt counts cannot overflow.
    TORCH_CHECK(count <= input.size(0) - total, "tokens_per_expert sum must match input rows");
    total += count;
  }
  TORCH_CHECK(total == input.size(0), "tokens_per_expert sum must match input rows");
}

inline std::vector<int64_t> host_counts(const at::Tensor& counts, const at::Tensor& input, const at::Tensor& weight) {
  TORCH_CHECK(counts.device() == input.device(), "tokens_per_expert must be on the input device");
  TORCH_CHECK(counts.dim() == 1, "areno_grouped_linear tokens_per_expert must be 1D");
  TORCH_CHECK(counts.scalar_type() == at::kLong || counts.scalar_type() == at::kInt,
              "areno_grouped_linear tokens_per_expert must be int32 or int64");
  TORCH_CHECK(weight.dim() == 3 && counts.numel() == weight.size(0), "tokens_per_expert must match expert count");
  // Preserve the existing CUDA counts entry's synchronization contract. This
  // is shared metadata handling, not a device-specific numerical fallback.
  auto cpu = counts.to(at::kCPU).contiguous();
  std::vector<int64_t> result(static_cast<size_t>(cpu.numel()));
  for (int64_t i = 0; i < cpu.numel(); ++i) {
    result[i] = cpu.scalar_type() == at::kLong ? cpu.const_data_ptr<int64_t>()[i] : cpu.const_data_ptr<int32_t>()[i];
  }
  return result;
}

template <typename CheckTensor, typename Matmul>
at::Tensor forward(const at::Tensor& input, const at::Tensor& weight, const std::vector<int64_t>& counts,
                    CheckTensor check_tensor, Matmul mm) {
  check_tensor(input, input);
  check_tensor(weight, input);
  check_layout(input, weight, counts);
  const int64_t k = input.size(1), n = weight.size(1);
  auto output = at::empty({input.size(0), n}, input.options());
  int64_t offset = 0;
  for (int64_t expert = 0; expert < weight.size(0); ++expert) {
    int64_t m = counts[expert];
    if (m > 0) {
      auto x = input.narrow(0, offset, m), w = weight.select(0, expert), y = output.narrow(0, offset, m);
      if (y.numel() > 0) {
        if (k == 0) y.zero_();
        else mm(x, w, y, m, n, k, false, true);
      }
    }
    offset += m;
  }
  return output;
}

template <typename CheckTensor, typename Matmul>
std::vector<at::Tensor> backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
    const std::vector<int64_t>& counts, bool need_input, bool need_weight, CheckTensor check_tensor, Matmul mm) {
  check_tensor(input, input);
  check_tensor(weight, input);
  check_tensor(grad, input);
  check_layout(input, weight, counts);
  TORCH_CHECK(grad.scalar_type() == input.scalar_type(), "areno_grouped_linear grad dtype must match input");
  TORCH_CHECK(grad.dim() == 2 && grad.size(0) == input.size(0) && grad.size(1) == weight.size(1),
              "areno_grouped_linear gradient shape mismatch");
  TORCH_CHECK(grad.is_contiguous(), "areno_grouped_linear native gradient must be contiguous");
  auto dx = need_input ? at::empty(input.sizes(), input.options()) : at::empty({0}, input.options());
  auto dw = need_weight ? at::zeros(weight.sizes(), weight.options()) : at::empty({0}, weight.options());
  const int64_t k = input.size(1), n = weight.size(1);
  int64_t offset = 0;
  for (int64_t expert = 0; expert < weight.size(0); ++expert) {
    int64_t m = counts[expert];
    if (m > 0) {
      auto g = grad.narrow(0, offset, m);
      if (need_input) {
        auto w = weight.select(0, expert), local_dx = dx.narrow(0, offset, m);
        if (local_dx.numel() > 0) {
          if (n == 0) local_dx.zero_();
          else mm(g, w, local_dx, m, k, n, false, false);
        }
      }
      if (need_weight) {
        auto x = input.narrow(0, offset, m), local_dw = dw.select(0, expert);
        if (local_dw.numel() > 0) mm(g, x, local_dw, n, k, m, true, false);
      }
    }
    offset += m;
  }
  return {dx, dw};
}
} // namespace grouped_linear
} // namespace areno_accel
