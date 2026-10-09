#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include "grouped_linear_common.h"

namespace {
void check_cpu(const at::Tensor& tensor, const at::Tensor&) {
  TORCH_CHECK(tensor.device().is_cpu(), "test adapter requires CPU tensors");
}

// Only the GEMM adapter is replaced in this test extension. Compile and run
// the exact production host code shared by CUDA and NPU, with real tensors.
void mm(const at::Tensor& a, const at::Tensor& b, at::Tensor& out,
        int64_t m, int64_t n, int64_t k, bool transpose_a, bool transpose_b) {
  auto left = transpose_a ? a.t() : a;
  auto right = transpose_b ? b.t() : b;
  TORCH_CHECK(left.sizes() == at::IntArrayRef({m, k}) && right.sizes() == at::IntArrayRef({k, n}),
              "shared grouped GEMM dimensions do not match slices");
  at::mm_out(out, left, right);
}

at::Tensor forward(const at::Tensor& x, const at::Tensor& w, const std::vector<int64_t>& counts) {
  return areno_accel::grouped_linear::forward(x, w, counts, check_cpu, mm);
}

std::vector<at::Tensor> backward(const at::Tensor& g, const at::Tensor& x, const at::Tensor& w,
    const std::vector<int64_t>& counts, bool need_x, bool need_w) {
  return areno_accel::grouped_linear::backward(g, x, w, counts, need_x, need_w, check_cpu, mm);
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("areno_grouped_linear_forward", &forward);
  m.def("areno_grouped_linear_backward", &backward);
  m.def("areno_grouped_linear_forward_counts", [](const at::Tensor& x, const at::Tensor& w, const at::Tensor& c) {
    return forward(x, w, areno_accel::grouped_linear::host_counts(c, x, w));
  });
  m.def("areno_grouped_linear_backward_counts", [](const at::Tensor& g, const at::Tensor& x, const at::Tensor& w,
      const at::Tensor& c, bool need_x, bool need_w) {
    return backward(g, x, w, areno_accel::grouped_linear::host_counts(c, x, w), need_x, need_w);
  });
}
