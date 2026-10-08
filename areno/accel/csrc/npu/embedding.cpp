#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <limits>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tensor_format.h"
#include "embedding_launch.h"

namespace areno_npu {
namespace {
void check_embedding_tensor(const at::Tensor& x, const at::Tensor& weight, bool contiguous) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.device() == weight.device(),
                "vocab embedding tensors must be on the same Ascend NPU device");
    TORCH_CHECK(!contiguous || x.is_contiguous(), "vocab embedding native inputs must be contiguous");
    TORCH_CHECK(is_base_format(x), "vocab embedding requires base NPU storage format");
}

void check_embedding(const at::Tensor& ids, const at::Tensor& weight, int64_t start, int64_t end, bool backward) {
    check_embedding_tensor(weight, weight, !backward);
    check_embedding_tensor(ids, weight, true);
    TORCH_CHECK(ids.scalar_type() == at::kLong, "vocab embedding ids must be int64");
    TORCH_CHECK(weight.dim() == 2, "vocab embedding weight must be 2D");
    TORCH_CHECK(weight.scalar_type() == at::kFloat || weight.scalar_type() == at::kHalf || weight.scalar_type() == at::kBFloat16,
                "vocab embedding weights must be FP32, FP16 or BF16");
    TORCH_CHECK(start <= std::numeric_limits<int64_t>::max() - weight.size(0) && end == start + weight.size(0),
                "vocab embedding vocabulary range must match weight rows");
}

void run_embedding(const at::Tensor& ids, const at::Tensor& input, at::Tensor& output, int64_t hidden,
                    int64_t start, int64_t end, bool backward) {
    if (ids.numel() == 0 || hidden == 0 || output.numel() == 0) return;
    int64_t tasks = ids.numel() * ((hidden - 1) / kEmbeddingTile + 1);
    auto stream = c10_npu::getCurrentNPUStream(ids.device().index()).stream(true);
    uint32_t storage = input.scalar_type() == at::kFloat ? 0 : input.scalar_type() == at::kHalf ? 1 : 2;
    launch_embedding(static_cast<uint32_t>(std::min<int64_t>(tasks, 32)), stream, storage, backward,
        ids.const_data_ptr<int64_t>(), input.const_data_ptr(), output.data_ptr(), ids.numel(), hidden, start, end);
}

at::Tensor embedding_forward(const at::Tensor& ids, const at::Tensor& weight, int64_t start, int64_t end) {
    check_embedding(ids, weight, start, end, false);
    const c10_npu::NPUGuard guard(weight.device());
    auto shape = ids.sizes().vec();
    shape.push_back(weight.size(1));
    auto output = at::empty(shape, weight.options());
    run_embedding(ids, weight, output, weight.size(1), start, end, false);
    return output;
}

at::Tensor embedding_backward(const at::Tensor& grad, const at::Tensor& ids, const at::Tensor& weight,
                               int64_t start, int64_t end) {
    check_embedding(ids, weight, start, end, true);
    check_embedding_tensor(grad, weight, true);
    auto shape = ids.sizes().vec();
    shape.push_back(weight.size(1));
    TORCH_CHECK(grad.scalar_type() == weight.scalar_type() && grad.sizes().vec() == shape,
                "vocab embedding gradient shape and dtype must match the forward output");
    const c10_npu::NPUGuard guard(weight.device());
    // Allocate dense local storage even when the saved weight is a strided
    // view; autograd applies this logical gradient to the original layout.
    auto result = at::zeros(weight.sizes(), weight.options());
    run_embedding(ids, grad, result, weight.size(1), start, end, true);
    return result;
}
} // namespace
} // namespace areno_npu

void register_embedding(pybind11::module_& m) {
    m.def("areno_vocab_embedding_forward", &areno_npu::embedding_forward);
    m.def("areno_vocab_embedding_backward", &areno_npu::embedding_backward);
}
