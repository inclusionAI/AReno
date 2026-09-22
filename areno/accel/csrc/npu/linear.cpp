#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <limits>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_mm.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tensor_format.h"
#include "linear_launch.h"
#include "../grouped_linear_common.h"

namespace areno_npu {
namespace {
void check_linear_tensor(const at::Tensor& tensor, const at::Tensor& input) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device() == input.device(),
                "linear tensors must be on the same Ascend NPU device");
    TORCH_CHECK(tensor.scalar_type() == input.scalar_type(), "linear tensor dtype must match input");
    TORCH_CHECK(tensor.is_contiguous(), "linear native inputs must be contiguous");
    TORCH_CHECK(is_base_format(tensor), "linear requires base NPU storage format");
}

int64_t check_linear(const at::Tensor& input, const at::Tensor& weight) {
    check_linear_tensor(input, input);
    check_linear_tensor(weight, input);
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "linear supports FP32, FP16 or BF16");
    TORCH_CHECK(input.dim() >= 2 && weight.dim() == 2 && input.size(-1) == weight.size(1),
                "linear input and weight shape mismatch");
    // Multiplying leading dimensions also handles a zero input feature count.
    int64_t rows = 1;
    for (int64_t dim = 0; dim < input.dim() - 1; ++dim) {
        TORCH_CHECK(input.size(dim) == 0 || rows <= std::numeric_limits<int64_t>::max() / input.size(dim),
                    "linear leading dimensions overflow");
        rows *= input.size(dim);
    }
    return rows;
}

struct TensorDeleter {
    void operator()(aclTensor* tensor) const { aclDestroyTensor(tensor); }
};
using AclTensor = std::unique_ptr<aclTensor, TensorDeleter>;

AclTensor matrix(const at::Tensor& data, int64_t rows, int64_t columns, bool transpose = false) {
    int64_t storage[2] = {rows, columns};
    int64_t shape[2] = {transpose ? columns : rows, transpose ? rows : columns};
    int64_t strides[2] = {transpose ? 1 : columns, transpose ? columns : 1};
    auto dtype = data.scalar_type() == at::kFloat ? ACL_FLOAT : data.scalar_type() == at::kHalf ? ACL_FLOAT16 : ACL_BF16;
    // The shared Python wrapper supplies contiguous slices. data_ptr already
    // includes their storage offset; transpose is metadata, not a layout copy.
    AclTensor result(aclCreateTensor(shape, 2, dtype, strides, 0, ACL_FORMAT_ND, storage, 2, data.data_ptr()));
    TORCH_CHECK(result, "linear aclCreateTensor failed");
    return result;
}

void mm(const at::Tensor& a, const at::Tensor& b, at::Tensor& out, int64_t m, int64_t n, int64_t k,
        bool transpose_a, bool transpose_b) {
    if (out.numel() == 0) return;
    if (k == 0) { out.zero_(); return; }
    auto left = matrix(a, transpose_a ? k : m, transpose_a ? m : k, transpose_a);
    auto right = matrix(b, transpose_b ? n : k, transpose_b ? k : n, transpose_b);
    auto result = matrix(out, m, n);
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    // KEEP_DTYPE: match CUDA's FP32 compute without silently lowering FP32
    // inputs to HF32/FP16. CANN owns the Cube implementation, like cuBLAS.
    auto status = aclnnMmGetWorkspaceSize(left.get(), right.get(), result.get(), int8_t{0}, &bytes, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "linear aclnnMmGetWorkspaceSize failed: ", status);
    TORCH_CHECK(bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), "linear workspace is too large");
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, out.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream(out.device().index()).stream(true);
    status = aclnnMm(bytes ? workspace.data_ptr() : nullptr, bytes, executor, stream);
    TORCH_CHECK(status == ACL_SUCCESS, "linear aclnnMm failed: ", status);
    // CANN releases a non-repeatable executor after phase two. Tensor metadata
    // is now disposable; workspace storage remains ordered by TorchNPU's
    // current-stream allocator, just as in its native op API adapter.
}

void bias_kernel(const at::Tensor& input, const at::Tensor& bias, at::Tensor& output,
                 int64_t rows, int64_t columns, bool backward) {
    if (output.numel() == 0) return;
    int64_t tiles = (columns - 1) / kLinearTile + 1;
    int64_t tasks = backward ? tiles : rows * tiles;
    uint32_t storage = input.scalar_type() == at::kFloat ? 0 : input.scalar_type() == at::kHalf ? 1 : 2;
    auto stream = c10_npu::getCurrentNPUStream(input.device().index()).stream(true);
    launch_linear_bias(static_cast<uint32_t>(std::min<int64_t>(tasks, 32)), stream, storage, backward,
        input.const_data_ptr(), backward ? nullptr : bias.const_data_ptr(), output.data_ptr(), rows, columns);
}

at::Tensor linear_forward(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, bool use_bias) {
    int64_t rows = check_linear(input, weight), k = weight.size(1), n = weight.size(0);
    if (use_bias) {
        check_linear_tensor(bias, input);
        TORCH_CHECK(bias.numel() == n, "linear bias size mismatch");
    }
    const c10_npu::NPUGuard guard(input.device());
    auto shape = input.sizes().vec();
    shape.back() = n;
    auto output = at::empty(shape, input.options());
    mm(input, weight, output, rows, n, k, false, true);
    // Keep CUDA's two-stage rounding: store GEMM in output dtype, then bias.
    if (use_bias) bias_kernel(output, bias, output, rows, n, false);
    return output;
}

std::vector<at::Tensor> linear_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
    bool use_bias, bool need_input, bool need_weight, bool need_bias) {
    int64_t rows = check_linear(input, weight), k = weight.size(1), n = weight.size(0);
    check_linear_tensor(grad, input);
    auto shape = input.sizes().vec();
    shape.back() = n;
    TORCH_CHECK(grad.sizes().vec() == shape, "linear gradient shape mismatch");
    const c10_npu::NPUGuard guard(input.device());
    auto dx = need_input ? at::empty(input.sizes(), input.options()) : at::empty({0}, input.options());
    auto dw = need_weight ? at::empty(weight.sizes(), weight.options()) : at::empty({0}, weight.options());
    auto db = at::empty({need_bias ? n : 0}, grad.options());
    if (need_input) mm(grad, weight, dx, rows, k, n, false, false);
    if (need_weight) mm(grad, input, dw, n, k, rows, true, false);
    if (need_bias) bias_kernel(grad, grad, db, rows, n, true);
    return {dx, dw, db};
}

at::Tensor grouped_forward(const at::Tensor& input, const at::Tensor& weight, const std::vector<int64_t>& counts) {
    check_linear_tensor(input, input);
    const c10_npu::NPUGuard guard(input.device());
    return areno_accel::grouped_linear::forward(input, weight, counts, check_linear_tensor, mm);
}

std::vector<at::Tensor> grouped_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
    const std::vector<int64_t>& counts, bool need_input, bool need_weight) {
    check_linear_tensor(input, input);
    const c10_npu::NPUGuard guard(input.device());
    return areno_accel::grouped_linear::backward(grad, input, weight, counts, need_input, need_weight,
                                               check_linear_tensor, mm);
}
} // namespace
} // namespace areno_npu

void register_linear(pybind11::module_& m) {
    m.def("areno_linear_forward", &areno_npu::linear_forward);
    m.def("areno_linear_backward", &areno_npu::linear_backward);
    m.def("areno_grouped_linear_forward", &areno_npu::grouped_forward);
    m.def("areno_grouped_linear_backward", &areno_npu::grouped_backward);
    m.def("areno_grouped_linear_forward_counts", [](const at::Tensor& input, const at::Tensor& weight,
        const at::Tensor& counts) {
        return areno_npu::grouped_forward(input, weight, areno_accel::grouped_linear::host_counts(counts, input, weight));
    });
    m.def("areno_grouped_linear_backward_counts", [](const at::Tensor& grad, const at::Tensor& input,
        const at::Tensor& weight, const at::Tensor& counts, bool need_input, bool need_weight) {
        return areno_npu::grouped_backward(grad, input, weight, areno_accel::grouped_linear::host_counts(counts, input, weight),
                                          need_input, need_weight);
    });
}
