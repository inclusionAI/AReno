#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <algorithm>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tensor_format.h"
#include "../routing_common.h"
#undef ARENO_ROUTING_INLINE
#include "routing_launch.h"

namespace areno_npu {
namespace {
using areno_accel::routing::kMaxExperts;
using areno_accel::routing::kMaxGroups;
using areno_accel::routing::kMaxTopK;

void check_tensor(const at::Tensor& tensor, const at::Tensor& logits, at::ScalarType dtype) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device() == logits.device(),
                "routing tensors must be on the same Ascend NPU device");
    TORCH_CHECK(tensor.scalar_type() == dtype, "routing tensor dtype mismatch");
    TORCH_CHECK(tensor.is_contiguous(), "routing native tensors must be contiguous");
    TORCH_CHECK(is_base_format(tensor), "routing requires base NPU storage format");
}

void check_logits(const at::Tensor& logits, int64_t top_k) {
    check_tensor(logits, logits, logits.scalar_type());
    TORCH_CHECK(logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kHalf || logits.scalar_type() == at::kBFloat16,
                "routing supports FP32, FP16 or BF16 logits");
    TORCH_CHECK(logits.dim() == 2, "routing logits must be 2D");
    TORCH_CHECK(logits.size(1) > 0 && logits.size(1) <= kMaxExperts, "routing unsupported expert count");
    TORCH_CHECK(top_k > 0 && top_k <= kMaxTopK && top_k <= logits.size(1), "routing unsupported top_k");
}

void run(RoutingOp op, const at::Tensor& logits, const at::Tensor& bias, at::Tensor& indices,
         at::Tensor& weights, at::Tensor& grad, bool renormalize, int groups = 1, int topk_group = 1) {
    if (logits.size(0) == 0) return;
    auto stream = c10_npu::getCurrentNPUStream(logits.device().index()).stream(true);
    uint32_t storage = logits.scalar_type() == at::kFloat ? 0 : logits.scalar_type() == at::kHalf ? 1 : 2;
    launch_routing(static_cast<uint32_t>(std::min<int64_t>(logits.size(0), 32)), stream, storage, op,
        logits.const_data_ptr(), bias.defined() ? bias.const_data_ptr<float>() : nullptr,
        indices.data_ptr<int64_t>(), weights.data_ptr<float>(), grad.defined() ? grad.data_ptr() : nullptr,
        logits.size(0), static_cast<int>(logits.size(1)), static_cast<int>(indices.size(1)), renormalize, groups, topk_group);
}

std::vector<at::Tensor> topk_forward(const at::Tensor& logits, int64_t top_k, bool renormalize) {
    check_logits(logits, top_k);
    const c10_npu::NPUGuard guard(logits.device());
    auto indices = at::empty({logits.size(0), top_k}, logits.options().dtype(at::kLong));
    auto weights = at::empty({logits.size(0), top_k}, logits.options().dtype(at::kFloat));
    at::Tensor grad;
    run(TopKForward, logits, {}, indices, weights, grad, renormalize);
    return {indices, weights};
}

at::Tensor topk_backward(at::Tensor grad_weight, const at::Tensor& logits, at::Tensor indices, bool renormalize) {
    TORCH_CHECK(indices.dim() == 2, "routing indices must be 2D");
    check_logits(logits, indices.size(1));
    check_tensor(indices, logits, at::kLong);
    check_tensor(grad_weight, logits, at::kFloat);
    TORCH_CHECK(indices.size(0) == logits.size(0) && grad_weight.sizes() == indices.sizes(), "routing gradient shape mismatch");
    const c10_npu::NPUGuard guard(logits.device());
    auto grad = at::empty(logits.sizes(), logits.options());
    run(TopKBackward, logits, {}, indices, grad_weight, grad, renormalize);
    return grad;
}

std::vector<at::Tensor> grouped_router(const at::Tensor& logits, const at::Tensor& bias,
    int64_t top_k, int64_t groups, int64_t topk_group) {
    check_logits(logits, top_k);
    check_tensor(bias, logits, at::kFloat);
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == logits.size(1), "routing bias shape mismatch");
    TORCH_CHECK(groups > 0 && groups <= kMaxGroups && logits.size(1) % groups == 0, "routing unsupported groups");
    TORCH_CHECK(topk_group > 0 && topk_group <= groups, "routing unsupported topk_group");
    TORCH_CHECK(top_k / topk_group > 0 && top_k / topk_group <= logits.size(1) / groups
                && top_k <= topk_group * (logits.size(1) / groups), "routing insufficient experts in selected groups");
    const c10_npu::NPUGuard guard(logits.device());
    auto indices = at::empty({logits.size(0), top_k}, logits.options().dtype(at::kLong));
    auto weights = at::empty({logits.size(0), top_k}, logits.options().dtype(at::kFloat));
    at::Tensor grad;
    run(GroupedRouter, logits, bias, indices, weights, grad, true, static_cast<int>(groups), static_cast<int>(topk_group));
    return {indices, weights};
}
} // namespace
} // namespace areno_npu

void register_routing(pybind11::module_& m) {
    m.def("areno_topk_softmax_forward", &areno_npu::topk_forward);
    m.def("areno_topk_softmax_backward", &areno_npu::topk_backward);
    m.def("areno_grouped_topk_router", &areno_npu::grouped_router);
}
