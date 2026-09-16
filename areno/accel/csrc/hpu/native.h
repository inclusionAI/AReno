#pragma once
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"

namespace areno_hpu {
using Tensor = at::Tensor;
using Tensors = std::vector<Tensor>;
using Descriptor = habana::custom_op::UserCustomOpDescriptor;
using Metadata = habana::PartialOutputMetaDataVector;

inline const char* dtype_suffix(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat: return "f32";
        case at::kBFloat16: return "bf16";
        case at::kHalf: return "f16";
        default: TORCH_CHECK(false, "HPU native kernels require float32, bfloat16, or float16 storage");
    }
}

inline void check_tensor(const Tensor& tensor, const Tensor& reference, at::ScalarType dtype) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::HPU && tensor.device() == reference.device(),
                "native HPU tensors must be on the same HPU device");
    TORCH_CHECK(tensor.is_contiguous(), "native HPU tensors must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, "native HPU tensor dtype mismatch");
}

inline Tensors call(const std::string& name, at::ScalarType dtype, const at::Stack& args) {
    auto descriptor = Descriptor::getUserCustomOpDescriptor("custom_op::" + name + "_" + dtype_suffix(dtype));
    return descriptor.execute(args);
}

inline void execute_boxed(const c10::OperatorHandle& op, at::Stack* stack) {
    auto descriptor = Descriptor::getUserCustomOpDescriptor(op.schema().name());
    auto output = descriptor.execute(*stack);
    stack->clear();
    for (auto& tensor : output) stack->emplace_back(std::move(tensor));
}

inline void meta_boxed(const c10::OperatorHandle& op, at::Stack* stack) {
    const auto& descriptor = Descriptor::getUserCustomOpDescriptor(op.schema().name());
    auto meta = descriptor.getOutputMetaFn()(*stack);
    stack->clear();
    for (auto& output : meta) {
        stack->emplace_back(at::empty(output.shape, at::TensorOptions().device(at::kMeta).dtype(output.dtype)));
    }
}
Tensor linear_forward(Tensor x, Tensor weight, Tensor bias, bool use_bias);
Tensors linear_backward(Tensor grad, Tensor x, Tensor weight, bool use_bias, bool need_x, bool need_w, bool need_bias);
Tensors moe_topk_permute(Tensor x,Tensor ids,Tensor weights,int64_t start,int64_t experts);
Tensor moe_scatter(Tensor x,Tensor ids,int64_t tokens);
Tensor recurrent_state_update(Tensor old,Tensor updated,Tensor indices);
Tensor grouped_forward(Tensor x,Tensor weight,Tensor counts);
} // namespace areno_hpu

void bind_normalization(pybind11::module_& m);
void bind_linear(pybind11::module_& m);
void bind_optimizer(pybind11::module_& m);
void bind_quantized_optimizer(pybind11::module_& m);
void bind_factored_optimizer(pybind11::module_& m);
void bind_embedding(pybind11::module_& m);
void bind_attention(pybind11::module_& m);
void bind_paged_cache(pybind11::module_& m);
void bind_grouped_linear(pybind11::module_& m);
void bind_conv(pybind11::module_& m);
void bind_topk(pybind11::module_& m);
void bind_moe(pybind11::module_& m);
void bind_moe_align(pybind11::module_& m);
void bind_fused_moe(pybind11::module_& m);

void bind_kda(pybind11::module_& m);

void bind_seg_la(pybind11::module_& m);
