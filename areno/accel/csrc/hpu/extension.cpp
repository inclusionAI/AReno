// Native activation bindings. Tensor math is performed by activation.c.
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"
#include "native.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace {
using habana::custom_op::UserCustomOpDescriptor;

struct Activation {
    const char* name;
    bool backward;
    bool gated;
};

constexpr Activation activations[] = {
#define ARENO_ACTIVATION(name, kind, backward, gated) {#name, backward, gated},
#include "activation_ops.inc"
#undef ARENO_ACTIVATION
};

const char* dtype_suffix(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat: return "f32";
        case at::kBFloat16: return "bf16";
        case at::kHalf: return "f16";
        default: TORCH_CHECK(false, "HPU activations require float32, bfloat16, or float16 tensors");
    }
}

std::string schema(const Activation& op, at::ScalarType dtype) {
    return std::string("custom_op::") + op.name + "_" + dtype_suffix(dtype);
}

at::Tensor execute(const Activation& op, const at::Tensor& input, const at::Tensor& grad = {}) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::HPU, "activation input must be HPU");
    TORCH_CHECK(input.is_contiguous(), "activation input must be contiguous");
    dtype_suffix(input.scalar_type());
    auto output_shape = input.sizes().vec();
    int64_t channels = op.gated ? 2 : 1;
    if (op.gated) {
        TORCH_CHECK(input.dim() > 0 && input.size(-1) % 2 == 0, "gated activation requires an even last dimension");
        if (!op.backward) output_shape.back() /= 2;
    }
    if (op.backward) {
        TORCH_CHECK(grad.device() == input.device() && grad.scalar_type() == input.scalar_type(),
                    "activation gradient must match the input device and dtype");
        TORCH_CHECK(grad.is_contiguous() && grad.numel() * channels == input.numel(),
                    "activation gradient must be contiguous with the forward output shape");
    }
    if (input.numel() == 0) return at::empty(output_shape, input.options());
    // Explicit channel dimension gives the TPC tensor engine a boundary at H
    // for both halves, including widths not divisible by the vector length.
    int64_t width = op.gated ? input.size(-1) / 2 : input.numel();
    int64_t rows = input.numel() / (channels * width);
    std::vector<c10::IValue> inputs{input.view({rows, channels, width})};
    if (op.backward) inputs.emplace_back(grad.view({rows, 1, width}));
    auto descriptor = UserCustomOpDescriptor::getUserCustomOpDescriptor(schema(op, input.scalar_type()));
    return descriptor.execute(inputs).at(0).view(output_shape);
}

void execute_out(const Activation& op, at::Tensor output, const at::Tensor& input, const at::Tensor& grad = {}) {
    TORCH_CHECK(output.device() == input.device() && output.scalar_type() == input.scalar_type(),
                "activation output must match the input device and dtype");
    auto expected_shape = input.sizes().vec();
    TORCH_CHECK(!expected_shape.empty() && expected_shape.back() % 2 == 0,
                "gated activation requires an even last dimension");
    if (!op.backward) expected_shape.back() /= 2;
    TORCH_CHECK(output.sizes().vec() == expected_shape, "activation output shape mismatch");
    // CustomOp allocates its result. A native copy preserves the caller's
    // preallocated buffer and aliases; the activation itself is one TPC kernel.
    output.copy_(execute(op, input, grad));
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    for (const auto& op : activations) {
        for (const auto* suffix : {"f32", "bf16", "f16"}) {
            const std::string name = std::string(op.name) + "_" + suffix;
            m.def((name + (op.backward ? "(Tensor input, Tensor grad) -> Tensor" : "(Tensor input) -> Tensor")).c_str());
            habana::custom_op::registerUserCustomOp(
                "custom_op::" + name, name,
                [op](const at::Stack& inputs) {
                    const auto input = inputs[0].toTensor();
                    auto shape = input.sizes().vec();
                    if (op.gated && !op.backward) shape[1] = 1;
                    return habana::PartialOutputMetaDataVector{{input.scalar_type(), shape}};
                }, nullptr);
        }
    }
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    for (const auto& op : activations) {
        for (const auto* suffix : {"f32", "bf16", "f16"}) {
            const std::string name = std::string(op.name) + "_" + suffix;
            if (op.backward) {
                m.impl(name.c_str(), [name](const at::Tensor& input, const at::Tensor& grad) {
                    auto descriptor = UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::" + name);
                    return descriptor.execute({input, grad}).at(0);
                });
            } else {
                m.impl(name.c_str(), [name](const at::Tensor& input) {
                    auto descriptor = UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::" + name);
                    return descriptor.execute({input}).at(0);
                });
            }
        }
    }
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    for (const auto& op : activations) {
        for (const auto* suffix : {"f32", "bf16", "f16"}) {
            const std::string name = std::string(op.name) + "_" + suffix;
            if (op.backward) {
                m.impl(name.c_str(), [](const at::Tensor& input, const at::Tensor&) {
                    return at::empty_like(input);
                });
            } else {
                m.impl(name.c_str(), [op](const at::Tensor& input) {
                    auto shape = input.sizes().vec();
                    if (op.gated) shape[1] = 1;
                    return at::empty(shape, input.options());
                });
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    const auto* mode = std::getenv("PT_HPU_LAZY_MODE");
    TORCH_CHECK(mode != nullptr && std::string(mode) == std::to_string(ARENO_HPU_LAZY_MODE),
                "PT_HPU_LAZY_MODE must match the mode used to build the AReno HPU extension");
    bind_normalization(m);
    bind_linear(m);
    bind_optimizer(m);
    bind_quantized_optimizer(m);
    bind_factored_optimizer(m);
    bind_embedding(m);
    bind_attention(m);
    bind_paged_cache(m);
    bind_grouped_linear(m);
    bind_conv(m);
    bind_topk(m);
    bind_moe(m);
    bind_moe_align(m);
    bind_fused_moe(m);
    bind_kda(m);
    bind_seg_la(m);
    for (const auto& op : activations) {
        if (op.gated && op.backward) {
            m.def(op.name, [op](at::Tensor output, const at::Tensor& grad, const at::Tensor& input) {
                execute_out(op, output, input, grad);
            });
        } else if (op.gated) {
            m.def(op.name, [op](at::Tensor output, const at::Tensor& input) { execute_out(op, output, input); });
        } else if (op.backward) {
            m.def(op.name, [op](const at::Tensor& grad, const at::Tensor& input) { return execute(op, input, grad); });
        } else {
            m.def(op.name, [op](const at::Tensor& input) { return execute(op, input); });
        }
    }
    m.def("__getattr__", [](const std::string& name) -> pybind11::object {
        if (name.rfind("__", 0) == 0) throw pybind11::attribute_error(name);
        PyErr_SetString(PyExc_NotImplementedError, ("AReno native HPU operator is not implemented: " + name).c_str());
        throw pybind11::error_already_set();
    });
}
