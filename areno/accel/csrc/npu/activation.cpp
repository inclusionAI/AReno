#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <torch/csrc/utils/pybind.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "activation_launch.h"

namespace areno_npu {
namespace {
void check(const at::Tensor& input, bool contiguous = true) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
                "AReno NPU activation requires an Ascend NPU tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
                input.scalar_type() == at::kBFloat16, "NPU activation requires FP32, FP16 or BF16");
    TORCH_CHECK(!contiguous || input.is_contiguous(), "NPU activation input must be contiguous");
    TORCH_CHECK(at_npu::native::FormatHelper::IsBaseFormatType(input),
                "NPU activation requires a base storage format; convert packed NPU formats before calling accel");
}

void match(const at::Tensor& tensor, const at::Tensor& input, at::IntArrayRef shape, bool contiguous = true) {
    check(tensor, contiguous);
    TORCH_CHECK(tensor.device() == input.device() && tensor.scalar_type() == input.scalar_type() &&
                tensor.sizes() == shape, "NPU activation tensor shape, dtype and device must match");
}

void run(at::Tensor& output, const at::Tensor& input, const at::Tensor& grad,
         Activation op, int64_t rows, int64_t width) {
    if (output.numel() == 0) return;
    const c10_npu::NPUGuard device_guard(input.device());
    // A user-provided out buffer or empty_like(saved_view) can have strides.
    // Only that layout copy uses ATen; activation arithmetic runs in one kernel.
    auto packed = output.is_contiguous() ? output : at::empty(output.sizes(), output.options());
    const auto tiles = rows * ((width + kActivationTile - 1) / kActivationTile);
    const auto blocks = static_cast<uint32_t>(std::min<int64_t>(tiles, 32));
    const uint32_t storage = input.scalar_type() == at::kFloat ? 0 : input.scalar_type() == at::kHalf ? 1 : 2;
    auto stream = c10_npu::getCurrentNPUStream(input.device().index()).stream(true);
    launch_activation(blocks, stream, storage, op, packed.data_ptr(), input.const_data_ptr(),
                      grad.defined() ? grad.const_data_ptr() : nullptr, rows, width);
    if (!output.is_contiguous()) output.copy_(packed);
}

at::Tensor unary(const at::Tensor& input, Activation op) {
    check(input);
    const c10_npu::NPUGuard device_guard(input.device());
    auto output = at::empty(input.sizes(), input.options());
    run(output, input, {}, op, 1, input.numel());
    return output;
}

at::Tensor unary_backward(const at::Tensor& grad, const at::Tensor& saved, Activation op) {
    check(saved);
    match(grad, saved, saved.sizes());
    const c10_npu::NPUGuard device_guard(saved.device());
    auto output = at::empty(saved.sizes(), saved.options());
    run(output, saved, grad, op, 1, saved.numel());
    return output;
}

void gated(at::Tensor output, const at::Tensor& input, const at::Tensor& grad, Activation op) {
    check(input);
    TORCH_CHECK(input.dim() > 0 && input.size(-1) % 2 == 0, "NPU gated activation requires an even last dimension");
    auto shape = input.sizes().vec();
    const auto width = shape.back() /= 2;
    const bool backward = (op & 1) != 0;
    match(output, input, backward ? input.sizes() : at::IntArrayRef(shape), false);
    at::assert_no_internal_overlap(output);
    at::assert_no_overlap(output, input);
    if (backward) {
        match(grad, input, shape);
        at::assert_no_overlap(output, grad);
    }
    run(output, input, grad, op, width ? input.numel() / (2 * width) : 0, width);
}
} // namespace
} // namespace areno_npu

void register_activations(pybind11::module_& m) {
    using namespace areno_npu;
    m.def("areno_silu", [](const at::Tensor& x) { return unary(x, Silu); });
    m.def("areno_d_silu", [](const at::Tensor& g, const at::Tensor& x) { return unary_backward(g, x, DSilu); });
    m.def("areno_sigmoid", [](const at::Tensor& x) { return unary(x, Sigmoid); });
    m.def("areno_d_sigmoid", [](const at::Tensor& g, const at::Tensor& y) { return unary_backward(g, y, DSigmoid); });
    m.def("areno_softplus", [](const at::Tensor& x) { return unary(x, Softplus); });
    m.def("areno_d_softplus", [](const at::Tensor& g, const at::Tensor& x) { return unary_backward(g, x, DSoftplus); });
    m.def("areno_silu_and_mul", [](at::Tensor out, const at::Tensor& x) { gated(out, x, {}, SiluMul); });
    m.def("areno_d_silu_and_mul", [](at::Tensor out, const at::Tensor& g, const at::Tensor& x) {
        gated(out, x, g, DSiluMul);
    });
    m.def("areno_gelu_tanh_and_mul", [](at::Tensor out, const at::Tensor& x) { gated(out, x, {}, GeluTanhMul); });
    m.def("areno_d_gelu_tanh_and_mul", [](at::Tensor out, const at::Tensor& g, const at::Tensor& x) {
        gated(out, x, g, DGeluTanhMul);
    });
}
