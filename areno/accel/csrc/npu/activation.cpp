// Ascend activation entry points. ATen dispatches to torch_npu's CANN kernels.
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

namespace {
void check(const at::Tensor& input) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
                "AReno NPU activation requires an Ascend NPU tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
                input.scalar_type() == at::kBFloat16, "NPU activation requires FP32, FP16 or BF16");
}

void check_grad(const at::Tensor& grad, const at::Tensor& saved) {
    check(saved);
    TORCH_CHECK(grad.device() == saved.device() && grad.scalar_type() == saved.scalar_type() &&
                grad.sizes() == saved.sizes(), "NPU activation gradient must match the saved tensor");
}

at::Tensor silu(const at::Tensor& x) {
    check(x);
    return at::silu(x.to(at::kFloat)).to(x.scalar_type());
}
at::Tensor d_silu(const at::Tensor& grad, const at::Tensor& x) {
    check_grad(grad, x);
    return at::silu_backward(grad.to(at::kFloat), x.to(at::kFloat)).to(x.scalar_type());
}
at::Tensor sigmoid(const at::Tensor& x) {
    check(x);
    return at::sigmoid(x.to(at::kFloat)).to(x.scalar_type());
}
at::Tensor d_sigmoid(const at::Tensor& grad, const at::Tensor& out) {
    check_grad(grad, out);
    return at::sigmoid_backward(grad.to(at::kFloat), out.to(at::kFloat)).to(out.scalar_type());
}
at::Tensor softplus(const at::Tensor& x) {
    check(x);
    return at::softplus(x.to(at::kFloat), 1, 20).to(x.scalar_type());
}
at::Tensor d_softplus(const at::Tensor& grad, const at::Tensor& x) {
    check_grad(grad, x);
    return at::softplus_backward(grad.to(at::kFloat), x.to(at::kFloat), 1, 20).to(x.scalar_type());
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("areno_silu", &silu);
    m.def("areno_d_silu", &d_silu);
    m.def("areno_sigmoid", &sigmoid);
    m.def("areno_d_sigmoid", &d_sigmoid);
    m.def("areno_softplus", &softplus);
    m.def("areno_d_softplus", &d_softplus);
    m.attr("supports_training_and_serving") = false;
}
