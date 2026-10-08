#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <cmath>
#include <tuple>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tensor_format.h"
#include "normalization_launch.h"

namespace areno_npu {
namespace {
void check_norm_tensor(const at::Tensor& x, bool contiguous = true) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "RMSNorm requires an Ascend NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "RMSNorm requires FP32, FP16 or BF16 tensors");
    TORCH_CHECK(!contiguous || x.is_contiguous(), "RMSNorm native input must be contiguous");
    TORCH_CHECK(is_base_format(x), "RMSNorm requires base NPU storage format");
}

void check_like(const at::Tensor& other, const at::Tensor& x, bool contiguous = true) {
    check_norm_tensor(other, contiguous);
    TORCH_CHECK(other.device() == x.device() && other.scalar_type() == x.scalar_type() && other.sizes() == x.sizes(),
                "RMSNorm tensor shape, dtype and device must match input");
}

void check_input(const at::Tensor& x, const at::Tensor& weight, bool scale) {
    check_norm_tensor(x);
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "RMSNorm requires a nonempty feature dimension");
    if (scale) {
        check_norm_tensor(weight);
        TORCH_CHECK(weight.device() == x.device() && weight.scalar_type() == at::kFloat &&
                    weight.numel() == x.size(-1), "RMSNorm weight must be FP32 with one value per feature on input device");
    }
}

uint32_t storage(const at::Tensor& x) {
    return x.scalar_type() == at::kFloat ? 0 : x.scalar_type() == at::kHalf ? 1 : 2;
}

void run_norm(const at::Tensor& x, const at::Tensor& gate, const at::Tensor& weight, const at::Tensor& grad,
              at::Tensor& out, at::Tensor& grad_gate, at::Tensor& inv, at::Tensor& grad_weight,
              bool backward, bool scale, uint32_t gate_kind, int64_t groups, float eps) {
    const int64_t width = x.size(-1), rows = x.numel() / width;
    if (rows == 0) return;
    auto stream = c10_npu::getCurrentNPUStream(x.device().index()).stream(true);
    launch_normalization(static_cast<uint32_t>(std::min<int64_t>(rows, 32)), stream,
        storage(x), scale ? storage(weight) : 0, backward, scale, gate_kind,
        x.const_data_ptr(), gate.defined() ? gate.const_data_ptr() : nullptr,
        scale ? weight.const_data_ptr() : nullptr, backward ? grad.const_data_ptr() : nullptr,
        out.data_ptr(), grad_gate.defined() ? grad_gate.data_ptr() : nullptr, inv.data_ptr<float>(),
        backward && scale ? grad_weight.data_ptr<float>() : nullptr, rows, width, groups, eps);
}

std::vector<at::Tensor> norm_forward(const at::Tensor& x, const at::Tensor& gate,
                                    const at::Tensor& weight, double eps, bool scale) {
    check_input(x, weight, scale);
    TORCH_CHECK(std::isfinite(eps) && eps >= 0, "RMSNorm epsilon must be finite and nonnegative");
    if (gate.defined()) check_like(gate, x);
    const c10_npu::NPUGuard guard(x.device());
    auto out = at::empty(x.sizes(), x.options());
    auto inv = at::empty({x.numel() / x.size(-1)}, x.options().dtype(at::kFloat));
    at::Tensor unused;
    run_norm(x, gate, weight, {}, out, unused, inv, unused, false, scale, gate.defined() ? 1 : 0, 1, eps);
    return {out, inv};
}

std::vector<at::Tensor> norm_backward(const at::Tensor& grad, const at::Tensor& x, const at::Tensor& gate,
                                     const at::Tensor& weight, at::Tensor inv, bool scale) {
    check_input(x, weight, scale);
    check_like(grad, x);
    if (gate.defined()) check_like(gate, x);
    check_norm_tensor(inv);
    TORCH_CHECK(inv.device() == x.device() && inv.scalar_type() == at::kFloat && inv.dim() == 1 &&
                inv.numel() == x.numel() / x.size(-1), "RMSNorm saved inv_rms must be FP32 with one value per row");
    const c10_npu::NPUGuard guard(x.device());
    auto dx = at::empty(x.sizes(), x.options());
    auto dg = gate.defined() ? at::empty(x.sizes(), x.options()) : at::Tensor{};
    // Like CUDA, dw is FP32 and accumulates atomically over rows. Its clear is
    // ordered before the native launch by the framework's current stream.
    auto dw = scale ? at::zeros(weight.sizes(), weight.options()) : at::empty({0}, x.options().dtype(at::kFloat));
    run_norm(x, gate, weight, grad, dx, dg, inv, dw, true, scale, gate.defined() ? 1 : 0, 1, 0);
    return gate.defined() ? std::vector<at::Tensor>{dx, dg, dw} : std::vector<at::Tensor>{dx, dw};
}

std::tuple<at::Tensor, at::Tensor> group_norm_gate(const at::Tensor& x, const at::Tensor& gate,
                                                const at::Tensor& weight, double eps) {
    check_norm_tensor(x, false);
    check_like(gate, x, false);
    check_norm_tensor(weight, false);
    TORCH_CHECK(x.dim() == 3 && x.size(2) > 0, "rms_norm_gate_fwd expects (M, G, N) with N > 0");
    TORCH_CHECK(weight.device() == x.device() && weight.dim() == 2 &&
                weight.size(0) == x.size(1) && weight.size(1) == x.size(2),
                "group RMSNorm weight must have shape (G, N) on input device");
    TORCH_CHECK(x.size(2) <= 65536 / x.element_size(), "fused group RMSNorm does not support this feature size");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0, "RMSNorm epsilon must be finite and nonnegative");
    const c10_npu::NPUGuard guard(x.device());
    auto packed_x = x.contiguous(), packed_gate = gate.contiguous(), packed_weight = weight.contiguous();
    auto out = at::empty(x.sizes(), x.options());
    auto inv = at::empty({x.size(0), x.size(1)}, x.options().dtype(at::kFloat));
    at::Tensor unused;
    // This entry matches the CUDA Triton operator: sigmoid(gate), not SiLU.
    run_norm(packed_x, packed_gate, packed_weight, {}, out, unused, inv, unused, false, true, 2, x.size(1), eps);
    return {out, inv};
}
} // namespace
} // namespace areno_npu

void register_normalization(pybind11::module_& m) {
    using namespace areno_npu;
    m.def("areno_rmsnorm_forward", [](const at::Tensor& x, const at::Tensor& w, double eps) {
        return norm_forward(x, {}, w, eps, true);
    });
    m.def("areno_optional_scale_rmsnorm_forward", [](const at::Tensor& x, const at::Tensor& w, double eps, bool scale) {
        return norm_forward(x, {}, w, eps, scale);
    });
    m.def("areno_rmsnorm_silu_gate_forward", [](const at::Tensor& x, const at::Tensor& g, const at::Tensor& w, double eps) {
        return norm_forward(x, g, w, eps, true);
    });
    m.def("areno_rmsnorm_backward", [](const at::Tensor& dy, const at::Tensor& x, const at::Tensor& w, at::Tensor inv) {
        return norm_backward(dy, x, {}, w, inv, true);
    });
    m.def("areno_optional_scale_rmsnorm_backward", [](const at::Tensor& dy, const at::Tensor& x, const at::Tensor& w,
                                                    at::Tensor inv, bool scale) {
        return norm_backward(dy, x, {}, w, inv, scale);
    });
    m.def("areno_rmsnorm_silu_gate_backward", [](const at::Tensor& dy, const at::Tensor& x, const at::Tensor& g,
                                               const at::Tensor& w, at::Tensor inv) {
        return norm_backward(dy, x, g, w, inv, true);
    });
    m.def("rms_norm_gate_fwd", &group_norm_gate);
}
