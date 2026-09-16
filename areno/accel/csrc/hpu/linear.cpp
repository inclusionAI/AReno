#include "native.h"
#include "normalization_params.h"
#include <synapse_common_types.h>
#include <limits>

namespace {
using namespace areno_hpu;

template<class F> void for_each_linear(F&& fn) {
    for (auto suffix : {"f32", "bf16", "f16"}) {
        for (const auto* name : {"areno_mme_gemm", "areno_linear_bias", "areno_linear_bias_grad"}) {
            fn(std::string(name) + "_" + suffix, std::string(name));
        }
    }
}

Tensor gemm(Tensor left, Tensor right, bool transpose_left, bool transpose_right) {
    return call("areno_mme_gemm", left.scalar_type(), {left, right, transpose_left, transpose_right})[0];
}

void check_linear(Tensor x, Tensor w) {
    check_tensor(x, x, x.scalar_type());
    check_tensor(w, x, x.scalar_type());
    dtype_suffix(x.scalar_type());
    TORCH_CHECK(x.dim() > 0 && w.dim() == 2 && w.size(1) > 0 && x.size(-1) == w.size(1), "linear shape mismatch");
    TORCH_CHECK(w.size(0) <= std::numeric_limits<int>::max()
                && x.numel() / w.size(1) <= std::numeric_limits<int>::max(), "linear dimensions exceed TPC indexing");
}

Tensor forward(Tensor x, Tensor w, Tensor bias, bool use_bias) {
    check_linear(x, w);
    if (use_bias) {
        check_tensor(bias, x, x.scalar_type());
        TORCH_CHECK(bias.numel() == w.size(0), "linear bias shape mismatch");
    }
    auto shape = x.sizes().vec();
    shape.back() = w.size(0);
    const auto rows = x.numel() / w.size(1);
    if (rows == 0 || w.size(0) == 0) return at::empty(shape, x.options());
    auto output = gemm(x.view({rows, w.size(1)}), w, false, true);
    if (use_bias) output = call("areno_linear_bias", x.scalar_type(), {output, bias.view({w.size(0)})})[0];
    return output.view(shape);
}

Tensors backward(Tensor dy, Tensor x, Tensor w, bool use_bias, bool need_x, bool need_w, bool need_bias) {
    check_linear(x, w);
    check_tensor(dy, x, x.scalar_type());
    auto shape = x.sizes().vec();
    shape.back() = w.size(0);
    TORCH_CHECK(dy.sizes().vec() == shape, "linear gradient shape mismatch");
    const auto rows = x.numel() / w.size(1);
    Tensors result(3);
    if (rows == 0 || w.size(0) == 0) {
        if (need_x) result[0] = at::zeros_like(x);
        if (need_w) result[1] = at::zeros_like(w);
        if (use_bias && need_bias) result[2] = at::zeros({w.size(0)}, x.options());
        return result;
    }
    auto grad = dy.view({rows, w.size(0)});
    if (need_x) result[0] = gemm(grad, w, false, false).view(x.sizes());
    if (need_w) result[1] = gemm(grad, x.view({rows, w.size(1)}), true, false);
    if (use_bias && need_bias) result[2] = call("areno_linear_bias_grad", x.scalar_type(), {grad})[0];
    return result;
}
} // namespace

namespace areno_hpu {
Tensor linear_forward(Tensor x, Tensor weight, Tensor bias, bool use_bias) { return forward(x, weight, bias, use_bias); }
Tensors linear_backward(Tensor grad, Tensor x, Tensor weight, bool use_bias, bool need_x, bool need_w, bool need_bias) {
    return backward(grad, x, weight, use_bias, need_x, need_w, need_bias);
}
} // namespace areno_hpu

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    for_each_linear([&](const std::string& name, const std::string& base) {
        const bool gemm = base == "areno_mme_gemm";
        const bool bias_grad = base == "areno_linear_bias_grad";
        m.def((name + (gemm ? "(Tensor left, Tensor right, bool transpose_left, bool transpose_right) -> Tensor"
            : bias_grad ? "(Tensor input) -> Tensor" : "(Tensor input, Tensor bias) -> Tensor")).c_str());
        // UserCustomOpBackend passes the GUID directly to Synapse without a
        // dtype suffix. GEMM therefore uses the native MME node, like the bridge's mm.
        habana::custom_op::registerUserCustomOp("custom_op::" + name, gemm ? "gemm" : name,
            [gemm, bias_grad](const at::Stack& args) {
                auto left = args[0].toTensor();
                TORCH_CHECK(left.dim() == 2, "internal linear input must be a matrix");
                if (!gemm) return Metadata{{left.scalar_type(), bias_grad ? std::vector<int64_t>{left.size(1)} : left.sizes().vec()}};
                auto right = args[1].toTensor();
                const bool ta = args[2].toBool(), tb = args[3].toBool();
                TORCH_CHECK(right.dim() == 2 && left.size(ta ? 0 : 1) == right.size(tb ? 1 : 0), "GEMM inner dimension mismatch");
                TORCH_CHECK(left.scalar_type() == right.scalar_type(), "GEMM dtype mismatch");
                return Metadata{{left.scalar_type(), {left.size(ta ? 1 : 0), right.size(tb ? 0 : 1)}}};
            }, [gemm](const at::Stack& args, size_t& size) -> std::shared_ptr<void> {
                if (gemm) {
                    size = sizeof(synGEMMParams);
                    return std::make_shared<synGEMMParams>(synGEMMParams{args[2].toBool(), args[3].toBool()});
                }
                auto input = args[0].toTensor();
                size = sizeof(NormalizationParams);
                return std::make_shared<NormalizationParams>(NormalizationParams{
                    static_cast<int>(input.size(1)), static_cast<int>(input.size(0)), 0.0f});
            });
    });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    for_each_linear([&](const std::string& name, const std::string&) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>());
    });
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    for_each_linear([&](const std::string& name, const std::string&) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>());
    });
}

void bind_linear(pybind11::module_& m) {
    m.def("areno_linear_forward", &forward);
    m.def("areno_linear_backward", &backward);
}
