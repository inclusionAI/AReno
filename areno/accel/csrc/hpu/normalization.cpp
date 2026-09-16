#include "native.h"
#include "normalization_params.h"
#include <limits>

namespace {
using namespace areno_hpu;

template<class F> void for_each_norm(F&& fn) {
    for (auto suffix : {"f32", "bf16", "f16"}) {
#define ARENO_NORMALIZATION(name, kind, direction) fn(std::string(#name) + "_" + suffix, kind, direction);
#include "normalization_ops.inc"
#undef ARENO_NORMALIZATION
    }
}

std::string norm_name(int kind, int direction) {
    return std::string("areno_rmsnorm_") + (kind == 0 ? "unscaled" : kind == 1 ? "scaled" : "gated")
        + (direction == 0 ? "_fwd" : direction == 1 ? "_bwd" : "_dw");
}

void check_input(const Tensor& input, const Tensor& weight, const Tensor& gate, int kind) {
    check_tensor(input, input, input.scalar_type());
    dtype_suffix(input.scalar_type());
    TORCH_CHECK(input.dim() > 0 && input.size(-1) > 0, "RMSNorm requires a nonempty last dimension");
    TORCH_CHECK(input.size(-1) <= std::numeric_limits<int>::max()
                && input.numel() / input.size(-1) <= std::numeric_limits<int>::max(), "RMSNorm dimensions exceed TPC indexing");
    if (kind >= 1) {
        check_tensor(weight, input, at::kFloat);
        TORCH_CHECK(weight.numel() == input.size(-1), "RMSNorm weight size mismatch");
    }
    if (kind == 2) {
        check_tensor(gate, input, input.scalar_type());
        TORCH_CHECK(gate.sizes() == input.sizes(), "RMSNorm gate shape mismatch");
    }
}

Tensors forward(Tensor input, Tensor weight, Tensor gate, double eps, int kind) {
    check_input(input, weight, gate, kind);
    const auto rows = input.numel() / input.size(-1);
    if (rows == 0) return {at::empty_like(input), at::empty({0}, input.options().dtype(at::kFloat))};
    at::Stack args{input.view({rows, input.size(-1)})};
    if (kind >= 1) args.emplace_back(weight.view({input.size(-1)}));
    if (kind == 2) args.emplace_back(gate.view({rows, input.size(-1)}));
    args.emplace_back(eps);
    auto result = call(norm_name(kind, 0), input.scalar_type(), args);
    return {result[0].view(input.sizes()), result[1].view({rows})};
}

Tensors backward(Tensor grad, Tensor input, Tensor weight, Tensor gate, Tensor inv, int kind) {
    check_input(input, weight, gate, kind);
    check_tensor(grad, input, input.scalar_type());
    check_tensor(inv, input, at::kFloat);
    TORCH_CHECK(grad.sizes() == input.sizes(), "RMSNorm gradient shape mismatch");
    const auto rows = input.numel() / input.size(-1);
    TORCH_CHECK(inv.numel() == rows, "RMSNorm inverse RMS size mismatch");
    if (rows == 0) {
        Tensors result{at::empty_like(input)};
        if (kind == 2) result.emplace_back(at::empty_like(gate));
        result.emplace_back(kind >= 1 ? at::zeros_like(weight) : at::empty({0}, input.options().dtype(at::kFloat)));
        return result;
    }
    at::Stack args{input.view({rows, input.size(-1)}), grad.view({rows, input.size(-1)}), inv.view({rows, 1})};
    if (kind >= 1) args.emplace_back(weight.view({input.size(-1)}));
    if (kind == 2) args.emplace_back(gate.view({rows, input.size(-1)}));
    args.emplace_back(0.0);
    auto result = call(norm_name(kind, 1), input.scalar_type(), args);
    for (auto& tensor : result) tensor = tensor.view(input.sizes());
    if (kind >= 1) {
        at::Stack dw_args{args[0], args[1], args[2]};
        if (kind == 2) dw_args.emplace_back(args[4]);
        dw_args.emplace_back(0.0);
        result.emplace_back(call(norm_name(kind, 2), input.scalar_type(), dw_args)[0].view(weight.sizes()));
    } else {
        result.emplace_back(at::empty({0}, input.options().dtype(at::kFloat)));
    }
    return result;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    for_each_norm([&](const std::string& name, int kind, int direction) {
        std::string args = "(Tensor input";
        if (direction != 0) args += ", Tensor grad, Tensor inv";
        if (kind >= 1 && direction != 2) args += ", Tensor weight";
        if (kind >= 2) args += ", Tensor gate";
        args += kind == 3 ? ", float eps, int groups) -> " : ", float eps) -> ";
        args += direction == 0 || (direction == 1 && kind == 2) ? "(Tensor, Tensor)" : "Tensor";
        m.def((name + args).c_str());
        habana::custom_op::registerUserCustomOp("custom_op::" + name, name,
            [kind, direction](const at::Stack& args) {
                const auto input = args[0].toTensor();
                TORCH_CHECK(input.dim() == 2, "internal RMSNorm input must be a matrix");
                if (direction == 2) return Metadata{{at::kFloat, {input.size(1)}}};
                Metadata output{{input.scalar_type(), input.sizes().vec()}};
                if (direction == 0) output.push_back({at::kFloat, {input.size(0), 1}});
                else if (kind == 2) output.push_back({input.scalar_type(), input.sizes().vec()});
                return output;
            }, [kind](const at::Stack& args, size_t& size) -> std::shared_ptr<void> {
                const auto input = args[0].toTensor();
                if (kind == 3) {
                    size=sizeof(GroupNormParams);
                    return std::make_shared<GroupNormParams>(GroupNormParams{static_cast<int>(input.size(1)),static_cast<int>(input.size(0)),
                        static_cast<float>(args[3].toDouble()),static_cast<int>(args[4].toInt())});
                }
                size = sizeof(NormalizationParams);
                return std::make_shared<NormalizationParams>(NormalizationParams{
                    static_cast<int>(input.size(1)), static_cast<int>(input.size(0)), static_cast<float>(args.back().toDouble())});
            });
    });
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    for_each_norm([&](const std::string& name, int, int) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>());
    });
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    for_each_norm([&](const std::string& name, int, int) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>());
    });
}

void bind_normalization(pybind11::module_& m) {
    m.def("rms_norm_gate_fwd",[](Tensor x,Tensor gate,Tensor weight,double eps) {
        x=x.contiguous(); gate=gate.contiguous(); weight=weight.contiguous();
        check_tensor(x,x,x.scalar_type()); dtype_suffix(x.scalar_type());
        check_tensor(gate,x,x.scalar_type()); check_tensor(weight,x,weight.scalar_type()); dtype_suffix(weight.scalar_type());
        TORCH_CHECK(x.dim() == 3 && gate.sizes() == x.sizes() && weight.dim() == 2
            && weight.size(0) == x.size(1) && weight.size(1) == x.size(2) && x.size(2) > 0,"group RMSNorm shape mismatch");
        auto rows=x.size(0)*x.size(1),hidden=x.size(2);
        TORCH_CHECK(rows <= std::numeric_limits<int>::max() && hidden <= std::numeric_limits<int>::max(),"group RMSNorm exceeds TPC indexing");
        if (!rows) return Tensors{at::empty_like(x),at::empty({x.size(0),x.size(1)},x.options().dtype(at::kFloat))};
        auto result=call("areno_rmsnorm_grouped_fwd",x.scalar_type(),{x.view({rows,hidden}),weight.to(at::kFloat),gate.view({rows,hidden}),eps,x.size(1)});
        return Tensors{result[0].view(x.sizes()),result[1].view({x.size(0),x.size(1)})};
    });
    m.def("areno_rmsnorm_forward", [](Tensor x, Tensor w, double eps) { return forward(x, w, {}, eps, 1); });
    m.def("areno_optional_scale_rmsnorm_forward", [](Tensor x, Tensor w, double eps, bool scale) {
        return forward(x, w, {}, eps, scale ? 1 : 0);
    });
    m.def("areno_rmsnorm_silu_gate_forward", [](Tensor x, Tensor g, Tensor w, double eps) { return forward(x, w, g, eps, 2); });
    m.def("areno_rmsnorm_backward", [](Tensor dy, Tensor x, Tensor w, Tensor inv) { return backward(dy, x, w, {}, inv, 1); });
    m.def("areno_optional_scale_rmsnorm_backward", [](Tensor dy, Tensor x, Tensor w, Tensor inv, bool scale) {
        return backward(dy, x, w, {}, inv, scale ? 1 : 0);
    });
    m.def("areno_rmsnorm_silu_gate_backward", [](Tensor dy, Tensor x, Tensor g, Tensor w, Tensor inv) {
        return backward(dy, x, w, g, inv, 2);
    });
}
