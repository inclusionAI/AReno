#include "native.h"
#include "optimizer_params.h"
#include <limits>

namespace {
using namespace areno_hpu;

template<class F> void for_each_optimizer(F&& fn) {
    for (auto grad : {"f32", "bf16"}) {
        for (auto model : {"f32", "bf16"}) {
            fn(std::string("areno_adamw_state_g") + grad + "_" + model, false);
        }
        fn(std::string("areno_adamw_master_g") + grad + "_bf16", true);
    }
}

void check_state(Tensor model, Tensor grad, Tensor avg, Tensor variance) {
    for (const auto& tensor : {model, grad}) {
        TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kBFloat16,
                    "native AdamW model and gradient must be float32 or bfloat16");
        check_tensor(tensor, model, tensor.scalar_type());
    }
    check_tensor(avg, model, at::kFloat);
    check_tensor(variance, model, at::kFloat);
    TORCH_CHECK(model.numel() == grad.numel() && model.numel() == avg.numel() && model.numel() == variance.numel(),
                "native AdamW tensor sizes must match");
    TORCH_CHECK(model.numel() <= std::numeric_limits<int>::max(), "AdamW shard exceeds TPC indexing");
}

void state_step(Tensor model, Tensor grad, Tensor avg, Tensor variance, double beta1, double beta2,
                double lr, double decay, double eps, double step_size, double correction) {
    check_state(model, grad, avg, variance);
    if (model.numel() == 0) return;
    auto result = call(std::string("areno_adamw_state_g") + dtype_suffix(grad.scalar_type()), model.scalar_type(),
        {model.view({-1}), grad.view({-1}), avg.view({-1}), variance.view({-1}), 0, beta1, beta2, lr, decay, eps, step_size, correction});
    model.copy_(result[0].view(model.sizes()));
    avg.copy_(result[1].view(avg.sizes()));
    variance.copy_(result[2].view(variance.sizes()));
}

void master_step(Tensor model, Tensor low, Tensor carry, Tensor grad, Tensor avg, Tensor variance,
                 int64_t offset, double beta1, double beta2, double lr, double decay, double eps,
                 double step_size, double correction) {
    const auto count = model.numel();
    TORCH_CHECK(offset >= 0 && offset <= low.numel() && count <= low.numel() - offset, "AdamW master state slice is out of bounds");
    TORCH_CHECK(avg.numel() == low.numel() && variance.numel() == low.numel(), "AdamW master state sizes must match");
    check_tensor(low, model, at::kUInt16);
    check_tensor(carry, model, at::kByte);
    TORCH_CHECK(carry.numel() >= (low.numel() + 7) / 8, "AdamW master carry buffer is too short");
    auto m = avg.view({-1}).narrow(0, offset, count);
    auto v = variance.view({-1}).narrow(0, offset, count);
    check_state(model, grad, m, v);
    if (count == 0) return;
    if (model.scalar_type() == at::kFloat) {
        state_step(model, grad, m, v, beta1, beta2, lr, decay, eps, step_size, correction);
        return;
    }
    auto low_slice = low.view({-1}).narrow(0, offset, count);
    auto carry_slice = carry.view({-1}).narrow(0, offset / 8, (offset % 8 + count + 7) / 8);
    auto result = call(std::string("areno_adamw_master_g") + dtype_suffix(grad.scalar_type()), model.scalar_type(),
        {model.view({-1}), grad.view({-1}), low_slice, carry_slice, m, v, offset % 8,
         beta1, beta2, lr, decay, eps, step_size, correction});
    model.copy_(result[0].view(model.sizes()));
    low_slice.copy_(result[1]);
    carry_slice.copy_(result[2]);
    m.copy_(result[3]);
    v.copy_(result[4]);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    for_each_optimizer([&](const std::string& name, bool master) {
        std::string schema = "(Tensor model, Tensor grad, ";
        if (master) schema += "Tensor low, Tensor carry, ";
        schema += "Tensor avg, Tensor variance, int carry_offset, float beta1, float beta2, float lr, float decay, float eps, float step_size, float correction) -> ";
        schema += master ? "(Tensor, Tensor, Tensor, Tensor, Tensor)" : "(Tensor, Tensor, Tensor)";
        m.def((name + schema).c_str());
        habana::custom_op::registerUserCustomOp("custom_op::" + name, name,
            [master](const at::Stack& args) {
                Metadata result;
                for (int index : master ? std::vector<int>{0, 2, 3, 4, 5} : std::vector<int>{0, 2, 3}) {
                    auto tensor = args[index].toTensor();
                    result.push_back({tensor.scalar_type(), tensor.sizes().vec()});
                }
                return result;
            }, [master](const at::Stack& args, size_t& size) -> std::shared_ptr<void> {
                const int first = master ? 6 : 4;
                size = sizeof(OptimizerParams);
                return std::make_shared<OptimizerParams>(OptimizerParams{
                    static_cast<int>(args[0].toTensor().numel()), static_cast<int>(args[first].toInt()),
                    static_cast<float>(args[first + 1].toDouble()), static_cast<float>(args[first + 2].toDouble()),
                    static_cast<float>(args[first + 3].toDouble()), static_cast<float>(args[first + 4].toDouble()),
                    static_cast<float>(args[first + 5].toDouble()), static_cast<float>(args[first + 6].toDouble()),
                    static_cast<float>(args[first + 7].toDouble())});
            });
    });
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    for_each_optimizer([&](const std::string& name, bool) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>());
    });
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    for_each_optimizer([&](const std::string& name, bool) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>());
    });
}
void bind_optimizer(pybind11::module_& m) {
    m.def("areno_adamw_fp32_state_step", &state_step);
    m.def("areno_adamw_fp32_master_step", &master_step);
}
