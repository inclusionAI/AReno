#include "native.h"
#include "factored_stats_params.h"
#include "quantized_optimizer_params.h"
#include <limits>

namespace {
using namespace areno_hpu;

void check_matrix(int64_t count, int64_t start, int64_t rows, int64_t columns) {
    const int64_t limit = std::numeric_limits<int>::max();
    TORCH_CHECK(rows > 0 && columns > 0 && rows <= limit && columns <= limit / rows && rows <= limit - columns,
                "factored AdamW matrix exceeds TPC indexing");
    TORCH_CHECK(start >= 0 && start <= rows * columns && count <= rows * columns - start,
                "factored AdamW parameter shard is out of bounds");
}

void check_float(const Tensor& tensor, const Tensor& reference) {
    TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kBFloat16,
                "factored AdamW model and gradient must be float32 or bfloat16");
    check_tensor(tensor, reference, tensor.scalar_type());
}

Tensor slice(Tensor tensor, int64_t offset, int64_t count) {
    TORCH_CHECK(offset >= 0 && offset <= tensor.numel() && count <= tensor.numel() - offset,
                "factored AdamW moment slice is out of bounds");
    return tensor.view({-1}).narrow(0, offset, count);
}

template<class F> void for_each_factored(F&& fn) {
    for (auto dtype : {"f32", "bf16"}) fn(std::string("areno_adamw4_stats_") + dtype, 0);
    fn("areno_adamw4_invalid_f32", 1);
    for (auto grad : {"f32", "bf16"}) for (auto model : {"f32", "bf16"}) {
        fn(std::string("areno_adamw4_factored_g") + grad + "_" + model, 2);
    }
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    for_each_factored([&](const std::string& name, int kind) {
        std::string signature;
        if (kind == 0) signature = "(Tensor grad, Tensor sums, int start, int rows, int columns) -> (Tensor, Tensor)";
        else if (kind == 1) signature = "(Tensor mask, Tensor invalid, int count, int start, int rows, int columns) -> Tensor";
        else signature = "(Tensor model, Tensor grad, Tensor mq, Tensor ms, Tensor factors, Tensor mean, Tensor invalid, "
            "int block, int start, int rows, int columns, float beta1, float lr, float decay, float eps, float step_size, float correction) -> (Tensor, Tensor, Tensor)";
        m.def((name + signature).c_str());
        habana::custom_op::registerUserCustomOp("custom_op::" + name, name,
            [kind](const at::Stack& args) {
                if (kind == 0) {
                    auto shape = args[1].toTensor().sizes().vec();
                    return Metadata{{at::kFloat, shape}, {at::kInt, shape}};
                }
                if (kind == 1) return Metadata{{at::kInt, {1}}};
                Metadata result;
                for (int index : {0, 2, 3}) {
                    auto tensor = args[index].toTensor();
                    result.push_back({tensor.scalar_type(), tensor.sizes().vec()});
                }
                return result;
            }, [kind](const at::Stack& args, size_t& size) -> std::shared_ptr<void> {
                if (kind < 2) {
                    size = sizeof(FactoredStatsParams);
                    int first = kind == 0 ? 2 : 3;
                    auto count = kind == 0 ? args[0].toTensor().numel() : args[2].toInt();
                    return std::make_shared<FactoredStatsParams>(FactoredStatsParams{
                        static_cast<int>(count), static_cast<int>(args[first].toInt()),
                        static_cast<int>(args[first+1].toInt()), static_cast<int>(args[first+2].toInt())});
                }
                size = sizeof(QuantizedOptimizerParams);
                return std::make_shared<QuantizedOptimizerParams>(QuantizedOptimizerParams{
                    static_cast<int>(args[0].toTensor().numel()), static_cast<int>(args[7].toInt()),
                    static_cast<float>(args[11].toDouble()), 0.0f,
                    static_cast<float>(args[12].toDouble()), static_cast<float>(args[13].toDouble()),
                    static_cast<float>(args[14].toDouble()), static_cast<float>(args[15].toDouble()),
                    static_cast<float>(args[16].toDouble()), static_cast<int>(args[8].toInt()),
                    static_cast<int>(args[9].toInt()), static_cast<int>(args[10].toInt())});
            });
    });
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    for_each_factored([&](const std::string& name, int) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>());
    });
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    for_each_factored([&](const std::string& name, int) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>());
    });
}

void bind_factored_optimizer(pybind11::module_& m) {
    m.def("areno_adamw_4bit_factored_stats", [](Tensor grad, Tensor sums, Tensor invalid,
        int64_t start, int64_t rows, int64_t columns) {
        check_float(grad, grad);
        check_tensor(sums, grad, at::kFloat);
        check_tensor(invalid, grad, at::kInt);
        check_matrix(grad.numel(), start, rows, columns);
        TORCH_CHECK(sums.numel() == rows + columns && invalid.numel() == 1, "factored AdamW statistics size mismatch");
        if (grad.numel() == 0) return;
        auto result = call("areno_adamw4_stats", grad.scalar_type(), {grad.view({-1}), sums.view({-1}), start, rows, columns});
        auto flag = call("areno_adamw4_invalid", at::kFloat,
                         {result[1], invalid.view({-1}), grad.numel(), start, rows, columns});
        sums.copy_(result[0].view(sums.sizes()));
        invalid.copy_(flag[0].view(invalid.sizes()));
    });
    m.def("areno_adamw_4bit_factored_step", [](Tensor model, Tensor grad, Tensor mq, Tensor ms,
        Tensor factors, Tensor mean, Tensor invalid, int64_t mq_offset, int64_t ms_offset, int64_t start,
        int64_t block, int64_t rows, int64_t columns, double beta1, double lr, double decay,
        double eps, double step_size, double correction) {
        check_float(model, model);
        check_float(grad, model);
        check_tensor(mq, model, at::kByte);
        for (auto tensor : {ms, factors, mean}) check_tensor(tensor, model, at::kFloat);
        check_tensor(invalid, model, at::kInt);
        check_matrix(model.numel(), start, rows, columns);
        TORCH_CHECK(block >= 32 && block <= 1024 && (block & (block-1)) == 0, "invalid factored AdamW4 block size");
        TORCH_CHECK(grad.numel() == model.numel() && factors.numel() == rows + columns
                    && mean.numel() == 1 && invalid.numel() == 1, "factored AdamW state size mismatch");
        auto q = slice(mq, mq_offset, (model.numel()+1)/2);
        auto scale = slice(ms, ms_offset, (model.numel()+block-1)/block);
        if (model.numel() == 0) return;
        auto result = call(std::string("areno_adamw4_factored_g") + dtype_suffix(grad.scalar_type()), model.scalar_type(),
            {model.view({-1}), grad.view({-1}), q, scale, factors.view({-1}), mean.view({-1}), invalid.view({-1}),
             block, start, rows, columns, beta1, lr, decay, eps, step_size, correction});
        model.copy_(result[0].view(model.sizes()));
        q.copy_(result[1]);
        scale.copy_(result[2]);
    });
}
