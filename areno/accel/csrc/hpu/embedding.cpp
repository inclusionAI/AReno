#include "native.h"
#include "embedding_params.h"
#include <limits>

namespace {
using namespace areno_hpu;
template<class F> void for_each_embedding(F&& fn) {
    for (bool backward : {false, true}) for (auto dtype : {"f32", "bf16", "f16"}) {
        fn(std::string(backward ? "areno_embedding_grad_" : "areno_embedding_") + dtype, backward);
    }
}
void check_embedding(Tensor ids, Tensor weight, int64_t start, int64_t end) {
    check_tensor(ids, weight, at::kLong);
    check_tensor(weight, weight, weight.scalar_type());
    dtype_suffix(weight.scalar_type());
    const auto limit = std::numeric_limits<int>::max();
    TORCH_CHECK(start >= 0 && end >= start && end <= limit, "embedding vocabulary range exceeds TPC indexing");
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == end-start && weight.size(1) <= limit && ids.numel() <= limit,
                "embedding weight shape or token count mismatch");
}
Tensor forward(Tensor ids, Tensor weight, int64_t start, int64_t end) {
    check_embedding(ids, weight, start, end);
    auto shape = ids.sizes().vec();
    shape.push_back(weight.size(1));
    if (ids.numel() == 0 || weight.numel() == 0) return at::zeros(shape, weight.options());
    return call("areno_embedding", weight.scalar_type(), {ids.view({-1}), weight, start, end})[0].view(shape);
}
Tensor backward(Tensor grad, Tensor ids, Tensor weight, int64_t start, int64_t end) {
    check_embedding(ids, weight, start, end);
    check_tensor(grad, weight, weight.scalar_type());
    auto shape = ids.sizes().vec();
    shape.push_back(weight.size(1));
    TORCH_CHECK(grad.sizes().vec() == shape, "embedding gradient shape mismatch");
    if (ids.numel() == 0 || weight.numel() == 0) return at::zeros_like(weight);
    return call("areno_embedding_grad", weight.scalar_type(),
                {ids.view({-1}), grad.view({ids.numel(), weight.size(1)}), start, end})[0];
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    for_each_embedding([&](const std::string& name, bool backward) {
        m.def((name + "(Tensor ids, Tensor values, int start, int end) -> Tensor").c_str());
        habana::custom_op::registerUserCustomOp("custom_op::" + name, name,
            [backward](const at::Stack& args) {
                auto values = args[1].toTensor();
                auto rows = backward ? args[3].toInt()-args[2].toInt() : args[0].toTensor().numel();
                return Metadata{{values.scalar_type(), {rows, values.size(1)}}};
            }, [](const at::Stack& args, size_t& size) -> std::shared_ptr<void> {
                size = sizeof(EmbeddingParams);
                return std::make_shared<EmbeddingParams>(EmbeddingParams{static_cast<int>(args[0].toTensor().numel()),
                    static_cast<int>(args[1].toTensor().size(1)), static_cast<int>(args[2].toInt()), static_cast<int>(args[3].toInt())});
            });
    });
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    for_each_embedding([&](const std::string& name, bool) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>());
    });
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    for_each_embedding([&](const std::string& name, bool) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>());
    });
}
void bind_embedding(pybind11::module_& m) {
    m.def("areno_vocab_embedding_forward", &forward);
    m.def("areno_vocab_embedding_backward", &backward);
}
