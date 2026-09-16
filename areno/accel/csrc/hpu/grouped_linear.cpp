// Like CUDA's grouped cublas loop, dispatch one native MME GEMM per expert.
#include "native.h"

namespace {
using namespace areno_hpu;
using Counts = std::vector<int64_t>;

void check_grouped(Tensor x, Tensor weight, const Counts& counts) {
    check_tensor(x, x, x.scalar_type());
    check_tensor(weight, x, x.scalar_type());
    dtype_suffix(x.scalar_type());
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3 && x.size(1) == weight.size(2), "grouped linear shape mismatch");
    TORCH_CHECK(static_cast<int64_t>(counts.size()) == weight.size(0), "grouped linear expert count mismatch");
    int64_t total = 0;
    for (auto count : counts) {
        TORCH_CHECK(count >= 0 && count <= x.size(0)-total, "grouped linear token counts are out of bounds");
        total += count;
    }
    TORCH_CHECK(total == x.size(0), "grouped linear token counts must sum to input rows");
}

Counts materialize_counts(Tensor counts, Tensor weight) {
    TORCH_CHECK(counts.scalar_type() == at::kLong || counts.scalar_type() == at::kInt, "grouped linear counts must be int32 or int64");
    check_tensor(counts, weight, counts.scalar_type());
    TORCH_CHECK(counts.dim() == 1 && counts.numel() == weight.size(0), "grouped linear counts shape mismatch");
    // CUDA's counts entry point also materializes the small count vector on CPU.
    auto host = counts.to(at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    auto data = host.data_ptr<int64_t>();
    return Counts(data, data+host.numel());
}

Tensor forward(Tensor x, Tensor weight, const Counts& counts) {
    check_grouped(x, weight, counts);
    auto output = at::empty({x.size(0), weight.size(1)}, x.options());
    int64_t offset = 0;
    for (int64_t expert=0; expert<weight.size(0); ++expert) {
        auto count = counts[expert];
        if (count) output.narrow(0, offset, count).copy_(linear_forward(x.narrow(0, offset, count), weight.select(0, expert), {}, false));
        offset += count;
    }
    return output;
}

Tensors backward(Tensor grad, Tensor x, Tensor weight, const Counts& counts, bool need_x, bool need_w) {
    check_grouped(x, weight, counts);
    check_tensor(grad, x, x.scalar_type());
    TORCH_CHECK(grad.dim() == 2 && grad.size(0) == x.size(0) && grad.size(1) == weight.size(1), "grouped linear gradient shape mismatch");
    Tensors result(2);
    if (need_x) result[0] = at::empty_like(x);
    if (need_w) result[1] = at::zeros_like(weight);
    int64_t offset = 0;
    for (int64_t expert=0; expert<weight.size(0); ++expert) {
        auto count = counts[expert];
        if (count && (need_x || need_w)) {
            auto gradients = linear_backward(grad.narrow(0, offset, count), x.narrow(0, offset, count), weight.select(0, expert), false, need_x, need_w, false);
            if (need_x) result[0].narrow(0, offset, count).copy_(gradients[0]);
            if (need_w) result[1].select(0, expert).copy_(gradients[1]);
        }
        offset += count;
    }
    return result;
}
} // namespace

namespace areno_hpu {
Tensor grouped_forward(Tensor x,Tensor weight,Tensor counts) { return forward(x,weight,materialize_counts(counts,weight)); }
} // namespace areno_hpu

void bind_grouped_linear(pybind11::module_& m) {
    m.def("areno_grouped_linear_forward", &forward);
    m.def("areno_grouped_linear_backward", &backward);
    m.def("areno_grouped_linear_forward_counts", [](Tensor x, Tensor weight, Tensor counts) {
        return forward(x, weight, materialize_counts(counts, weight));
    });
    m.def("areno_grouped_linear_backward_counts", [](Tensor grad, Tensor x, Tensor weight, Tensor counts, bool need_x, bool need_w) {
        return backward(grad, x, weight, materialize_counts(counts, weight), need_x, need_w);
    });
}
