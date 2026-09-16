#include "native.h"
#include "attention_params.h"
#include <limits>

namespace {
using namespace areno_hpu;
template<class F> void for_each_attention(F&& fn) {
    for (int kind : {0, 1, 2}) for (bool backward : {false, true}) for (auto dtype : {"f32", "bf16", "f16"}) {
        if (kind == 2 && backward) continue;
        auto base = kind == 0 ? "areno_attention" : kind == 1 ? "areno_attention_packed" : "areno_attention_paged";
        fn(std::string(base) + (backward ? "_grad_" : "_") + dtype, kind, backward);
    }
}

Tensors execute(Tensor q, Tensor k, Tensor v, Tensor cu, Tensor grad, Tensor out,
                int64_t query_start, int64_t window, double scale, bool packed, bool backward) {
    for (auto tensor : {q, k, v}) check_tensor(tensor, q, q.scalar_type());
    dtype_suffix(q.scalar_type());
    int rank = packed ? 3 : 4;
    TORCH_CHECK(q.dim() == rank && k.dim() == rank && v.sizes() == k.sizes(), "attention q/k/v shape mismatch");
    TORCH_CHECK(q.size(-1) > 0 && q.size(-1) == k.size(-1), "attention head dimension mismatch");
    auto q_heads = q.size(1), kv_heads = k.size(1);
    auto q_length = q.size(packed ? 0 : 2), k_length = k.size(packed ? 0 : 2);
    const auto limit = std::numeric_limits<int>::max();
    TORCH_CHECK(q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0, "attention q heads must be divisible by kv heads");
    TORCH_CHECK(q.size(0) == k.size(0), "attention batch/token count mismatch");
    TORCH_CHECK(q.numel()/q.size(-1) <= limit && k.numel()/k.size(-1) <= limit && q.size(-1) <= limit,
                "attention shape exceeds TPC indexing");
    TORCH_CHECK(window >= -1 && window <= limit, "attention window exceeds TPC indexing");
    int64_t sequences = 0;
    if (packed) {
        check_tensor(cu, q, at::kInt);
        TORCH_CHECK(cu.dim() == 1 && cu.numel() >= 2 && cu.numel() <= limit, "attention requires int32 sequence boundaries");
        sequences = cu.numel()-1;
    } else {
        TORCH_CHECK(q_heads == kv_heads && query_start >= 0 && query_start <= k_length-q_length,
                    "dense attention query positions or head counts mismatch");
    }
    if (backward) {
        for (auto tensor : {grad, out}) {
            check_tensor(tensor, q, q.scalar_type());
            TORCH_CHECK(tensor.sizes() == q.sizes(), "attention output/gradient shape mismatch");
        }
    }
    if (q.numel() == 0) {
        if (!backward) return {at::empty_like(q)};
        return {at::zeros_like(q, q.options().dtype(at::kFloat)), at::zeros_like(k, k.options().dtype(at::kFloat)),
                at::zeros_like(v, v.options().dtype(at::kFloat))};
    }
    at::Stack args{q.view({-1, q.size(-1)}), k.view({-1, k.size(-1)}), v.view({-1, v.size(-1)})};
    if (backward) { args.emplace_back(grad.view({-1, q.size(-1)})); args.emplace_back(out.view({-1, q.size(-1)})); }
    if (packed) args.emplace_back(cu);
    for (auto value : {q_heads, kv_heads, q_length, k_length, query_start, window, sequences}) args.emplace_back(value);
    args.emplace_back(scale);
    args.emplace_back(0);
    args.emplace_back(0);
    auto result = call(std::string(packed ? "areno_attention_packed" : "areno_attention") + (backward ? "_grad" : ""), q.scalar_type(), args);
    result[0] = result[0].view(q.sizes());
    if (backward) { result[1] = result[1].view(k.sizes()); result[2] = result[2].view(v.sizes()); }
    return result;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    for_each_attention([&](const std::string& name, int kind, bool backward) {
        std::string signature = "(Tensor q, Tensor k, Tensor v, ";
        if (backward) signature += "Tensor grad, Tensor output, ";
        if (kind == 1) signature += "Tensor cu, ";
        if (kind == 2) signature += "Tensor table, Tensor lengths, ";
        signature += "int q_heads, int kv_heads, int q_length, int k_length, int query_start, int window, int sequences, float scale, int block_size, int max_blocks) -> ";
        signature += backward ? "(Tensor, Tensor, Tensor)" : "Tensor";
        m.def((name+signature).c_str());
        habana::custom_op::registerUserCustomOp("custom_op::"+name, name,
            [backward](const at::Stack& args) {
                Metadata result;
                for (int i=0; i<(backward ? 3 : 1); ++i) {
                    auto tensor = args[i].toTensor();
                    result.push_back({backward ? at::kFloat : tensor.scalar_type(), tensor.sizes().vec()});
                }
                return result;
            }, [kind, backward](const at::Stack& args, size_t& size) -> std::shared_ptr<void> {
                int first = (backward ? 5 : 3)+kind;
                size = sizeof(AttentionParams);
                return std::make_shared<AttentionParams>(AttentionParams{
                    static_cast<int>(args[0].toTensor().size(0)), static_cast<int>(args[1].toTensor().size(0)),
                    static_cast<int>(args[0].toTensor().size(1)),
                    static_cast<int>(args[first].toInt()), static_cast<int>(args[first+1].toInt()),
                    static_cast<int>(args[first+2].toInt()), static_cast<int>(args[first+3].toInt()),
                    static_cast<int>(args[first+4].toInt()), static_cast<int>(args[first+5].toInt()),
                    static_cast<int>(args[first+6].toInt()), static_cast<float>(args[first+7].toDouble()),
                    static_cast<int>(args[first+8].toInt()), static_cast<int>(args[first+9].toInt())});
            });
    });
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    for_each_attention([&](const std::string& name, int, bool) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>());
    });
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    for_each_attention([&](const std::string& name, int, bool) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>());
    });
}
void bind_attention(pybind11::module_& m) {
    m.def("areno_causal_attention_forward", [](Tensor q, Tensor k, Tensor v, int64_t start, int64_t window, double scale) {
        return execute(q,k,v,{},{},{},start,window,scale,false,false)[0];
    });
    m.def("areno_causal_attention_backward", [](Tensor grad, Tensor q, Tensor k, Tensor v, Tensor out, int64_t start, int64_t window, double scale) {
        return execute(q,k,v,{},grad,out,start,window,scale,false,true);
    });
    m.def("areno_varlen_causal_attention_forward", [](Tensor q, Tensor k, Tensor v, Tensor cu, int64_t window, double scale) {
        return execute(q,k,v,cu,{},{},0,window,scale,true,false)[0];
    });
    m.def("areno_varlen_causal_attention_backward", [](Tensor grad, Tensor q, Tensor k, Tensor v, Tensor out, Tensor cu, int64_t window, double scale) {
        return execute(q,k,v,cu,grad,out,0,window,scale,true,true);
    });
}
