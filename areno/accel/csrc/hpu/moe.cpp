#include "native.h"
#include "moe_params.h"
#include <limits>

namespace {
using namespace areno_hpu;
const char* names[]={"areno_moe_dense_counts","areno_moe_topk_counts","areno_moe_dense_permute","areno_moe_topk_permute","areno_moe_weight_grad"};
template<class F> void for_each_moe(F&& fn) {
    for (int kind=0; kind<5; ++kind) for (auto dtype : {"f32","bf16","f16"}) {
        if ((kind < 2 || kind == 4) && std::string(dtype) != "f32") continue;
        fn(std::string(names[kind])+"_"+dtype,kind);
    }
}
void check_matrix(Tensor x) {
    check_tensor(x,x,x.scalar_type()); dtype_suffix(x.scalar_type());
    TORCH_CHECK(x.dim() == 2 && x.size(1) > 0 && x.size(0) <= std::numeric_limits<int>::max()
        && x.size(1) <= std::numeric_limits<int>::max(),"MoE input shape exceeds TPC indexing");
}
Tensors permute(Tensor x,Tensor routes,Tensor weights,int64_t start,int64_t experts,int64_t expected_rows,bool topk) {
    check_matrix(x); check_tensor(weights,x,at::kFloat); check_tensor(routes,x,topk ? at::kLong : at::kBool);
    TORCH_CHECK(routes.dim() == 2 && routes.sizes() == weights.sizes() && routes.size(0) == x.size(0),"MoE routing shape mismatch");
    const auto limit=std::numeric_limits<int>::max();
    TORCH_CHECK(experts >= 0 && start >= 0 && start <= limit && experts <= limit-start && routes.numel() <= limit,
                "MoE routing dimensions exceed TPC indexing");
    auto map=topk ? routes : routes.to(at::kByte);
    auto tokens=x.size(0),hidden=x.size(1),k=routes.size(1);
    Tensor counts;
    int64_t rows=0;
    if (tokens && k && experts) {
        counts=call(names[topk ? 1 : 0],at::kFloat,{map,weights,tokens,hidden,experts,k,start,0})[0];
        // CUDA also transfers route counts before allocating data-dependent outputs.
        auto host=counts.to(at::kCPU);
        auto data=host.data_ptr<int64_t>();
        for (int64_t e=0; e<experts; ++e) rows+=data[e];
    } else counts=at::zeros({experts},x.options().dtype(at::kLong));
    TORCH_CHECK(topk || expected_rows == rows,"MoE num_out_tokens must equal the number of active routes");
    Tensors result;
    if (!rows) {
        result={at::empty({0,hidden},x.options()),at::empty({0},x.options().dtype(at::kFloat)),at::empty({0},x.options().dtype(at::kLong))};
        if (topk) result.emplace_back(at::empty({0},x.options().dtype(at::kInt)));
    } else result=call(names[topk ? 3 : 2],x.scalar_type(),{x,map,weights,counts,tokens,hidden,experts,k,start,rows});
    if (topk) result.emplace_back(counts);
    return result;
}
Tensor scatter(Tensor x,Tensor ids,int64_t tokens,int64_t hidden) {
    check_matrix(x); check_tensor(ids,x,at::kLong);
    TORCH_CHECK(ids.dim() == 1 && ids.numel() == x.size(0) && hidden == x.size(1)
        && tokens >= 0 && tokens <= std::numeric_limits<int>::max(),"MoE scatter shape mismatch");
    if (x.size(0) == 0 || tokens == 0) return at::zeros({tokens,hidden},x.options());
    return call("areno_embedding_grad",x.scalar_type(),{ids,x,0,tokens})[0];
}
Tensor gather(Tensor x,Tensor ids) {
    check_matrix(x); check_tensor(ids,x,at::kLong);
    TORCH_CHECK(ids.dim() == 1 && ids.numel() <= std::numeric_limits<int>::max(),"MoE gather indices shape mismatch");
    if (ids.numel() == 0 || x.size(0) == 0) return at::zeros({ids.numel(),x.size(1)},x.options());
    return call("areno_embedding",x.scalar_type(),{ids,x,0,x.size(0)})[0];
}
Tensor weight_backward(Tensor grad,Tensor ids,Tensor positions,int64_t tokens,int64_t k) {
    check_tensor(grad,grad,at::kFloat); check_tensor(ids,grad,at::kLong); check_tensor(positions,grad,at::kInt);
    const auto limit=std::numeric_limits<int>::max();
    TORCH_CHECK(grad.dim() == 1 && ids.sizes() == grad.sizes() && positions.sizes() == grad.sizes()
        && grad.numel() <= limit && tokens >= 0 && tokens <= limit && k > 0 && k <= limit,"MoE route gradient shape mismatch");
    if (grad.numel() == 0) return at::zeros({tokens,k},grad.options());
    return call(names[4],at::kFloat,{grad,ids,positions,tokens,0,0,k,0,grad.numel()})[0];
}
} // namespace

namespace areno_hpu {
Tensors moe_topk_permute(Tensor x,Tensor ids,Tensor weights,int64_t start,int64_t experts) { return permute(x,ids,weights,start,experts,0,true); }
Tensor moe_scatter(Tensor x,Tensor ids,int64_t tokens) { return scatter(x,ids,tokens,x.size(1)); }
} // namespace areno_hpu

TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    for_each_moe([&](const std::string& name,int kind) {
        std::string signature=kind < 2 ? "(Tensor routes, Tensor weights, " : kind < 4 ? "(Tensor input, Tensor routes, Tensor weights, Tensor counts, "
            : "(Tensor grad, Tensor ids, Tensor positions, ";
        signature+="int tokens, int hidden, int experts, int top_k, int start, int rows) -> ";
        signature+=kind < 2 || kind == 4 ? "Tensor" : kind == 2 ? "(Tensor, Tensor, Tensor)" : "(Tensor, Tensor, Tensor, Tensor)";
        m.def((name+signature).c_str());
        habana::custom_op::registerUserCustomOp("custom_op::"+name,name,
            [kind](const at::Stack& args) {
                int first=kind < 2 ? 2 : kind < 4 ? 4 : 3;
                if (kind < 2) return Metadata{{at::kLong,{args[first+2].toInt()}}};
                if (kind == 4) return Metadata{{at::kFloat,{args[first].toInt(),args[first+3].toInt()}}};
                int64_t rows=args[first+5].toInt(),hidden=args[first+1].toInt();
                Metadata result{{args[0].toTensor().scalar_type(),{rows,hidden}},{at::kFloat,{rows}},{at::kLong,{rows}}};
                if (kind == 3) result.push_back({at::kInt,{rows}});
                return result;
            }, [kind](const at::Stack& args,size_t& size)->std::shared_ptr<void> {
                int first=kind < 2 ? 2 : kind < 4 ? 4 : 3; size=sizeof(MoeParams);
                return std::make_shared<MoeParams>(MoeParams{static_cast<int>(args[first].toInt()),static_cast<int>(args[first+1].toInt()),
                    static_cast<int>(args[first+2].toInt()),static_cast<int>(args[first+3].toInt()),static_cast<int>(args[first+4].toInt()),static_cast<int>(args[first+5].toInt())});
            });
    });
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    for_each_moe([&](const std::string& name,int) { m.impl(name.c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>()); });
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    for_each_moe([&](const std::string& name,int) { m.impl(name.c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>()); });
}
void bind_moe(pybind11::module_& m) {
    m.def("areno_moe_permute_forward",[](Tensor x,Tensor weights,Tensor map,int64_t rows) { return permute(x,map,weights,0,map.size(1),rows,false); });
    m.def("areno_moe_topk_permute_forward",&areno_hpu::moe_topk_permute);
    m.def("areno_moe_unpermute_forward",&scatter);
    m.def("areno_moe_gather_by_token_index",&gather);
    m.def("areno_moe_topk_weight_backward",&weight_backward);
}
