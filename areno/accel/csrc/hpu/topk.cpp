#include "native.h"
#include "topk_params.h"
#include <limits>

namespace {
using namespace areno_hpu;
template<class F> void for_each_topk(F&& fn) {
    for (int kind : {0,1,2}) for (auto dtype : {"f32","bf16","f16"}) {
        auto base=kind == 0 ? "areno_topk" : kind == 1 ? "areno_topk_grad" : "areno_topk_grouped";
        fn(std::string(base)+"_"+dtype,kind);
    }
}
void check_logits(Tensor logits,int64_t top_k) {
    check_tensor(logits,logits,logits.scalar_type()); dtype_suffix(logits.scalar_type());
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) <= std::numeric_limits<int>::max()
        && logits.size(1) > 0 && logits.size(1) <= 512 && top_k > 0 && top_k <= 16 && top_k <= logits.size(1),
        "top-k supports at most 512 experts and 16 selected experts");
}
Tensors forward(Tensor logits,Tensor bias,int64_t top_k,bool renormalize,int64_t groups,int64_t top_groups) {
    check_logits(logits,top_k);
    bool grouped=bias.defined();
    if (grouped) {
        check_tensor(bias,logits,at::kFloat);
        TORCH_CHECK(bias.dim() == 1 && bias.numel() == logits.size(1) && groups > 0 && groups <= 64
            && logits.size(1)%groups == 0 && top_groups > 0 && top_groups <= groups && top_k/top_groups > 0,
            "grouped top-k group configuration mismatch");
    }
    if (logits.size(0) == 0) return {at::empty({0,top_k},logits.options().dtype(at::kLong)),at::empty({0,top_k},logits.options().dtype(at::kFloat))};
    at::Stack args{logits}; if (grouped) args.emplace_back(bias);
    args.emplace_back(top_k); args.emplace_back(renormalize); args.emplace_back(groups); args.emplace_back(top_groups);
    return call(grouped ? "areno_topk_grouped" : "areno_topk",logits.scalar_type(),args);
}
Tensor backward(Tensor grad,Tensor logits,Tensor ids,bool renormalize) {
    TORCH_CHECK(ids.dim() == 2,"top-k indices must be a matrix"); check_logits(logits,ids.size(1));
    check_tensor(ids,logits,at::kLong); check_tensor(grad,logits,at::kFloat);
    TORCH_CHECK(ids.size(0) == logits.size(0) && grad.sizes() == ids.sizes(),"top-k gradient shape mismatch");
    if (logits.size(0) == 0) return at::empty_like(logits);
    return call("areno_topk_grad",logits.scalar_type(),{logits,ids,grad,ids.size(1),renormalize,0,0})[0];
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    for_each_topk([&](const std::string& name,int kind) {
        std::string signature="(Tensor logits, ";
        if (kind == 1) signature+="Tensor ids, Tensor grad, ";
        if (kind == 2) signature+="Tensor bias, ";
        signature+="int top_k, bool renormalize, int groups, int top_groups) -> ";
        signature+=kind == 1 ? "Tensor" : "(Tensor, Tensor)";
        m.def((name+signature).c_str());
        habana::custom_op::registerUserCustomOp("custom_op::"+name,name,
            [kind](const at::Stack& args) {
                auto x=args[0].toTensor(); int first=kind == 1 ? 3 : kind == 2 ? 2 : 1;
                if (kind == 1) return Metadata{{x.scalar_type(),x.sizes().vec()}};
                auto shape=std::vector<int64_t>{x.size(0),args[first].toInt()};
                return Metadata{{at::kLong,shape},{at::kFloat,shape}};
            }, [kind](const at::Stack& args,size_t& size)->std::shared_ptr<void> {
                int first=kind == 1 ? 3 : kind == 2 ? 2 : 1; size=sizeof(TopkParams);
                return std::make_shared<TopkParams>(TopkParams{static_cast<int>(args[0].toTensor().size(0)),static_cast<int>(args[0].toTensor().size(1)),
                    static_cast<int>(args[first].toInt()),args[first+1].toBool(),static_cast<int>(args[first+2].toInt()),static_cast<int>(args[first+3].toInt())});
            });
    });
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    for_each_topk([&](const std::string& name,int) { m.impl(name.c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>()); });
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    for_each_topk([&](const std::string& name,int) { m.impl(name.c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>()); });
}
void bind_topk(pybind11::module_& m) {
    m.def("areno_topk_softmax_forward",[](Tensor x,int64_t top_k,bool renormalize) { return forward(x,{},top_k,renormalize,0,0); });
    m.def("areno_topk_softmax_backward",&backward);
    m.def("areno_grouped_topk_router",[](Tensor x,Tensor bias,int64_t top_k,int64_t groups,int64_t top_groups) {
        return forward(x,bias,top_k,true,groups,top_groups);
    });
}
