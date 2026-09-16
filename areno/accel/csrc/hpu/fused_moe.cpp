#include "native.h"
#include "fused_moe_params.h"
#include <limits>

namespace {
using namespace areno_hpu;
template<class F> void each(F&& fn) {
    for (int kind=0; kind<2; ++kind) for (auto suffix : {"f32","bf16","f16"})
        fn(std::string(kind == 0 ? "areno_moe_weighted_" : "areno_moe_reduce_")+suffix,kind);
}
Tensor fused(Tensor x,Tensor w1,Tensor w2,Tensor weights,Tensor ids,pybind11::object config,const std::string& activation) {
    x=x.contiguous(); weights=weights.contiguous(); ids=ids.to(at::kLong).contiguous();
    check_tensor(x,x,x.scalar_type()); check_tensor(w1,x,x.scalar_type()); check_tensor(w2,x,x.scalar_type());
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,"fused_experts requires fp16/bf16 hidden_states");
    TORCH_CHECK(x.dim() == 2 && w1.dim() == 3 && w2.dim() == 3 && ids.dim() == 2,"fused_experts expects matrix inputs and grouped weights");
    int64_t experts=w1.size(0),hidden=x.size(1),intermediate=w2.size(2),tokens=x.size(0),k=ids.size(1);
    TORCH_CHECK(w2.size(0) == experts && w1.size(1) == 2*intermediate && w1.size(2) == hidden && w2.size(1) == hidden
        && ids.size(0) == tokens && ids.sizes() == weights.sizes() && k > 0 && experts > 0 && intermediate > 0,"fused_experts shape mismatch");
    TORCH_CHECK(config.attr("num_experts").cast<int64_t>() == experts && config.attr("hidden_size").cast<int64_t>() == hidden
        && config.attr("intermediate_size").cast<int64_t>() == intermediate && config.attr("top_k").cast<int64_t>() == k,"fused_experts config shape mismatch");
    TORCH_CHECK(activation == "silu" || activation == "gelu_tanh","unsupported fused MoE activation: ",activation);
    double scale=config.attr("routed_scaling_factor").cast<double>();
    if (!tokens) return at::empty_like(x);
    auto routed=moe_topk_permute(x,ids,weights,0,experts);
    int64_t rows=routed[0].size(0);
    if (!rows) return at::zeros_like(x);
    auto up=grouped_forward(routed[0],w1,routed[4]);
    auto activated=call(activation == "silu" ? "areno_silu_and_mul" : "areno_gelu_tanh_and_mul",x.scalar_type(),
        {up.view({rows,2,intermediate})})[0].view({rows,intermediate});
    // Keep the down accumulator in FP32 until routing weights are applied, as
    // CUDA does. The native MME FP32 path needs extra cast buffers on HPU.
    auto down=grouped_forward(activated.to(at::kFloat),w2.to(at::kFloat),routed[4]);
    auto expanded=call("areno_moe_weighted",x.scalar_type(),{down,routed[1],routed[2],routed[3],rows,tokens,hidden,k,scale})[0];
    return call("areno_moe_reduce",x.scalar_type(),{expanded,rows,tokens,hidden,k,scale})[0];
}
} // namespace
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    each([&](const std::string& name,int kind) {
        m.def((name+(kind == 0 ? "(Tensor down, Tensor weights, Tensor ids, Tensor positions, " : "(Tensor input, ")
            +"int rows, int tokens, int hidden, int top_k, float scale) -> Tensor").c_str());
        auto suffix=name.substr(name.rfind('_')+1);
        at::ScalarType dtype=suffix == "f32" ? at::kFloat : suffix == "bf16" ? at::kBFloat16 : at::kHalf;
        habana::custom_op::registerUserCustomOp("custom_op::"+name,name,
            [kind,dtype](const at::Stack& args) {
                int offset=kind == 0 ? 4 : 1;
                return Metadata{{dtype,{args[offset+1].toInt()*(kind == 0 ? args[offset+3].toInt() : 1),args[offset+2].toInt()}}};
            },[kind](const at::Stack& args,size_t& size)->std::shared_ptr<void> {
                int o=kind == 0 ? 4 : 1; size=sizeof(FusedMoeParams);
                return std::make_shared<FusedMoeParams>(FusedMoeParams{static_cast<int>(args[o].toInt()),static_cast<int>(args[o+1].toInt()),
                    static_cast<int>(args[o+2].toInt()),static_cast<int>(args[o+3].toInt()),static_cast<float>(args[o+4].toDouble())});
            });
    });
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) { each([&](const std::string& n,int) { m.impl(n.c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>()); }); }
TORCH_LIBRARY_IMPL(custom_op,Meta,m) { each([&](const std::string& n,int) { m.impl(n.c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>()); }); }
void bind_fused_moe(pybind11::module_& m) {
    m.def("areno_fused_experts",&fused,pybind11::arg("hidden_states"),pybind11::arg("w1"),pybind11::arg("w2"),
        pybind11::arg("topk_weights"),pybind11::arg("topk_ids"),pybind11::arg("config"),pybind11::kw_only(),pybind11::arg("activation")="silu");
}
