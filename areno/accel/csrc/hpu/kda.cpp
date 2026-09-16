#include "native.h"
#include "kda_params.h"
#include "state_update_params.h"
#include <torch/csrc/autograd/custom_function.h>
#include <cmath>
#include <limits>

namespace {
using namespace areno_hpu;
using Dims=std::vector<int64_t>;
using Context=torch::autograd::AutogradContext;
using Variables=torch::autograd::variable_list;

at::Stack prepare_args(const Tensors& inputs,const Dims& dims,double bound,double beta,double threshold) {
    at::Stack args; for (auto& input : inputs) args.emplace_back(input);
    for (auto dim : dims) args.emplace_back(dim);
    args.emplace_back(bound); args.emplace_back(beta); args.emplace_back(threshold); return args;
}
at::Stack recurrent_args(const Tensors& inputs,const Dims& dims,double scale) {
    at::Stack args; for (auto& input : inputs) args.emplace_back(input);
    for (auto dim : dims) args.emplace_back(dim);
    args.emplace_back(scale); return args;
}
class Prepare : public torch::autograd::Function<Prepare> {
public:
    static Tensor forward(Context* ctx,Tensor q,Tensor k,Tensor g,Tensor a,Tensor bias,Tensor beta,Dims dims,double bound,double sb,double threshold) {
        ctx->save_for_backward({q,k,g,a,bias});
        ctx->saved_data["dims"]=dims; ctx->saved_data["bound"]=bound; ctx->saved_data["sb"]=sb; ctx->saved_data["threshold"]=threshold;
        return call("areno_kda_prepare",at::kFloat,prepare_args({q,k,g,a,bias,beta},dims,bound,sb,threshold))[0];
    }
    static Variables backward(Context* ctx,Variables grads) {
        auto inputs=ctx->get_saved_variables(); inputs.push_back(grads[0].contiguous());
        auto result=call("areno_kda_prepare_grad",at::kFloat,prepare_args(inputs,ctx->saved_data["dims"].toIntVector(),
            ctx->saved_data["bound"].toDouble(),ctx->saved_data["sb"].toDouble(),ctx->saved_data["threshold"].toDouble()));
        result.resize(10); return result;
    }
};
class Recurrence : public torch::autograd::Function<Recurrence> {
public:
    static Variables forward(Context* ctx,Tensor prepared,Tensor v,Tensor initial,Tensor cu,Tensor indices,Dims dims,double scale) {
        auto result=call("areno_kda",at::kFloat,recurrent_args({prepared,v,initial,cu,indices},dims,scale));
        ctx->save_for_backward({prepared,v,result[2],cu,indices}); ctx->saved_data["dims"]=dims; ctx->saved_data["scale"]=scale;
        ctx->set_materialize_grads(false);
        return {result[0],result[1]};
    }
    static Variables backward(Context* ctx,Variables grads) {
        auto saved=ctx->get_saved_variables(); auto dims=ctx->saved_data["dims"].toIntVector();
        if (!grads[0].defined()) grads[0]=at::zeros_like(saved[1]);
        if (!grads[1].defined()) grads[1]=at::zeros({dims[4]*dims[1]*dims[3],dims[2]},saved[0].options());
        auto result=call("areno_kda_grad",at::kFloat,recurrent_args({saved[0],saved[1],saved[2],grads[0].contiguous(),grads[1].contiguous(),saved[3],saved[4]},
            dims,ctx->saved_data["scale"].toDouble()));
        result.resize(7); return result;
    }
};
int storage_dtype(Tensor value) {
    dtype_suffix(value.scalar_type()); return value.scalar_type() == at::kFloat ? 0 : value.scalar_type() == at::kBFloat16 ? 1 : 2;
}
Tensor canonical(Tensor input,Tensor reference) {
    TORCH_CHECK(input.device() == reference.device() && input.device().type() == c10::DeviceType::HPU,"KDA tensors must be on the same HPU");
    dtype_suffix(input.scalar_type()); return input.to(at::kFloat).contiguous();
}
Tensors run(Tensor q,Tensor k,Tensor v,Tensor g,Tensor beta,c10::optional<Tensor> initial,c10::optional<Tensor> indices,
            bool normalize,c10::optional<double> scale,c10::optional<Tensor> cu,Tensor a,Tensor bias,c10::optional<double> bound,
            bool recurrent,double softplus_beta,double threshold) {
    TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() && v.dim() == 4 && v.size(0) == q.size(0) && v.size(1) == q.size(1),"KDA expects [batch, tokens, heads, channels]");
    const int64_t batch=q.size(0),length=q.size(1),qh=q.size(2),key=q.size(3),heads=v.size(2),value=v.size(3),tokens=batch*length;
    TORCH_CHECK(batch > 0 && qh > 0 && heads > 0 && heads%qh == 0 && key > 0 && key <= 512 && value > 0,"KDA head shape is unsupported (key dimension must be <= 512)");
    TORCH_CHECK(g.numel() == tokens*heads*key && beta.numel() == tokens*heads && a.numel() == heads && bias.numel() == heads*key,"KDA gate shape mismatch");
    TORCH_CHECK(softplus_beta > 0 && (!scale || *scale > 0),"KDA scale and softplus_beta must be positive");
    int qdtype=storage_dtype(q),gdtype=storage_dtype(g);
    auto query=canonical(q,q).view({tokens*qh,key}),keys=canonical(k,q).view({tokens*qh,key});
    auto values=canonical(v,q).view({tokens*heads,value}),gate=canonical(g,q).view({tokens*heads,key});
    auto betas=canonical(beta,q).view({tokens*heads,1}),rates=canonical(a,q).view({heads}),biases=canonical(bias,q).view({heads,key});
    Tensor boundaries;
    int64_t sequences=batch;
    if (cu) {
        TORCH_CHECK(cu->device() == q.device() && cu->dim() == 1 && cu->numel() > 1 && (cu->scalar_type() == at::kInt || cu->scalar_type() == at::kLong)
            && batch == 1,"packed KDA requires batch=1 and integral cu_seqlens");
        sequences=cu->numel()-1; boundaries=cu->to(at::kInt).contiguous();
    } else boundaries=at::arange(0,(batch+1)*std::max<int64_t>(length,1),std::max<int64_t>(length,1),q.options().dtype(at::kInt));
    Tensor state;
    int64_t slots=sequences;
    if (initial) {
        TORCH_CHECK(initial->dim() == 4 && initial->size(1) == heads && initial->size(2) == value && initial->size(3) == key && initial->size(0) > 0,"KDA state expects [slots, heads, value_dim, key_dim]");
        slots=initial->size(0); state=canonical(*initial,q).view({slots*heads*value,key});
    } else state=at::zeros({slots*heads*value,key},q.options().dtype(at::kFloat));
    Tensor mapping;
    if (indices) {
        TORCH_CHECK(indices->device() == q.device() && indices->dim() == 1 && indices->numel() == sequences
            && (indices->scalar_type() == at::kLong || indices->scalar_type() == at::kInt),"KDA state indices must match sequences");
        mapping=indices->to(at::kLong).contiguous();
    } else {
        TORCH_CHECK(slots >= sequences,"KDA state has fewer slots than sequences");
        mapping=at::arange(sequences,q.options().dtype(at::kLong));
    }
    const int64_t limit=std::numeric_limits<int>::max();
    TORCH_CHECK(tokens <= limit/heads && sequences <= limit/heads/value && slots <= limit/heads/value && tokens <= limit/heads/value,"KDA dimensions exceed native indexing");
    TORCH_CHECK(tokens > 0,"KDA requires at least one token; empty segments within packed batches are supported");
    Dims prep_dims{tokens,qh,heads,key,normalize,qdtype,gdtype,recurrent,bound.has_value()};
    auto prepared=Prepare::apply(query,keys,gate,rates,biases,betas,prep_dims,bound.value_or(0),softplus_beta,threshold);
    bool save=at::GradMode::is_enabled() && (prepared.requires_grad() || values.requires_grad() || state.requires_grad());
    Dims dims{tokens,heads,key,value,sequences,slots,save};
    auto result=Recurrence::apply(prepared,values,state,boundaries,mapping,dims,scale.value_or(1/std::sqrt(double(key))));
    return {result[0].view(v.sizes()).to(q.scalar_type()),result[1].view({sequences,heads,value,key}),mapping};
}

void register_prepare(torch::Library& m,bool backward) {
    std::string name=backward ? "areno_kda_prepare_grad_f32" : "areno_kda_prepare_f32";
    m.def((name+"(Tensor q, Tensor k, Tensor gate, Tensor a, Tensor bias, Tensor beta_or_grad, int tokens, int q_heads, int heads, int key_dim, int normalize, int storage_dtype, int gate_dtype, int recurrent, int bounded, float lower_bound, float softplus_beta, float threshold) -> "+(backward ? "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)" : "Tensor")).c_str());
    habana::custom_op::registerUserCustomOp("custom_op::"+name,name,[backward](const at::Stack& a) {
        int64_t rows=a[6].toInt()*a[8].toInt(),key=a[9].toInt();
        if (!backward) return Metadata{{at::kFloat,{rows,3*key+1}}};
        Metadata result; for (int i=0; i<5; ++i) result.push_back({at::kFloat,a[i].toTensor().sizes().vec()});
        result.push_back({at::kFloat,{rows,1}}); return result;
    },[](const at::Stack& a,size_t& size)->std::shared_ptr<void> {
        size=sizeof(KdaPrepareParams); auto p=std::make_shared<KdaPrepareParams>();
        p->tokens=a[6].toInt(); p->q_heads=a[7].toInt(); p->heads=a[8].toInt(); p->key_dim=a[9].toInt(); p->normalize=a[10].toInt();
        p->storage_dtype=a[11].toInt(); p->gate_dtype=a[12].toInt(); p->recurrent=a[13].toInt(); p->bounded=a[14].toInt();
        p->lower_bound=a[15].toDouble(); p->softplus_beta=a[16].toDouble(); p->softplus_threshold=a[17].toDouble(); return p;
    });
}
void register_recurrence(torch::Library& m,bool backward) {
    std::string name=backward ? "areno_kda_grad_f32" : "areno_kda_f32";
    std::string tensors=backward ? "Tensor history, Tensor grad_output, Tensor grad_final, " : "Tensor initial, ";
    m.def((name+"(Tensor prepared, Tensor v, "+tensors+"Tensor cu, Tensor indices, int tokens, int heads, int key_dim, int value_dim, int sequences, int slots, int save_history, float scale) -> (Tensor, Tensor, Tensor)").c_str());
    habana::custom_op::registerUserCustomOp("custom_op::"+name,name,[backward](const at::Stack& a) {
        int first=backward ? 7 : 5;
        int64_t rows=a[first].toInt()*a[first+1].toInt(),key=a[first+2].toInt(),value=a[first+3].toInt();
        if (backward) return Metadata{{at::kFloat,{rows,3*key+1}},{at::kFloat,{rows,value}},
            {at::kFloat,{a[first+5].toInt()*a[first+1].toInt()*value,key}}};
        return Metadata{{at::kFloat,{rows,value}},{at::kFloat,{a[first+4].toInt()*a[first+1].toInt()*value,key}},
            {at::kFloat,{a[first+6].toInt() ? rows*value : 1,key}}};
    },[backward](const at::Stack& a,size_t& size)->std::shared_ptr<void> {
        int f=backward ? 7 : 5; size=sizeof(KdaParams);
        return std::make_shared<KdaParams>(KdaParams{static_cast<int>(a[f].toInt()),static_cast<int>(a[f+1].toInt()),static_cast<int>(a[f+2].toInt()),
            static_cast<int>(a[f+3].toInt()),static_cast<int>(a[f+4].toInt()),static_cast<int>(a[f+5].toInt()),static_cast<int>(a[f+6].toInt()),static_cast<float>(a[f+7].toDouble())});
    });
}
const char* names[]={"areno_kda_prepare_f32","areno_kda_prepare_grad_f32","areno_kda_f32","areno_kda_grad_f32"};
} // namespace

namespace areno_hpu {
Tensor recurrent_state_update(Tensor old,Tensor updated,Tensor indices) {
    TORCH_CHECK(old.dim() >= 2 && updated.dim() == old.dim() && old.size(0) > 0 && updated.size(0) == indices.numel(),"recurrent state update shape mismatch");
    int64_t width=old.numel()/old.size(0);
    TORCH_CHECK(width <= std::numeric_limits<int>::max() && updated.numel() == updated.size(0)*width,"recurrent state update width mismatch");
    return call("areno_state_update",old.scalar_type(),{old.contiguous().view({old.size(0),width}),updated.contiguous().view({updated.size(0),width}),
        indices,old.size(0),updated.size(0),width})[0].view(old.sizes());
}
} // namespace areno_hpu
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    for (bool backward : {false,true}) { register_prepare(m,backward); register_recurrence(m,backward); }
    for (auto suffix : {"f32","bf16","f16"}) {
        std::string name=std::string("areno_state_update_")+suffix;
        m.def((name+"(Tensor old_state, Tensor new_state, Tensor indices, int slots, int sequences, int width) -> Tensor").c_str());
        habana::custom_op::registerUserCustomOp("custom_op::"+name,name,[](const at::Stack& a) { return Metadata{{a[0].toTensor().scalar_type(),a[0].toTensor().sizes().vec()}}; },
            [](const at::Stack& a,size_t& size)->std::shared_ptr<void> { size=sizeof(StateUpdateParams);
                return std::make_shared<StateUpdateParams>(StateUpdateParams{static_cast<int>(a[3].toInt()),static_cast<int>(a[4].toInt()),static_cast<int>(a[5].toInt())}); });
    }
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    for (auto name : names) m.impl(name,torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>());
    for (auto suffix : {"f32","bf16","f16"}) m.impl((std::string("areno_state_update_")+suffix).c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>());
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    for (auto name : names) m.impl(name,torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>());
    for (auto suffix : {"f32","bf16","f16"}) m.impl((std::string("areno_state_update_")+suffix).c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>());
}
void bind_kda(pybind11::module_& m) {
    namespace py=pybind11;
    m.def("chunk_kda",[](Tensor q,Tensor k,Tensor v,Tensor g,Tensor beta,c10::optional<Tensor> initial,c10::optional<Tensor> indices,bool final,
        bool normalize,c10::optional<double> scale,c10::optional<Tensor> cu,Tensor a,Tensor bias,c10::optional<double> bound) {
        auto out=run(q,k,v,g,beta,initial,indices,normalize,scale,cu,a,bias,bound,false,1,20);
        return std::make_tuple(out[0],final ? c10::optional<Tensor>(out[1]) : c10::nullopt);
    },py::arg("q"),py::arg("k"),py::arg("v"),py::arg("g"),py::arg("beta"),py::kw_only(),py::arg("initial_state")=py::none(),
        py::arg("initial_state_indices")=py::none(),py::arg("output_final_state")=false,py::arg("use_qk_l2norm_in_kernel")=true,
        py::arg("scale")=py::none(),py::arg("cu_seqlens")=py::none(),py::arg("A_log"),py::arg("dt_bias"),py::arg("lower_bound")=py::none());
    m.def("fused_sigmoid_gating_delta_rule_update",[](Tensor a,Tensor gate,Tensor bias,double sb,double threshold,Tensor q,Tensor k,Tensor v,Tensor beta,
        Tensor state,Tensor indices,c10::optional<double> scale,bool normalize,c10::optional<Tensor> cu,bool is_kda,c10::optional<double> bound) {
        TORCH_CHECK(is_kda,"the AReno recurrent entry point requires is_kda=True");
        at::NoGradGuard guard;
        auto out=run(q,k,v,gate,beta,state,indices,normalize,scale,cu,a,bias,bound,true,sb,threshold);
        state.copy_(recurrent_state_update(state,out[1].to(state.scalar_type()),out[2]));
        return out[0];
    },py::arg("A_log"),py::arg("a"),py::arg("dt_bias"),py::arg("softplus_beta"),py::arg("softplus_threshold"),py::arg("q"),py::arg("k"),
        py::arg("v"),py::arg("b"),py::arg("initial_state_source"),py::arg("initial_state_indices"),py::arg("scale")=py::none(),
        py::arg("use_qk_l2norm_in_kernel")=false,py::arg("cu_seqlens")=py::none(),py::arg("is_kda")=true,py::arg("lower_bound")=py::none());
}
