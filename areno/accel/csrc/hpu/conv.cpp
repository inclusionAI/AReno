#include "native.h"
#include "conv_params.h"
#include <limits>

namespace {
using namespace areno_hpu;
std::string base_name(int kind) { return kind == 0 ? "areno_conv" : kind == 1 ? "areno_conv_packed" : "areno_conv_decode"; }
template<class F> void for_each_conv(F&& fn) {
    for (int kind : {0,1,2}) for (bool backward : {false,true}) for (auto dtype : {"f32","bf16","f16"}) {
        if (kind == 2 && backward) continue;
        fn(base_name(kind)+(backward ? "_grad_" : "_")+dtype,kind,backward);
    }
}
Tensors execute(Tensor x, Tensor w, Tensor cu, Tensor history, Tensor grad, Tensor preact, int kind, bool backward) {
    check_tensor(x,x,x.scalar_type()); dtype_suffix(x.scalar_type()); check_tensor(w,x,at::kFloat);
    TORCH_CHECK(x.dim() == (kind == 2 ? 2 : 3) && w.dim() == 3 && w.size(1) == 1
        && x.size(-1) > 0 && x.size(-1) == w.size(0) && w.size(2) > 0, "causal convolution shape mismatch");
    const int64_t channels=x.size(-1), tokens=x.numel()/channels, kernel=w.size(2), length=kind == 2 ? 1 : x.size(1);
    const auto limit=std::numeric_limits<int>::max();
    TORCH_CHECK(channels <= limit && tokens <= limit && kernel <= limit && length <= limit, "convolution shape exceeds TPC indexing");
    int64_t sequences=0;
    if (kind == 1) {
        check_tensor(cu,x,at::kInt);
        TORCH_CHECK(x.size(0) == 1 && cu.dim() == 1 && cu.numel() >= 2 && cu.numel() <= limit, "packed convolution boundaries mismatch");
        sequences=cu.numel()-1;
    }
    if (kind == 2) {
        check_tensor(history,x,x.scalar_type());
        TORCH_CHECK(history.dim() == 3 && history.size(0) == x.size(0) && history.size(1) == channels
            && history.size(2) == kernel-1 && history.numel()/channels <= limit, "convolution history shape mismatch");
        if (kernel == 1) kind=0;
    }
    if (backward) {
        check_tensor(grad,x,x.scalar_type()); check_tensor(preact,x,at::kFloat);
        TORCH_CHECK(grad.sizes() == x.sizes() && preact.sizes() == x.sizes(), "convolution gradient shape mismatch");
    }
    if (tokens == 0) return {at::empty_like(x), backward ? at::zeros_like(w) : at::empty_like(x,x.options().dtype(at::kFloat))};
    auto weight=w.view({channels,kernel}).transpose(0,1).contiguous();
    at::Stack args{x.view({tokens,channels}),weight};
    if (backward) { args.emplace_back(grad.view({tokens,channels})); args.emplace_back(preact.view({tokens,channels})); }
    if (kind == 1) args.emplace_back(cu);
    if (kind == 2) args.emplace_back(history.transpose(1,2).contiguous().view({tokens*(kernel-1),channels}));
    args.emplace_back(length); args.emplace_back(sequences);
    auto result=call(base_name(kind)+(backward ? "_grad" : ""),x.scalar_type(),args);
    result[0]=result[0].view(x.sizes());
    result[1]=backward ? result[1].transpose(0,1).contiguous().view(w.sizes()) : result[1].view(x.sizes());
    return result;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    for_each_conv([&](const std::string& name,int kind,bool backward) {
        std::string signature="(Tensor input, Tensor weight, ";
        if (backward) signature+="Tensor grad, Tensor preact, ";
        if (kind == 1) signature+="Tensor cu, ";
        if (kind == 2) signature+="Tensor history, ";
        signature+="int length, int sequences) -> (Tensor, Tensor)";
        m.def((name+signature).c_str());
        habana::custom_op::registerUserCustomOp("custom_op::"+name,name,
            [backward](const at::Stack& args) {
                auto x=args[0].toTensor();
                return Metadata{{x.scalar_type(),x.sizes().vec()}, {at::kFloat,args[backward ? 1 : 0].toTensor().sizes().vec()}};
            }, [kind,backward](const at::Stack& args,size_t& size)->std::shared_ptr<void> {
                int first=(backward ? 4 : 2)+(kind != 0);
                size=sizeof(ConvParams);
                return std::make_shared<ConvParams>(ConvParams{static_cast<int>(args[0].toTensor().size(0)),
                    static_cast<int>(args[0].toTensor().size(1)),static_cast<int>(args[1].toTensor().size(0)),
                    static_cast<int>(args[first].toInt()),static_cast<int>(args[first+1].toInt())});
            });
    });
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    for_each_conv([&](const std::string& name,int,bool) { m.impl(name.c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>()); });
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    for_each_conv([&](const std::string& name,int,bool) { m.impl(name.c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>()); });
}
void bind_conv(pybind11::module_& m) {
    m.def("areno_depthwise_causal_conv1d_silu_forward",[](Tensor x,Tensor w) { return execute(x,w,{},{},{},{},0,false); });
    m.def("areno_depthwise_causal_conv1d_silu_decode",[](Tensor x,Tensor history,Tensor w) { return execute(x,w,{},history,{},{},2,false); });
    m.def("areno_packed_depthwise_causal_conv1d_silu_forward",[](Tensor x,Tensor w,Tensor cu) { return execute(x,w,cu,{},{},{},1,false); });
    m.def("areno_depthwise_causal_conv1d_silu_backward",[](Tensor grad,Tensor x,Tensor w,Tensor preact) { return execute(x,w,{},{},grad,preact,0,true); });
    m.def("areno_packed_depthwise_causal_conv1d_silu_backward",[](Tensor grad,Tensor x,Tensor w,Tensor cu,Tensor preact) { return execute(x,w,cu,{},grad,preact,1,true); });
}
