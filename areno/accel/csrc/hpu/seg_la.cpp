#include "native.h"
#include "seg_la_params.h"
#include <cmath>
#include <limits>
namespace {
using namespace areno_hpu;
const char* names[]={"areno_seg_la_prefill","areno_seg_la_decode","areno_seg_la_mtp","areno_seg_la_spec"};
Tensor canonical(Tensor x,Tensor q,at::ScalarType dtype) {
    TORCH_CHECK(x.device() == q.device() && q.device().type() == c10::DeviceType::HPU,"seg_la tensors must be on the same HPU");
    return x.to(dtype).contiguous();
}
Tensor forward(Tensor q,Tensor k,Tensor v,Tensor state,Tensor decay,pybind11::object meta,c10::optional<Tensor> caches,c10::optional<double> scale) {
    TORCH_CHECK(q.dim() == 3 && q.sizes() == k.sizes() && q.sizes() == v.sizes(),"seg_la requires matching [tokens, heads, hidden] Q/K/V; GQA is unsupported");
    dtype_suffix(q.scalar_type()); dtype_suffix(state.scalar_type());
    int64_t tokens=q.size(0),heads=q.size(1),hidden=q.size(2),sequences=meta.attr("batch_size").cast<int64_t>();
    TORCH_CHECK(heads > 0 && hidden > 0 && hidden <= 512 && sequences > 0,"seg_la requires positive dimensions and head_dim <= 512");
    TORCH_CHECK(state.dim() == 4 && state.size(0) > 0 && state.size(1) == heads && state.size(2) == hidden && state.size(3) == hidden
        && decay.numel() == heads,"seg_la state/decay shape mismatch");
    auto offsets=canonical(meta.attr("q_offsets").cast<Tensor>(),q,at::kInt).view({-1});
    auto lengths=canonical(meta.attr("q_lengths").cast<Tensor>(),q,at::kInt).view({-1});
    auto slots=canonical(meta.attr("s_offsets").cast<Tensor>(),q,at::kLong).view({-1});
    auto scales=canonical(meta.attr("s_scales").cast<Tensor>(),q,at::kFloat).view({-1});
    TORCH_CHECK(offsets.numel() == sequences+1 && lengths.numel() == sequences && slots.numel() == sequences && scales.numel() == sequences,"seg_la metadata shape mismatch");
    if (!tokens) return at::empty_like(q);
    int kind=(tokens+sequences-1)/sequences <= 1 ? 1 : caches ? 2 : !meta.attr("mask").is_none() ? 3 : 0;
    int64_t steps=kind == 2 ? tokens/sequences : 0,mask_size=0;
    if (kind == 1) TORCH_CHECK(tokens == sequences,"seg_la decode requires one token per request");
    if (caches) TORCH_CHECK(caches->device() == q.device(),"seg_la caches must be on the same HPU");
    if (kind == 2) TORCH_CHECK(tokens%sequences == 0 && caches->dim() == 5 && caches->size(0) == state.size(0)
        && caches->size(1) >= steps && caches->size(2) == heads && caches->size(3) == hidden && caches->size(4) == hidden,"seg_la MTP cache shape mismatch");
    const int64_t limit=std::numeric_limits<int>::max();
    TORCH_CHECK(tokens <= limit/heads/hidden && state.size(0) <= limit/heads/hidden && sequences <= limit/heads/hidden,"seg_la exceeds native indexing");
    at::Stack args{canonical(q,q,at::kFloat).view({tokens*heads,hidden}),canonical(k,q,at::kFloat).view({tokens*heads,hidden}),
        canonical(v,q,at::kFloat).view({tokens*heads,hidden}),canonical(state.transpose(-1,-2),q,at::kFloat).view({state.size(0)*heads*hidden,hidden}),
        canonical(decay,q,at::kFloat).view({heads}),offsets,lengths,slots,scales};
    if (kind == 3) {
        auto mask=meta.attr("mask").cast<Tensor>();
        TORCH_CHECK(mask.dim() == 3 && mask.size(0) == sequences && mask.size(1) == mask.size(2) && mask.size(1) > 0,"seg_la tree mask must be [batch, window, window]");
        mask_size=mask.size(1); args.emplace_back(canonical(mask,q,at::kByte).view({sequences*mask_size,mask_size}));
    }
    int dtype=q.scalar_type() == at::kFloat ? 0 : q.scalar_type() == at::kBFloat16 ? 1 : 2;
    for (auto dim : {tokens,heads,hidden,sequences,state.size(0),steps,mask_size,int64_t(dtype)}) args.emplace_back(dim);
    args.emplace_back(scale.value_or(1/std::sqrt(double(hidden))));
    auto result=call(names[kind],at::kFloat,args);
    if (kind < 2) {
        auto updated=result[1].view({sequences,heads,hidden,hidden}).transpose(-1,-2).to(state.scalar_type());
        state.copy_(recurrent_state_update(state,updated,slots));
    } else if (kind == 2) {
        auto updated=result[2].view({sequences,steps,heads,hidden,hidden}).transpose(-1,-2).to(caches->scalar_type());
        auto destination=caches->narrow(1,0,steps);
        destination.copy_(recurrent_state_update(destination,updated,slots));
    }
    return result[0].view(q.sizes()).to(q.scalar_type());
}
} // namespace
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    for (int kind=0; kind<4; ++kind) {
        std::string name=std::string(names[kind])+"_f32";
        m.def((name+"(Tensor q, Tensor k, Tensor v, Tensor initial, Tensor decay, Tensor offsets, Tensor lengths, Tensor slots, Tensor state_scales, "
            +(kind == 3 ? "Tensor mask, " : "")+"int tokens, int heads, int hidden, int sequences, int slot_count, int steps, int mask_size, int storage_dtype, float scale) -> "
            +(kind == 2 ? "(Tensor, Tensor, Tensor)" : "(Tensor, Tensor)")).c_str());
        habana::custom_op::registerUserCustomOp("custom_op::"+name,name,[kind](const at::Stack& a) {
            int f=kind == 3 ? 10 : 9; int64_t n=a[f].toInt(),h=a[f+1].toInt(),d=a[f+2].toInt(),seq=a[f+3].toInt();
            Metadata result{{at::kFloat,{n*h,d}},{at::kFloat,{seq*h*d,d}}};
            if (kind == 2) result.push_back({at::kFloat,{seq*a[f+5].toInt()*h*d,d}}); return result;
        },[kind](const at::Stack& a,size_t& size)->std::shared_ptr<void> {
            int f=kind == 3 ? 10 : 9; size=sizeof(SegLaParams);
            return std::make_shared<SegLaParams>(SegLaParams{static_cast<int>(a[f].toInt()),static_cast<int>(a[f+1].toInt()),static_cast<int>(a[f+2].toInt()),
                static_cast<int>(a[f+3].toInt()),static_cast<int>(a[f+4].toInt()),static_cast<int>(a[f+5].toInt()),static_cast<int>(a[f+6].toInt()),static_cast<int>(a[f+7].toInt()),static_cast<float>(a[f+8].toDouble())});
        });
    }
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) { for (auto name : names) m.impl((std::string(name)+"_f32").c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>()); }
TORCH_LIBRARY_IMPL(custom_op,Meta,m) { for (auto name : names) m.impl((std::string(name)+"_f32").c_str(),torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>()); }
void bind_seg_la(pybind11::module_& m) { m.def("seg_la_fwd",&forward,pybind11::arg("q"),pybind11::arg("k"),pybind11::arg("v"),pybind11::arg("s"),
    pybind11::arg("decay_scales"),pybind11::arg("meta"),pybind11::arg("caches")=pybind11::none(),pybind11::arg("softmax_scale")=pybind11::none()); }
