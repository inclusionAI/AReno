#pragma once
#include "topk_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_topk(
    tpc_lib_api::HabanaKernelParams* params,tpc_lib_api::HabanaKernelInstantiation* instance,const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    bool backward=binary.direction == 1,grouped=binary.kind == 1;
    unsigned inputs=backward ? 3 : grouped ? 2 : 1,outputs=backward ? 1 : 2;
    if (params->inputTensorNr != inputs) { params->inputTensorNr=inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != outputs) { params->outputTensorNr=outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* c=static_cast<const TopkParams*>(params->nodeParams.nodeParams);
    if (!c || c->tokens <= 0 || c->experts <= 0 || c->experts > 512 || c->top_k <= 0 || c->top_k > 16
        || c->top_k > c->experts || (grouped && (c->groups <= 0 || c->groups > 64 || c->experts%c->groups
            || c->top_groups <= 0 || c->top_groups > c->groups || c->top_k/c->top_groups == 0))) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    auto dtype=binary.dtype == 0 ? DATA_F32 : binary.dtype == 1 ? DATA_BF16 : DATA_F16;
    bool size_mismatch=false,dtype_mismatch=false;
    auto geometry=[&](auto& tensor,auto type,uint64_t width,uint64_t rows,unsigned dims=2) {
        dtype_mismatch |= tensor.geometry.dataType != type; tensor.geometry.dataType=type;
        size_mismatch |= tensor.geometry.dims != dims || tensor.geometry.maxSizes[0] != width
            || (dims == 2 && tensor.geometry.maxSizes[1] != rows);
        tensor.geometry.dims=dims; tensor.geometry.maxSizes[0]=width;
        if (dims == 2) tensor.geometry.maxSizes[1]=rows;
    };
    geometry(params->inputTensors[0],dtype,c->experts,c->tokens);
    if (grouped) geometry(params->inputTensors[1],DATA_F32,c->experts,1,1);
    if (backward) { geometry(params->inputTensors[1],DATA_I64,c->top_k,c->tokens); geometry(params->inputTensors[2],DATA_F32,c->top_k,c->tokens); }
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    geometry(params->outputTensors[0],backward ? dtype : DATA_I64,backward ? c->experts : c->top_k,c->tokens);
    if (!backward) geometry(params->outputTensors[1],DATA_F32,c->top_k,c->tokens);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank=1; instance->indexSpaceGeometry[0]=c->tokens;
    auto map=[](auto& pattern,int width) {
        pattern={}; pattern.mapping[0].indexSpaceDim=0; pattern.mapping[0].end_b=width-1;
        pattern.mapping[1].indexSpaceDim=0; pattern.mapping[1].a=1;
    };
    for (unsigned i=0; i<inputs; ++i) {
        if (grouped && i == 1) { instance->inputTensorAccessPattern[i]={}; instance->inputTensorAccessPattern[i].allRequired=true; }
        else map(instance->inputTensorAccessPattern[i],i == 0 ? c->experts : c->top_k);
    }
    for (unsigned i=0; i<outputs; ++i) map(instance->outputTensorAccessPattern[i],backward ? c->experts : c->top_k);
    instance->kernel.paramsNr=sizeof(TopkParams)/sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams,c,sizeof(TopkParams));
    return GLUE_SUCCESS;
}
