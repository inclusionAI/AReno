#pragma once
#include "moe_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_moe(
    tpc_lib_api::HabanaKernelParams* params,tpc_lib_api::HabanaKernelInstantiation* instance,const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    int kind=binary.kind;
    unsigned inputs=kind < 2 ? 2 : kind < 4 ? 4 : 3,outputs=kind < 2 || kind == 4 ? 1 : kind == 2 ? 3 : 4;
    if (params->inputTensorNr != inputs) { params->inputTensorNr=inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != outputs) { params->outputTensorNr=outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* c=static_cast<const MoeParams*>(params->nodeParams.nodeParams);
    if (!c || c->tokens <= 0 || c->top_k <= 0 || (kind < 4 && c->experts <= 0)
        || (kind >= 2 && c->rows <= 0) || (kind >= 2 && kind < 4 && c->hidden <= 0)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    auto dtype=binary.dtype == 0 ? DATA_F32 : binary.dtype == 1 ? DATA_BF16 : DATA_F16;
    bool size_mismatch=false,dtype_mismatch=false;
    auto geometry=[&](auto& tensor,auto type,uint64_t width,uint64_t rows=1,unsigned dims=1) {
        dtype_mismatch |= tensor.geometry.dataType != type; tensor.geometry.dataType=type;
        size_mismatch |= tensor.geometry.dims != dims || tensor.geometry.maxSizes[0] != width || (dims == 2 && tensor.geometry.maxSizes[1] != rows);
        tensor.geometry.dims=dims; tensor.geometry.maxSizes[0]=width; if (dims == 2) tensor.geometry.maxSizes[1]=rows;
    };
    if (kind < 4) {
        unsigned first=kind < 2 ? 0 : 1;
        if (kind >= 2) geometry(params->inputTensors[0],dtype,c->hidden,c->tokens,2);
        geometry(params->inputTensors[first],kind%2 ? DATA_I64 : DATA_U8,c->top_k,c->tokens,2);
        geometry(params->inputTensors[first+1],DATA_F32,c->top_k,c->tokens,2);
        if (kind >= 2) geometry(params->inputTensors[3],DATA_I64,c->experts);
    } else {
        geometry(params->inputTensors[0],DATA_F32,c->rows); geometry(params->inputTensors[1],DATA_I64,c->rows);
        geometry(params->inputTensors[2],DATA_I32,c->rows);
    }
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (kind < 2) geometry(params->outputTensors[0],DATA_I64,c->experts);
    else if (kind < 4) {
        geometry(params->outputTensors[0],dtype,c->hidden,c->rows,2);
        geometry(params->outputTensors[1],DATA_F32,c->rows); geometry(params->outputTensors[2],DATA_I64,c->rows);
        if (kind == 3) geometry(params->outputTensors[3],DATA_I32,c->rows);
    } else geometry(params->outputTensors[0],DATA_F32,c->top_k,c->tokens,2);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank=kind >= 2 && kind < 4 ? 2 : 1;
    instance->indexSpaceGeometry[0]=kind < 2 ? c->experts : kind < 4 ? (uint64_t(c->hidden)+127)/128 : c->rows;
    if (instance->indexSpaceRank == 2) instance->indexSpaceGeometry[1]=c->experts;
    for (unsigned i=0; i<inputs; ++i) { instance->inputTensorAccessPattern[i]={}; instance->inputTensorAccessPattern[i].allRequired=true; }
    for (unsigned i=0; i<outputs; ++i) {
        auto& pattern=instance->outputTensorAccessPattern[i]; pattern={};
        if (kind < 2) { pattern.mapping[0].indexSpaceDim=0; pattern.mapping[0].a=1; }
        else { pattern.allRequired=true; pattern.memsetBeforeExecution=kind == 4; }
    }
    instance->kernel.paramsNr=sizeof(MoeParams)/sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams,c,sizeof(MoeParams));
    return GLUE_SUCCESS;
}
