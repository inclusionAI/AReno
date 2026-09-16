#pragma once
#include "fused_moe_params.h"
static tpc_lib_api::GlueCodeReturn instantiate_fused_moe(
    tpc_lib_api::HabanaKernelParams* params,tpc_lib_api::HabanaKernelInstantiation* instance,const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    unsigned inputs=binary.kind == 0 ? 4 : 1;
    if (params->inputTensorNr != inputs) { params->inputTensorNr=inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != 1) { params->outputTensorNr=1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* c=static_cast<const FusedMoeParams*>(params->nodeParams.nodeParams);
    if (!c || c->rows <= 0 || c->tokens <= 0 || c->hidden <= 0 || c->top_k <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    auto dtype=binary.dtype == 0 ? DATA_F32 : binary.dtype == 1 ? DATA_BF16 : DATA_F16;
    bool sizes=false,types=false;
    auto geometry=[&](auto& tensor,auto type,uint64_t width,uint64_t rows=1,unsigned dims=1) {
        types |= tensor.geometry.dataType != type; tensor.geometry.dataType=type;
        sizes |= tensor.geometry.dims != dims || tensor.geometry.maxSizes[0] != width || (dims == 2 && tensor.geometry.maxSizes[1] != rows);
        tensor.geometry.dims=dims; tensor.geometry.maxSizes[0]=width; if (dims == 2) tensor.geometry.maxSizes[1]=rows;
    };
    geometry(params->inputTensors[0],binary.kind == 0 ? DATA_F32 : dtype,c->hidden,binary.kind == 0 ? c->rows : uint64_t(c->tokens)*c->top_k,2);
    if (binary.kind == 0) {
        geometry(params->inputTensors[1],DATA_F32,c->rows); geometry(params->inputTensors[2],DATA_I64,c->rows);
        geometry(params->inputTensors[3],DATA_I32,c->rows);
    }
    if (sizes) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (types) return GLUE_INCOMPATIBLE_DATA_TYPE;
    geometry(params->outputTensors[0],dtype,c->hidden,uint64_t(c->tokens)*(binary.kind == 0 ? c->top_k : 1),2);
    if (sizes) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (types) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank=2; instance->indexSpaceGeometry[0]=(uint64_t(c->hidden)+127)/128;
    instance->indexSpaceGeometry[1]=binary.kind == 0 ? c->rows : c->tokens;
    for (unsigned i=0; i<inputs; ++i) { instance->inputTensorAccessPattern[i]={}; instance->inputTensorAccessPattern[i].allRequired=true; }
    auto& output=instance->outputTensorAccessPattern[0]; output={}; output.allRequired=true; output.memsetBeforeExecution=binary.kind == 0;
    instance->kernel.paramsNr=sizeof(FusedMoeParams)/sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams,c,sizeof(FusedMoeParams));
    return GLUE_SUCCESS;
}
