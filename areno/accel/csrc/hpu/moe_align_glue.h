#pragma once
#include "moe_align_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_moe_align(
    tpc_lib_api::HabanaKernelParams* params,tpc_lib_api::HabanaKernelInstantiation* instance,const KernelBinary&
) {
    using namespace tpc_lib_api;
    if (params->inputTensorNr != 4) { params->inputTensorNr=4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != 4) { params->outputTensorNr=4; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* c=static_cast<const MoeAlignParams*>(params->nodeParams.nodeParams);
    if (!c || c->routes <= 0 || c->expert_slots <= 0 || c->block <= 0 || c->scratch_size <= c->expert_slots
        || uint64_t(c->capacity) < uint64_t(c->routes)+uint64_t(c->expert_slots)*(c->block-1)
        || uint64_t(c->block_capacity)*c->block < uint64_t(c->capacity)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    bool size_mismatch=false,dtype_mismatch=false;
    auto geometry=[&](auto& tensor,auto type,uint64_t count) {
        dtype_mismatch |= tensor.geometry.dataType != type; tensor.geometry.dataType=type;
        size_mismatch |= tensor.geometry.dims != 1 || tensor.geometry.maxSizes[0] != count;
        tensor.geometry.dims=1; tensor.geometry.maxSizes[0]=count;
    };
    geometry(params->inputTensors[0],DATA_I64,c->routes);
    geometry(params->inputTensors[1],DATA_I32,c->capacity);
    geometry(params->inputTensors[2],DATA_I32,c->block_capacity);
    geometry(params->inputTensors[3],DATA_I32,c->scratch_size);
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    geometry(params->outputTensors[0],DATA_I32,c->capacity);
    geometry(params->outputTensors[1],DATA_I32,c->block_capacity);
    geometry(params->outputTensors[2],DATA_I32,1);
    geometry(params->outputTensors[3],DATA_I32,c->scratch_size);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank=1; instance->indexSpaceGeometry[0]=1;
    for (unsigned i=0; i<4; ++i) {
        instance->inputTensorAccessPattern[i]={}; instance->inputTensorAccessPattern[i].allRequired=true;
        instance->outputTensorAccessPattern[i]={}; instance->outputTensorAccessPattern[i].allRequired=true;
    }
    instance->kernel.paramsNr=sizeof(MoeAlignParams)/sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams,c,sizeof(MoeAlignParams));
    return GLUE_SUCCESS;
}
