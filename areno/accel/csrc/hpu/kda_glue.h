#pragma once
#include "kda_params.h"
#include "state_update_params.h"

// These kernels use canonical FP32 matrices. Only metadata/index tensors differ.
struct RecurrentGeometry {
    bool sizes=false,types=false;
    template<class T,class D> void set(T& tensor,D type,uint64_t width,uint64_t rows=1,unsigned dims=2) {
        types |= tensor.geometry.dataType != type; tensor.geometry.dataType=type;
        sizes |= tensor.geometry.dims != dims || tensor.geometry.maxSizes[0] != width || (dims == 2 && tensor.geometry.maxSizes[1] != rows);
        tensor.geometry.dims=dims; tensor.geometry.maxSizes[0]=width; if (dims == 2) tensor.geometry.maxSizes[1]=rows;
    }
};
static void recurrent_access(tpc_lib_api::HabanaKernelInstantiation* instance,unsigned inputs,unsigned outputs) {
    for (unsigned i=0; i<inputs; ++i) { instance->inputTensorAccessPattern[i]={}; instance->inputTensorAccessPattern[i].allRequired=true; }
    for (unsigned i=0; i<outputs; ++i) { instance->outputTensorAccessPattern[i]={}; instance->outputTensorAccessPattern[i].allRequired=true; }
}
static tpc_lib_api::GlueCodeReturn instantiate_kda_prepare(
    tpc_lib_api::HabanaKernelParams* params,tpc_lib_api::HabanaKernelInstantiation* instance,const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    unsigned outputs=binary.direction ? 6 : 1;
    if (params->inputTensorNr != 6) { params->inputTensorNr=6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != outputs) { params->outputTensorNr=outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* c=static_cast<const KdaPrepareParams*>(params->nodeParams.nodeParams);
    if (!c || c->tokens <= 0 || c->q_heads <= 0 || c->heads%c->q_heads || c->heads <= 0 || c->key_dim <= 0 || c->key_dim > 512)
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    RecurrentGeometry g;
    for (int i=0; i<2; ++i) g.set(params->inputTensors[i],DATA_F32,c->key_dim,uint64_t(c->tokens)*c->q_heads);
    g.set(params->inputTensors[2],DATA_F32,c->key_dim,uint64_t(c->tokens)*c->heads);
    g.set(params->inputTensors[3],DATA_F32,c->heads,1,1); g.set(params->inputTensors[4],DATA_F32,c->key_dim,c->heads);
    g.set(params->inputTensors[5],DATA_F32,binary.direction ? 3*c->key_dim+1 : 1,uint64_t(c->tokens)*c->heads);
    if (g.sizes) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (g.types) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (!binary.direction) g.set(params->outputTensors[0],DATA_F32,3*c->key_dim+1,uint64_t(c->tokens)*c->heads);
    else {
        for (int i=0; i<3; ++i) g.set(params->outputTensors[i],DATA_F32,c->key_dim,uint64_t(c->tokens)*(i == 2 ? c->heads : c->q_heads));
        g.set(params->outputTensors[3],DATA_F32,c->heads,1,1); g.set(params->outputTensors[4],DATA_F32,c->key_dim,c->heads);
        g.set(params->outputTensors[5],DATA_F32,1,uint64_t(c->tokens)*c->heads);
    }
    if (g.sizes) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (g.types) return GLUE_INCOMPATIBLE_DATA_TYPE;
    recurrent_access(instance,6,outputs);
    if (binary.direction) for (int i : {0,1,3,4}) instance->outputTensorAccessPattern[i].memsetBeforeExecution=true;
    instance->indexSpaceRank=1; instance->indexSpaceGeometry[0]=uint64_t(c->tokens)*c->heads;
    instance->kernel.paramsNr=sizeof(KdaPrepareParams)/sizeof(uint32_t); std::memcpy(instance->kernel.scalarParams,c,sizeof(KdaPrepareParams));
    return GLUE_SUCCESS;
}
static tpc_lib_api::GlueCodeReturn instantiate_kda(
    tpc_lib_api::HabanaKernelParams* params,tpc_lib_api::HabanaKernelInstantiation* instance,const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    unsigned inputs=binary.direction ? 7 : 5;
    if (params->inputTensorNr != inputs) { params->inputTensorNr=inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != 3) { params->outputTensorNr=3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* c=static_cast<const KdaParams*>(params->nodeParams.nodeParams);
    if (!c || c->tokens <= 0 || c->heads <= 0 || c->key_dim <= 0 || c->key_dim > 512 || c->value_dim <= 0 || c->sequences <= 0 || c->slots <= 0)
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    uint64_t rows=uint64_t(c->tokens)*c->heads,state_rows=uint64_t(c->slots)*c->heads*c->value_dim;
    uint64_t final_rows=uint64_t(c->sequences)*c->heads*c->value_dim;
    RecurrentGeometry g;
    g.set(params->inputTensors[0],DATA_F32,3*c->key_dim+1,rows); g.set(params->inputTensors[1],DATA_F32,c->value_dim,rows);
    g.set(params->inputTensors[2],DATA_F32,c->key_dim,binary.direction ? rows*c->value_dim : state_rows);
    unsigned metadata=binary.direction ? 5 : 3;
    if (binary.direction) { g.set(params->inputTensors[3],DATA_F32,c->value_dim,rows); g.set(params->inputTensors[4],DATA_F32,c->key_dim,final_rows); }
    g.set(params->inputTensors[metadata],DATA_I32,c->sequences+1,1,1); g.set(params->inputTensors[metadata+1],DATA_I64,c->sequences,1,1);
    if (g.sizes) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (g.types) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (!binary.direction) {
        g.set(params->outputTensors[0],DATA_F32,c->value_dim,rows); g.set(params->outputTensors[1],DATA_F32,c->key_dim,final_rows);
        g.set(params->outputTensors[2],DATA_F32,c->key_dim,c->save_history ? rows*c->value_dim : 1);
    } else {
        g.set(params->outputTensors[0],DATA_F32,3*c->key_dim+1,rows); g.set(params->outputTensors[1],DATA_F32,c->value_dim,rows);
        g.set(params->outputTensors[2],DATA_F32,c->key_dim,state_rows);
    }
    if (g.sizes) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (g.types) return GLUE_INCOMPATIBLE_DATA_TYPE;
    recurrent_access(instance,inputs,3);
    if (binary.direction) for (int i : {0,2}) instance->outputTensorAccessPattern[i].memsetBeforeExecution=true;
    instance->indexSpaceRank=1; instance->indexSpaceGeometry[0]=final_rows;
    instance->kernel.paramsNr=sizeof(KdaParams)/sizeof(uint32_t); std::memcpy(instance->kernel.scalarParams,c,sizeof(KdaParams));
    return GLUE_SUCCESS;
}
static tpc_lib_api::GlueCodeReturn instantiate_state_update(
    tpc_lib_api::HabanaKernelParams* params,tpc_lib_api::HabanaKernelInstantiation* instance,const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    if (params->inputTensorNr != 3) { params->inputTensorNr=3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != 1) { params->outputTensorNr=1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* c=static_cast<const StateUpdateParams*>(params->nodeParams.nodeParams);
    if (!c || c->slots <= 0 || c->sequences <= 0 || c->width <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    auto dtype=binary.dtype == 0 ? DATA_F32 : binary.dtype == 1 ? DATA_BF16 : DATA_F16;
    RecurrentGeometry g;
    g.set(params->inputTensors[0],dtype,c->width,c->slots); g.set(params->inputTensors[1],dtype,c->width,c->sequences);
    g.set(params->inputTensors[2],DATA_I64,c->sequences,1,1);
    if (g.sizes) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (g.types) return GLUE_INCOMPATIBLE_DATA_TYPE;
    g.set(params->outputTensors[0],dtype,c->width,c->slots);
    if (g.sizes) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (g.types) return GLUE_INCOMPATIBLE_DATA_TYPE;
    recurrent_access(instance,3,1);
    instance->indexSpaceRank=2; instance->indexSpaceGeometry[0]=(uint64_t(c->width)+127)/128; instance->indexSpaceGeometry[1]=c->slots;
    instance->kernel.paramsNr=sizeof(StateUpdateParams)/sizeof(uint32_t); std::memcpy(instance->kernel.scalarParams,c,sizeof(StateUpdateParams));
    return GLUE_SUCCESS;
}
