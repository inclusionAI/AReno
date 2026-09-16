#pragma once
#include "seg_la_params.h"
static tpc_lib_api::GlueCodeReturn instantiate_seg_la(
    tpc_lib_api::HabanaKernelParams* params,tpc_lib_api::HabanaKernelInstantiation* instance,const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    unsigned inputs=binary.kind == 3 ? 10 : 9,outputs=binary.kind == 2 ? 3 : 2;
    if (params->inputTensorNr != inputs) { params->inputTensorNr=inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != outputs) { params->outputTensorNr=outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* c=static_cast<const SegLaParams*>(params->nodeParams.nodeParams);
    if (!c || c->tokens <= 0 || c->heads <= 0 || c->hidden <= 0 || c->hidden > 512 || c->sequences <= 0 || c->slots <= 0
        || (binary.kind == 2 && c->steps <= 0) || (binary.kind == 3 && c->mask_size <= 0)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    RecurrentGeometry g;
    for (int i=0; i<3; ++i) g.set(params->inputTensors[i],DATA_F32,c->hidden,uint64_t(c->tokens)*c->heads);
    g.set(params->inputTensors[3],DATA_F32,c->hidden,uint64_t(c->slots)*c->heads*c->hidden);
    g.set(params->inputTensors[4],DATA_F32,c->heads,1,1); g.set(params->inputTensors[5],DATA_I32,c->sequences+1,1,1);
    g.set(params->inputTensors[6],DATA_I32,c->sequences,1,1); g.set(params->inputTensors[7],DATA_I64,c->sequences,1,1);
    g.set(params->inputTensors[8],DATA_F32,c->sequences,1,1);
    if (binary.kind == 3) g.set(params->inputTensors[9],DATA_U8,c->mask_size,uint64_t(c->sequences)*c->mask_size);
    if (g.sizes) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (g.types) return GLUE_INCOMPATIBLE_DATA_TYPE;
    g.set(params->outputTensors[0],DATA_F32,c->hidden,uint64_t(c->tokens)*c->heads);
    g.set(params->outputTensors[1],DATA_F32,c->hidden,uint64_t(c->sequences)*c->heads*c->hidden);
    if (binary.kind == 2) g.set(params->outputTensors[2],DATA_F32,c->hidden,uint64_t(c->sequences)*c->steps*c->heads*c->hidden);
    if (g.sizes) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (g.types) return GLUE_INCOMPATIBLE_DATA_TYPE;
    recurrent_access(instance,inputs,outputs);
    // Skipped request slots have defined zero outputs; state write-back ignores them.
    for (unsigned i=0; i<outputs; ++i) instance->outputTensorAccessPattern[i].memsetBeforeExecution=true;
    instance->indexSpaceRank=1; instance->indexSpaceGeometry[0]=uint64_t(c->sequences)*c->heads*c->hidden;
    instance->kernel.paramsNr=sizeof(SegLaParams)/sizeof(uint32_t); std::memcpy(instance->kernel.scalarParams,c,sizeof(SegLaParams));
    return GLUE_SUCCESS;
}
