#pragma once
#include "conv_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_conv(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance, const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    bool backward=binary.direction == 1;
    unsigned inputs=(backward ? 4 : 2)+(binary.kind != 0);
    if (params->inputTensorNr != inputs) { params->inputTensorNr=inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != 2) { params->outputTensorNr=2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* c=static_cast<const ConvParams*>(params->nodeParams.nodeParams);
    if (!c || c->tokens <= 0 || c->channels <= 0 || c->kernel <= 0 || c->length <= 0
        || (binary.kind == 1 && c->sequences <= 0) || (binary.kind == 2 && (backward || c->kernel < 2))) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    auto dtype=binary.dtype == 0 ? DATA_F32 : binary.dtype == 1 ? DATA_BF16 : DATA_F16;
    bool size_mismatch=false, dtype_mismatch=false;
    auto geometry=[&](auto& tensor,auto type,uint64_t width,uint64_t rows,unsigned dims=2) {
        dtype_mismatch |= tensor.geometry.dataType != type; tensor.geometry.dataType=type;
        size_mismatch |= tensor.geometry.dims != dims || tensor.geometry.maxSizes[0] != width
            || (dims == 2 && tensor.geometry.maxSizes[1] != rows);
        tensor.geometry.dims=dims; tensor.geometry.maxSizes[0]=width;
        if (dims == 2) tensor.geometry.maxSizes[1]=rows;
    };
    geometry(params->inputTensors[0],dtype,c->channels,c->tokens);
    geometry(params->inputTensors[1],DATA_F32,c->channels,c->kernel);
    if (backward) {
        geometry(params->inputTensors[2],dtype,c->channels,c->tokens);
        geometry(params->inputTensors[3],DATA_F32,c->channels,c->tokens);
    }
    if (binary.kind == 1) geometry(params->inputTensors[inputs-1],DATA_I32,uint64_t(c->sequences)+1,1,1);
    if (binary.kind == 2) geometry(params->inputTensors[inputs-1],dtype,c->channels,uint64_t(c->tokens)*(c->kernel-1));
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    geometry(params->outputTensors[0],dtype,c->channels,c->tokens);
    geometry(params->outputTensors[1],DATA_F32,c->channels,backward ? c->kernel : c->tokens);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank=2;
    instance->indexSpaceGeometry[0]=(uint64_t(c->channels)+127)/128;
    instance->indexSpaceGeometry[1]=backward ? std::max(c->tokens,c->kernel) : c->tokens;
    for (unsigned i=0; i<inputs; ++i) { instance->inputTensorAccessPattern[i]={}; instance->inputTensorAccessPattern[i].allRequired=true; }
    for (unsigned i=0; i<2; ++i) {
        auto& pattern=instance->outputTensorAccessPattern[i]; pattern={};
        pattern.mapping[0].indexSpaceDim=0; pattern.mapping[0].a=128; pattern.mapping[0].end_b=127;
        pattern.mapping[1].indexSpaceDim=1; pattern.mapping[1].a=1;
    }
    instance->kernel.paramsNr=sizeof(ConvParams)/sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams,c,sizeof(ConvParams));
    return GLUE_SUCCESS;
}
