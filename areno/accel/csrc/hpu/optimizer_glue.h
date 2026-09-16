#pragma once
#include "optimizer_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_optimizer(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance,
    const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    const bool master = binary.kind == 1;
    const unsigned inputs = master ? 6 : 4, outputs = master ? 5 : 3;
    if (params->inputTensorNr != inputs) {
        params->inputTensorNr = inputs;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (params->outputTensorNr != outputs) {
        params->outputTensorNr = outputs;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const auto* config = static_cast<const OptimizerParams*>(params->nodeParams.nodeParams);
    if (config == nullptr || config->count <= 0 || config->carry_offset < 0 || config->carry_offset >= 8) {
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    const uint64_t count = config->count, carries = (config->carry_offset + count + 7) / 8;
    const auto model_dtype = binary.dtype == 0 ? DATA_F32 : DATA_BF16;
    const auto grad_dtype = binary.direction == 0 ? DATA_F32 : DATA_BF16;
    bool size_mismatch = false, dtype_mismatch = false;
    auto geometry = [&](auto& tensor, auto dtype, uint64_t size) {
        dtype_mismatch |= tensor.geometry.dataType != dtype;
        tensor.geometry.dataType = dtype;
        size_mismatch |= tensor.geometry.dims != 1 || tensor.geometry.maxSizes[0] != size;
        tensor.geometry.dims = 1;
        tensor.geometry.maxSizes[0] = size;
    };
    geometry(params->inputTensors[0], model_dtype, count);
    geometry(params->inputTensors[1], grad_dtype, count);
    if (master) {
        geometry(params->inputTensors[2], DATA_U16, count);
        geometry(params->inputTensors[3], DATA_U8, carries);
    }
    geometry(params->inputTensors[inputs - 2], DATA_F32, count);
    geometry(params->inputTensors[inputs - 1], DATA_F32, count);
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    geometry(params->outputTensors[0], model_dtype, count);
    if (master) {
        geometry(params->outputTensors[1], DATA_U16, count);
        geometry(params->outputTensors[2], DATA_U8, carries);
    }
    geometry(params->outputTensors[outputs - 2], DATA_F32, count);
    geometry(params->outputTensors[outputs - 1], DATA_F32, count);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = master ? carries : (count + 127) / 128;
    auto map = [&](auto& pattern, bool carry) {
        pattern = {};
        auto& mapping = pattern.mapping[0];
        mapping.indexSpaceDim = 0;
        mapping.a = master ? (carry ? 1 : 8) : 128;
        mapping.start_b = master && !carry ? -config->carry_offset : 0;
        mapping.end_b = mapping.start_b + mapping.a - 1;
    };
    for (unsigned i=0; i<inputs; ++i) map(instance->inputTensorAccessPattern[i], master && i == 3);
    for (unsigned i=0; i<outputs; ++i) map(instance->outputTensorAccessPattern[i], master && i == 2);
    instance->kernel.paramsNr = sizeof(OptimizerParams) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, config, sizeof(OptimizerParams));
    return GLUE_SUCCESS;
}
