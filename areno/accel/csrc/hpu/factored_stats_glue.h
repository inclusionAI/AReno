#pragma once
#include "factored_stats_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_factored_stats(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance, const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    const bool reduce = binary.kind == 1;
    const unsigned outputs = reduce ? 1 : 2;
    if (params->inputTensorNr != 2) { params->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != outputs) { params->outputTensorNr = outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* config = static_cast<const FactoredStatsParams*>(params->nodeParams.nodeParams);
    if (config == nullptr || config->count <= 0 || config->rows <= 0 || config->columns <= 0
        || config->parameter_shard_start < 0
        || uint64_t(config->parameter_shard_start) + config->count > uint64_t(config->rows) * config->columns) {
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    const uint64_t factors = uint64_t(config->rows) + config->columns;
    const auto dtype = binary.dtype == 0 ? DATA_F32 : DATA_BF16;
    bool size_mismatch = false, dtype_mismatch = false;
    auto geometry = [&](auto& tensor, auto type, uint64_t size) {
        dtype_mismatch |= tensor.geometry.dataType != type;
        tensor.geometry.dataType = type;
        size_mismatch |= tensor.geometry.dims != 1 || tensor.geometry.maxSizes[0] != size;
        tensor.geometry.dims = 1;
        tensor.geometry.maxSizes[0] = size;
    };
    geometry(params->inputTensors[0], reduce ? DATA_I32 : dtype, reduce ? factors : config->count);
    geometry(params->inputTensors[1], reduce ? DATA_I32 : DATA_F32, reduce ? 1 : factors);
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    geometry(params->outputTensors[0], reduce ? DATA_I32 : DATA_F32, reduce ? 1 : factors);
    if (!reduce) geometry(params->outputTensors[1], DATA_I32, factors);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = reduce ? 1 : factors;
    for (unsigned i=0; i<2; ++i) {
        instance->inputTensorAccessPattern[i] = {};
        instance->inputTensorAccessPattern[i].allRequired = true;
    }
    for (unsigned i=0; i<outputs; ++i) {
        auto& pattern = instance->outputTensorAccessPattern[i];
        pattern = {};
        pattern.mapping[0].indexSpaceDim = 0;
        pattern.mapping[0].a = 1;
    }
    instance->kernel.paramsNr = sizeof(FactoredStatsParams)/sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, config, sizeof(FactoredStatsParams));
    return GLUE_SUCCESS;
}
