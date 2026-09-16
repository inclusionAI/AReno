#pragma once
#include "normalization_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_linear(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance,
    const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    const bool forward = binary.direction == 0;
    const unsigned inputs = forward ? 2 : 1;
    if (params->inputTensorNr != inputs) {
        params->inputTensorNr = inputs;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (params->outputTensorNr != 1) {
        params->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const auto* config = static_cast<const NormalizationParams*>(params->nodeParams.nodeParams);
    if (config == nullptr || config->hidden <= 0 || config->rows <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    const uint64_t hidden = config->hidden, rows = config->rows;
    const auto dtype = binary.dtype == 0 ? DATA_F32 : binary.dtype == 1 ? DATA_BF16 : DATA_F16;
    bool size_mismatch = false, dtype_mismatch = false;
    auto shape = [&](auto& tensor, bool matrix) {
        auto& geometry = tensor.geometry;
        dtype_mismatch |= geometry.dataType != dtype;
        geometry.dataType = dtype;
        size_mismatch |= geometry.dims != (matrix ? 2 : 1) || geometry.maxSizes[0] != hidden
            || (matrix && geometry.maxSizes[1] != rows);
        geometry.dims = matrix ? 2 : 1;
        geometry.maxSizes[0] = hidden;
        if (matrix) geometry.maxSizes[1] = rows;
    };
    shape(params->inputTensors[0], true);
    if (forward) shape(params->inputTensors[1], false);
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    shape(params->outputTensors[0], forward);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank = forward ? 2 : 1;
    instance->indexSpaceGeometry[0] = (hidden + 127) / 128;
    if (forward) instance->indexSpaceGeometry[1] = rows;
    auto map = [&](auto& pattern, bool matrix) {
        pattern = {};
        pattern.mapping[0].indexSpaceDim = 0;
        pattern.mapping[0].a = 128;
        pattern.mapping[0].end_b = 127;
        if (matrix) {
            pattern.mapping[1].indexSpaceDim = forward ? 1 : 0;
            pattern.mapping[1].a = forward ? 1 : 0;
            pattern.mapping[1].end_b = forward ? 0 : rows - 1;
        }
    };
    map(instance->inputTensorAccessPattern[0], true);
    if (forward) map(instance->inputTensorAccessPattern[1], false);
    map(instance->outputTensorAccessPattern[0], forward);
    instance->kernel.paramsNr = sizeof(NormalizationParams) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, config, sizeof(NormalizationParams));
    return GLUE_SUCCESS;
}
