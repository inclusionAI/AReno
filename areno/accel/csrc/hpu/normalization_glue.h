#pragma once
#include "normalization_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_normalization(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance,
    const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    const bool forward = binary.direction == 0;
    const bool dw = binary.direction == 2;
    const bool scaled = binary.kind >= 1;
    const bool gated = binary.kind >= 2;
    const unsigned input_count = forward ? 1 + scaled + gated : 3 + (scaled && !dw) + gated;
    const unsigned output_count = forward || (!dw && gated) ? 2 : 1;
    if (params->inputTensorNr != input_count) {
        params->inputTensorNr = input_count;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (params->outputTensorNr != output_count) {
        params->outputTensorNr = output_count;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const auto* config = static_cast<const NormalizationParams*>(params->nodeParams.nodeParams);
    if (config == nullptr || config->hidden <= 0 || config->rows <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    const uint64_t hidden = config->hidden, rows = config->rows;
    const auto storage_dtype = binary.dtype == 0 ? DATA_F32 : binary.dtype == 1 ? DATA_BF16 : DATA_F16;
    bool size_mismatch = false, dtype_mismatch = false;
    auto geometry = [&](auto& tensor, auto dtype, std::initializer_list<uint64_t> sizes) {
        dtype_mismatch |= tensor.geometry.dataType != dtype;
        tensor.geometry.dataType = dtype;
        size_mismatch |= tensor.geometry.dims != sizes.size();
        tensor.geometry.dims = sizes.size();
        unsigned dim = 0;
        for (auto size : sizes) {
            size_mismatch |= tensor.geometry.maxSizes[dim] != size;
            tensor.geometry.maxSizes[dim++] = size;
        }
    };
    geometry(params->inputTensors[0], storage_dtype, {hidden, rows});
    if (!forward) {
        geometry(params->inputTensors[1], storage_dtype, {hidden, rows});
        geometry(params->inputTensors[2], DATA_F32, {1, rows});
    }
    if (binary.kind == 3) {
        const auto* group_config=static_cast<const GroupNormParams*>(params->nodeParams.nodeParams);
        if (!forward || group_config->groups <= 0 || rows%group_config->groups != 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        geometry(params->inputTensors[1],DATA_F32,{hidden,uint64_t(group_config->groups)});
    } else if (scaled && !dw) geometry(params->inputTensors[forward ? 1 : 3], DATA_F32, {hidden});
    if (gated) geometry(params->inputTensors[input_count - 1], storage_dtype, {hidden, rows});
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (dw) geometry(params->outputTensors[0], DATA_F32, {hidden});
    else geometry(params->outputTensors[0], storage_dtype, {hidden, rows});
    if (forward) geometry(params->outputTensors[1], DATA_F32, {1, rows});
    else if (gated && !dw) geometry(params->outputTensors[1], storage_dtype, {hidden, rows});
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = dw ? (hidden + 127) / 128 : rows;
    auto map = [&](auto& pattern, const auto& tensor) {
        pattern = {};
        for (unsigned dim = 0; dim < tensor.geometry.dims; ++dim) {
            auto& mapping = pattern.mapping[dim];
            mapping.indexSpaceDim = 0;
            mapping.start_b = 0;
            if (dw && dim == 0 && tensor.geometry.maxSizes[0] != 1) {
                mapping.a = 128;
                mapping.end_b = 127;
            } else if (!dw && dim == 1) {
                mapping.a = 1;
                mapping.end_b = 0;
            } else {
                mapping.a = 0;
                mapping.end_b = tensor.geometry.maxSizes[dim] - 1;
            }
        }
    };
    for (unsigned i = 0; i < input_count; ++i) map(instance->inputTensorAccessPattern[i], params->inputTensors[i]);
    if (binary.kind == 3) { instance->inputTensorAccessPattern[1]={}; instance->inputTensorAccessPattern[1].allRequired=true; }
    for (unsigned i = 0; i < output_count; ++i) map(instance->outputTensorAccessPattern[i], params->outputTensors[i]);
    auto parameter_size=binary.kind == 3 ? sizeof(GroupNormParams) : sizeof(NormalizationParams);
    instance->kernel.paramsNr = parameter_size / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, config, parameter_size);
    return GLUE_SUCCESS;
}
