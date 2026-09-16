#pragma once
#include "quantized_optimizer_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_quantized_optimizer(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance, const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    const bool factored = binary.kind == 3, packed = binary.kind != 8;
    const unsigned inputs = factored ? 7 : packed ? 6 : 8, outputs = factored ? 3 : 5;
    if (params->inputTensorNr != inputs) { params->inputTensorNr = inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != outputs) { params->outputTensorNr = outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* config = static_cast<const QuantizedOptimizerParams*>(params->nodeParams.nodeParams);
    if (config == nullptr || config->count <= 0 || config->block_size <= 0 || config->block_size > 4096
        || (packed && (config->block_size < 32 || config->block_size > 1024 || (config->block_size & (config->block_size-1))))) {
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    const uint64_t count = config->count, block = config->block_size;
    if (factored && (config->rows <= 0 || config->columns <= 0 || config->parameter_shard_start < 0
        || uint64_t(config->parameter_shard_start) + count > uint64_t(config->rows) * config->columns)) {
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    const uint64_t codes = packed ? (count+1)/2 : count, blocks = (count+block-1)/block;
    const auto model_dtype = binary.dtype == 0 ? DATA_F32 : DATA_BF16, grad_dtype = binary.direction == 0 ? DATA_F32 : DATA_BF16;
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
    geometry(params->inputTensors[2], DATA_U8, codes);
    geometry(params->inputTensors[3], DATA_F32, blocks);
    if (factored) {
        geometry(params->inputTensors[4], DATA_F32, uint64_t(config->rows) + config->columns);
        geometry(params->inputTensors[5], DATA_F32, 1);
        geometry(params->inputTensors[6], DATA_I32, 1);
    } else {
        geometry(params->inputTensors[4], DATA_U8, codes);
        geometry(params->inputTensors[5], DATA_F32, blocks);
    }
    if (!packed) for (unsigned i : {6u, 7u}) geometry(params->inputTensors[i], DATA_F32, 256);
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    geometry(params->outputTensors[0], model_dtype, count);
    for (unsigned i=1; i<outputs; ++i) geometry(params->outputTensors[i], i % 2 ? DATA_U8 : DATA_F32, i % 2 ? codes : blocks);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = blocks;
    auto map = [&](auto& pattern, uint64_t stride, uint64_t span) {
        pattern = {};
        pattern.mapping[0].indexSpaceDim = 0;
        pattern.mapping[0].a = stride;
        pattern.mapping[0].end_b = span-1;
    };
    for (unsigned i=0; i<inputs; ++i) {
        if (factored && i >= 4) {
            instance->inputTensorAccessPattern[i] = {};
            instance->inputTensorAccessPattern[i].allRequired = true;
            continue;
        }
        uint64_t stride = i < 2 ? block : i >= 6 ? 0 : (i % 2 == 0 ? (packed ? block/2 : block) : 1);
        map(instance->inputTensorAccessPattern[i], stride, i >= 6 ? 256 : stride);
    }
    for (unsigned i=0; i<outputs; ++i) {
        uint64_t stride = i == 0 ? block : i % 2 == 1 ? (packed ? block/2 : block) : 1;
        map(instance->outputTensorAccessPattern[i], stride, stride);
    }
    instance->kernel.paramsNr = sizeof(QuantizedOptimizerParams)/sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, config, sizeof(QuantizedOptimizerParams));
    return GLUE_SUCCESS;
}
