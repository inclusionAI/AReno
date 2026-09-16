#pragma once
#include "embedding_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_embedding(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance, const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    if (params->inputTensorNr != 2) { params->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != 1) { params->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* config = static_cast<const EmbeddingParams*>(params->nodeParams.nodeParams);
    if (!config || config->tokens <= 0 || config->hidden <= 0 || config->vocab_start < 0 || config->vocab_end <= config->vocab_start) {
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    const bool backward = binary.direction == 1;
    const auto dtype = binary.dtype == 0 ? DATA_F32 : binary.dtype == 1 ? DATA_BF16 : DATA_F16;
    bool size_mismatch = false, dtype_mismatch = false;
    auto geometry = [&](auto& tensor, auto type, uint64_t width, uint64_t rows, unsigned dims) {
        dtype_mismatch |= tensor.geometry.dataType != type;
        tensor.geometry.dataType = type;
        size_mismatch |= tensor.geometry.dims != dims || tensor.geometry.maxSizes[0] != width
            || (dims == 2 && tensor.geometry.maxSizes[1] != rows);
        tensor.geometry.dims = dims;
        tensor.geometry.maxSizes[0] = width;
        if (dims == 2) tensor.geometry.maxSizes[1] = rows;
    };
    geometry(params->inputTensors[0], DATA_I64, config->tokens, 1, 1);
    geometry(params->inputTensors[1], dtype, config->hidden, backward ? config->tokens : config->vocab_end-config->vocab_start, 2);
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    geometry(params->outputTensors[0], dtype, config->hidden, backward ? config->vocab_end-config->vocab_start : config->tokens, 2);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank = 2;
    instance->indexSpaceGeometry[0] = (uint64_t(config->hidden)+127)/128;
    instance->indexSpaceGeometry[1] = config->tokens;
    auto& ids = instance->inputTensorAccessPattern[0];
    ids = {};
    ids.mapping[0].indexSpaceDim = 1;
    ids.mapping[0].a = 1;
    auto map = [](auto& pattern, bool indirect) {
        pattern = {};
        pattern.allRequired = indirect;
        if (indirect) return;
        pattern.mapping[0].indexSpaceDim = 0;
        pattern.mapping[0].a = 128;
        pattern.mapping[0].end_b = 127;
        pattern.mapping[1].indexSpaceDim = 1;
        pattern.mapping[1].a = 1;
    };
    map(instance->inputTensorAccessPattern[1], !backward);
    map(instance->outputTensorAccessPattern[0], backward);
    instance->outputTensorAccessPattern[0].memsetBeforeExecution = backward;
    instance->kernel.paramsNr = sizeof(EmbeddingParams)/sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, config, sizeof(EmbeddingParams));
    return GLUE_SUCCESS;
}
