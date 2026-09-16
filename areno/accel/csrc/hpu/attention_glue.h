#pragma once
#include "attention_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_attention(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance, const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    const bool backward = binary.direction == 1, packed = binary.kind == 1, paged = binary.kind == 2;
    const unsigned extra = packed ? 1 : paged ? 2 : 0;
    const unsigned inputs = (backward ? 5 : 3) + extra, outputs = backward ? 3 : 1;
    if (params->inputTensorNr != inputs) { params->inputTensorNr = inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != outputs) { params->outputTensorNr = outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* config = static_cast<const AttentionParams*>(params->nodeParams.nodeParams);
    if (!config || config->q_rows <= 0 || config->k_rows <= 0 || config->hidden <= 0
        || config->q_heads <= 0 || config->kv_heads <= 0 || config->q_heads % config->kv_heads != 0
        || (paged ? backward || config->block_size <= 0 || config->max_blocks <= 0
            : packed ? config->sequences <= 0 : config->q_length <= 0 || config->query_start < 0
            || config->query_start > config->k_length-config->q_length)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
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
    for (unsigned i=0; i<inputs-extra; ++i) {
        geometry(params->inputTensors[i], dtype, config->hidden, i == 1 || i == 2 ? config->k_rows : config->q_rows, 2);
    }
    if (packed) geometry(params->inputTensors[inputs-1], DATA_I32, uint64_t(config->sequences)+1, 1, 1);
    if (paged) {
        geometry(params->inputTensors[3], DATA_I32, config->max_blocks, config->q_rows/config->q_heads, 2);
        geometry(params->inputTensors[4], DATA_I32, config->q_rows/config->q_heads, 1, 1);
    }
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    for (unsigned i=0; i<outputs; ++i) geometry(params->outputTensors[i], backward ? DATA_F32 : dtype,
        config->hidden, i == 0 ? config->q_rows : config->k_rows, 2);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank = 2;
    instance->indexSpaceGeometry[0] = (uint64_t(config->hidden)+127)/128;
    instance->indexSpaceGeometry[1] = config->q_rows;
    for (unsigned i=0; i<inputs; ++i) {
        instance->inputTensorAccessPattern[i] = {};
        instance->inputTensorAccessPattern[i].allRequired = true;
    }
    for (unsigned i=0; i<outputs; ++i) {
        auto& pattern = instance->outputTensorAccessPattern[i];
        pattern = {};
        if (i != 0) {
            pattern.allRequired = true;
            pattern.memsetBeforeExecution = true;
        } else {
            pattern.mapping[0].indexSpaceDim = 0;
            pattern.mapping[0].a = 128;
            pattern.mapping[0].end_b = 127;
            pattern.mapping[1].indexSpaceDim = 1;
            pattern.mapping[1].a = 1;
        }
    }
    instance->kernel.paramsNr = sizeof(AttentionParams)/sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, config, sizeof(AttentionParams));
    return GLUE_SUCCESS;
}
