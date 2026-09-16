#pragma once
#include "paged_cache_params.h"

static tpc_lib_api::GlueCodeReturn instantiate_paged_cache(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance, const KernelBinary& binary
) {
    using namespace tpc_lib_api;
    bool owners = binary.kind == 0;
    unsigned inputs = owners ? 2 : 5, outputs = owners ? 1 : 2;
    if (params->inputTensorNr != inputs) { params->inputTensorNr = inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (params->outputTensorNr != outputs) { params->outputTensorNr = outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* c = static_cast<const PagedCacheParams*>(params->nodeParams.nodeParams);
    if (!c || c->batch <= 0 || c->slots <= 0 || c->block_size <= 0 || c->kv_heads <= 0 || c->hidden <= 0
        || (owners && c->max_blocks <= 0)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    auto dtype = binary.dtype == 0 ? DATA_F32 : binary.dtype == 1 ? DATA_BF16 : DATA_F16;
    bool size_mismatch = false, dtype_mismatch = false;
    auto geometry = [&](auto& tensor, auto type, uint64_t width, uint64_t rows, unsigned dims) {
        dtype_mismatch |= tensor.geometry.dataType != type;
        tensor.geometry.dataType = type;
        size_mismatch |= tensor.geometry.dims != dims || tensor.geometry.maxSizes[0] != width
            || (dims == 2 && tensor.geometry.maxSizes[1] != rows);
        tensor.geometry.dims = dims; tensor.geometry.maxSizes[0] = width;
        if (dims == 2) tensor.geometry.maxSizes[1] = rows;
    };
    if (owners) {
        geometry(params->inputTensors[0], DATA_I32, c->max_blocks, c->batch, 2);
        geometry(params->inputTensors[1], DATA_I32, c->batch, 1, 1);
    } else {
        for (unsigned i=0; i<4; ++i) geometry(params->inputTensors[i], dtype, c->hidden,
            uint64_t(i<2 ? c->slots : c->batch)*c->kv_heads, 2);
        geometry(params->inputTensors[4], DATA_I32, c->slots, 1, 1);
    }
    if (size_mismatch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (owners) geometry(params->outputTensors[0], DATA_I32, c->slots, 1, 1);
    else for (unsigned i=0; i<2; ++i) geometry(params->outputTensors[i], dtype, c->hidden, uint64_t(c->slots)*c->kv_heads, 2);
    if (size_mismatch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (dtype_mismatch) return GLUE_INCOMPATIBLE_DATA_TYPE;
    instance->indexSpaceRank = owners ? 1 : 2;
    instance->indexSpaceGeometry[0] = owners ? c->batch : (uint64_t(c->hidden)+127)/128;
    if (!owners) instance->indexSpaceGeometry[1] = uint64_t(c->slots)*c->kv_heads;
    for (unsigned i=0; i<inputs; ++i) {
        instance->inputTensorAccessPattern[i] = {};
        instance->inputTensorAccessPattern[i].allRequired = true;
    }
    for (unsigned i=0; i<outputs; ++i) {
        auto& pattern = instance->outputTensorAccessPattern[i];
        pattern = {};
        if (owners) {
            pattern.allRequired = true;
            pattern.memsetBeforeExecution = true;
            pattern.memsetValue.i32Value = -1;
        } else {
            pattern.mapping[0].indexSpaceDim = 0;
            pattern.mapping[0].a = 128;
            pattern.mapping[0].end_b = 127;
            pattern.mapping[1].indexSpaceDim = 1;
            pattern.mapping[1].a = 1;
        }
    }
    instance->kernel.paramsNr = sizeof(PagedCacheParams)/sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, c, sizeof(PagedCacheParams));
    return GLUE_SUCCESS;
}
