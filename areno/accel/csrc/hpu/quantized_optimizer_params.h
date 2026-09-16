#pragma once
struct QuantizedOptimizerParams {
    int count;
    int block_size;
    float beta1;
    float beta2;
    float effective_lr;
    float weight_decay;
    float eps;
    float step_size;
    float bias_correction2_sqrt;
    int parameter_shard_start = 0;
    int rows = 0;
    int columns = 0;
};
