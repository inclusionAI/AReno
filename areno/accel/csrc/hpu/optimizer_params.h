#pragma once
struct OptimizerParams {
    int count;
    int carry_offset;
    float beta1;
    float beta2;
    float effective_lr;
    float weight_decay;
    float eps;
    float step_size;
    float bias_correction2_sqrt;
};
