#pragma once
struct KdaPrepareParams {
    int tokens,q_heads,heads,key_dim,normalize,storage_dtype,gate_dtype,recurrent,bounded;
    float lower_bound,softplus_beta,softplus_threshold;
};
struct KdaParams { int tokens,heads,key_dim,value_dim,sequences,slots,save_history; float scale; };
