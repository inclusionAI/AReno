// FP32 formulas shared by the TPC vector kernel and the scalar numerical test.
// ARENO_FLOAT and ARENO_* math primitives are supplied by the caller.
#pragma once

static inline ARENO_FLOAT areno_sigmoid_value(ARENO_FLOAT x) {
    return ARENO_RECIP(1.0f + ARENO_EXP(-x));
}

static inline ARENO_FLOAT areno_activation_value(int kind, ARENO_FLOAT x, ARENO_FLOAT up) {
    if (kind == 1) return areno_sigmoid_value(x);
    if (kind == 2) {
        ARENO_FLOAT e = ARENO_EXP(x);
        ARENO_FLOAT u = 1.0f + e;
        // Correct cancellation in log(1 + exp(x)) when exp(x) is small.
        ARENO_FLOAT value = ARENO_LOG(u) * e * ARENO_RECIP(u - 1.0f);
        value = ARENO_SELECT_EQ(u, 1.0f, e, value);
        return ARENO_SELECT_GT(x, 20.0f, x, value);
    }
    if (kind == 4) {
        ARENO_FLOAT inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return x * (0.5f * (1.0f + ARENO_TANH(inner))) * up;
    }
    ARENO_FLOAT value = x * areno_sigmoid_value(x);
    return kind == 3 ? value * up : value;
}

static inline ARENO_FLOAT areno_activation_grad_x(
    int kind, ARENO_FLOAT x, ARENO_FLOAT up, ARENO_FLOAT grad
) {
    // Sigmoid backward consumes the saved output, matching the CUDA ABI.
    if (kind == 1) return grad * x * (1.0f - x);
    if (kind == 2) return grad * ARENO_SELECT_GT(x, 20.0f, 1.0f, areno_sigmoid_value(x));
    if (kind == 4) {
        ARENO_FLOAT x2 = x * x;
        ARENO_FLOAT inner = 0.7978845608028654f * (x + 0.044715f * x * x2);
        ARENO_FLOAT t = ARENO_TANH(inner);
        ARENO_FLOAT cdf = 0.5f * (1.0f + t);
        ARENO_FLOAT derivative = cdf + 0.5f * x * (1.0f - t * t)
            * (0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x2));
        return grad * up * derivative;
    }
    ARENO_FLOAT sigmoid = areno_sigmoid_value(x);
    if (kind == 3) grad = grad * up;
    return grad * sigmoid * (1.0f + x * (1.0f - sigmoid));
}

static inline ARENO_FLOAT areno_activation_grad_up(int kind, ARENO_FLOAT x, ARENO_FLOAT grad) {
    return grad * areno_activation_value(kind, x, 1.0f);
}
