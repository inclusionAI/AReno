#include "kernel_operator.h"
#include <math.h>
#define ARENO_ROUTING_INLINE __aicore__ inline
#include "../routing_common.h"
#undef ARENO_ROUTING_INLINE
#include "routing_launch.h"

namespace areno_npu {
using namespace AscendC;
using areno_accel::routing::insert_topk;
using areno_accel::routing::kMaxExperts;
using areno_accel::routing::kMaxGroups;
using areno_accel::routing::kMaxTopK;

template <typename T, RoutingOp Op>
class RoutingKernel {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue, idxInQueue;
    TQue<QuePosition::VECOUT, 1> idxOutQueue, weightOutQueue, gradOutQueue;
    TBuf<QuePosition::VECCALC> scratch;
    GlobalTensor<T> logits, gradLogits;
    GlobalTensor<float> bias, weights;
    GlobalTensor<int64_t> indices;
    LocalTensor<float> probs, route, dprob, gradient, reduced, work;

    template <HardEvent Event>
    __aicore__ inline void Sync() {
        auto id = pipe.FetchEventID(Event);
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }

    template <typename S>
    __aicore__ inline void Read(GlobalTensor<S>& gm, LocalTensor<float> dst, int64_t offset, uint32_t n) {
        auto local = inQueue.template AllocTensor<S>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(S)), 0, 0, 0};
        DataCopyPadExtParams<S> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, pad);
        inQueue.EnQue(local);
        local = inQueue.template DeQue<S>();
        if constexpr (sizeof(S) == sizeof(float)) Adds(dst, local, 0.0f, n);
        else Cast(dst, local, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(local);
    }

    __aicore__ inline void Softmax(uint32_t experts) {
        Sync<HardEvent::V_S>();
        float maximum = -INFINITY;
        for (uint32_t i = 0; i < experts; ++i) {
            float v = probs.GetValue(i);
            if (v > maximum) maximum = v;
        }
        Sync<HardEvent::S_V>();
        Adds(probs, probs, -maximum, experts);
        PipeBarrier<PIPE_V>();
        Exp(probs, probs, experts);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduced, probs, work, experts);
        Sync<HardEvent::V_S>();
        float sum = reduced.GetValue(0);
        sum = sum > 1.0e-20f ? sum : 1.0e-20f;
        Sync<HardEvent::S_V>();
        Duplicate(route, sum, experts);
        PipeBarrier<PIPE_V>();
        Div(probs, probs, route, experts);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SaveSelection(int64_t token, int k, int* selected, float* selectedWeight) {
        auto ids = idxOutQueue.template AllocTensor<int64_t>();
        auto values = weightOutQueue.template AllocTensor<float>();
        Sync<HardEvent::MTE3_S>();
        for (int pos = 0; pos < k; ++pos) {
            ids.SetValue(pos, static_cast<int64_t>(selected[pos]));
            values.SetValue(pos, selectedWeight[pos]);
        }
        Sync<HardEvent::S_MTE3>();
        idxOutQueue.EnQue(ids);
        weightOutQueue.EnQue(values);
        ids = idxOutQueue.template DeQue<int64_t>();
        values = weightOutQueue.template DeQue<float>();
        DataCopyExtParams idsCopy{1, static_cast<uint32_t>(k * sizeof(int64_t)), 0, 0, 0};
        DataCopyExtParams weightCopy{1, static_cast<uint32_t>(k * sizeof(float)), 0, 0, 0};
        DataCopyPad(indices[token * k], ids, idsCopy);
        DataCopyPad(weights[token * k], values, weightCopy);
        idxOutQueue.FreeTensor(ids);
        weightOutQueue.FreeTensor(values);
    }

    __aicore__ inline void Select(int64_t token, int experts, int k, bool renormalize, int groups, int topk_group) {
        Sync<HardEvent::V_S>();
        int selected[kMaxTopK];
        float values[kMaxTopK];
        for (int pos = 0; pos < k; ++pos) { values[pos] = -INFINITY; selected[pos] = 0; }
        if constexpr (Op == GroupedRouter) {
            int groupIndices[kMaxGroups], enabled[kMaxGroups];
            float groupValues[kMaxGroups];
            for (int group = 0; group < groups; ++group) {
                groupValues[group] = -INFINITY;
                groupIndices[group] = groups;
                enabled[group] = 0;
            }
            int per_group = experts / groups, score_k = k / topk_group;
            for (int group = 0; group < groups; ++group) {
                float localValues[kMaxTopK];
                int localIndices[kMaxTopK];
                for (int pos = 0; pos < score_k; ++pos) { localValues[pos] = -INFINITY; localIndices[pos] = experts; }
                for (int i = 0; i < per_group; ++i) {
                    int expert = group * per_group + i;
                    insert_topk(route.GetValue(expert), expert, localValues, localIndices, score_k);
                }
                float score = 0.0f;
                for (int pos = 0; pos < score_k; ++pos) score += localValues[pos];
                insert_topk(score, group, groupValues, groupIndices, topk_group);
            }
            for (int pos = 0; pos < topk_group; ++pos) {
                if (groupIndices[pos] < groups) enabled[groupIndices[pos]] = 1;
            }
            for (int expert = 0; expert < experts; ++expert) {
                if (enabled[expert / per_group]) insert_topk(route.GetValue(expert), expert, values, selected, k);
            }
            // Bias affects selection only, never the returned sigmoid weights.
            for (int pos = 0; pos < k; ++pos) values[pos] = probs.GetValue(selected[pos]);
        } else {
            for (int expert = 0; expert < experts; ++expert) insert_topk(probs.GetValue(expert), expert, values, selected, k);
        }
        float sum = 0.0f;
        for (int pos = 0; pos < k; ++pos) sum += values[pos];
        sum = sum > 1.0e-20f ? sum : 1.0e-20f;
        if (renormalize) for (int pos = 0; pos < k; ++pos) values[pos] /= sum;
        SaveSelection(token, k, selected, values);
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void Backward(int64_t token, int experts, int k, bool renormalize) {
        Read(weights, gradient, token * k, k);
        auto ids = idxInQueue.template AllocTensor<int64_t>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(k * sizeof(int64_t)), 0, 0, 0};
        DataCopyPadExtParams<int64_t> pad{false, 0, 0, 0};
        DataCopyPad(ids, indices[token * k], copy, pad);
        idxInQueue.EnQue(ids);
        ids = idxInQueue.template DeQue<int64_t>();
        Duplicate(dprob, 0.0f, experts);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::V_S>();
        float selected_sum = 0.0f, weighted_grad_sum = 0.0f;
        for (int pos = 0; pos < k; ++pos) {
            int expert = static_cast<int>(ids.GetValue(pos));
            float p = probs.GetValue(expert), g = gradient.GetValue(pos);
            selected_sum += p;
            weighted_grad_sum += g * p;
        }
        selected_sum = selected_sum > 1.0e-20f ? selected_sum : 1.0e-20f;
        float dot = 0.0f;
        for (int pos = 0; pos < k; ++pos) {
            int expert = static_cast<int>(ids.GetValue(pos));
            float p = probs.GetValue(expert), g = gradient.GetValue(pos);
            float dp = renormalize ? (g * selected_sum - weighted_grad_sum) / (selected_sum * selected_sum) : g;
            dprob.SetValue(expert, dprob.GetValue(expert) + dp);
            dot += dp * p;
        }
        Sync<HardEvent::S_MTE2>();
        idxInQueue.FreeTensor(ids);
        Sync<HardEvent::S_V>();
        Adds(dprob, dprob, -dot, experts);
        PipeBarrier<PIPE_V>();
        Mul(dprob, probs, dprob, experts);
        PipeBarrier<PIPE_V>();
        auto local = gradOutQueue.template AllocTensor<T>();
        if constexpr (sizeof(T) == sizeof(float)) Adds(local, dprob, 0.0f, experts);
        else Cast(local, dprob, RoundMode::CAST_RINT, experts);
        gradOutQueue.EnQue(local);
        local = gradOutQueue.template DeQue<T>();
        DataCopyExtParams outCopy{1, static_cast<uint32_t>(experts * sizeof(T)), 0, 0, 0};
        DataCopyPad(gradLogits[token * experts], local, outCopy);
        gradOutQueue.FreeTensor(local);
    }

public:
    __aicore__ inline RoutingKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR b, GM_ADDR ids, GM_ADDR w, GM_ADDR dx) {
        logits.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ids));
        weights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
        if constexpr (Op == GroupedRouter) bias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        pipe.InitBuffer(inQueue, 1, kMaxExperts * sizeof(float));
        if constexpr (Op == TopKBackward) {
            gradLogits.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dx));
            pipe.InitBuffer(idxInQueue, 1, kMaxTopK * sizeof(int64_t));
            pipe.InitBuffer(gradOutQueue, 1, kMaxExperts * sizeof(T));
        } else {
            pipe.InitBuffer(idxOutQueue, 1, kMaxTopK * sizeof(int64_t));
            pipe.InitBuffer(weightOutQueue, 1, kMaxTopK * sizeof(float));
        }
        pipe.InitBuffer(scratch, 6 * kMaxExperts * sizeof(float));
        probs = scratch.Get<float>();
        route = probs[kMaxExperts];
        dprob = probs[2 * kMaxExperts];
        gradient = probs[3 * kMaxExperts];
        reduced = probs[4 * kMaxExperts];
        work = probs[5 * kMaxExperts];
    }

    __aicore__ inline void Process(int64_t tokens, int experts, int k, bool renormalize, int groups, int topk_group) {
        for (int64_t token = GetBlockIdx(); token < tokens; token += GetBlockNum()) {
            Read(logits, probs, token * experts, experts);
            if constexpr (Op == GroupedRouter) {
                Muls(probs, probs, -1.0f, experts);
                PipeBarrier<PIPE_V>();
                Exp(probs, probs, experts);
                PipeBarrier<PIPE_V>();
                Adds(probs, probs, 1.0f, experts);
                PipeBarrier<PIPE_V>();
                Reciprocal(probs, probs, experts);
                PipeBarrier<PIPE_V>();
                Read(bias, route, 0, experts);
                Add(route, probs, route, experts);
                PipeBarrier<PIPE_V>();
            } else Softmax(experts);
            if constexpr (Op == TopKBackward) Backward(token, experts, k, renormalize);
            else Select(token, experts, k, renormalize, groups, topk_group);
        }
    }
};

template <typename T, RoutingOp Op>
__global__ __aicore__ void routing_kernel(GM_ADDR x, GM_ADDR b, GM_ADDR ids, GM_ADDR w, GM_ADDR dx,
    int64_t tokens, int experts, int k, bool renormalize, int groups, int topk_group) {
    RoutingKernel<T, Op> kernel;
    kernel.Init(x, b, ids, w, dx);
    kernel.Process(tokens, experts, k, renormalize, groups, topk_group);
}

template <typename T>
void launch_routing_typed(uint32_t blocks, void* stream, RoutingOp op, const void* logits, const float* bias,
    int64_t* indices, float* weights, void* grad_logits, int64_t tokens, int experts, int k,
    bool renormalize, int groups, int topk_group) {
#define ARENO_ROUTING(OP) case OP: routing_kernel<T, OP><<<blocks, nullptr, stream>>>( \
    (uint8_t*)logits, (uint8_t*)bias, (uint8_t*)indices, (uint8_t*)weights, (uint8_t*)grad_logits, \
    tokens, experts, k, renormalize, groups, topk_group); break
    switch (op) { ARENO_ROUTING(TopKForward); ARENO_ROUTING(TopKBackward); ARENO_ROUTING(GroupedRouter); }
#undef ARENO_ROUTING
}

void launch_routing(uint32_t blocks, void* stream, uint32_t storage, RoutingOp op,
    const void* logits, const float* bias, int64_t* indices, float* weights, void* grad_logits,
    int64_t tokens, int experts, int k, bool renormalize, int groups, int topk_group) {
#define ARENO_ROUTING_TYPE(T) launch_routing_typed<T>(blocks, stream, op, logits, bias, indices, weights, grad_logits, \
    tokens, experts, k, renormalize, groups, topk_group)
    switch (storage) {
        case 0: { ARENO_ROUTING_TYPE(float); } break;
        case 1: { ARENO_ROUTING_TYPE(half); } break;
        case 2: { ARENO_ROUTING_TYPE(bfloat16_t); } break;
    }
#undef ARENO_ROUTING_TYPE
}
} // namespace areno_npu
