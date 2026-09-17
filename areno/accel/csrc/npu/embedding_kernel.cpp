#include "kernel_operator.h"
#include "embedding_launch.h"

namespace areno_npu {
using namespace AscendC;

template <typename T, bool Backward>
class EmbeddingKernel {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> idQueue, dataQueue;
    TQue<QuePosition::VECOUT, 1> zeroQueue;
    GlobalTensor<int64_t> ids;
    GlobalTensor<T> input, output;

    template <HardEvent Event>
    __aicore__ inline void Sync() {
        auto id = pipe.FetchEventID(Event);
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }

    __aicore__ inline int64_t Token(int64_t position) {
        auto local = idQueue.template AllocTensor<int64_t>();
        DataCopyExtParams copy{1, sizeof(int64_t), 0, 0, 0};
        DataCopyPadExtParams<int64_t> padding{false, 0, 0, 0};
        DataCopyPad(local, ids[position], copy, padding);
        idQueue.EnQue(local);
        local = idQueue.template DeQue<int64_t>();
        Sync<HardEvent::MTE2_S>();
        int64_t result = local.GetValue(0);
        Sync<HardEvent::S_MTE2>();
        idQueue.FreeTensor(local);
        return result;
    }

    __aicore__ inline void Transfer(int64_t src, int64_t dst, uint32_t n) {
        auto local = dataQueue.template AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        DataCopyPad(local, input[src], copy, padding);
        dataQueue.EnQue(local);
        local = dataQueue.template DeQue<T>();
        // DMA-to-DMA copying preserves the stored bits in forward, including
        // signed zero and nonfinite weights. Backward adds in storage dtype,
        // as CUDA's embedding atomic_add does for repeated token ids.
        Sync<HardEvent::MTE2_MTE3>();
        if constexpr (Backward) SetAtomicAdd<T>();
        DataCopyPad(output[dst], local, copy);
        if constexpr (Backward) DisableDmaAtomic();
        Sync<HardEvent::MTE3_MTE2>();
        dataQueue.FreeTensor(local);
    }

    __aicore__ inline void Zero(int64_t dst, uint32_t n) {
        auto local = zeroQueue.template AllocTensor<T>();
        Duplicate(local.template ReinterpretCast<int32_t>(), int32_t{0}, (n * sizeof(T) + 3) / 4);
        zeroQueue.EnQue(local);
        local = zeroQueue.template DeQue<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPad(output[dst], local, copy);
        zeroQueue.FreeTensor(local);
    }

public:
    __aicore__ inline EmbeddingKernel() {}

    __aicore__ inline void Init(GM_ADDR idIn, GM_ADDR dataIn, GM_ADDR dataOut) {
        ids.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(idIn));
        input.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dataIn));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dataOut));
        pipe.InitBuffer(idQueue, 1, 32);
        pipe.InitBuffer(dataQueue, 1, kEmbeddingTile * sizeof(T));
        if constexpr (!Backward) pipe.InitBuffer(zeroQueue, 1, kEmbeddingTile * sizeof(T));
    }

    __aicore__ inline void Process(int64_t tokens, int64_t hidden, int64_t start, int64_t end) {
        int64_t tiles = (hidden - 1) / kEmbeddingTile + 1;
        for (int64_t task = GetBlockIdx(); task < tokens * tiles; task += GetBlockNum()) {
            int64_t token = task / tiles, col = task % tiles * kEmbeddingTile;
            uint32_t n = hidden - col < kEmbeddingTile ? hidden - col : kEmbeddingTile;
            int64_t id = Token(token);
            if (id >= start && id < end) {
                int64_t table = (id - start) * hidden + col, dense = token * hidden + col;
                Transfer(Backward ? dense : table, Backward ? table : dense, n);
            } else if constexpr (!Backward) Zero(token * hidden + col, n);
        }
    }
};

} // namespace areno_npu

template <typename T, bool Backward>
__global__ __aicore__ void embedding_kernel(GM_ADDR ids, GM_ADDR input, GM_ADDR output,
    int64_t tokens, int64_t hidden, int64_t start, int64_t end) {
    using namespace AscendC;
    using namespace areno_npu;
    EmbeddingKernel<T, Backward> kernel;
    kernel.Init(ids, input, output);
    kernel.Process(tokens, hidden, start, end);
}

namespace areno_npu {

void launch_embedding(uint32_t blocks, void* stream, uint32_t storage, bool backward,
    const int64_t* ids, const void* input, void* output, int64_t tokens, int64_t hidden, int64_t start, int64_t end) {
#define ARENO_EMBEDDING_LAUNCH(T, B) embedding_kernel<T, B><<<blocks, nullptr, stream>>>( \
    (uint8_t*)ids, (uint8_t*)input, (uint8_t*)output, tokens, hidden, start, end)
#define ARENO_EMBEDDING_TYPE(T) \
    if (backward) { ARENO_EMBEDDING_LAUNCH(T, true); } else { ARENO_EMBEDDING_LAUNCH(T, false); }
    switch (storage) {
        case 0: { ARENO_EMBEDDING_TYPE(float); } break;
        case 1: { ARENO_EMBEDDING_TYPE(half); } break;
        case 2: { ARENO_EMBEDDING_TYPE(bfloat16_t); } break;
    }
#undef ARENO_EMBEDDING_TYPE
#undef ARENO_EMBEDDING_LAUNCH
}
} // namespace areno_npu
