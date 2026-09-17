#include "kernel_operator.h"
#include "optimizer_launch.h"

namespace areno_npu {
using namespace AscendC;

// Each task is a row fragment contained in both a 1024-element column tile
// and the caller's parameter shard. Columns accumulate by vector DMA; each
// fragment contributes one reduced row sum. Other shards can add to the same
// factors without expanding a parameter-sized variance tensor.
template <typename Grad>
class FactoredStatsKernel {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECCALC> scratch;
    GlobalTensor<Grad> grad;
    GlobalTensor<float> sums;
    GlobalTensor<int32_t> invalid;
    LocalTensor<float> squared, work, reduced;

    template <HardEvent Event>
    __aicore__ inline void Sync() {
        auto id = pipe.FetchEventID(Event);
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }

    __aicore__ inline void Read(int64_t offset, uint32_t n) {
        auto local = inQueue.template AllocTensor<Grad>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(Grad)), 0, 0, 0};
        DataCopyPadExtParams<Grad> padding{false, 0, 0, 0};
        DataCopyPad(local, grad[offset], copy, padding);
        inQueue.EnQue(local);
        local = inQueue.template DeQue<Grad>();
        if constexpr (sizeof(Grad) == sizeof(float)) Adds(squared, local, 0.0f, n);
        else Cast(squared, local, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(local);
        Mul(squared, squared, squared, n);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddToFactors(LocalTensor<float> src, int64_t offset, uint32_t n) {
        auto out = outQueue.template AllocTensor<float>();
        Adds(out, src, 0.0f, n);
        outQueue.EnQue(out);
        out = outQueue.template DeQue<float>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
        SetAtomicAdd<float>();
        DataCopyPad(sums[offset], out, copy);
        DisableDmaAtomic();
        outQueue.FreeTensor(out);
    }

public:
    __aicore__ inline FactoredStatsKernel() {}

    __aicore__ inline void Init(GM_ADDR gradient, GM_ADDR factors, GM_ADDR flag) {
        grad.SetGlobalBuffer(reinterpret_cast<__gm__ Grad*>(gradient));
        sums.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(factors));
        invalid.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(flag));
        pipe.InitBuffer(inQueue, 1, kAdamTile * sizeof(Grad));
        pipe.InitBuffer(outQueue, 1, kAdamTile * sizeof(float));
        pipe.InitBuffer(scratch, 2 * kAdamTile * sizeof(float) + 32);
        squared = scratch.Get<float>();
        work = squared[kAdamTile];
        reduced = squared[2 * kAdamTile];
    }

    __aicore__ inline void Process(int64_t numel, int64_t start, int64_t rows, int64_t columns) {
        const int64_t columnTiles = (columns - 1) / kAdamTile + 1;
        const int64_t end = start + numel;
        const int64_t first = start / columns * columnTiles + start % columns / kAdamTile;
        const int64_t last = (end - 1) / columns * columnTiles + (end - 1) % columns / kAdamTile;
        bool bad = false;
        for (int64_t task = first + GetBlockIdx(); task <= last; task += GetBlockNum()) {
            const int64_t row = task / columnTiles;
            const int64_t col = (task % columnTiles) * kAdamTile;
            int64_t begin = row * columns + col;
            int64_t stop = row * columns + (col + kAdamTile < columns ? col + kAdamTile : columns);
            if (begin < start) begin = start;
            if (stop > end) stop = end;
            const uint32_t n = stop - begin;
            Read(begin - start, n);
            Sync<HardEvent::V_S>();
            auto words = squared.template ReinterpretCast<uint32_t>();
            for (uint32_t i = 0; i < n; ++i) {
                if ((words.GetValue(i) & 0x7f800000u) == 0x7f800000u) {
                    // Match CUDA: skip only the nonfinite contribution while
                    // marking the whole parameter invalid for its update.
                    words.SetValue(i, 0u);
                    bad = true;
                }
            }
            Sync<HardEvent::S_V>();
            AddToFactors(squared, rows + begin % columns, n);
            ReduceSum(reduced, squared, work, n);
            PipeBarrier<PIPE_V>();
            AddToFactors(reduced, row, 1);
        }
        if (bad) {
            auto out = outQueue.template AllocTensor<int32_t>();
            Duplicate(out, int32_t{1}, 1);
            outQueue.EnQue(out);
            out = outQueue.template DeQue<int32_t>();
            DataCopyExtParams copy{1, sizeof(int32_t), 0, 0, 0};
            SetAtomicMax<int32_t>();
            DataCopyPad(invalid, out, copy);
            DisableDmaAtomic();
            outQueue.FreeTensor(out);
        }
    }
};

} // namespace areno_npu

template <typename Grad>
__global__ __aicore__ void factored_stats_kernel(GM_ADDR grad, GM_ADDR factors, GM_ADDR invalid,
    int64_t n, int64_t start, int64_t rows, int64_t columns) {
    using namespace AscendC;
    using namespace areno_npu;
    FactoredStatsKernel<Grad> kernel;
    kernel.Init(grad, factors, invalid);
    kernel.Process(n, start, rows, columns);
}

namespace areno_npu {

void launch_adamw_factored_stats(uint32_t blocks, void* stream, bool grad_bf16, const void* grad,
                                 float* sums, int32_t* invalid, int64_t n, int64_t start,
                                 int64_t rows, int64_t columns) {
#define ARENO_FACTORED_STATS(G) factored_stats_kernel<G><<<blocks, nullptr, stream>>>( \
    (uint8_t*)grad, (uint8_t*)sums, (uint8_t*)invalid, n, start, rows, columns)
    if (grad_bf16) { ARENO_FACTORED_STATS(bfloat16_t); }
    else { ARENO_FACTORED_STATS(float); }
#undef ARENO_FACTORED_STATS
}
} // namespace areno_npu
