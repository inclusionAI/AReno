"""模型加载各阶段的进度与耗时追踪。

Issue #230：为模型加载的五个阶段（引用解析、配置/分词器加载、权重分片读取、
设备放置、worker 分发）输出每个阶段的进度和耗时。输出既人类可读又结构化，
方便非交互式消费者解析；仅在 rank 0 打印日志，避免多 rank 运行时刷屏。
失败时保留最后完成的阶段，调用方可以据此定位是哪个阶段、哪个输入出了问题。

追踪器刻意不引入任何依赖：只用 ``time.perf_counter`` 和现有的 ``areno`` logger。
除新增几行 INFO 级日志外，不改变任何默认行为，保持向后兼容。
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("areno.engine.load_progress")

# 固定的阶段顺序，调用方和测试通过名字引用阶段，无需引入额外枚举；
# 追踪器会记录最后一个完成的阶段。
STAGE_REFERENCE_RESOLUTION = "reference_resolution"
STAGE_CONFIG_TOKENIZER = "config_tokenizer_load"
STAGE_WEIGHT_SHARD_READING = "weight_shard_reading"
STAGE_DEVICE_PLACEMENT = "device_placement"
STAGE_WORKER_DISTRIBUTION = "worker_distribution"


class ModelLoadTracker:
    """记录每个阶段的耗时，并在 rank 0 输出有界的进度日志。

    ``rank0`` 门控日志输出，分布式运行时只有一个进程打印；
    调用方传入 ``get_tp_context().rank == 0``。追踪器维护
    ``last_completed_stage``，失败时无需重跑即可定位是哪个阶段、哪个输入出问题。
    """

    def __init__(self, *, rank0: bool = True):
        self._rank0 = bool(rank0)
        self.last_completed_stage: str | None = None
        self.start_time: float = time.perf_counter()

    @contextmanager
    def stage(self, name: str, *, detail: str | None = None) -> Iterator[None]:
        """为一个加载阶段计时，并在 rank 0 输出 start/done 日志。

        退出时（成功或异常）记录耗时并更新 ``last_completed_stage``。
        异常在输出结构化的 ``status=failed`` 行后被原样 re-raise，
        不会隐藏原始错误。
        """

        if self._rank0:
            prefix = f"model_load stage={name}"
            if detail:
                logger.info("%s status=start detail=%s", prefix, detail)
            else:
                logger.info("%s status=start", prefix)
        begin = time.perf_counter()
        try:
            yield
        except BaseException:
            elapsed = time.perf_counter() - begin
            if self._rank0:
                logger.info(
                    "model_load stage=%s status=failed elapsed=%.3fs",
                    name,
                    elapsed,
                )
            raise
        elapsed = time.perf_counter() - begin
        self.last_completed_stage = name
        if self._rank0:
            logger.info(
                "model_load stage=%s status=done elapsed=%.3fs",
                name,
                elapsed,
            )

    def summary(self) -> dict[str, object]:
        """返回结构化快照，供非交互式消费者使用。

        包含最后完成的阶段和追踪器构造以来的总耗时，
        这样 dashboard 或 CLI 无需解析日志行即可展示一条记录。
        """

        return {
            "last_completed_stage": self.last_completed_stage,
            "total_elapsed_s": time.perf_counter() - self.start_time,
        }


@contextmanager
def tracked_stage(tracker: ModelLoadTracker | None, name: str, *, detail: str | None = None) -> Iterator[None]:
    """当 ``tracker`` 存在时为 ``name`` 计时，否则不追踪直接运行。

    让调用点在追踪可选时保持单行写法（例如在可能在追踪加载流程之外
    被调用的 registry 辅助函数里）。
    """

    if tracker is None:
        yield
        return
    with tracker.stage(name, detail=detail):
        yield