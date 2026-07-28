#!/usr/bin/env python3
"""Stall watcher 日志可观测性 demo。

不依赖 torch / GPU，纯 Python + 假时钟。直接运行：

    python scripts/demo_stall_warning.py

你会在终端看到带时间戳的 ``WARNING areno.stall_watch`` 日志行，以及
每一步 ``train_stats`` 里被注入的 ``stall_stage`` / ``stall_wait_s`` 字段。

可选环境变量：
    ARENO_LOG_LEVEL=DEBUG  # 调低全局级别看更多细节
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# 0) 把仓库根目录加进 sys.path，让 ``import areno`` 在任何 cwd 下都成立。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 1) 手动安装 areno logger（绕过 areno/__init__.py → torch 的依赖链）。
#    生产环境用 ``import areno`` 即可，它会自动调用 configure_default_logging()。
from areno.engine.log import configure_default_logging

configure_default_logging()
# 把级别交给环境变量，默认 INFO；WARNING 级别的 stall 日志会显示。
logging.getLogger("areno").setLevel(
    getattr(logging, os.environ.get("ARENO_LOG_LEVEL", "INFO").upper(), logging.INFO)
)

# 2) stall watcher 本体 —— 纯 Python，无 torch。
from areno.engine.runtime.stall_watch import (
    StallWatchConfig,
    StallWatcher,
    StallWarning,
    make_stall_watcher,
    _LoggerStallSink,
)


def main() -> int:
    # ----- 假时钟 -----
    clock = [0.0]

    def fake_now() -> float:
        return clock[0]

    # ----- 收集 sink：除了走默认 logger，还把 warning 收集起来便于打印 train_stats -----
    # 默认 sink (_LoggerStallSink) 已经会打 WARNING 日志，但一旦传入自定义 sink
    # 就会覆盖它。这里用 MultiSink 把两者串起来：既走 logger（生产格式），
    # 又收集到 list（便于本地展示）。
    collected: list[StallWarning] = []

    class MultiSink:
        def __init__(self, *sinks):
            self._sinks = sinks

        def __call__(self, w: StallWarning) -> None:
            for s in self._sinks:
                try:
                    s(w)
                except Exception:
                    pass

    logger_sink = _LoggerStallSink()

    def collecting_sink(w: StallWarning) -> None:
        collected.append(w)

    combined_sink = MultiSink(logger_sink, collecting_sink)

    # ----- 配置：threshold=100s, min_interval=60s -----
    # 对应 CLI: --stall-warn-interval-s 100 --stall-warn-min-interval-s 60
    cfg = StallWatchConfig(
        interval_s=100.0,
        min_interval_s=60.0,
        stages=("loading", "rollout", "training"),
        now=fake_now,
    )
    cfg.validate()  # Trainer 构造时会调这一步
    watcher = StallWatcher(cfg, sink=combined_sink)

    print("=" * 72, file=sys.stderr)
    print("Stall Watcher 日志 demo", file=sys.stderr)
    print("threshold=100s  min_interval=60s", file=sys.stderr)
    print("下面每一步会: tick -> advance clock -> check -> 打印 train_stats", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    # ----- 模拟训练循环 -----
    # t=0:   init 完成, tick loading
    # t=5:   rollout 正常推进
    # t=60:  rollout 最后一次 tick, 之后卡住
    # t=90:  training 正常 tick 一次, 之后也卡住
    # 每 30s 在 "train step" 末尾 check 一次
    ticks = [
        (0, "loading"),
        (5, "rollout"),
        (30, "rollout"),
        (60, "rollout"),   # 之后 rollout 卡死
        (90, "training"),  # 之后 training 也卡死
    ]
    checks = [120, 150, 180, 210, 240, 270, 300]

    events = sorted(
        [(t, "tick", s) for t, s in ticks] + [(t, "check", "") for t in checks]
    )

    step = 0
    for t, kind, stage in events:
        clock[0] = t
        if kind == "tick":
            watcher.tick(stage)
            print(f"[t={t:4d}s] tick({stage!r})", file=sys.stderr)
        else:
            step += 1
            warnings = watcher.check()
            # 模拟 Trainer.train() 末尾把最后一条 warning 注入 train_stats
            train_stats: dict = {"loss": 0.1 * step, "step": step}
            if warnings:
                latest = warnings[-1]
                train_stats["stall_stage"] = latest.stage
                train_stats["stall_wait_s"] = round(latest.wait_s, 3)
                train_stats["stall_threshold_s"] = latest.threshold_s
            # 这就是 trainers 里每步都会打的那行：
            print(f"[t={t:4d}s] step={step} train_stats={train_stats}", file=sys.stderr)

    print("=" * 72, file=sys.stderr)
    print(f"总共触发 {len(collected)} 条 stall warning（已通过 areno.stall_watch logger 输出）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

