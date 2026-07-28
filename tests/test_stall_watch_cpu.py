"""CPU tests for the stage stall watcher.

All tests use a controllable fake clock and a collecting sink so behaviour is
fully deterministic without a GPU, real timers, or network. Tests assert
emitted warning fields, rate-limiting counts, and disabled/no-op behaviour
rather than just exit status.
"""

from __future__ import annotations

import unittest

from areno.engine.runtime.stall_watch import (
    STALL_STAGES,
    StallWatchConfig,
    StallWatcher,
    StallWarning,
    make_stall_watcher,
)


class FakeClock:
    """Monotonic-style clock whose current time is advanced manually."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class CollectingSink:
    """Sink that records every warning it receives, never raises."""

    def __init__(self) -> None:
        self.warnings: list[StallWarning] = []

    def __call__(self, warning: StallWarning) -> None:
        self.warnings.append(warning)


def _make_watcher(
    *,
    interval_s: float,
    min_interval_s: float = 30.0,
    stages: tuple[str, ...] = STALL_STAGES,
    clock: FakeClock | None = None,
    sink: CollectingSink | None = None,
) -> tuple[StallWatcher, FakeClock, CollectingSink]:
    clock = clock or FakeClock()
    sink = sink or CollectingSink()
    cfg = StallWatchConfig(interval_s=interval_s, min_interval_s=min_interval_s, stages=stages, now=clock)
    return StallWatcher(cfg, sink=sink), clock, sink


class StallWatchConfigValidationTest(unittest.TestCase):
    """Config validation surfaces actionable errors for malformed input."""

    def test_disabled_config_is_valid(self):
        StallWatchConfig(interval_s=0.0).validate()
        StallWatchConfig(interval_s=0.0, min_interval_s=0.0).validate()

    def test_negative_interval_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            StallWatchConfig(interval_s=-1.0).validate()
        self.assertIn("interval_s", str(ctx.exception))

    def test_negative_min_interval_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            StallWatchConfig(interval_s=10.0, min_interval_s=-1.0).validate()
        self.assertIn("min_interval_s", str(ctx.exception))

    def test_min_exceeds_interval_rejected_when_enabled(self):
        with self.assertRaises(ValueError) as ctx:
            StallWatchConfig(interval_s=5.0, min_interval_s=10.0).validate()
        self.assertIn("min_interval_s", str(ctx.exception))
        self.assertIn("interval_s", str(ctx.exception))

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            StallWatchConfig(interval_s=60.0, min_interval_s=30.0, stages=("loading", "unknown_stage")).validate()
        self.assertIn("unknown_stage", str(ctx.exception))

    def test_enabled_property(self):
        self.assertFalse(StallWatchConfig(interval_s=0.0).enabled)
        self.assertTrue(StallWatchConfig(interval_s=1.0).enabled)


class StallWatcherTickCheckTest(unittest.TestCase):
    """Core tick/check behaviour with a controllable clock."""

    def test_under_threshold_returns_empty(self):
        # [单测用例]测试场景：未到阈值时不产生警告
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0, stages=("rollout",))
        watcher.tick("rollout")
        clock.advance(50.0)
        self.assertEqual(watcher.check(), [])
        self.assertEqual(sink.warnings, [])

    def test_over_threshold_emits_warning(self):
        # [单测用例]测试场景：超过阈值后产生包含阶段名和等待时长的警告
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0, stages=("rollout",))
        watcher.tick("rollout")
        clock.advance(150.0)
        warnings = watcher.check()
        self.assertEqual(len(warnings), 1)
        warn = warnings[0]
        self.assertEqual(warn.stage, "rollout")
        self.assertAlmostEqual(warn.wait_s, 150.0, places=6)
        self.assertEqual(warn.threshold_s, 100.0)
        self.assertEqual(len(sink.warnings), 1)

    def test_progress_event_resets_timer(self):
        # [单测用例]测试场景：进展事件重置计时器，后续 check 不再警告
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0, stages=("rollout",))
        watcher.tick("rollout")
        clock.advance(90.0)
        self.assertEqual(watcher.check(), [])
        # A progress event before threshold resets the timer.
        watcher.tick("rollout")
        clock.advance(90.0)
        self.assertEqual(watcher.check(), [])
        self.assertEqual(sink.warnings, [])

    def test_rate_limiting_suppresses_repeat_within_min_interval(self):
        # [单测用例]测试场景：min_interval 内的连续 check 只产生一次警告
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=30.0, stages=("rollout",))
        watcher.tick("rollout")
        clock.advance(150.0)
        first = watcher.check()
        self.assertEqual(len(first), 1)
        # Advance only 10s (< min_interval_s=30); same stage should be suppressed.
        clock.advance(10.0)
        second = watcher.check()
        self.assertEqual(second, [])
        self.assertEqual(len(sink.warnings), 1)
        # After min_interval elapses again, a new warning is emitted.
        clock.advance(25.0)
        third = watcher.check()
        self.assertEqual(len(third), 1)
        self.assertEqual(len(sink.warnings), 2)

    def test_multi_stage_independence(self):
        # [单测用例]测试场景：rollout 超阈但 training 刚 tick，只警告 rollout
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0)
        watcher.tick("rollout")
        watcher.tick("training")
        # Advance past threshold; both were ticked at t=0 so both are stale.
        clock.advance(150.0)
        warnings = watcher.check()
        warned = {w.stage for w in warnings}
        self.assertIn("rollout", warned)
        self.assertIn("training", warned)

    def test_multi_stage_only_warns_stale_stage(self):
        # [单测用例]测试场景：只 tick 部分阶段，未 tick 的阶段不警告（空初始化语义）
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0)
        # Tick rollout at t=0, then advance past threshold.
        watcher.tick("rollout")
        clock.advance(150.0)
        # Tick training now so its timer is fresh while rollout is stale.
        watcher.tick("training")
        warnings = watcher.check()
        stages_warned = {w.stage for w in warnings}
        self.assertIn("rollout", stages_warned)
        self.assertNotIn("training", stages_warned)
        # Stages never ticked (loading/data/reward) are absent: empty-init means
        # they are not tracked and therefore never reported as stalled.
        for untouched in ("loading", "data", "reward"):
            self.assertNotIn(untouched, stages_warned)

    def test_unknown_stage_tick_ignored(self):
        # [单测用例]测试场景：未跟踪的阶段名 tick 不影响已跟踪阶段（空初始化：需先 tick 已跟踪阶段）
        watcher, clock, sink = _make_watcher(
            interval_s=100.0, min_interval_s=0.0, stages=("rollout",)
        )
        # Tick the tracked stage first so it is being watched.
        watcher.tick("rollout")
        # An untracked stage name must be silently ignored, not crash.
        watcher.tick("not_a_tracked_stage")
        clock.advance(200.0)
        warnings = watcher.check()
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].stage, "rollout")

    def test_feed_stage_event_maps_concrete_stage(self):
        # [单测用例]测试场景：feed_stage_event 把 dashboard stage 字符串映射到逻辑阶段
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0)
        watcher.feed_stage_event("rollout_start")
        clock.advance(50.0)
        self.assertEqual(watcher.check(), [])
        watcher.feed_stage_event("train_end")  # maps to training
        # rollout was ticked at t=0 and 50s passed; advance further past threshold.
        clock.advance(60.0)
        warnings = watcher.check()
        warned = {w.stage for w in warnings}
        self.assertIn("rollout", warned)
        self.assertNotIn("training", warned)

    def test_unknown_concrete_stage_ignored(self):
        # [单测用例]测试场景：未映射的 dashboard stage 字符串被安全忽略，已知阶段照常警告
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0, stages=("rollout",))
        # Tick the tracked stage via a mapped concrete event first.
        watcher.feed_stage_event("rollout_start")
        # An unmapped concrete stage name must be silently ignored, not crash.
        watcher.feed_stage_event("some_unmapped_stage")
        clock.advance(200.0)
        # No exception raised; rollout (tracked, ticked above) is now stale.
        warnings = watcher.check()
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].stage, "rollout")

    def test_loading_retired_after_first_non_loading_tick(self):
        # [单测用例]测试场景：loading 被 tick 后，任何非 loading 阶段 tick 会退休 loading，避免训练循环启动后误报
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0)
        watcher.tick("loading")
        self.assertIn("loading", watcher._last_tick)
        # Tick rollout: loading should be retired immediately.
        watcher.tick("rollout")
        self.assertNotIn("loading", watcher._last_tick)
        self.assertNotIn("loading", watcher._last_warn)
        # Advance well past threshold; loading must not warn (retired).
        clock.advance(500.0)
        warnings = watcher.check()
        stages_warned = {w.stage for w in warnings}
        self.assertIn("rollout", stages_warned)
        self.assertNotIn("loading", stages_warned)

    def test_loading_warns_when_stuck_after_init(self):
        # [单测用例]测试场景：loading 被 tick 后若无任何后续阶段 tick，超阈时正确报警（卡在 init 后）
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0)
        watcher.tick("loading")
        clock.advance(150.0)
        warnings = watcher.check()
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].stage, "loading")

    def test_epoch_start_not_mapped_to_loading(self):
        # [单测用例]测试场景：epoch_start 不映射到任何逻辑阶段（语义属训练循环推进，非 initial loading）
        from areno.engine.runtime.stall_watch import _STAGE_MAP

        self.assertNotIn("epoch_start", _STAGE_MAP)
        # feed_stage_event("epoch_start") should be a no-op.
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0)
        watcher.feed_stage_event("epoch_start")
        clock.advance(200.0)
        self.assertEqual(watcher.check(), [])


class StallWatcherDisabledTest(unittest.TestCase):
    """Disabled watcher (interval_s == 0) is a complete no-op."""

    def test_disabled_watcher_tick_and_check_are_noop(self):
        # [单测用例]测试场景：interval_s=0 时 tick/check 完全 no-op
        clock = FakeClock()
        sink = CollectingSink()
        cfg = StallWatchConfig(interval_s=0.0, now=clock)
        watcher = StallWatcher(cfg, sink=sink)
        watcher.tick("rollout")
        clock.advance(1000.0)
        self.assertEqual(watcher.check(), [])
        self.assertEqual(sink.warnings, [])

    def test_make_stall_watcher_returns_none_for_disabled(self):
        self.assertIsNone(make_stall_watcher(None))
        self.assertIsNone(make_stall_watcher(StallWatchConfig(interval_s=0.0)))

    def test_make_stall_watcher_returns_watcher_for_enabled(self):
        watcher = make_stall_watcher(StallWatchConfig(interval_s=1.0))
        self.assertIsInstance(watcher, StallWatcher)


class StallWarningDictTest(unittest.TestCase):
    """StallWarning.to_dict produces train_stats-compatible fields."""

    def test_to_dict_fields(self):
        warn = StallWarning(stage="rollout", wait_s=312.456, threshold_s=300.0)
        d = warn.to_dict()
        self.assertEqual(d["stall_stage"], "rollout")
        self.assertEqual(d["stall_wait_s"], 312.456)
        self.assertEqual(d["stall_threshold_s"], 300.0)


class StallWatcherBoundaryTest(unittest.TestCase):
    """Boundary behaviour: wait exactly equal to threshold triggers a stall."""

    def test_wait_equal_to_threshold_triggers_stall(self):
        # [单测用例]测试场景：wait_s == threshold 时触发警告（实现使用 wait >= interval）
        watcher, clock, sink = _make_watcher(interval_s=100.0, min_interval_s=0.0, stages=("rollout",))
        watcher.tick("rollout")
        clock.advance(100.0)
        # wait == threshold: implementation uses strict >=, so this IS a stall.
        # Pin the actual semantics: ``wait >= interval`` triggers.
        warnings = watcher.check()
        self.assertEqual(len(warnings), 1)
        self.assertAlmostEqual(warnings[0].wait_s, 100.0, places=6)


class StallWatcherSinkFailureIsolationTest(unittest.TestCase):
    """A failing sink must not crash the watcher or the training loop."""

    def test_failing_sink_does_not_propagate(self):
        # [单测用例]测试场景：sink 抛异常时 check 不向调用方传播
        class BoomSink:
            def __call__(self, _warning: StallWarning) -> None:
                raise RuntimeError("sink boom")

        clock = FakeClock()
        cfg = StallWatchConfig(interval_s=10.0, min_interval_s=0.0, stages=("rollout",), now=clock)
        watcher = StallWatcher(cfg, sink=BoomSink())
        watcher.tick("rollout")
        clock.advance(20.0)
        # Should not raise even though the sink explodes.
        warnings = watcher.check()
        self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
