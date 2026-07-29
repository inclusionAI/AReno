"""CPU tests for serve readiness functionality.

Tests cover:
- Normal state transition flow
- Timeout handling
- Worker/Router failure scenarios
- Input validation
- Default behavior (backward compatibility)
- Metrics isolation
- Output formats
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from areno.engine.runtime.readiness import ReadinessState, ReadinessStateMachine, STATE_ORDER
from areno.engine.runtime.readiness_metrics import ReadinessMetricsCollector, is_probe_request
from areno.engine.runtime.readiness_validation import ValidationError, validate_readiness_options


class TestReadinessStateMachine:
    """Tests for the readiness state machine."""

    def test_initial_state_when_enabled(self):
        """CPU-01: State machine starts in MODEL_LOADING when enabled."""
        sm = ReadinessStateMachine(enabled=True)

        assert sm.enabled is True
        assert sm.current_state == ReadinessState.MODEL_LOADING

    def test_initial_state_when_disabled(self):
        """CPU-01: State machine has no state when disabled."""
        sm = ReadinessStateMachine(enabled=False)

        assert sm.enabled is False
        assert sm.current_state is None

    def test_normal_state_transition_flow(self):
        """CPU-01: Test normal progression through all states."""
        sm = ReadinessStateMachine(enabled=True)

        # Progress through states
        sm.mark_stage_complete(ReadinessState.MODEL_LOADING)
        assert sm.current_state == ReadinessState.WORKER_READY

        sm.mark_stage_complete(ReadinessState.WORKER_READY)
        assert sm.current_state == ReadinessState.ROUTER_READY

        sm.mark_stage_complete(ReadinessState.ROUTER_READY)
        assert sm.current_state == ReadinessState.MINIMAL_PROBE

        sm.mark_stage_complete(ReadinessState.MINIMAL_PROBE)
        assert sm.current_state == ReadinessState.READY

    def test_invalid_transition_blocked(self):
        """CPU-01: Invalid state transitions are blocked."""
        sm = ReadinessStateMachine(enabled=True)

        # Can't jump from MODEL_LOADING to READY
        result = sm.transition_to(ReadinessState.READY)
        assert result is False
        assert sm.current_state == ReadinessState.MODEL_LOADING

    def test_failed_state_is_terminal(self):
        """CPU-01: FAILED state is terminal."""
        sm = ReadinessStateMachine(enabled=True)

        sm.mark_failed(error="Test error")
        assert sm.current_state == ReadinessState.FAILED

        # Can't transition out of FAILED
        result = sm.transition_to(ReadinessState.READY)
        assert result is False

    def test_timeout_detection(self):
        """CPU-02: Timeout detection works correctly."""
        sm = ReadinessStateMachine(
            enabled=True,
            timeout_per_stage_seconds=0.01,  # Very short timeout
        )

        # Wait for timeout
        time.sleep(0.02)

        # Check timeout should transition to FAILED
        timed_out = sm.check_timeout()
        assert timed_out is True
        assert sm.current_state == ReadinessState.FAILED

    def test_timeout_not_triggered_when_disabled(self):
        """CPU-02: Timeout not checked when disabled."""
        sm = ReadinessStateMachine(
            enabled=False,
            timeout_per_stage_seconds=0.01,
        )

        time.sleep(0.02)

        timed_out = sm.check_timeout()
        assert timed_out is False

    def test_worker_failure(self):
        """CPU-03: Worker failure is tracked correctly."""
        sm = ReadinessStateMachine(enabled=True)

        # Progress to WORKER_READY
        sm.mark_stage_complete(ReadinessState.MODEL_LOADING)
        assert sm.current_state == ReadinessState.WORKER_READY

        # Fail at worker stage
        sm.mark_failed(error="Worker initialization failed")
        assert sm.current_state == ReadinessState.FAILED

        status = sm.get_status()
        assert status.status == "failed"
        assert "Worker initialization failed" in status.error

    def test_router_failure(self):
        """CPU-04: Router failure is tracked correctly."""
        sm = ReadinessStateMachine(enabled=True)

        # Progress to ROUTER_READY
        sm.mark_stage_complete(ReadinessState.MODEL_LOADING)
        sm.mark_stage_complete(ReadinessState.WORKER_READY)
        assert sm.current_state == ReadinessState.ROUTER_READY

        # Fail at router stage
        sm.mark_failed(error="Router initialization failed")
        assert sm.current_state == ReadinessState.FAILED

        status = sm.get_status()
        assert status.status == "failed"
        assert "Router initialization failed" in status.error

    def test_status_reporting(self):
        """CPU-01: Status reporting includes all stages."""
        sm = ReadinessStateMachine(enabled=True)

        status = sm.get_status()
        assert status.status == "not_ready"
        assert status.current_stage == "model_loading"
        assert "model_loading" in status.stages
        assert "worker_ready" in status.stages
        assert "router_ready" in status.stages
        assert "minimal_probe" in status.stages

    def test_status_to_dict(self):
        """CPU-10: Status can be serialized to dict."""
        sm = ReadinessStateMachine(enabled=True)
        sm.mark_stage_complete(ReadinessState.MODEL_LOADING)

        status = sm.get_status()
        d = status.to_dict()

        assert "status" in d
        assert "current_stage" in d
        assert "stages" in d
        assert "last_completed_stage" in d
        assert "error" in d

    def test_callback_on_state_change(self):
        """CPU-01: Callback is invoked on state change."""
        callback_calls = []

        def callback(old_state, new_state, duration_ms):
            callback_calls.append((old_state, new_state, duration_ms))

        sm = ReadinessStateMachine(
            enabled=True,
            on_state_change=callback,
        )

        sm.mark_stage_complete(ReadinessState.MODEL_LOADING)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == ReadinessState.MODEL_LOADING
        assert callback_calls[0][1] == ReadinessState.WORKER_READY

    def test_metrics_state_values(self):
        """CPU-08: Metrics state values are correct."""
        sm = ReadinessStateMachine(enabled=True)

        # Check initial state
        assert sm.get_state_for_metrics() == 0  # MODEL_LOADING

        # Progress and check
        sm.mark_stage_complete(ReadinessState.MODEL_LOADING)
        assert sm.get_state_for_metrics() == 1  # WORKER_READY

        sm.mark_stage_complete(ReadinessState.WORKER_READY)
        assert sm.get_state_for_metrics() == 2  # ROUTER_READY

        sm.mark_stage_complete(ReadinessState.ROUTER_READY)
        assert sm.get_state_for_metrics() == 3  # MINIMAL_PROBE

        sm.mark_stage_complete(ReadinessState.MINIMAL_PROBE)
        assert sm.get_state_for_metrics() == 4  # READY

        sm.mark_failed()
        assert sm.get_state_for_metrics() == 5  # FAILED

    def test_stage_durations_for_metrics(self):
        """CPU-08: Stage durations are tracked for metrics."""
        sm = ReadinessStateMachine(enabled=True)

        # Initially no durations
        durations = sm.get_stage_durations_for_metrics()
        assert "model_loading" not in durations

        # Complete a stage
        time.sleep(0.01)
        sm.mark_stage_complete(ReadinessState.MODEL_LOADING)

        durations = sm.get_stage_durations_for_metrics()
        assert "model_loading" in durations
        assert durations["model_loading"] >= 10  # At least 10ms


class TestReadinessValidation:
    """Tests for readiness input validation."""

    def test_validate_positive_timeout(self):
        """CPU-05: Valid positive timeout is accepted."""
        config = validate_readiness_options(enabled=True, timeout=30)

        assert config["enabled"] is True
        assert config["timeout_per_stage_seconds"] == 30

    def test_validate_timeout_rejects_negative(self):
        """CPU-05: Negative timeout is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_readiness_options(enabled=True, timeout=-1)

        assert "must be at least" in str(exc_info.value)

    def test_validate_timeout_rejects_zero(self):
        """CPU-05: Zero timeout is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_readiness_options(enabled=True, timeout=0)

        assert "must be at least" in str(exc_info.value)

    def test_validate_timeout_rejects_non_numeric_string(self):
        """CPU-06: Non-numeric string timeout is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_readiness_options(enabled=True, timeout="not_a_number")

        assert "invalid format" in str(exc_info.value)

    def test_validate_timeout_accepts_numeric_string(self):
        """CPU-05: Numeric string timeout is accepted."""
        config = validate_readiness_options(enabled=True, timeout="30")

        assert config["timeout_per_stage_seconds"] == 30

    def test_validate_timeout_rejects_boolean(self):
        """CPU-06: Boolean timeout is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_readiness_options(enabled=True, timeout=True)

        assert "not a boolean" in str(exc_info.value)

    def test_validate_enabled_accepts_true(self):
        """CPU-05: Enabled=true is accepted."""
        config = validate_readiness_options(enabled=True)
        assert config["enabled"] is True

    def test_validate_enabled_accepts_false(self):
        """CPU-05: Enabled=false is accepted."""
        config = validate_readiness_options(enabled=False)
        assert config["enabled"] is False

    def test_validate_enabled_accepts_string_true(self):
        """CPU-05: String 'true' is accepted."""
        config = validate_readiness_options(enabled="true")
        assert config["enabled"] is True

    def test_validate_enabled_accepts_string_false(self):
        """CPU-05: String 'false' is accepted."""
        config = validate_readiness_options(enabled="false")
        assert config["enabled"] is False

    def test_validation_error_to_dict(self):
        """CPU-05: ValidationError can be serialized."""
        try:
            validate_readiness_options(enabled=True, timeout=-1)
        except ValidationError as e:
            d = e.to_dict()
            assert "field" in d
            assert "value" in d
            assert "message" in d


class TestReadinessMetrics:
    """Tests for readiness metrics collection."""

    def test_metrics_collector_initialization(self):
        """CPU-08: Metrics collector initializes correctly."""
        sm = ReadinessStateMachine(enabled=True)
        collector = ReadinessMetricsCollector(sm)

        assert collector._readiness is sm
        assert collector._probe_request_count == 0

    def test_probe_request_counting(self):
        """CPU-08: Probe requests are counted separately."""
        sm = ReadinessStateMachine(enabled=True)
        collector = ReadinessMetricsCollector(sm)

        collector.record_probe_request()
        collector.record_probe_request()
        collector.record_probe_request()

        assert collector._probe_request_count == 3

    def test_metrics_output_format(self):
        """CPU-08: Metrics output is in Prometheus format."""
        sm = ReadinessStateMachine(enabled=True)
        collector = ReadinessMetricsCollector(sm)

        metrics = collector.get_metrics()

        assert "areno_serve_readiness_state" in metrics
        assert "areno_serve_readiness_stage_duration_ms" in metrics
        assert "areno_serve_probe_requests_total" in metrics
        assert "areno_serve_uptime_seconds" in metrics

    def test_metrics_dict_format(self):
        """CPU-08: Metrics can be returned as dict."""
        sm = ReadinessStateMachine(enabled=True)
        collector = ReadinessMetricsCollector(sm)

        metrics = collector.get_metrics_dict()

        assert "readiness_state" in metrics
        assert "stage_durations" in metrics
        assert "probe_requests_total" in metrics
        assert "uptime_seconds" in metrics

    def test_is_probe_request_detection(self):
        """CPU-08: Probe request detection works correctly."""
        assert is_probe_request("/health") is True
        assert is_probe_request("/ready") is True
        assert is_probe_request("/readiness/status") is True
        assert is_probe_request("/metrics") is True

        assert is_probe_request("/v1/chat/completions") is False
        assert is_probe_request("/v1/models") is False

    def test_metrics_without_readiness(self):
        """CPU-08: Metrics work even without readiness enabled."""
        collector = ReadinessMetricsCollector(None)

        metrics = collector.get_metrics()
        assert "areno_serve_readiness_state" in metrics
        assert "areno_serve_readiness_state 0" in metrics


class TestReadinessBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_disabled_by_default(self):
        """CPU-07: Readiness is disabled by default."""
        config = validate_readiness_options()
        assert config["enabled"] is False

    def test_no_state_when_disabled(self):
        """CPU-07: No state tracking when disabled."""
        sm = ReadinessStateMachine(enabled=False)

        assert sm.current_state is None
        status = sm.get_status()
        assert status.status == "not_enabled"

    def test_disabled_state_machine_ignores_operations(self):
        """CPU-07: Disabled state machine ignores all operations."""
        sm = ReadinessStateMachine(enabled=False)

        # These should all be no-ops
        sm.mark_stage_complete(ReadinessState.MODEL_LOADING)
        sm.mark_failed(error="Test")
        sm.check_timeout()

        # State should still be None
        assert sm.current_state is None


class TestReadinessIntegration:
    """Integration tests for readiness components."""

    def test_full_lifecycle_with_metrics(self):
        """INT-01: Full lifecycle with metrics collection."""
        sm = ReadinessStateMachine(enabled=True)
        collector = ReadinessMetricsCollector(sm)

        # Progress through all stages
        for state in STATE_ORDER[:-1]:  # Exclude READY
            sm.mark_stage_complete(state)
            collector.record_probe_request()

        # Check final state
        assert sm.current_state == ReadinessState.READY

        # Check metrics
        metrics = collector.get_metrics_dict()
        assert metrics["readiness_state"] == 4  # READY
        assert metrics["probe_requests_total"] == 4
        assert len(metrics["stage_durations"]) > 0

    def test_failure_recovery_tracking(self):
        """INT-02: Failure is tracked correctly."""
        sm = ReadinessStateMachine(enabled=True)

        # Fail during model loading
        sm.mark_failed(error="CUDA OOM")

        status = sm.get_status()
        assert status.status == "failed"
        assert status.error == "CUDA OOM"
        assert status.current_stage == "failed"

        # Check that failed stage is recorded
        assert status.stages["model_loading"].state == "failed"

    def test_metrics_fields_correct(self):
        """INT-03: Metrics fields are correctly populated."""
        sm = ReadinessStateMachine(enabled=True)
        sm.mark_stage_complete(ReadinessState.MODEL_LOADING)
        sm.mark_stage_complete(ReadinessState.WORKER_READY)

        collector = ReadinessMetricsCollector(sm)
        metrics = collector.get_metrics()

        # Check that stage durations are present
        assert 'stage="model_loading"' in metrics
        assert 'stage="worker_ready"' in metrics

        # Check state value
        lines = metrics.split("\n")
        state_line = [l for l in lines if l.startswith("areno_serve_readiness_state ")]
        assert len(state_line) == 1
        assert "2" in state_line[0]  # ROUTER_READY state
