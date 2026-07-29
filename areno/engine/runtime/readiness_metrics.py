"""Metrics export for serve readiness.

Provides Prometheus-compatible metrics for readiness state and stage durations.
These metrics are separate from business request metrics to avoid pollution.
"""

from __future__ import annotations

import time
from typing import Any

from areno.engine.runtime.readiness import ReadinessStateMachine


class ReadinessMetricsCollector:
    """Collector for readiness-related metrics.

    Emits metrics in Prometheus format for scraping by monitoring systems.
    These metrics are kept separate from business request metrics.
    """

    def __init__(self, readiness: ReadinessStateMachine | None = None):
        """Initialize the collector.

        Args:
            readiness: The readiness state machine to collect from
        """
        self._readiness = readiness
        self._probe_request_count = 0
        self._probe_start_time = time.time()

    def record_probe_request(self) -> None:
        """Record a probe request (not counted in business metrics)."""
        self._probe_request_count += 1

    def get_metrics(self) -> str:
        """Get readiness metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Readiness state gauge
        lines.append("# HELP areno_serve_readiness_state Current serve readiness state")
        lines.append("# TYPE areno_serve_readiness_state gauge")
        state_value = 0
        if self._readiness is not None:
            state_value = self._readiness.get_state_for_metrics()
        lines.append(f"areno_serve_readiness_state {state_value}")

        # Stage duration gauges
        lines.append("# HELP areno_serve_readiness_stage_duration_ms Duration of each readiness stage in milliseconds")
        lines.append("# TYPE areno_serve_readiness_stage_duration_ms gauge")

        if self._readiness is not None:
            durations = self._readiness.get_stage_durations_for_metrics()
            for stage_name, duration_ms in durations.items():
                lines.append(f'areno_serve_readiness_stage_duration_ms{{stage="{stage_name}"}} {duration_ms}')

        # Probe request count (not business requests)
        lines.append("# HELP areno_serve_probe_requests_total Total number of probe requests")
        lines.append("# TYPE areno_serve_probe_requests_total counter")
        lines.append(f"areno_serve_probe_requests_total {self._probe_request_count}")

        # Uptime since probe start
        lines.append("# HELP areno_serve_uptime_seconds Uptime since server start")
        lines.append("# TYPE areno_serve_uptime_seconds gauge")
        uptime = time.time() - self._probe_start_time
        lines.append(f"areno_serve_uptime_seconds {uptime:.3f}")

        return "\n".join(lines) + "\n"

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get readiness metrics as a dictionary.

        Returns:
            Dictionary of metrics
        """
        result: dict[str, Any] = {
            "readiness_state": 0,
            "stage_durations": {},
            "probe_requests_total": self._probe_request_count,
            "uptime_seconds": time.time() - self._probe_start_time,
        }

        if self._readiness is not None:
            result["readiness_state"] = self._readiness.get_state_for_metrics()
            result["stage_durations"] = self._readiness.get_stage_durations_for_metrics()

        return result


def is_probe_request(path: str) -> bool:
    """Check if a request path is a probe endpoint.

    Args:
        path: Request path

    Returns:
        True if this is a probe request
    """
    probe_paths = {"/health", "/ready", "/readiness/status", "/metrics"}
    return path in probe_paths or path.rstrip("/") in probe_paths
