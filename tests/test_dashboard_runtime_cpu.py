"""Dashboard runtime reports must follow the diagnostics backend."""

import pytest

from areno.dashboard import server
from tests.test_cli_diagnostics_cpu import _mlx_report, _ready_report


@pytest.fixture
def runtime(monkeypatch):
    for name in ("ENV_REPORT_CACHE", "ENV_CHECKS_CACHE", "ENV_CHECK_COUNTS_CACHE"):
        monkeypatch.setattr(server, name, None)

    def configure(report, gpu_output=""):
        calls = []
        monkeypatch.setattr(server, "collect_env", lambda: report)

        def run_text(command):
            calls.append(command)
            return gpu_output if command[0] == "nvidia-smi" else "test"

        monkeypatch.setattr(server, "run_text", run_text)
        return calls

    return configure


def test_mlx_runtime_reports_metal_without_cuda_probe(tmp_path, runtime):
    calls = runtime(_mlx_report(str(tmp_path)))

    result = server.runtime_env()

    assert result["ready"]
    assert result["check_counts"]["fail"] == 0
    assert result["gpus"] == [{"name": "Apple Silicon GPU", "backend": "mlx"}]
    assert result["gpu_summary"] == "Apple Silicon GPU (Metal)"
    assert not any(command[0] == "nvidia-smi" for command in calls)
    assert server.runtime_attention()["attention"] is None


def test_metal_unavailable_stays_not_ready(tmp_path, runtime):
    report = _mlx_report(str(tmp_path))
    report["metal"].update(available=False, error="Metal unavailable")
    calls = runtime(report)

    result = server.runtime_env()

    assert not result["ready"]
    assert result["gpus"] == []
    assert result["gpu_summary"] == "Metal unavailable"
    assert not any(command[0] == "nvidia-smi" for command in calls)
    attention = server.runtime_attention()["attention"]
    assert attention["name"] == "MLX Metal availability"
    assert attention["status"] == "fail"


def test_cuda_gpu_metrics_unchanged(tmp_path, runtime):
    calls = runtime(_ready_report(str(tmp_path)), "NVIDIA H100, 1024, 81920, 25\n")

    result = server.runtime_env()

    assert result["ready"]
    assert result["gpus"] == [
        {"name": "NVIDIA H100", "memory_used_mb": 1024, "memory_total_mb": 81920, "utilization": 25}
    ]
    assert result["gpu_summary"] == "NVIDIA H100 1024/81920MB"
    assert any(command[0] == "nvidia-smi" for command in calls)
