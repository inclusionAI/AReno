"""CPU tests for the watch command.

These tests verify the core logic of the watch module without requiring
GPU or actual training runs.
"""

import json
import os
import re
import signal
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from areno.cli.watch import (
    ARENO_RUNTIME_DIR,
    DEFAULT_INTERVAL,
    RUNS_DIR,
    GracefulExit,
    RunStatus,
    WatchConfig,
    calculate_eta,
    check_training_active,
    find_latest_run_id,
    find_status_file,
    format_eta,
    is_process_running,
    read_status,
    render_json,
    render_line,
    render_tty,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_runs_dir(tmp_path):
    """Create a temporary runs directory."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return runs_dir


@pytest.fixture
def valid_status_content():
    """Valid dashboard state file content."""
    return {
        "pid": 12345,
        "stage": "train",
        "status": "running",
        "updated_at": time.time(),
        "step": 150,
        "epoch": 3,
        "role": "worker-0",
        "loss": 0.2345,
        "reward_mean": 0.8923,
        "throughput": 1200,
        "total_steps": 1000,
    }


# =============================================================================
# Test: find_latest_run_id
# =============================================================================


def test_find_latest_run_id_empty(temp_runs_dir):
    """Test finding latest run ID when directory is empty."""
    with patch("areno.cli.watch.RUNS_DIR", temp_runs_dir):
        result = find_latest_run_id()
        assert result is None


def test_find_latest_run_id_single(temp_runs_dir):
    """Test finding latest run ID with single run."""
    run_dir = temp_runs_dir / "run_001"
    run_dir.mkdir()

    with patch("areno.cli.watch.RUNS_DIR", temp_runs_dir):
        result = find_latest_run_id()
        assert result == "run_001"


def test_find_latest_run_id_multiple(temp_runs_dir):
    """Test finding latest run ID with multiple runs."""
    # Create runs with different modification times
    run_1 = temp_runs_dir / "run_001"
    run_2 = temp_runs_dir / "run_002"
    run_3 = temp_runs_dir / "run_003"

    run_1.mkdir()
    time.sleep(0.1)  # Ensure different mtime
    run_2.mkdir()
    time.sleep(0.1)
    run_3.mkdir()

    with patch("areno.cli.watch.RUNS_DIR", temp_runs_dir):
        result = find_latest_run_id()
        # Most recent should be run_003
        assert result == "run_003"


# =============================================================================
# Test: find_status_file
# =============================================================================


def test_find_status_file_not_found(temp_runs_dir):
    """Test finding status file when run doesn't exist."""
    with patch("areno.cli.watch.RUNS_DIR", temp_runs_dir):
        result = find_status_file("nonexistent")
        # Falls back to looking for dashboard_state.*.json in runs dir
        assert result is None


def test_find_status_file_exists(temp_runs_dir, valid_status_content):
    """Test finding status file when it exists."""
    run_dir = temp_runs_dir / "run_001"
    run_dir.mkdir()

    status_file = run_dir / "dashboard_state.12345.json"
    status_file.write_text(json.dumps(valid_status_content))

    with patch("areno.cli.watch.RUNS_DIR", temp_runs_dir):
        result = find_status_file("run_001")
        assert result == status_file


# =============================================================================
# Test: read_status
# =============================================================================


def test_read_status_valid(tmp_path, valid_status_content):
    """Test reading valid status file."""
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps(valid_status_content))

    result = read_status(status_file)

    assert result is not None
    assert result.pid == 12345
    assert result.stage == "train"
    assert result.status == "running"
    assert result.step == 150
    assert result.loss == 0.2345
    assert result.reward_mean == 0.8923
    assert result.throughput == 1200
    assert result.total_steps == 1000


def test_read_status_missing_fields(tmp_path):
    """Test reading status file with missing fields."""
    minimal_content = {
        "pid": 12345,
        "stage": "rollout",
        "status": "running",
    }
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps(minimal_content))

    result = read_status(status_file)

    assert result is not None
    assert result.pid == 12345
    assert result.stage == "rollout"
    assert result.status == "running"
    assert result.step is None
    assert result.loss is None


def test_read_status_invalid_json(tmp_path):
    """Test reading invalid JSON file."""
    status_file = tmp_path / "status.json"
    status_file.write_text("{ invalid json")

    result = read_status(status_file)

    assert result is None


def test_read_status_nonexistent(tmp_path):
    """Test reading non-existent file."""
    status_file = tmp_path / "nonexistent.json"

    result = read_status(status_file)

    assert result is None


# =============================================================================
# Test: is_process_running
# =============================================================================


def test_is_process_running_invalid():
    """Test checking non-existent process."""
    # Use an unlikely PID
    result = is_process_running(999999)
    assert result is False


def test_is_process_running_current():
    """Test checking current process."""
    result = is_process_running(os.getpid())
    assert result is True


# =============================================================================
# Test: check_training_active
# =============================================================================


def test_check_training_active_running():
    """Test checking active training."""
    status = RunStatus(
        pid=os.getpid(),
        stage="train",
        status="running",
        updated_at=time.time(),
        step=100,
    )
    assert check_training_active(status) is True


def test_check_training_active_completed():
    """Test checking completed training."""
    status = RunStatus(
        pid=12345,
        stage="train",
        status="completed",
        updated_at=time.time(),
        step=1000,
    )
    assert check_training_active(status) is False


# =============================================================================
# Test: calculate_eta
# =============================================================================


def test_calculate_eta_basic():
    """Test basic ETA calculation."""
    # At step 50, with 100 total, after 10 seconds
    result = calculate_eta(current_step=50, total_steps=100, start_time=0, current_time=10)
    assert result == 10  # 50 steps in 10 seconds = 5 steps/s, remaining 50 steps = 10s


def test_calculate_eta_zero_step():
    """Test ETA with zero current step."""
    result = calculate_eta(current_step=0, total_steps=100, start_time=0, current_time=10)
    assert result is None


def test_calculate_eta_completed():
    """Test ETA when training is completed."""
    result = calculate_eta(current_step=100, total_steps=100, start_time=0, current_time=10)
    assert result == 0


def test_calculate_eta_invalid_total():
    """Test ETA with invalid total steps."""
    result = calculate_eta(current_step=50, total_steps=0, start_time=0, current_time=10)
    assert result is None


def test_calculate_eta_invalid_time():
    """Test ETA with invalid time values."""
    result = calculate_eta(current_step=50, total_steps=100, start_time=10, current_time=0)
    assert result is None


# =============================================================================
# Test: format_eta
# =============================================================================


def test_format_eta_none():
    """Test formatting None ETA."""
    assert format_eta(None) == "N/A"


def test_format_eta_zero():
    """Test formatting zero ETA."""
    assert format_eta(0) == "done"


def test_format_eta_seconds():
    """Test formatting ETA in seconds."""
    assert format_eta(45) == "45s"


def test_format_eta_minutes():
    """Test formatting ETA in minutes."""
    assert format_eta(120) == "2m 0s"


def test_format_eta_hours():
    """Test formatting ETA in hours."""
    assert format_eta(3665) == "1h 1m"


# =============================================================================
# Test: render_line
# =============================================================================


def test_render_line_basic():
    """Test basic line rendering."""
    status = RunStatus(
        pid=12345,
        stage="train",
        status="running",
        updated_at=time.time(),
        step=150,
        total_steps=1000,
        loss=0.2345,
        reward_mean=0.8923,
        throughput=1200,
    )

    result = render_line(status, eta=10)

    assert "Step 150/1000" in result
    assert "Loss 0.2345" in result
    assert "Reward 0.8923" in result
    assert "tok/s 1200" in result
    assert "ETA 10s" in result
    assert "Stage=train" in result
    assert "Status=running" in result


def test_render_line_minimal():
    """Test line rendering with minimal data."""
    status = RunStatus(
        pid=12345,
        stage="rollout",
        status="running",
        updated_at=time.time(),
    )

    result = render_line(status, eta=None)

    assert "Stage=rollout" in result
    assert "Status=running" in result


# =============================================================================
# Test: render_json
# =============================================================================


def test_render_json_basic():
    """Test JSON rendering."""
    status = RunStatus(
        pid=12345,
        stage="train",
        status="running",
        updated_at=1000.0,
        step=150,
        total_steps=1000,
        loss=0.2345,
        reward_mean=0.8923,
        throughput=1200,
    )

    result = render_json(status, eta=10)

    data = json.loads(result)

    assert data["step"] == 150
    assert data["total_steps"] == 1000
    assert data["loss"] == 0.2345
    assert data["reward"] == 0.8923
    assert data["throughput"] == 1200
    assert data["eta_seconds"] == 10
    assert data["stage"] == "train"
    assert data["status"] == "running"
    assert data["pid"] == 12345


# =============================================================================
# Test: render_tty (basic smoke test)
# =============================================================================


def test_render_tty_basic():
    """Test basic TTY rendering."""
    status = RunStatus(
        pid=12345,
        stage="train",
        status="running",
        updated_at=time.time(),
        step=150,
        total_steps=1000,
        loss=0.2345,
        reward_mean=0.8923,
        throughput=1200,
    )

    result = render_tty(status, eta=60, elapsed=10)

    # Strip ANSI codes for basic content checking
    plain_result = re.sub(r"\x1b\[[0-9;]*m", "", result)

    assert "AReno Watch" in plain_result
    assert "Step: 150/1000" in plain_result
    assert "Loss: 0.2345" in plain_result
    assert "Reward: 0.8923" in plain_result
    assert "Throughput: 1200 tok/s" in plain_result
    assert "ETA: 1m 0s" in plain_result


# =============================================================================
# Test: GracefulExit
# =============================================================================


def test_graceful_exit_init():
    """Test GracefulExit initialization."""
    handler = GracefulExit()
    assert handler.exit_requested is False


def test_graceful_exit_trigger():
    """Test triggering graceful exit via signal."""
    handler = GracefulExit()

    # Simulate signal
    handler._handler(signal.SIGINT, None)

    assert handler.exit_requested is True


# =============================================================================
# Test: WatchConfig validation
# =============================================================================


def test_watch_config_defaults():
    """Test WatchConfig default values."""
    config = WatchConfig(
        run_id=None,
        latest=False,
        interval=1,
        json_output=False,
        quiet=False,
        timeout=0,
    )

    assert config.run_id is None
    assert config.latest is False
    assert config.interval == 1
    assert config.json_output is False
    assert config.quiet is False
    assert config.timeout == 0


def test_watch_config_custom():
    """Test WatchConfig with custom values."""
    config = WatchConfig(
        run_id="test_run",
        latest=True,
        interval=5,
        json_output=True,
        quiet=True,
        timeout=3600,
    )

    assert config.run_id == "test_run"
    assert config.latest is True
    assert config.interval == 5
    assert config.json_output is True
    assert config.quiet is True
    assert config.timeout == 3600


# =============================================================================
# Test: Constants
# =============================================================================


def test_default_interval():
    """Test DEFAULT_INTERVAL is reasonable."""
    assert DEFAULT_INTERVAL == 1
    assert DEFAULT_INTERVAL > 0


def test_directory_constants():
    """Test directory constants exist and are Path objects."""
    assert isinstance(ARENO_RUNTIME_DIR, Path)
    assert isinstance(RUNS_DIR, Path)
    assert "areno" in str(ARENO_RUNTIME_DIR)
