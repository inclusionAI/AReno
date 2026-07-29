#!/usr/bin/env python3
"""Watch command demo - lightweight simulation without AReno dependencies.

This file demonstrates the watch rendering logic in isolation for testing
and documentation purposes. It mocks the status data flow that the actual
`areno watch` command reads from dashboard_state.{pid}.json files.
"""

import json
import time
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# =============================================================================
# Data Model
# =============================================================================


@dataclass
class RunStatus:
    """Parsed status from dashboard state file.
    
    Mirrors the RunStatus dataclass in areno/cli/watch.py for demo purposes.
    """
    pid: int
    stage: str
    status: str
    updated_at: float
    step: int = None
    epoch: int = None
    role: str = None
    loss: float = None
    reward_mean: float = None
    throughput: int = None
    total_steps: int = None


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"


def green(text): return f"{Colors.GREEN}{text}{Colors.RESET}"
def cyan(text): return f"{Colors.CYAN}{text}{Colors.RESET}"
def magenta(text): return f"{Colors.MAGENTA}{text}{Colors.RESET}"
def bold(text): return f"{Colors.BOLD}{text}{Colors.RESET}"


# =============================================================================
# Rendering Helpers
# =============================================================================


def format_eta(seconds):
    """Format ETA seconds into human-readable string (e.g., '5m 30s')."""
    if seconds is None:
        return "N/A"
    if seconds <= 0:
        return "done"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def calculate_eta(current_step, total_steps, start_time, current_time):
    """Estimate remaining time based on current progress.
    
    Uses simple linear projection: (remaining_steps / steps_per_second).
    Returns None if insufficient data for estimation.
    """
    if current_step is None or total_steps is None or total_steps == 0:
        return None
    if current_step >= total_steps:
        return 0
    elapsed = current_time - start_time
    if elapsed <= 0 or current_step <= 0:
        return None
    rate = current_step / elapsed
    if rate <= 0:
        return None
    remaining_steps = total_steps - current_step
    return int(remaining_steps / rate)


def render_tty(status: RunStatus, eta: Optional[int], elapsed: float) -> str:
    width = 60
    progress = ""
    if status.step is not None and status.total_steps is not None:
        percent = status.step / max(status.total_steps, 1)
        bar_width = 15
        filled = int(percent * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        progress = f"Step: {status.step}/{status.total_steps}  {bar}  {percent * 100:.1f}%"

    loss_str = f"{cyan('Loss:')} {status.loss:.4f}" if status.loss else f"{cyan('Loss:')} N/A"
    reward_str = f"{magenta('Reward:')} {status.reward_mean:.4f}" if status.reward_mean else f"{magenta('Reward:')} N/A"
    throughput_str = f"{cyan('Throughput:')} {status.throughput} tok/s" if status.throughput else ""

    lines = [
        "╔" + "═" * (width - 2) + "╗",
        f"║  {bold('AReno Watch')} - Run status" + " " * (width - len("  AReno Watch - Run status") - 4) + "║",
        "╠" + "═" * (width - 2) + "╣",
    ]
    if progress:
        lines.append(f"║  {green(progress)}" + " " * (width - len(green(progress)) - 4) + "║")
    lines.append(f"║  {loss_str}    {reward_str}" + " " * (width - len(f"{loss_str}    {reward_str}") - 4) + "║")
    if throughput_str:
        lines.append(f"║  {throughput_str}" + " " * (width - len(throughput_str) - 4) + "║")
    lines.append(f"║  {cyan('Stage:')} {status.stage}    {bold('ETA:')} {format_eta(eta)}" + " " * (width - len(f"Stage: {status.stage}    ETA: {format_eta(eta)}") - 4) + "║")
    lines.append("╚" + "═" * (width - 2) + "╝")
    return "\n".join(lines)


def render_json(status: RunStatus, eta: Optional[int]) -> str:
    data = {
        "step": status.step,
        "total_steps": status.total_steps,
        "loss": status.loss,
        "reward": status.reward_mean,
        "throughput": status.throughput,
        "eta_seconds": eta,
        "stage": status.stage,
        "status": status.status,
    }
    return json.dumps(data, ensure_ascii=False)


def render_line(status: RunStatus, eta: Optional[int]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [f"[{timestamp}]"]
    if status.step:
        parts.append(f"Step {status.step}/{status.total_steps}")
    if status.loss:
        parts.append(f"Loss {status.loss:.4f}")
    if status.reward_mean:
        parts.append(f"Reward {status.reward_mean:.4f}")
    if status.throughput:
        parts.append(f"tok/s {status.throughput}")
    if eta is not None:
        parts.append(f"ETA {format_eta(eta)}")
    parts.append(f"Stage={status.stage}")
    parts.append(f"Status={status.status}")
    return " | ".join(parts)


# =============================================================================
# Demo Entry Point
# =============================================================================

# Mock status data for demo - simulates a training run at step 150/1000
mock_status = RunStatus(
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


class GracefulExit:
    """Handles SIGINT/SIGTERM for graceful shutdown.
    
    Unlike the actual watch command which reads from files,
    this demo version just mocks the signal handling behavior.
    """
    def __init__(self):
        self.exit_requested = False

    def setup(self):
        """Register signal handlers for Ctrl+C and termination."""
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        print("\n\n👋 Watch stopped (training continues)")
        self.exit_requested = True


def demo_mode(mode="tty"):
    """演示不同模式的输出"""
    start_time = time.time() - 10  # 假设已经运行了 10 秒
    exit_handler = GracefulExit()
    exit_handler.setup()

    print(f"\n🚀 Starting {mode.upper()} mode demo...")
    print("Press Ctrl+C to stop\n")

    step = 150
    while not exit_handler.exit_requested and step < 1000:
        current_time = time.time()
        status = RunStatus(
            pid=12345,
            stage="train" if step % 2 == 0 else "rollout",
            status="running",
            updated_at=current_time,
            step=step,
            total_steps=1000,
            loss=0.25 - (step * 0.0001),
            reward_mean=0.85 + (step * 0.0001),
            throughput=1000 + step,
        )

        eta = calculate_eta(status.step, status.total_steps, start_time, current_time)

        # 清屏并显示
        print("\033[2J\033[H", end="")

        if mode == "tty":
            print(render_tty(status, eta, current_time - start_time))
        elif mode == "json":
            print(render_json(status, eta))
        else:
            print(render_line(status, eta))

        print("\n" + "─" * 40)
        print(f"Mode: {mode.upper()}")
        print(f"Step: {step}/1000 | Press Ctrl+C to exit")

        step += 50
        time.sleep(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\n" + "=" * 50)
        print("🎯 AReno Watch Demo")
        print("=" * 50)
        print("\nAvailable modes:")
        print("  1. tty   - Colorful TTY display (default)")
        print("  2. json  - JSON Lines output")
        print("  3. line  - Single line output")
        print("\nUsage: python test_watch_demo.py [mode]")
        print("\nExample:")
        print("  python test_watch_demo.py tty")
        print("  python test_watch_demo.py json")
        print("  python test_watch_demo.py line")
        mode = "tty"

    demo_mode(mode)