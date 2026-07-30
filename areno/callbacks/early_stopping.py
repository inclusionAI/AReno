"""Early Stopping callback for AReno training.

This module provides early stopping functionality to automatically stop training
when validation metrics stop improving for a specified number of consecutive rounds.

Example:
    >>> from areno.callbacks import EarlyStopping
    >>> es = EarlyStopping(monitor="eval_loss", patience=3, mode="min")
    >>> for epoch in range(100):
    ...     metrics = trainer.evaluate()
    ...     if es(metrics):
    ...         print(f"Early stopping at epoch {epoch}")
    ...         break
"""

import math
from typing import Any, Dict, Optional, Callable


class EarlyStopping:
    """Early stopping handler for training optimization.

    Monitors a specified metric and stops training when it stops improving
    for a configured number of consecutive evaluations (patience).

    Args:
        monitor: Metric name to monitor (default: "eval_loss").
        patience: Number of evaluations with no improvement to wait before stopping (default: 3).
        mode: One of {"min", "max"}. In "min" mode, lower metric values are better.
            In "max" mode, higher values are better (default: "min").
        min_delta: Minimum change in the monitored metric to qualify as an improvement.
            Ignores changes smaller than min_delta (default: 0.0).
        verbose: Whether to print messages when improvement is detected or stopping is triggered (default: True).
        warmup: Number of initial evaluations to skip before starting early stopping checks.
            Useful when initial unstable metrics should be ignored (default: 0).
        baseline: Baseline value for the monitored metric.
            Training will stop if the model doesn't beat the baseline by at least min_delta (default: None).

    Attributes:
        counter: Current count of consecutive non-improving evaluations.
        best_score: Best metric value seen so far.
        early_stop: Whether early stopping has been triggered.
        warmup_counter: Current warmup evaluation count.
    """

    def __init__(
        self,
        monitor: str = "eval_loss",
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True,
        warmup: int = 0,
        baseline: Optional[float] = None,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.warmup = warmup
        self.baseline = baseline

        # State variables
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.warmup_counter = 0
        self.total_evals = 0

        # Validate mode
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        # Validate patience
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")

        # Validate warmup
        if warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {warmup}")

        # Set up comparison function based on mode
        if mode == "min":
            self.is_better: Callable[[float, float], bool] = (
                lambda current, best: current < best - min_delta
            )
            self.is_better_than_baseline: Callable[[float], bool] = (
                lambda current: baseline is None or current < baseline - min_delta
            )
        else:  # mode == "max"
            self.is_better = lambda current, best: current > best + min_delta
            self.is_better_than_baseline = lambda current: baseline is None or current > baseline + min_delta

    def __call__(self, metrics: Dict[str, Any]) -> bool:
        """Check if early stopping should be triggered.

        Args:
            metrics: Dictionary containing metric values. Must include the monitored metric.

        Returns:
            True if early stopping should be triggered, False otherwise.
        """
        if self.early_stop:
            return True

        # Check if monitored metric exists
        if self.monitor not in metrics:
            if self.verbose:
                print(f"[EarlyStopping] Warning: '{self.monitor}' not found in metrics. "
                      f"Available: {list(metrics.keys())}")
            return False

        current_score = metrics[self.monitor]
        self.total_evals += 1

        # Handle NaN values
        if isinstance(current_score, float) and math.isnan(current_score):
            if self.verbose:
                print(f"[EarlyStopping] Warning: {self.monitor} is NaN, treating as no improvement")
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] Stopping due to NaN values for {self.patience} consecutive evaluations")
            return self.early_stop

        # Handle non-numeric values
        if not isinstance(current_score, (int, float)):
            if self.verbose:
                print(f"[EarlyStopping] Warning: {self.monitor} has non-numeric value {current_score}")
            return False

        # Warmup phase: skip early stopping checks
        if self.warmup_counter < self.warmup:
            self.warmup_counter += 1
            if self.verbose:
                print(f"[EarlyStopping] Warmup {self.warmup_counter}/{self.warmup}: {self.monitor}={current_score:.4f}")
            # Still track best score during warmup
            if self.best_score is None or self.is_better(current_score, self.best_score):
                self.best_score = current_score
            return False

        # First valid evaluation after warmup
        if self.best_score is None:
            self.best_score = current_score
            if self.verbose:
                print(f"[EarlyStopping] {self.monitor} initialized to {current_score:.4f}")
            return False

        # Check for improvement
        if self.is_better(current_score, self.best_score):
            # Improvement detected
            if self.best_score != current_score:  # Not a tie
                improvement = abs(current_score - self.best_score)
                if self.verbose:
                    print(f"[EarlyStopping] {self.monitor} improved from {self.best_score:.4f} "
                          f"to {current_score:.4f} (Δ{improvement:.4f})")
            else:
                if self.verbose:
                    print(f"[EarlyStopping] {self.monitor} tied at {current_score:.4f}")
            self.best_score = current_score
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] {self.monitor} did not improve. "
                      f"Current: {current_score:.4f}, Best: {self.best_score:.4f}. "
                      f"Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] Stopping! Best {self.monitor}: {self.best_score:.4f}")

        # Check baseline condition
        if self.baseline is not None and not self.is_better_than_baseline(current_score):
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"[EarlyStopping] Did not beat baseline {self.baseline:.4f}")

        return self.early_stop

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary for checkpoint saving.

        Returns:
            Dictionary containing the current state of early stopping.
        """
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "warmup_counter": self.warmup_counter,
            "total_evals": self.total_evals,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: Dictionary containing saved early stopping state.
        """
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.early_stop = state_dict["early_stop"]
        self.warmup_counter = state_dict.get("warmup_counter", 0)
        self.total_evals = state_dict.get("total_evals", 0)

    def reset(self) -> None:
        """Reset early stopping state to initial values."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.warmup_counter = 0
        self.total_evals = 0

    def get_status(self) -> Dict[str, Any]:
        """Get current status information.

        Returns:
            Dictionary with current status for logging/monitoring.
        """
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "warmup": self.warmup,
            "warmup_counter": self.warmup_counter,
            "remaining_patience": max(0, self.patience - self.counter),
        }
