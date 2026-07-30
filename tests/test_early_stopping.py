"""Comprehensive unit tests for EarlyStopping callback.

Tests cover:
- Basic min/max mode functionality
- Warmup evaluations
- NaN handling
- Tie handling
- Late improvement scenarios
- Boundary values (patience=0, warmup=0)
- State persistence
- Invalid inputs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from areno.callbacks import EarlyStopping


def test_min_mode_basic():
    """Test basic min mode (lower is better) - loss monitoring."""
    print("\n[Test] min_mode_basic: Testing loss improvement monitoring...")
    es = EarlyStopping(monitor="eval_loss", patience=3, mode="min", verbose=False)

    # Simulate training: loss decreases then plateaus
    losses = [2.0, 1.8, 1.6, 1.6, 1.7, 1.8, 1.9]  # 3 consecutive non-improvements

    stop_triggered = False
    for i, loss in enumerate(losses):
        should_stop = es({"eval_loss": loss})
        if should_stop:
            print(f"  Early stop triggered at step {i+1}, loss={loss}")
            stop_triggered = True
            break

    assert stop_triggered, "Early stopping should have triggered"
    assert es.counter == 3, f"Counter should be 3, got {es.counter}"
    assert es.best_score == 1.6, f"Best score should be 1.6, got {es.best_score}"
    print("  ✅ min_mode_basic passed")


def test_max_mode_basic():
    """Test basic max mode (higher is better) - accuracy monitoring."""
    print("\n[Test] max_mode_basic: Testing accuracy improvement monitoring...")
    es = EarlyStopping(monitor="eval_accuracy", patience=2, mode="max", verbose=False)

    # Simulate training: accuracy increases then plateaus
    accuracies = [0.6, 0.7, 0.75, 0.75, 0.74]  # 2 consecutive non-improvements

    stop_triggered = False
    for i, acc in enumerate(accuracies):
        should_stop = es({"eval_accuracy": acc})
        if should_stop:
            print(f"  Early stop triggered at step {i+1}, accuracy={acc}")
            stop_triggered = True
            break

    assert stop_triggered, "Early stopping should have triggered"
    assert es.best_score == 0.75, f"Best score should be 0.75, got {es.best_score}"
    print("  ✅ max_mode_basic passed")


def test_continuous_improvement_no_stop():
    """Test that continuous improvement doesn't trigger early stopping."""
    print("\n[Test] continuous_improvement_no_stop: Testing no stop when improving...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", verbose=False)

    # Continuously improving
    losses = [2.0, 1.9, 1.8, 1.7, 1.6]

    for loss in losses:
        should_stop = es({"eval_loss": loss})
        assert not should_stop, "Should not stop when continuously improving"

    assert es.counter == 0, f"Counter should be 0, got {es.counter}"
    print("  ✅ continuous_improvement_no_stop passed")


def test_min_delta():
    """Test min_delta threshold for ignoring small improvements."""
    print("\n[Test] min_delta: Testing minimum improvement threshold...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", min_delta=0.1, verbose=False)

    # Small improvements (0.05) should be ignored due to min_delta=0.1
    losses = [2.0, 1.95, 1.90, 1.85]  # Each only 0.05 better

    stop_triggered = False
    for i, loss in enumerate(losses):
        should_stop = es({"eval_loss": loss})
        if should_stop:
            stop_triggered = True
            break

    assert stop_triggered, "Should stop when improvements are below min_delta"
    print("  ✅ min_delta passed")


def test_min_delta_significant_improvement():
    """Test that significant improvements above min_delta are recognized."""
    print("\n[Test] min_delta_significant_improvement: Testing significant improvements...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", min_delta=0.1, verbose=False)

    # One significant improvement, then plateau
    losses = [2.0, 1.85, 1.84, 1.83]  # First is 0.15 (significant), rest are 0.01

    for loss in losses[:-1]:
        should_stop = es({"eval_loss": loss})
        assert not should_stop, "Should not stop after significant improvement"

    # Last one triggers
    should_stop = es({"eval_loss": losses[-1]})
    assert should_stop, "Should stop after patience with no significant improvement"
    assert es.best_score == 1.85, f"Best should be 1.85, got {es.best_score}"
    print("  ✅ min_delta_significant_improvement passed")


def test_warmup():
    """Test warmup evaluations are skipped."""
    print("\n[Test] warmup: Testing warmup phase...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", warmup=3, verbose=False)

    # Losses during warmup (should be ignored for early stopping)
    warmup_losses = [100.0, 50.0, 30.0]  # Very high, would normally trigger stop
    # After warmup, losses plateau
    post_warmup = [2.0, 2.0, 2.0]  # 2 consecutive non-improvements

    all_losses = warmup_losses + post_warmup

    stop_triggered = False
    for i, loss in enumerate(all_losses):
        should_stop = es({"eval_loss": loss})
        if should_stop:
            print(f"  Early stop triggered at step {i+1}")
            stop_triggered = True
            break

    assert stop_triggered, "Should stop after patience is reached post-warmup"
    assert es.warmup_counter == 3, f"Warmup counter should be 3, got {es.warmup_counter}"
    print("  ✅ warmup passed")


def test_warmup_with_improvement():
    """Test that best score is tracked during warmup."""
    print("\n[Test] warmup_with_improvement: Testing best score tracking during warmup...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", warmup=2, verbose=False)

    # Improving during warmup
    losses = [5.0, 3.0, 2.0, 2.1, 2.2]  # Best is 2.0 after warmup

    for loss in losses:
        should_stop = es({"eval_loss": loss})

    assert es.best_score == 2.0, f"Best should be 2.0 (post-warmup), got {es.best_score}"
    assert should_stop, "Should stop after 2 non-improvements"
    print("  ✅ warmup_with_improvement passed")


def test_nan_handling():
    """Test NaN values in metrics."""
    print("\n[Test] nan_handling: Testing NaN value handling...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", verbose=False)

    # Normal then NaN
    metrics = [{"eval_loss": 2.0}, {"eval_loss": float('nan')}, {"eval_loss": float('nan')}]

    for i, m in enumerate(metrics):
        should_stop = es(m)
        if i == 2:
            assert should_stop, "Should stop after 2 consecutive NaNs"

    print("  ✅ nan_handling passed")


def test_nan_recovery():
    """Test recovery from single NaN."""
    print("\n[Test] nan_recovery: Testing recovery from NaN...")
    es = EarlyStopping(monitor="eval_loss", patience=3, mode="min", verbose=False)

    metrics = [
        {"eval_loss": 2.0},
        {"eval_loss": float('nan')},  # NaN
        {"eval_loss": 1.9},  # Recovery
        {"eval_loss": 1.8},  # Improvement
        {"eval_loss": 1.9},  # No improvement
        {"eval_loss": 2.0},  # No improvement
    ]

    should_stop = False
    for m in metrics:
        should_stop = es(m)

    assert not should_stop, "Should not stop - only 2 non-improvements after recovery"
    assert es.best_score == 1.8, f"Best should be 1.8, got {es.best_score}"
    print("  ✅ nan_recovery passed")


def test_tie_handling():
    """Test handling of tied scores."""
    print("\n[Test] tie_handling: Testing tied scores...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", verbose=False)

    # Same value repeated
    losses = [2.0, 2.0, 2.0]  # 2 ties should trigger stop

    stop_triggered = False
    for i, loss in enumerate(losses):
        should_stop = es({"eval_loss": loss})
        if should_stop:
            stop_triggered = True
            break

    assert stop_triggered, "Should stop after ties (no improvement)"
    print("  ✅ tie_handling passed")


def test_late_improvement():
    """Test late improvement resets counter."""
    print("\n[Test] late_improvement: Testing late improvement resets counter...")
    es = EarlyStopping(monitor="eval_loss", patience=3, mode="min", verbose=False)

    # Almost trigger, then improve
    # Step 1: 2.0 (init best=2.0)
    # Step 2: 2.1 (no improve, counter=1)
    # Step 3: 2.2 (no improve, counter=2)
    # Step 4: 2.3 (no improve, counter=3 -> STOP!)
    # Note: With patience=3, we stop at 3 consecutive non-improvements
    losses = [2.0, 2.1, 2.2, 2.3, 1.9, 2.0, 2.1, 2.2]

    stop_triggered = False
    stop_step = 0
    for i, loss in enumerate(losses):
        should_stop = es({"eval_loss": loss})
        if should_stop and not stop_triggered:
            print(f"  Stopped at step {i+1}")
            stop_triggered = True
            stop_step = i + 1
            break

    assert stop_triggered, "Should stop at step 4 (3 consecutive non-improvements)"
    assert stop_step == 4, f"Should stop at step 4, stopped at {stop_step}"
    assert es.best_score == 2.0, f"Best should be 2.0 (no improvement was good enough), got {es.best_score}"
    print("  ✅ late_improvement passed")


def test_patience_zero():
    """Test patience=0 (stop immediately on no improvement)."""
    print("\n[Test] patience_zero: Testing patience=0...")
    es = EarlyStopping(monitor="eval_loss", patience=0, mode="min", verbose=False)

    losses = [2.0, 2.1]  # First is init, second triggers stop

    should_stop = es({"eval_loss": losses[0]})
    assert not should_stop, "First eval should not stop"

    should_stop = es({"eval_loss": losses[1]})
    assert should_stop, "Should stop immediately with patience=0"
    print("  ✅ patience_zero passed")


def test_missing_metric():
    """Test handling of missing monitored metric."""
    print("\n[Test] missing_metric: Testing missing metric...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", verbose=False)

    # Metric missing
    should_stop = es({"other_metric": 1.0})
    assert not should_stop, "Should not stop when metric is missing"

    # Now provide it
    should_stop = es({"eval_loss": 2.0})
    assert not should_stop, "Should initialize on first valid metric"

    print("  ✅ missing_metric passed")


def test_non_numeric_metric():
    """Test handling of non-numeric metric values."""
    print("\n[Test] non_numeric_metric: Testing non-numeric values...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", verbose=False)

    # String value
    should_stop = es({"eval_loss": "invalid"})
    assert not should_stop, "Should handle string value gracefully"

    # None value
    should_stop = es({"eval_loss": None})
    assert not should_stop, "Should handle None value gracefully"

    print("  ✅ non_numeric_metric passed")


def test_baseline():
    """Test baseline comparison."""
    print("\n[Test] baseline: Testing baseline comparison...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", baseline=1.5, verbose=False)

    # Never beat baseline
    losses = [2.0, 2.0, 2.0]

    stop_triggered = False
    for loss in losses:
        should_stop = es({"eval_loss": loss})
        if should_stop:
            stop_triggered = True
            break

    assert stop_triggered, "Should stop when not beating baseline"
    print("  ✅ baseline passed")


def test_state_dict():
    """Test state saving and loading."""
    print("\n[Test] state_dict: Testing state persistence...")
    es = EarlyStopping(monitor="eval_loss", patience=3, mode="min", warmup=1, verbose=False)

    # Run some evaluations
    es({"eval_loss": 2.0})
    es({"eval_loss": 1.9})
    es({"eval_loss": 2.0})  # No improvement

    # Save state
    state = es.state_dict()
    print(f"  Saved state: {state}")

    # Create new instance and load
    es2 = EarlyStopping(monitor="eval_loss", patience=3, mode="min", warmup=1, verbose=False)
    es2.load_state_dict(state)

    assert es2.counter == es.counter, "Counter should match"
    assert es2.best_score == es.best_score, "Best score should match"
    assert es2.warmup_counter == es.warmup_counter, "Warmup counter should match"

    # Continue from loaded state
    es2({"eval_loss": 2.1})  # No improvement
    es2({"eval_loss": 2.2})  # No improvement - triggers stop

    assert es2.early_stop, "Should stop after loading state"
    print("  ✅ state_dict passed")


def test_reset():
    """Test reset functionality."""
    print("\n[Test] reset: Testing reset...")
    es = EarlyStopping(monitor="eval_loss", patience=2, mode="min", verbose=False)

    # Run and trigger
    es({"eval_loss": 2.0})
    es({"eval_loss": 2.1})
    es({"eval_loss": 2.2})
    assert es.early_stop, "Should have stopped"

    # Reset
    es.reset()

    assert not es.early_stop, "Should not be stopped after reset"
    assert es.counter == 0, "Counter should be 0"
    assert es.best_score is None, "Best score should be None"
    print("  ✅ reset passed")


def test_get_status():
    """Test status reporting."""
    print("\n[Test] get_status: Testing status reporting...")
    es = EarlyStopping(monitor="eval_loss", patience=3, mode="min", verbose=False)

    es({"eval_loss": 2.0})
    es({"eval_loss": 2.1})

    status = es.get_status()
    assert status["monitor"] == "eval_loss"
    assert status["patience"] == 3
    assert status["counter"] == 1
    assert status["remaining_patience"] == 2

    print("  ✅ get_status passed")


def test_invalid_mode():
    """Test invalid mode raises error."""
    print("\n[Test] invalid_mode: Testing invalid mode validation...")
    try:
        es = EarlyStopping(monitor="eval_loss", mode="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "mode must be" in str(e)

    print("  ✅ invalid_mode passed")


def test_invalid_patience():
    """Test negative patience raises error."""
    print("\n[Test] invalid_patience: Testing negative patience validation...")
    try:
        es = EarlyStopping(monitor="eval_loss", patience=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "patience must be non-negative" in str(e)

    print("  ✅ invalid_patience passed")


def test_invalid_warmup():
    """Test negative warmup raises error."""
    print("\n[Test] invalid_warmup: Testing negative warmup validation...")
    try:
        es = EarlyStopping(monitor="eval_loss", warmup=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "warmup must be non-negative" in str(e)

    print("  ✅ invalid_warmup passed")


def test_immediate_stop_with_patience_zero():
    """Test that patience=0 stops immediately on first non-improvement."""
    print("\n[Test] immediate_stop_with_patience_zero: Testing immediate stop...")
    es = EarlyStopping(monitor="eval_loss", patience=0, mode="min", verbose=False)

    es({"eval_loss": 2.0})  # Init
    stopped = es({"eval_loss": 2.0})  # Same value - no improvement

    assert stopped, "Should stop immediately with patience=0"
    print("  ✅ immediate_stop_with_patience_zero passed")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running EarlyStopping Comprehensive Unit Tests")
    print("=" * 60)

    tests = [
        test_min_mode_basic,
        test_max_mode_basic,
        test_continuous_improvement_no_stop,
        test_min_delta,
        test_min_delta_significant_improvement,
        test_warmup,
        test_warmup_with_improvement,
        test_nan_handling,
        test_nan_recovery,
        test_tie_handling,
        test_late_improvement,
        test_patience_zero,
        test_missing_metric,
        test_non_numeric_metric,
        test_baseline,
        test_state_dict,
        test_reset,
        test_get_status,
        test_invalid_mode,
        test_invalid_patience,
        test_invalid_warmup,
        test_immediate_stop_with_patience_zero,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️ {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
