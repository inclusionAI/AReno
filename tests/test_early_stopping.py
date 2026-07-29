"""EarlyStopping 单元测试"""
import sys
sys.path.insert(0, '/kaggle/working/AReno')

from areno.callbacks.early_stopping import EarlyStopping


def test_min_mode_basic():
    """测试 min 模式基本功能"""
    es = EarlyStopping(monitor="loss", patience=3, mode="min")
    assert es({"loss": 1.0}) == False
    assert es.best_score == 1.0
    assert es({"loss": 0.8}) == False
    assert es.best_score == 0.8
    assert es.counter == 0
    assert es({"loss": 0.85}) == False
    assert es.counter == 1
    assert es({"loss": 0.9}) == False
    assert es.counter == 2
    assert es({"loss": 0.95}) == True
    assert es.early_stop == True
    print("✅ test_min_mode_basic 通过")


def test_max_mode_basic():
    """测试 max 模式基本功能"""
    es = EarlyStopping(monitor="acc", patience=3, mode="max")
    assert es({"acc": 0.5}) == False
    assert es({"acc": 0.6}) == False
    assert es({"acc": 0.58}) == False
    assert es({"acc": 0.55}) == False
    assert es({"acc": 0.52}) == True
    print("✅ test_max_mode_basic 通过")


def test_continuous_improvement():
    """测试持续改善不触发早停"""
    es = EarlyStopping(monitor="loss", patience=2, mode="min")
    for loss in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        result = es({"loss": loss})
        assert result == False, f"持续改善不应触发早停, loss={loss}"
    assert es.counter == 0
    print("✅ test_continuous_improvement 通过")


def test_min_delta():
    """测试 min_delta 阈值"""
    es = EarlyStopping(monitor="loss", patience=2, mode="min", min_delta=0.1)
    es({"loss": 1.0})
    assert es({"loss": 0.95}) == False
    assert es.counter == 1
    assert es({"loss": 0.8}) == False
    assert es.counter == 0
    print("✅ test_min_delta 通过")


def test_missing_metric():
    """测试监控指标不存在的情况"""
    es = EarlyStopping(monitor="val_loss", patience=2)
    assert es({"loss": 1.0}) == False
    assert es.best_score is None
    print("✅ test_missing_metric 通过")


def test_state_dict():
    """测试状态保存和恢复"""
    es = EarlyStopping(monitor="loss", patience=3, mode="min")
    es({"loss": 1.0})
    es({"loss": 1.1})
    state = es.state_dict()
    assert state["counter"] == 1
    assert state["best_score"] == 1.0
    assert state["early_stop"] == False
    es2 = EarlyStopping(monitor="loss", patience=3, mode="min")
    es2.load_state_dict(state)
    assert es2.counter == 1
    assert es2.best_score == 1.0
    print("✅ test_state_dict 通过")


def run_all_tests():
    print("\n" + "="*50)
    print("运行 EarlyStopping 单元测试")
    print("="*50 + "\n")
    test_min_mode_basic()
    test_max_mode_basic()
    test_continuous_improvement()
    test_min_delta()
    test_missing_metric()
    test_state_dict()
    print("\n" + "="*50)
    print("🎉 所有测试通过！")
    print("="*50)


if __name__ == "__main__":
    run_all_tests()
