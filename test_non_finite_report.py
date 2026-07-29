"""Tests for non-finite value detection and reporting (Issue #238)."""

import os
import sys
import math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from areno.engine.runtime.non_finite import (
    check_loss_non_finite,
    detect_non_finite,
)

def test_normal_no_report():
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    report = detect_non_finite(model, opt, loss, grad_norm=1.0, step=10, lr=1e-3)
    assert report is None, "normal training should not produce a report"
    print("✅ test_normal_no_report passed")

def test_loss_nan():
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.tensor(float("nan"), requires_grad=True)
    assert check_loss_non_finite(loss) is True
    report = detect_non_finite(model, opt, loss, grad_norm=0.0, step=1, lr=1e-3)
    assert report is not None
    assert math.isnan(report.loss_value)
    print("✅ test_loss_nan passed")

def test_loss_inf():
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.tensor(float("inf"), requires_grad=True)
    assert check_loss_non_finite(loss) is True
    report = detect_non_finite(model, opt, loss, grad_norm=0.0, step=1, lr=1e-3)
    assert report is not None
    print(report.format_terminal())
    print("✅ test_loss_inf passed")

def test_param_inf():
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    with torch.no_grad():
        model.weight[0, 0] = float("inf")
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    report = detect_non_finite(model, opt, loss, grad_norm=1.0, step=50, lr=1e-4)
    assert report is not None
    assert any(e.inf_count > 0 for e in report.events)
    print(report.format_terminal())
    print("✅ test_param_inf passed")

def test_grad_explosion():
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    # Inject gradient explosion
    for p in model.parameters():
        if p.grad is not None:
            p.grad.fill_(1e8)
    report = detect_non_finite(
        model, opt, loss, grad_norm=1e8, step=200, lr=1e-3,
        grad_norm_threshold=1e6,
    )
    assert report is not None
    print(report.format_terminal())
    print("✅ test_grad_explosion passed")

if __name__ == "__main__":
    test_normal_no_report()
    test_loss_nan()
    test_loss_inf()
    test_param_inf()
    test_grad_explosion()
    print("\n🎉 全部测试通过!")
