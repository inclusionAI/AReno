from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areno.engine.runtime.device import configure_unified_memory_budget


def test_spark_allocator_budget_respects_cli_and_excludes_swap(monkeypatch):
    monkeypatch.setenv("ARENO_MST_MEM_FRAC", "0.8")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(name="NVIDIA GB10", total_memory=128 * 1024**3),
    )
    setter = Mock()
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", setter)
    device = torch.device("cuda", 0)
    assert configure_unified_memory_budget(device) == 0.8
    setter.assert_called_once_with(0.8, device)


def test_other_devices_keep_their_allocator_policy(monkeypatch):
    properties = Mock(return_value=SimpleNamespace(name="NVIDIA H100"))
    setter = Mock()
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", setter)
    assert configure_unified_memory_budget(torch.device("cpu")) is None
    properties.assert_not_called()
    assert configure_unified_memory_budget(torch.device("cuda")) is None
    setter.assert_not_called()


@pytest.mark.parametrize("value", ["0", "1.1", "nan"])
def test_invalid_budget_is_rejected_before_allocator_mutation(monkeypatch, value):
    monkeypatch.setenv("ARENO_MST_MEM_FRAC", value)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(name="NVIDIA GB10"))
    setter = Mock()
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", setter)
    with pytest.raises(ValueError):
        configure_unified_memory_budget(torch.device("cuda"))
    setter.assert_not_called()
