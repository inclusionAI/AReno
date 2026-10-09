"""Register NPU metadata for CPU tests without claiming hardware availability."""

import importlib.util
from types import ModuleType

import torch


def register_npu_device():
    if importlib.util.find_spec("torch_npu") is not None:
        import torch_npu  # noqa: F401

        return
    if torch._C._get_privateuse1_backend_name() == "privateuseone":
        torch.utils.rename_privateuse1_backend("npu")
    if not hasattr(torch, "npu"):
        # Renaming PrivateUse1 lasts for the entire process. Pair it with an
        # unavailable module so later CPU optimizer tests can query accelerators.
        module = ModuleType("torch.npu")
        module.is_available = lambda: False
        module.is_initialized = lambda: False
        torch._register_device_module("npu", module)
