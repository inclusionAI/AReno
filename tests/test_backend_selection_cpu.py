from __future__ import annotations

from unittest.mock import patch

import pytest

from areno.api.config import default_backend_type, resolve_backend_type
from areno.api.models import BackendType


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", BackendType.CUDA),
        ("Linux", "aarch64", BackendType.CUDA),
        ("Darwin", "arm64", BackendType.MLX),
    ],
)
def test_default_backend_follows_native_platform(system, machine, expected):
    with patch("areno.api.config.platform.system", return_value=system):
        with patch("areno.api.config.platform.machine", return_value=machine):
            assert default_backend_type() is expected
            assert resolve_backend_type(None, None) is expected


def test_default_backend_rejects_unsupported_platform():
    with patch("areno.api.config.platform.system", return_value="Darwin"):
        with patch("areno.api.config.platform.machine", return_value="x86_64"):
            with pytest.raises(RuntimeError, match="MLX requires Apple Silicon"):
                default_backend_type()


def test_explicit_backend_overrides_platform_detection():
    with patch("areno.api.config.platform.system", return_value="Darwin"):
        with patch("areno.api.config.platform.machine", return_value="arm64"):
            assert resolve_backend_type(BackendType.CUDA, None) is BackendType.CUDA
