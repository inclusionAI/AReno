"""CPU tests for host-resource preflight (fd / process / shm limits).

These tests inject deterministic limit values so the preflight logic is
exercised without touching the real host or any GPU/engine code. The probes
themselves use only the stdlib `resource` module and `/proc/sys/kernel/shmmax`,
so they import cleanly in any Python 3.10+ environment.
"""

from __future__ import annotations

import pytest
from click import UsageError
from click.testing import CliRunner

from areno.cli import diagnostics
from areno.cli import serve as serve_mod
from areno.cli import train as train_mod
from areno.cli.diagnostics import (
    RESOURCE_FAIL,
    RESOURCE_OK,
    RESOURCE_WARN,
    estimate_resource_demand,
    preflight_host_resources,
)


def _limits(
    *,
    fd_value: int | None = 4096,
    fd_unbounded: bool = False,
    nproc_value: int | None = 100,
    shm_value: int | None = 1 << 40,
    shm_available: bool = True,
) -> dict:
    """Build a deterministic limits dict for injection."""

    fd_available = fd_value is not None or fd_unbounded
    return {
        "file_descriptors": {
            "available": fd_available,
            "unbounded": fd_unbounded,
            "soft": fd_value,
            "hard": fd_value,
            "value": fd_value,
            "error": None if fd_available else "probe unavailable",
        },
        "processes": {
            "available": nproc_value is not None,
            "unbounded": False,
            "soft": nproc_value,
            "hard": nproc_value,
            "value": nproc_value,
            "error": None if nproc_value is not None else "probe unavailable",
        },
        "shared_memory": {
            "available": shm_available and shm_value is not None,
            "unbounded": False,
            "soft": shm_value,
            "hard": shm_value,
            "value": shm_value,
            "error": None if (shm_available and shm_value is not None) else "FileNotFoundError",
        },
    }


def _names(results):
    return [r.status for r in results]


# ---------------------------------------------------------------------------
# Demand estimate
# ---------------------------------------------------------------------------


def test_estimate_demand_formula_matches_documented_upper_bound():
    demand = estimate_resource_demand(world_size=8, tp_size=4)
    # RLIMIT_NOFILE is per process: 64 fds base + 7 cross-rank peers = 71.
    assert demand["file_descriptors"] == 64 + 7
    # world_size workers + 1 driver
    assert demand["processes"] == 9
    # 1 GiB * tp_size
    assert demand["shared_memory"] == (1 << 30) * 4


def test_estimate_demand_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        estimate_resource_demand(0, 1)
    with pytest.raises(ValueError):
        estimate_resource_demand(1, 0)


# ---------------------------------------------------------------------------
# Severities: success / boundary / failure / unbounded
# ---------------------------------------------------------------------------


def test_all_ok_when_limits_exceed_demand():
    results = preflight_host_resources(8, 4, policy="warn", limits=_limits())
    assert _names(results) == [RESOURCE_OK, RESOURCE_OK, RESOURCE_OK]
    fd = results[0]
    # Exact observed/required/delta values are emitted, not just a status.
    assert "observed=4096" in fd.detail
    demand = estimate_resource_demand(8, 4)
    assert f"required={demand['file_descriptors']}" in fd.detail
    assert f"delta={4096 - demand['file_descriptors']}" in fd.detail


def test_boundary_observed_equals_required_is_ok():
    demand = estimate_resource_demand(2, 1)
    limits = _limits(
        fd_value=demand["file_descriptors"], nproc_value=demand["processes"], shm_value=demand["shared_memory"]
    )
    results = preflight_host_resources(2, 1, policy="warn", limits=limits)
    assert _names(results) == [RESOURCE_OK, RESOURCE_OK, RESOURCE_OK]
    # fd/processes use unit="" -> "delta=0"; shm uses unit=" bytes" -> "delta bytes=0".
    assert "delta=0" in results[0].detail
    assert "delta=0" in results[1].detail
    assert "delta bytes=0" in results[2].detail


def test_low_file_descriptors_is_fail_with_delta_and_remediation():
    demand = estimate_resource_demand(8, 4)
    limits = _limits(fd_value=32)  # well below the per-process demand of 71
    results = preflight_host_resources(8, 4, policy="warn", limits=limits)
    fd = results[0]
    assert fd.status == RESOURCE_FAIL
    assert f"observed=32 required={demand['file_descriptors']}" in fd.detail
    assert f"delta={32 - demand['file_descriptors']}" in fd.detail
    assert "ulimit -n" in fd.next_step


def test_low_shared_memory_fail_points_at_sysctl():
    limits = _limits(shm_value=1024)
    results = preflight_host_resources(8, 4, policy="warn", limits=limits)
    shm = results[2]
    assert shm.status == RESOURCE_FAIL
    assert "bytes" in shm.detail
    assert "kernel.shmmax" in shm.next_step
    # The remediation hint carries the concrete required value, not a placeholder.
    demand = estimate_resource_demand(8, 4)
    assert f"kernel.shmmax={demand['shared_memory']}" in shm.next_step


def test_unbounded_rlimit_is_ok_not_unavailable():
    limits = _limits(fd_value=None, fd_unbounded=True)
    results = preflight_host_resources(8, 4, policy="warn", limits=limits)
    fd = results[0]
    assert fd.status == RESOURCE_OK
    assert "observed=unbounded" in fd.detail


# ---------------------------------------------------------------------------
# Graceful degradation when a probe cannot run
# ---------------------------------------------------------------------------


def test_unavailable_probe_degrades_to_warn_not_fail():
    # shmmax is Linux-only; on macOS/Windows the probe is unavailable.
    limits = _limits(shm_value=None, shm_available=False)
    results = preflight_host_resources(8, 4, policy="block", limits=limits)
    shm = results[2]
    assert shm.status == RESOURCE_WARN
    assert "probe unavailable" in shm.detail
    # An unavailable probe must never be a blocking failure.
    assert shm.status != RESOURCE_FAIL


def test_should_block_on_resources_only_counts_fail():
    limits = _limits(shm_value=None, shm_available=False)
    results = preflight_host_resources(8, 4, policy="block", limits=limits)
    # WARN from the unavailable probe does not block.
    assert diagnostics.should_block_on_resources(results) is False
    fail_limits = _limits(fd_value=1, nproc_value=1, shm_value=1)
    fail_results = preflight_host_resources(8, 4, policy="block", limits=fail_limits)
    assert diagnostics.should_block_on_resources(fail_results) is True


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_invalid_policy_raises():
    with pytest.raises(ValueError):
        preflight_host_resources(2, 1, policy="enforce")


def test_skip_short_circuits_without_probing(monkeypatch):
    # `skip` must not touch the host at all and returns no results.
    monkeypatch.setattr(diagnostics, "collect_host_limits", lambda: pytest.fail("host probed under skip"))
    assert preflight_host_resources(2, 1, policy="skip") == []


def test_format_resource_preflight_renders_status_and_next_steps():
    limits = _limits(fd_value=1, nproc_value=1000, shm_value=1 << 40)
    results = preflight_host_resources(2, 1, policy="warn", limits=limits)
    text = diagnostics.format_resource_preflight(results)
    assert "Host resource preflight:" in text
    assert "FAIL file descriptors" in text
    assert "OK   processes" in text
    assert "ulimit -n" in text  # next-step only rendered for WARN/FAIL


# ===========================================================================
# CLI integration -- train and serve honor --resource-check
#
# These tests need torch/FastAPI on the path because `areno.cli.train` and
# `areno.cli.serve` import them at module load. They mock the preflight probe
# (so no real host limits are read) and the heavyweight downstream steps
# (`run`, `create_app`, uvicorn) so only the preflight wiring is exercised.
# ===========================================================================


def _train_options(**overrides):
    """Valid gspo options that pass `_trainer_config_from_options` validation.

    Reuses the canonical defaults from the train-cli config test so the full
    validation surface (save_interval, lr, clip eps, ...) is satisfied.
    """

    from tests.test_train_cli_config_cpu import _options

    overrides.setdefault("resource_check", "warn")
    return _options(world_size=2, tp_size=1, **overrides)


@pytest.fixture
def patched_preflight(monkeypatch):
    """Patch the CLI preflight wrapper in train/serve with a recorder.

    The fake replaces `_preflight_host_resources` wholesale (it owns echo/raise
    behavior), so no real host limits are read. Under `block` it raises
    UsageError -- matching the real wrapper -- so block wiring is exercised.
    """

    calls: list[tuple] = []

    def _fake(world_size, tp_size, *, policy):
        calls.append((world_size, tp_size, policy))
        if policy == "skip":
            return
        if policy == "block":
            import click

            raise click.UsageError(
                "host resource limits are below the estimated demand for this run "
                f"(world_size={world_size}, tp_size={tp_size}); failing probes: ['file descriptors']"
            )

    # The CLI calls the module-level `_preflight_host_resources` wrapper.
    monkeypatch.setattr(train_mod, "_preflight_host_resources", _fake)
    monkeypatch.setattr(serve_mod, "_preflight_host_resources", _fake)
    return calls


def test_train_default_warn_does_not_abort_on_fail(monkeypatch, patched_preflight):
    # Real `_trainer_config_from_options` runs the preflight; we only stub `run`
    # so the test stops right after config resolution.
    monkeypatch.setattr(train_mod, "run", lambda trainer_config: None)
    cfg = train_mod._trainer_config_from_options(**_train_options())
    assert patched_preflight == [(2, 1, "warn")]
    assert cfg is not None  # preflight did not abort config construction


def test_train_block_aborts_before_run_on_fail(monkeypatch, patched_preflight):
    reached = {"run": False}
    monkeypatch.setattr(train_mod, "run", lambda trainer_config: reached.__setitem__("run", True))
    with pytest.raises(UsageError, match="host resource limits are below"):
        train_mod._trainer_config_from_options(**_train_options(resource_check="block"))
    assert patched_preflight == [(2, 1, "block")]
    assert reached["run"] is False  # aborted before run()


def test_train_skip_does_not_invoke_preflight(monkeypatch):
    # Under skip, the wrapper must not call the underlying probe at all.
    probe_calls = []
    monkeypatch.setattr(train_mod, "preflight_host_resources", lambda *a, **k: probe_calls.append((a, k)) or [])
    monkeypatch.setattr(train_mod, "run", lambda trainer_config: None)
    train_mod._trainer_config_from_options(**_train_options(resource_check="skip"))
    assert probe_calls == []


def test_serve_block_aborts_before_engine_init(monkeypatch, patched_preflight):
    reached = {"resolve": False, "create_app": False}
    monkeypatch.setattr(serve_mod, "resolve_model_ref", lambda *a, **k: reached.__setitem__("resolve", True) or "model")
    monkeypatch.setattr(serve_mod, "create_app", lambda **k: reached.__setitem__("create_app", True) or "app")
    import types

    monkeypatch.setattr(serve_mod, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None), raising=False)
    import areno.cli.dashboard_registry as dashboard_registry

    monkeypatch.setattr(dashboard_registry, "register_dashboard_job", lambda **k: None)

    runner = CliRunner()
    result = runner.invoke(
        serve_mod.serve_command,
        ["--model-path", "x", "--world-size", "2", "--tp-size", "1", "--resource-check", "block"],
    )
    assert patched_preflight == [(2, 1, "block")]
    assert reached["resolve"] is False  # preflight aborted before model resolution
    assert reached["create_app"] is False
    assert result.exit_code != 0
    assert "host resource limits are below" in result.output


def test_serve_warn_proceeds_to_engine_init(monkeypatch, patched_preflight):
    reached = {"resolve": False}
    monkeypatch.setattr(serve_mod, "resolve_model_ref", lambda *a, **k: reached.__setitem__("resolve", True) or "model")
    monkeypatch.setattr(serve_mod, "create_app", lambda **k: "app")
    import types

    monkeypatch.setattr(serve_mod, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None), raising=False)
    import areno.cli.dashboard_registry as dashboard_registry

    monkeypatch.setattr(dashboard_registry, "register_dashboard_job", lambda **k: None)

    runner = CliRunner()
    runner.invoke(
        serve_mod.serve_command,
        ["--model-path", "x", "--world-size", "2", "--tp-size", "1"],
    )
    assert patched_preflight == [(2, 1, "warn")]
    assert reached["resolve"] is True  # warn policy did not block engine init
