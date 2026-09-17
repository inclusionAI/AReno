"""Unit-safe estimates, totals and public pricing drift handling."""

from decimal import Decimal

import pytest

from areno.dashboard.flow.pricing import GPU_LABELS, Pricing, estimate, parse_rates


def fixture_html():
    # Deliberately artificial test rates; production reads all values from Modal.
    gpu = " ".join(f"{label} $0.001 / sec" for label in GPU_LABELS.values())
    return f"<main>{gpu} CPU Physical core $9 / core / sec Memory $9 / GiB / sec Modal Sandbox + Notebooks Pricing CPU Physical core (2 vCPU) $0.00001 / core / sec Memory $0.000002 / GiB / sec</main>"


def reservation():
    return {"gpu": "H100", "count": 2, "cpu": 4, "memory_gib": 32, "timeout_seconds": 14400}


def test_estimate_uses_sandbox_units_and_run_count():
    rates = parse_rates(fixture_html())
    result = estimate(reservation(), 2, rates, 3)
    # 2 GPUs + 4 physical cores + 32 GiB, each billed by the second.
    hourly = Decimal("0.002104") * 3600
    assert Decimal(result["hourly_cost"]) == hourly
    assert Decimal(result["planned_cost"]) == hourly * 2
    assert Decimal(result["total_cost"]) == hourly * 6
    assert sum(Decimal(v) for v in result["breakdown"].values()) == hourly * 2
    assert Decimal(result["lifetime_cost"]) == hourly * 4


@pytest.mark.parametrize("hours,count", [(float("nan"), 1), (0, 1), (5, 1), (1, 0), (1, 1.5)])
def test_invalid_assumptions_rejected(hours, count):
    with pytest.raises(ValueError):
        estimate(reservation(), hours, parse_rates(fixture_html()), count)


def test_changed_pricing_page_fails_instead_of_using_stale_constants():
    with pytest.raises(ValueError):
        parse_rates("<main>New pricing format</main>")
    with pytest.raises(ValueError):
        parse_rates(fixture_html().replace("Nvidia H100 SXM5", "Another accelerator"))


def test_aggregate_quotes_and_elapsed_time_do_not_include_deployments(monkeypatch):
    pricing = Pricing()
    quote = estimate(reservation(), 2, parse_rates(fixture_html()))
    jobs = [
        {
            "id": "one",
            "name": "SFT",
            "kind": "training",
            "status": "succeeded",
            "estimate": quote,
            "started_at": 100,
            "finished_at": 3700,
        },
        {"id": "two", "name": "GRPO", "kind": "training", "status": "queued", "estimate": quote},
        {"id": "endpoint", "name": "Serving", "kind": "deployment", "status": "ready", "estimate": quote},
    ]
    result = pricing.all_runs(jobs)
    assert result["run_count"] == 2
    assert Decimal(result["planned_total"]) == Decimal(quote["planned_cost"]) * 2
    assert Decimal(result["accrued_total"]) == Decimal(quote["hourly_cost"])


def test_pricing_failure_is_not_zero_for_unknown_run(monkeypatch):
    pricing = Pricing()
    monkeypatch.setattr(pricing, "rates", lambda: (_ for _ in ()).throw(RuntimeError("pricing offline")))
    result = pricing.all_runs([{"id": "one", "name": "Run", "kind": "training", "resources": reservation()}])
    assert result["priced_count"] == 0 and result["errors"][0]["error"] == "pricing offline"


def test_seconds_lifetime_and_legacy_hours_are_equivalent():
    rates = parse_rates(fixture_html())
    result = estimate({**reservation(), "timeout_seconds": 90}, 90 / 3600, rates)
    assert Decimal(result["lifetime_cost"]) == Decimal("0.002104") * 90
    legacy = {**reservation(), "timeout_hours": 4}
    del legacy["timeout_seconds"]
    assert estimate(legacy, 2, rates) == estimate(reservation(), 2, rates)


@pytest.mark.parametrize("seconds", [0, 86401, 1.5, True, float("nan")])
def test_invalid_timeout_seconds_rejected(seconds):
    from areno.dashboard.flow.workflows import resources

    with pytest.raises(ValueError):
        resources({"timeout_seconds": seconds})


def test_seconds_timeout_boundaries_and_precedence():
    from areno.dashboard.flow.workflows import resources

    assert resources({"timeout_seconds": 1})["timeout_seconds"] == 1
    assert resources({"timeout_seconds": 86400})["timeout_seconds"] == 86400
    assert resources({"timeout_seconds": 90, "timeout_hours": 4})["timeout_seconds"] == 90
