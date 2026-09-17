"""Compute-only estimates from Modal's live public Sandbox pricing, never invoices."""

from __future__ import annotations

import re
import threading
import time
import urllib.request
from decimal import Decimal
from html.parser import HTMLParser

from areno.dashboard.flow.workflows import bounded, resources, timeout_seconds

PRICING_URL = "https://modal.com/pricing"
GPU_LABELS = {
    "T4": "Nvidia T4",
    "L4": "Nvidia L4",
    "A10G": "Nvidia A10",
    "L40S": "Nvidia L40S",
    "A100-40GB": "Nvidia A100, 40 GB",
    "A100-80GB": "Nvidia A100, 80 GB",
    "H100": "Nvidia H100 SXM5",
    "H200": "Nvidia H200 SXM",
    "B200": "Nvidia B200",
}


class PricingText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def parse_rates(html):
    parser = PricingText()
    parser.feed(html)
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    sections = text.split("Modal Sandbox + Notebooks Pricing", 1)
    if len(sections) != 2:
        raise ValueError("Modal's Sandbox pricing format changed; estimates are unavailable")

    def rate(pattern, source):
        match = re.search(pattern, source)
        if not match:
            raise ValueError("A required Modal rate is missing; estimates are unavailable")
        value = Decimal(match.group(1))
        if not value.is_finite() or value <= 0:
            raise ValueError("Modal returned an invalid rate")
        return str(value)

    gpus = {
        gpu: rate(re.escape(label) + r"\s+\$([\d.]+)\s*/\s*sec\b", sections[0]) for gpu, label in GPU_LABELS.items()
    }
    return {
        "gpu_per_second": gpus,
        "cpu_per_core_second": rate(r"Physical core.*?\$([\d.]+)\s*/\s*core\s*/\s*sec", sections[1]),
        "memory_per_gib_second": rate(r"Memory\s+\$([\d.]+)\s*/\s*GiB\s*/\s*sec", sections[1]),
        "source": PRICING_URL,
        "fetched_at": time.time(),
        "currency": "USD",
        "basis": "Public Sandbox list rates; requested CPU and memory; compute only",
    }


def estimate(raw_resources, hours, rates, count=1):
    reserved = resources(raw_resources)
    hours = bounded(hours, "Estimated duration (hours)", 1 / 3600, 24)
    if hours > reserved["timeout_seconds"] / 3600:
        raise ValueError("Estimated duration cannot exceed the configured maximum lifetime")
    count = bounded(count, "Run count", 1, 10000, True)
    components = {
        "gpu": Decimal(rates["gpu_per_second"][reserved["gpu"]]) * reserved["count"],
        "cpu": Decimal(rates["cpu_per_core_second"]) * Decimal(str(reserved["cpu"])),
        "memory": Decimal(rates["memory_per_gib_second"]) * reserved["memory_gib"],
    }
    hourly = sum(components.values()) * 3600
    planned = hourly * Decimal(str(hours))
    return dict(
        hourly_cost=str(hourly),
        planned_cost=str(planned),
        total_cost=str(planned * count),
        hours=hours,
        run_count=count,
        currency="USD",
        rates=rates,
        breakdown={key: str(value * 3600 * Decimal(str(hours))) for key, value in components.items()},
        lifetime_cost=str(sum(components.values()) * reserved["timeout_seconds"]),
    )


class Pricing:
    def __init__(self):
        self.cache = None
        self.lock = threading.Lock()

    def rates(self):
        with self.lock:
            if self.cache and time.time() - self.cache["fetched_at"] < 900:
                return self.cache
            request = urllib.request.Request(PRICING_URL, headers={"User-Agent": "ARenoflow/0.1"})
            with urllib.request.urlopen(request, timeout=20) as response:
                self.cache = parse_rates(response.read().decode())
            return self.cache

    def quote(self, reservation, hours, count=1):
        return estimate(reservation, hours, self.rates(), count)

    def all_runs(self, jobs):
        rows, errors = [], []
        rates, rate_error = None, None
        if any(job.get("kind") == "training" and not job.get("estimate") for job in jobs):
            try:
                rates = self.rates()
            except Exception as exc:
                rate_error = str(exc)
        for job in jobs:
            if job.get("kind") != "training":
                continue
            try:
                if not job.get("estimate") and rate_error:
                    raise ValueError(rate_error)
                quote = job.get("estimate") or estimate(
                    job["resources"], timeout_seconds(job["resources"]) / 3600, rates
                )
                elapsed = (
                    max(0, (job.get("finished_at") or time.time()) - job["started_at"]) if job.get("started_at") else 0
                )
                accrued = Decimal(quote["hourly_cost"]) * Decimal(str(elapsed)) / 3600
                rows.append(
                    {
                        "id": job["id"],
                        "name": job["name"],
                        "status": job["status"],
                        "planned_cost": quote["planned_cost"],
                        "accrued_cost": str(accrued),
                        "hours": quote["hours"],
                        "source": quote["rates"]["source"],
                    }
                )
            except Exception as exc:
                errors.append({"id": job["id"], "error": str(exc)})
        return dict(
            rows=rows,
            errors=errors,
            run_count=len(rows) + len(errors),
            priced_count=len(rows),
            planned_total=str(sum((Decimal(row["planned_cost"]) for row in rows), Decimal(0))),
            accrued_total=str(sum((Decimal(row["accrued_cost"]) for row in rows), Decimal(0))),
            currency="USD",
        )
