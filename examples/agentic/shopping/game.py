"""Catalog helpers for the multi-turn shopping kit-planning example."""

from __future__ import annotations

from typing import Any

CATALOG: list[dict[str, Any]] = [
    {
        "id": "packable-rain-shell",
        "name": "Packable Rain Shell",
        "category": "jacket",
        "price": 89,
        "weight_g": 310,
        "rating": 4.6,
        "features": ["waterproof", "packable", "breathable"],
    },
    {
        "id": "trail-softshell",
        "name": "Trail Softshell",
        "category": "jacket",
        "price": 119,
        "weight_g": 520,
        "rating": 4.7,
        "features": ["windproof", "stretch", "warm"],
    },
    {
        "id": "ultralight-hiker",
        "name": "Ultralight Hiker",
        "category": "shoes",
        "price": 135,
        "weight_g": 680,
        "rating": 4.8,
        "features": ["trail", "water-resistant", "wide-toe"],
    },
    {
        "id": "city-commuter",
        "name": "City Commuter",
        "category": "shoes",
        "price": 95,
        "weight_g": 760,
        "rating": 4.3,
        "features": ["casual", "durable", "water-resistant"],
    },
    {
        "id": "insulated-bottle-750",
        "name": "Insulated Bottle 750",
        "category": "bottle",
        "price": 42,
        "weight_g": 390,
        "rating": 4.9,
        "features": ["insulated", "leakproof", "dishwasher-safe"],
    },
    {
        "id": "collapsible-bottle",
        "name": "Collapsible Bottle",
        "category": "bottle",
        "price": 28,
        "weight_g": 130,
        "rating": 4.2,
        "features": ["collapsible", "lightweight", "travel"],
    },
]


def make_prompt(record: dict[str, Any]) -> str:
    """Build the user request for one shopping task."""

    requirements = "; ".join(
        f"{category}: {', '.join(features)}" for category, features in record["required_features_by_category"].items()
    )
    categories = ", ".join(record["categories"])
    return (
        f"Build a {record['kit_name']} kit with one item from each category: {categories}. "
        f"The total budget is ${record['budget']}. Required features are: {requirements}. "
        "Use catalog tools to search, inspect, check the kit, and submit the final item ids."
    )


def search_catalog(category: str, query: str | None = None, max_price: int | None = None) -> list[dict[str, Any]]:
    """Return compact search results for matching catalog items."""

    query_terms = _terms(query or "")
    results = []
    for item in CATALOG:
        if item["category"] != category:
            continue
        if max_price is not None and int(item["price"]) > int(max_price):
            continue
        text = " ".join([item["name"], item["category"], *item["features"]]).lower()
        if query_terms and not all(term in text for term in query_terms):
            continue
        results.append(
            {
                "id": item["id"],
                "name": item["name"],
                "price": item["price"],
                "rating": item["rating"],
            }
        )
    return results


def search_catalog_many(categories: list[str], max_price: int | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return compact search results grouped by category."""

    return {category: search_catalog(category, max_price=max_price) for category in categories}


def inspect_item(item_id: str) -> dict[str, Any] | None:
    """Return full catalog details for one item."""

    for item in CATALOG:
        if item["id"] == item_id:
            return dict(item)
    return None


def inspect_items(item_ids: list[str]) -> list[dict[str, Any] | None]:
    """Return full catalog details for multiple items."""

    return [inspect_item(item_id) for item_id in item_ids]


def check_kit(record: dict[str, Any], item_ids: list[str]) -> dict[str, Any]:
    """Validate a proposed kit against budget and category feature constraints."""

    items = [item for item in inspect_items(item_ids) if item is not None]
    categories = {item["category"] for item in items}
    total = sum(int(item["price"]) for item in items)
    missing_categories = [category for category in record["categories"] if category not in categories]
    missing_features: dict[str, list[str]] = {}
    for item in CATALOG:
        if item["id"] not in item_ids:
            continue
        required = record["required_features_by_category"].get(item["category"], [])
        missing = [feature for feature in required if feature not in item["features"]]
        if missing:
            missing_features[item["category"]] = missing
    valid = (
        not missing_categories
        and not missing_features
        and total <= int(record["budget"])
        and len(items) == len(record["categories"])
    )
    return {
        "valid": valid,
        "total_price": total,
        "budget": record["budget"],
        "missing_categories": missing_categories,
        "missing_features": missing_features,
        "item_ids": [item["id"] for item in items],
    }


def best_bundle(record: dict[str, Any]) -> list[str]:
    """Return the highest-scoring valid item ids for the request."""

    choices_by_category: list[list[dict[str, Any]]] = []
    for category in record["categories"]:
        required = record["required_features_by_category"][category]
        choices = [
            item
            for item in CATALOG
            if item["category"] == category and all(feature in item["features"] for feature in required)
        ]
        choices_by_category.append(choices)
    scored = []
    for bundle in _product(choices_by_category):
        total = sum(int(item["price"]) for item in bundle)
        if total > int(record["budget"]):
            continue
        rating = sum(float(item["rating"]) for item in bundle)
        score = rating - (total / max(float(record["budget"]), 1.0)) * 0.1
        scored.append((score, [item["id"] for item in bundle]))
    if not scored:
        return []
    return max(scored)[1]


def score_bundle(record: dict[str, Any], item_ids: list[str] | None) -> float:
    """Score a submitted bundle against the shopping request."""

    if not item_ids:
        return -1.0
    status = check_kit(record, item_ids)
    if not status["valid"]:
        return -0.5
    return 1.0 if set(item_ids) == set(best_bundle(record)) else 0.4


def _product(groups: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    bundles: list[list[dict[str, Any]]] = [[]]
    for group in groups:
        bundles = [bundle + [item] for bundle in bundles for item in group]
    return bundles


def _terms(text: str) -> list[str]:
    return [part.strip().lower() for part in text.replace(",", " ").split() if part.strip()]
