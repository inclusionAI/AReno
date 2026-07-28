"""Near-duplicate training example detection (issue #218).

This module finds near-duplicate training examples **without deleting them**.
It uses memory-bounded normalized-text fingerprinting plus a lightweight
approximate similarity mode for prompt-only or full-sample comparisons.

Two detection modes are provided:

* **exact**: Normalises text (lowercasing, whitespace collapse, punctuation
  stripping) and groups records with identical fingerprints.  This catches
  exact duplicates and formatting-only variations.

* **near**: Uses character n-gram shingling with MinHash-style signatures to
  find approximate matches above a configurable Jaccard similarity threshold.
  This catches paraphrases, minor edits, and partial overlaps.

The module is pure Python with no external dependencies beyond the standard
library.  It never mutates source data.

Public API:

* ``find_duplicates(records, ...)`` — main entry point; returns
  :class:`DuplicateReport`.
* ``DuplicateReport`` — structured result with groups, summary, and
  ``to_dict()`` for JSON output.
* ``normalize_text(text)`` — text normalisation used by the exact mode.
* ``shingle_signature(text, ...)`` — n-gram shingling for the near mode.
* ``jaccard_similarity(sig_a, sig_b)`` — Jaccard similarity of two signatures.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Precompiled regexes for normalisation.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalise text for exact-duplicate fingerprinting.

    Applies: Unicode NFC normalisation, lowercasing, punctuation removal,
    and whitespace collapse.  The result is a canonical form that treats
    ``"Hello,  World!"`` and ``"hello world"`` as identical.

    Args:
        text: Raw input text.

    Returns:
        Normalised text string.
    """

    if not text:
        return ""
    normalised = unicodedata.normalize("NFC", text)
    normalised = normalised.lower()
    normalised = _PUNCT_RE.sub(" ", normalised)
    normalised = _MULTI_WS_RE.sub(" ", normalised).strip()
    return normalised


def _fingerprint(text: str) -> str:
    """Return a SHA-256 hex digest of normalised text for exact dedup."""

    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# N-gram shingling for near-duplicate detection
# ---------------------------------------------------------------------------


def shingle_signature(
    text: str,
    *,
    ngram_size: int = 5,
    max_features: int = 2048,
) -> frozenset[str]:
    """Compute a character n-gram shingle set for near-duplicate comparison.

    The text is normalised before shingling.  The resulting set is capped
    at ``max_features`` to bound memory usage for very long inputs.

    Args:
        text: Raw input text.
        ngram_size: Character n-gram length (default 5).
        max_features: Maximum number of shingles to retain (default 2048).

    Returns:
        A :class:`frozenset` of n-gram strings.
    """

    if not text:
        return frozenset()
    normalised = normalize_text(text)
    if len(normalised) < ngram_size:
        return frozenset({normalised}) if normalised else frozenset()
    shingles: set[str] = set()
    for i in range(len(normalised) - ngram_size + 1):
        shingles.add(normalised[i : i + ngram_size])
        if len(shingles) >= max_features:
            break
    return frozenset(shingles)


def jaccard_similarity(sig_a: frozenset[str], sig_b: frozenset[str]) -> float:
    """Compute Jaccard similarity between two shingle signatures.

    Args:
        sig_a: First signature (from :func:`shingle_signature`).
        sig_b: Second signature (from :func:`shingle_signature`).

    Returns:
        Similarity in [0.0, 1.0].  Returns 0.0 if both are empty.
    """

    if not sig_a and not sig_b:
        return 0.0
    intersection = len(sig_a & sig_b)
    union = len(sig_a | sig_b)
    if union == 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateGroup:
    """A group of near-duplicate records.

    Attributes:
        group_id: 0-based group identifier.
        record_indices: Indices into the original records list.
        similarity: Mean pairwise similarity within the group (1.0 for exact).
        match_type: "exact" or "near".
    """

    group_id: int
    record_indices: list[int]
    similarity: float
    match_type: Literal["exact", "near"]


@dataclass(frozen=True)
class DuplicateReport:
    """Structured result of duplicate detection.

    Attributes:
        groups: List of :class:`DuplicateGroup` objects, each containing
            indices of duplicate records.
        total_records: Number of records scanned.
        duplicate_records: Number of records that belong to at least one
            duplicate group (excluding the first representative of each group).
        unique_records: Number of unique records (not in any duplicate group).
        match_type: The detection mode used ("exact" or "near").
        threshold: Jaccard similarity threshold (only for "near" mode).
        ngram_size: Character n-gram size used for shingling.
    """

    groups: list[DuplicateGroup]
    total_records: int
    duplicate_records: int
    unique_records: int
    match_type: Literal["exact", "near"]
    threshold: float = 1.0
    ngram_size: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for structured output."""

        return {
            "groups": [
                {
                    "group_id": g.group_id,
                    "record_indices": g.record_indices,
                    "similarity": g.similarity,
                    "match_type": g.match_type,
                }
                for g in self.groups
            ],
            "total_records": self.total_records,
            "duplicate_records": self.duplicate_records,
            "unique_records": self.unique_records,
            "match_type": self.match_type,
            "threshold": self.threshold,
            "ngram_size": self.ngram_size,
        }

    @property
    def duplicate_fraction(self) -> float:
        """Fraction of records that are duplicates (excluding representatives)."""

        if self.total_records == 0:
            return 0.0
        return self.duplicate_records / self.total_records


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

# Field names checked in priority order for prompt/text extraction.
_DEFAULT_TEXT_KEYS: tuple[str, ...] = ("prompt", "question", "text", "content", "input", "query")


def _extract_text(record: dict[str, Any], text_keys: Sequence[str] | None) -> str:
    """Extract the primary text from a record for comparison.

    Args:
        record: A dataset row dict.
        text_keys: Ordered field names to check.  Defaults to
            ``("prompt", "question", "text", "content", "input", "query")``.

    Returns:
        The first string field found, or empty string if none.

    Raises:
        KeyError: If ``text_keys`` is provided but none of the keys exist
            and the record has no string values at all.
    """

    keys = tuple(text_keys) if text_keys is not None else _DEFAULT_TEXT_KEYS
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    # Fallback: try any string value in the record.
    for value in record.values():
        if isinstance(value, str):
            return value
    return ""


def _extract_full_sample(record: dict[str, Any], text_keys: Sequence[str] | None) -> str:
    """Extract a concatenation of all string fields for full-sample comparison."""

    parts: list[str] = []
    for value in record.values():
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def find_duplicates(
    records: list[dict[str, Any]],
    *,
    mode: Literal["exact", "near"] = "exact",
    scope: Literal["prompt", "full"] = "prompt",
    text_keys: Sequence[str] | None = None,
    threshold: float = 0.8,
    ngram_size: int = 5,
    max_features: int = 2048,
) -> DuplicateReport:
    """Find near-duplicate training examples without deleting them.

    Args:
        records: List of dataset row dicts.
        mode: Detection mode — ``"exact"`` for normalised fingerprinting or
            ``"near"`` for approximate n-gram similarity.
        scope: Comparison scope — ``"prompt"`` to compare only the primary
            text field, or ``"full"`` to compare all string fields.
        text_keys: Ordered field names for text extraction.  If ``None``,
            uses a default priority list.
        threshold: Jaccard similarity threshold for ``"near"`` mode (default 0.8).
        ngram_size: Character n-gram size for shingling (default 5).
        max_features: Maximum shingle features per record (default 2048).

    Returns:
        A :class:`DuplicateReport` with grouped duplicates and summary stats.

    Raises:
        ValueError: If parameters are invalid.
        TypeError: If records is not a list of dicts.
    """

    # --- Validate inputs ---
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    if mode not in ("exact", "near"):
        raise ValueError(f"mode must be 'exact' or 'near', got {mode!r}")
    if scope not in ("prompt", "full"):
        raise ValueError(f"scope must be 'prompt' or 'full', got {scope!r}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0.0, 1.0], got {threshold}")
    if ngram_size < 1:
        raise ValueError(f"ngram_size must be >= 1, got {ngram_size}")
    if max_features < 1:
        raise ValueError(f"max_features must be >= 1, got {max_features}")

    total = len(records)

    # --- Extract text for each record ---
    if scope == "prompt":
        texts = [_extract_text(rec, text_keys) for rec in records]
    else:
        texts = [_extract_full_sample(rec, text_keys) for rec in records]

    if mode == "exact":
        return _find_exact_duplicates(texts, total)
    return _find_near_duplicates(texts, total, threshold=threshold, ngram_size=ngram_size, max_features=max_features)


def _find_exact_duplicates(texts: list[str], total: int) -> DuplicateReport:
    """Group records by normalised-text fingerprint."""

    # Map fingerprint -> list of record indices.
    fp_to_indices: dict[str, list[int]] = {}
    for idx, text in enumerate(texts):
        fp = _fingerprint(text)
        fp_to_indices.setdefault(fp, []).append(idx)

    groups: list[DuplicateGroup] = []
    duplicate_count = 0
    for group_id, (fp, indices) in enumerate(item for item in fp_to_indices.items() if len(item[1]) > 1):
        groups.append(
            DuplicateGroup(
                group_id=group_id,
                record_indices=sorted(indices),
                similarity=1.0,
                match_type="exact",
            )
        )
        duplicate_count += len(indices) - 1  # Exclude the representative.

    # Re-number groups sequentially.
    groups = [
        DuplicateGroup(
            group_id=i,
            record_indices=g.record_indices,
            similarity=g.similarity,
            match_type=g.match_type,
        )
        for i, g in enumerate(groups)
    ]

    return DuplicateReport(
        groups=groups,
        total_records=total,
        duplicate_records=duplicate_count,
        unique_records=total - duplicate_count,
        match_type="exact",
        threshold=1.0,
        ngram_size=0,
    )


def _find_near_duplicates(
    texts: list[str],
    total: int,
    *,
    threshold: float,
    ngram_size: int,
    max_features: int,
) -> DuplicateReport:
    """Group records by approximate n-gram similarity using union-find."""

    # Compute signatures for all records.
    signatures: list[frozenset[str]] = [
        shingle_signature(text, ngram_size=ngram_size, max_features=max_features) for text in texts
    ]

    # Union-find (disjoint set) for grouping.
    parent = list(range(total))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Compare all pairs; O(n^2) but bounded by max_features per signature.
    # For very large datasets this is still memory-bounded because each
    # signature is a frozenset of at most max_features strings.
    for i in range(total):
        for j in range(i + 1, total):
            if find(i) == find(j):
                continue
            sim = jaccard_similarity(signatures[i], signatures[j])
            if sim >= threshold:
                union(i, j)

    # Collect groups from union-find roots.
    root_to_indices: dict[int, list[int]] = {}
    for idx in range(total):
        root = find(idx)
        root_to_indices.setdefault(root, []).append(idx)

    groups: list[DuplicateGroup] = []
    duplicate_count = 0
    group_id = 0
    for indices in root_to_indices.values():
        if len(indices) <= 1:
            continue
        # Compute mean pairwise similarity within the group.
        sims: list[float] = []
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                sims.append(jaccard_similarity(signatures[indices[a]], signatures[indices[b]]))
        mean_sim = sum(sims) / len(sims) if sims else 1.0

        groups.append(
            DuplicateGroup(
                group_id=group_id,
                record_indices=sorted(indices),
                similarity=round(mean_sim, 4),
                match_type="near",
            )
        )
        duplicate_count += len(indices) - 1
        group_id += 1

    return DuplicateReport(
        groups=groups,
        total_records=total,
        duplicate_records=duplicate_count,
        unique_records=total - duplicate_count,
        match_type="near",
        threshold=threshold,
        ngram_size=ngram_size,
    )


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------


def format_duplicate_report(report: DuplicateReport) -> str:
    """Return a human-readable summary of duplicate detection results.

    Args:
        report: A :class:`DuplicateReport` from :func:`find_duplicates`.

    Returns:
        A formatted multi-line string.
    """

    lines: list[str] = []
    lines.append(f"Duplicate detection ({report.match_type} mode):")
    lines.append(f"  Total records: {report.total_records}")
    lines.append(f"  Duplicate records: {report.duplicate_records}")
    lines.append(f"  Unique records: {report.unique_records}")
    lines.append(f"  Duplicate fraction: {report.duplicate_fraction:.2%}")
    if report.match_type == "near":
        lines.append(f"  Threshold: {report.threshold}")
        lines.append(f"  N-gram size: {report.ngram_size}")
    lines.append(f"  Duplicate groups: {len(report.groups)}")
    if report.groups:
        lines.append("")
        lines.append("Groups (first 10):")
        for g in report.groups[:10]:
            lines.append(
                f"  Group {g.group_id}: {len(g.record_indices)} records, "
                f"similarity={g.similarity:.4f}, indices={g.record_indices}"
            )
        if len(report.groups) > 10:
            lines.append(f"  ... and {len(report.groups) - 10} more groups")
    return "\n".join(lines)
