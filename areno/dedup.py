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
import heapq
import json
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
_DEFAULT_MAX_COMPARISONS = 250_000


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

    if not isinstance(text, str):
        raise TypeError("text must be a string")
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

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(ngram_size, int) or isinstance(ngram_size, bool) or ngram_size < 1:
        raise ValueError(f"ngram_size must be a positive integer, got {ngram_size!r}")
    if not isinstance(max_features, int) or isinstance(max_features, bool) or max_features < 1:
        raise ValueError(f"max_features must be a positive integer, got {max_features!r}")
    if not text:
        return frozenset()
    normalised = normalize_text(text)
    if len(normalised) < ngram_size:
        return frozenset({normalised}) if normalised else frozenset()
    # Keep the bottom-k shingles by a stable hash.  Retaining the first k
    # shingles would over-weight the beginning of long examples and could
    # make records with identical prefixes look artificially identical.
    selected: set[str] = set()
    heap: list[tuple[int, str]] = []
    for i in range(len(normalised) - ngram_size + 1):
        shingle = normalised[i : i + ngram_size]
        if shingle in selected:
            continue
        rank = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        entry = (-rank, shingle)
        if len(heap) < max_features:
            heapq.heappush(heap, entry)
            selected.add(shingle)
            continue
        if rank < -heap[0][0]:
            _, removed = heapq.heapreplace(heap, entry)
            selected.remove(removed)
            selected.add(shingle)
    return frozenset(selected)


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
        scope: Comparison scope used for the scan.
        text_keys: Prompt fields checked in priority order, or ``None`` for
            full-sample comparison.
        max_features: Maximum shingles retained per record in near mode.
        max_comparisons: Maximum candidate pairs evaluated in near mode.
        candidate_comparisons: Candidate pairs evaluated by the scan.
    """

    groups: list[DuplicateGroup]
    total_records: int
    duplicate_records: int
    unique_records: int
    match_type: Literal["exact", "near"]
    threshold: float = 1.0
    ngram_size: int = 5
    scope: Literal["prompt", "full"] = "prompt"
    text_keys: tuple[str, ...] | None = None
    max_features: int = 0
    max_comparisons: int = 0
    candidate_comparisons: int = 0

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
            "scope": self.scope,
            "text_keys": list(self.text_keys) if self.text_keys is not None else None,
            "max_features": self.max_features,
            "max_comparisons": self.max_comparisons,
            "candidate_comparisons": self.candidate_comparisons,
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
_DEFAULT_TEXT_KEYS: tuple[str, ...] = (
    "prompt",
    "question",
    "instruction",
    "problem",
    "text",
    "content",
    "input",
    "query",
)


def _normalise_text_keys(text_keys: Sequence[str] | None) -> tuple[str, ...]:
    """Validate prompt field names and return a deterministic tuple."""

    if text_keys is None:
        return _DEFAULT_TEXT_KEYS
    if isinstance(text_keys, str):
        raise TypeError("text_keys must be a sequence of field names, not a string")
    keys = tuple(text_keys)
    if not keys:
        raise ValueError("text_keys must contain at least one field name")
    normalised: list[str] = []
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("text_keys must contain only non-empty strings")
        stripped = key.strip()
        if stripped not in normalised:
            normalised.append(stripped)
    return tuple(normalised)


def _extract_text(record: dict[str, Any], text_keys: tuple[str, ...], record_index: int) -> str:
    """Extract the primary text from a record for comparison.

    Args:
        record: A dataset row dict.
        text_keys: Ordered field names to check.
        record_index: Index used in validation errors.

    Returns:
        The first non-empty string field found.

    Raises:
        ValueError: If none of the selected fields contains a non-empty string.
    """

    for key in text_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    fields = ", ".join(text_keys)
    raise ValueError(
        f"record at index {record_index} has no string prompt field with non-whitespace content; checked keys: {fields}"
    )


def _extract_full_sample(record: dict[str, Any], record_index: int) -> str:
    """Return a canonical JSON representation for full-sample comparison."""

    try:
        return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"record at index {record_index} must contain JSON-serializable values") from exc


def find_duplicates(
    records: list[dict[str, Any]],
    *,
    mode: Literal["exact", "near"] = "exact",
    scope: Literal["prompt", "full"] = "prompt",
    text_keys: Sequence[str] | None = None,
    threshold: float = 0.8,
    ngram_size: int = 5,
    max_features: int = 2048,
    max_comparisons: int = _DEFAULT_MAX_COMPARISONS,
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
        max_comparisons: Maximum shared-shingle candidate pairs evaluated in
            near mode (default 250,000).

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
    if not isinstance(ngram_size, int) or isinstance(ngram_size, bool) or ngram_size < 1:
        raise ValueError(f"ngram_size must be a positive integer, got {ngram_size!r}")
    if not isinstance(max_features, int) or isinstance(max_features, bool) or max_features < 1:
        raise ValueError(f"max_features must be a positive integer, got {max_features!r}")
    if not isinstance(max_comparisons, int) or isinstance(max_comparisons, bool) or max_comparisons < 1:
        raise ValueError(f"max_comparisons must be a positive integer, got {max_comparisons!r}")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"record at index {index} must be a dict")
        if any(not isinstance(key, str) for key in record):
            raise TypeError(f"record at index {index} must use string field names")

    total = len(records)

    # --- Extract text for each record ---
    if scope == "prompt":
        selected_text_keys = _normalise_text_keys(text_keys)
        texts = [_extract_text(record, selected_text_keys, index) for index, record in enumerate(records)]
    else:
        selected_text_keys = None
        texts = [_extract_full_sample(record, index) for index, record in enumerate(records)]

    if mode == "exact":
        return _find_exact_duplicates(
            texts,
            total,
            scope=scope,
            text_keys=selected_text_keys,
        )
    return _find_near_duplicates(
        texts,
        total,
        threshold=threshold,
        ngram_size=ngram_size,
        max_features=max_features,
        max_comparisons=max_comparisons,
        scope=scope,
        text_keys=selected_text_keys,
    )


def _find_exact_duplicates(
    texts: list[str],
    total: int,
    *,
    scope: Literal["prompt", "full"],
    text_keys: tuple[str, ...] | None,
) -> DuplicateReport:
    """Group records by normalised-text fingerprint."""

    # Map fingerprint -> list of record indices.
    fp_to_indices: dict[str, list[int]] = {}
    for idx, text in enumerate(texts):
        # Full-sample text is already canonical JSON. Hash it directly so
        # punctuation stripping cannot collapse distinct JSON types such as
        # the number 1 and the string "1".
        fp = hashlib.sha256(text.encode("utf-8")).hexdigest() if scope == "full" else _fingerprint(text)
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
        scope=scope,
        text_keys=text_keys,
        max_features=0,
    )


def _find_near_duplicates(
    texts: list[str],
    total: int,
    *,
    threshold: float,
    ngram_size: int,
    max_features: int,
    max_comparisons: int,
    scope: Literal["prompt", "full"],
    text_keys: tuple[str, ...] | None,
) -> DuplicateReport:
    """Group records using bounded shared-shingle candidates and union-find."""

    # Compute signatures for all records.
    signatures: list[frozenset[str]] = [
        shingle_signature(text, ngram_size=ngram_size, max_features=max_features) for text in texts
    ]

    # Build a deterministic inverted index. Two non-empty shingle sets with
    # positive Jaccard similarity must share at least one shingle, so disjoint
    # records never need an explicit comparison. The candidate set is capped
    # to bound both memory and similarity work.
    postings: dict[str, list[int]] = {}
    candidate_pairs: set[tuple[int, int]] = set()
    candidate_work = 0
    for index, signature in enumerate(signatures):
        for shingle in sorted(signature):
            prior_indices = postings.setdefault(shingle, [])
            candidate_work += len(prior_indices)
            if candidate_work > max_comparisons:
                raise ValueError(
                    "near-duplicate candidate comparison limit exceeded "
                    f"({max_comparisons}); reduce the input, increase max_comparisons, "
                    "reduce max_features, or use exact mode"
                )
            for prior_index in prior_indices:
                candidate_pairs.add((prior_index, index))
                if len(candidate_pairs) > max_comparisons:
                    raise ValueError(
                        "near-duplicate candidate comparison limit exceeded "
                        f"({max_comparisons}); reduce the input, increase max_comparisons, "
                        "reduce max_features, or use exact mode"
                    )
            prior_indices.append(index)

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

    # Compare each candidate exactly once. Retain the bounded similarity map
    # so group reporting does not repeat an unbounded all-pairs pass.
    pair_similarities: dict[tuple[int, int], float] = {}
    for i, j in sorted(candidate_pairs):
        sim = jaccard_similarity(signatures[i], signatures[j])
        pair_similarities[(i, j)] = sim
        if sim >= threshold:
            union(i, j)

    # Collect groups from union-find roots.
    root_to_indices: dict[int, list[int]] = {}
    for idx in range(total):
        root = find(idx)
        root_to_indices.setdefault(root, []).append(idx)

    # Missing candidate pairs have disjoint signatures and therefore exact
    # Jaccard similarity 0. Accumulating the retained candidate similarities
    # and dividing by all pairs preserves the original mean-pairwise metric
    # without another quadratic comparison pass.
    root_similarity_sums: dict[int, float] = {}
    for (left, right), similarity in pair_similarities.items():
        root = find(left)
        if root == find(right):
            root_similarity_sums[root] = root_similarity_sums.get(root, 0.0) + similarity

    groups: list[DuplicateGroup] = []
    duplicate_count = 0
    group_id = 0
    for indices in root_to_indices.values():
        if len(indices) <= 1:
            continue
        pair_count = len(indices) * (len(indices) - 1) // 2
        mean_sim = root_similarity_sums.get(find(indices[0]), 0.0) / pair_count

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
        scope=scope,
        text_keys=text_keys,
        max_features=max_features,
        max_comparisons=max_comparisons,
        candidate_comparisons=len(candidate_pairs),
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
    lines.append(f"  Scope: {report.scope}")
    if report.text_keys is not None:
        lines.append(f"  Prompt fields: {', '.join(report.text_keys)}")
    if report.match_type == "near":
        lines.append(f"  Threshold: {report.threshold}")
        lines.append(f"  N-gram size: {report.ngram_size}")
        lines.append(f"  Max features per record: {report.max_features}")
        lines.append(f"  Candidate comparisons: {report.candidate_comparisons} (limit: {report.max_comparisons})")
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
