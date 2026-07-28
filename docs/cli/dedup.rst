:orphan:

Dedup CLI reference
===================

``areno dedup``

Find near-duplicate training examples in a JSONL or JSON dataset without
deleting them.  The command scans a dataset file, groups duplicate records,
and prints a summary report.  Source data is never modified.

.. code-block:: bash

   areno dedup --data-path /path/to/dataset.jsonl

Two detection modes are available:

* **exact** (default): normalises text (lowercasing, whitespace collapse,
  punctuation removal) and groups records with identical fingerprints.
  Catches exact duplicates and formatting-only variations.
* **near**: uses character n-gram shingling with Jaccard similarity to
  find approximate matches above a configurable threshold.  Catches
  paraphrases, minor edits, and partial overlaps.

areno dedup
-----------

.. code-block:: bash

   areno dedup --data-path <path> [options]

Options
~~~~~~~

``--data-path`` (required)
    Path to a JSONL (``.jsonl``, ``.ndjson``) or JSON (``.json``) dataset file.
    JSONL files contain one JSON object per line.  JSON files must contain an
    array of objects.

``--mode`` (default: ``exact``)
    Detection mode: ``exact`` for normalised fingerprinting or ``near`` for
    n-gram similarity.

``--scope`` (default: ``prompt``)
    Comparison scope: ``prompt`` compares only the primary text field, or
    ``full`` compares all string fields concatenated.

``--threshold`` (default: ``0.8``)
    Jaccard similarity threshold for ``near`` mode.  Must be in (0.0, 1.0].
    Lower values find more approximate matches.

``--ngram-size`` (default: ``5``)
    Character n-gram size for shingling in ``near`` mode.

``--text-keys`` (default: ``prompt,question,text,content,input,query``)
    Comma-separated field names for text extraction in ``prompt`` scope.
    The first matching string field in each record is used.

``--json``
    Emit machine-readable JSON output instead of human-readable text.

Output fields
~~~~~~~~~~~~~

The human-readable output includes:

* Total records scanned.
* Duplicate records (records in duplicate groups, excluding one representative per group).
* Unique records.
* Duplicate fraction.
* Number of duplicate groups.
* Group details (group ID, record count, similarity, record indices).

The JSON output (``--json``) includes the same fields plus per-group
``record_indices``, ``similarity``, and ``match_type``.

Examples
~~~~~~~~

Exact mode with a JSONL dataset:

.. code-block:: bash

   areno dedup --data-path /path/to/train.jsonl

Near mode with a lower threshold:

.. code-block:: bash

   areno dedup --data-path /path/to/train.jsonl --mode near --threshold 0.5

Full-sample comparison with JSON output:

.. code-block:: bash

   areno dedup --data-path /path/to/train.jsonl --scope full --json

Custom text field:

.. code-block:: bash

   areno dedup --data-path /path/to/train.jsonl --text-keys question,problem

Limitations
~~~~~~~~~~~

* The ``near`` mode uses O(n^2) pairwise comparison, which may be slow for
  very large datasets (>10k records).  Memory is bounded by ``max_features``
  (default 2048 shingles per record).
* The command reports duplicates for review but does not remove, filter, or
  modify any records.
* Text extraction falls back to any string field if none of the specified
  ``text_keys`` are found in a record.