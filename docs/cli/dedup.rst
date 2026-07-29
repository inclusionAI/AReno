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

``--max-features`` (default: ``2048``)
    Maximum number of shingles retained per record in ``near`` mode.  AReno
    keeps the bottom-k shingles selected by a stable hash, so the cap does not
    favor the beginning of long records.

``--text-keys`` (default: ``prompt,question,instruction,problem,text,content,input,query``)
    Comma-separated field names for text extraction in ``prompt`` scope.
    The first matching string field in each record is used.  Missing fields
    produce an error containing only the record index and checked field names;
    AReno does not print the sample text.

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
``record_indices``, ``similarity``, and ``match_type``.  It also records
``scope``, ``text_keys``, ``threshold``, ``ngram_size``, and ``max_features``
so that a curation scan can be reproduced.

Examples
~~~~~~~~

Create a tiny JSONL fixture and run exact mode:

.. code-block:: bash

   printf '%s\n' \
     '{"prompt":"What is 2 + 2?","answer":"4"}' \
     '{"prompt":"what is 2 + 2","answer":"four"}' \
     '{"prompt":"Explain photosynthesis","answer":"..."}' \
     > /tmp/areno-dedup.jsonl
   areno dedup --data-path /tmp/areno-dedup.jsonl --json

Near mode with a lower threshold:

.. code-block:: bash

   areno dedup --data-path /tmp/areno-dedup.jsonl \
     --mode near --threshold 0.5 --max-features 1024

Full-sample comparison with JSON output:

.. code-block:: bash

   areno dedup --data-path /tmp/areno-dedup.jsonl --scope full --json

Custom text field:

.. code-block:: bash

   areno dedup --data-path /tmp/areno-dedup.jsonl --text-keys question,problem

Invalid resource or similarity controls fail before scanning the dataset.  For
example, this command exits with a Click validation error because the threshold
must be greater than zero:

.. code-block:: bash

   areno dedup --data-path /tmp/areno-dedup.jsonl --mode near --threshold 0

Limitations
~~~~~~~~~~~

* The ``near`` mode uses O(n^2) pairwise comparison, which may be slow for
  very large datasets (>10k records).  Signature memory is
  O(number-of-records * ``max_features``); the cap bounds each record rather
  than the whole input file.
* ``full`` scope uses a canonical, key-sorted JSON representation of the
  complete record, so JSON field order does not change the result.
* The command reports duplicates for review but does not remove, filter, or
  modify any records.
* In ``prompt`` scope, every record must contain a string value under one of
  the selected ``text_keys``.  Use ``--text-keys`` for custom dataset schemas.
