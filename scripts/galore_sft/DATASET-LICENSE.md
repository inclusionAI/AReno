# Dataset attribution

Databricks, Inc., databricks-dolly-15k (2023), including reference passages credited to Wikipedia editors and contributors. Dataset license: Creative Commons Attribution-ShareAlike 3.0 Unported, https://creativecommons.org/licenses/by-sa/3.0/legalcode . Retain attribution and share-alike terms when redistributing adaptations. This code bundle contains no Dolly examples or model weights.

Pinned source: ModelScope mirror AI-ModelScope/databricks-dolly-15k, revision 40789acc12fc986e9145f87d159f19e2f1d75c6a. Original JSONL SHA256: 2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec. Original dataset card README.md SHA256: db06c60090a11b20e79d1933b2f316f781605f07761ff2ae2795d6523e3e960b. The downloader retains the pinned card in the caller's raw directory.

Owner documentation: https://github.com/databrickslabs/dolly . Owner announcement: https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm . The public ModelScope mirror is used explicitly; there is no automatic alternative-hub fallback.

Transformations: strip outer field whitespace; remove 16 exact normalized full-example duplicates; render instruction/context/response headings; assign deterministic disjoint context/instruction groups to train/validation/test; save seed 0/1/2 training orders. Retained examples are not token-truncated, and no generated examples are added. The 90/5/5 grouping is an exploratory local split, not an official benchmark split or a guarantee of article-level or near-duplicate separation.
