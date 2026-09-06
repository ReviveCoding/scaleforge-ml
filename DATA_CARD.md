# ScaleForge-ML Data Card

## Dataset and intended use

The primary dataset is `openai/gsm8k`, configuration `main`, immutable revision `740312add88f781978c0658806c59bc2815b9866`, licensed MIT according to its Hugging Face card. ScaleForge uses the official train set for fitting, development selection, and policy development. The official test set is protected FINAL data. `HuggingFaceH4/MATH-500` is protected OOD/non-regression data. Both protected sets were accessed only under logged frozen qualification identities and are now CLOSED to scientific development.

The model/tokenizer workload is `Qwen/Qwen2.5-1.5B-Instruct`; token counts use tokenizer revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (`Qwen2Tokenizer`, vocabulary size 151,665).

## Acquisition and provenance

The successful development pipeline retrieves only the immutable `main/train-*.parquet` object directly through Hugging Face Hub. Raw source data, processed row-level Parquet, Hub caches, and model artifacts are gitignored. Version-controlled manifests retain revisions, counts, fingerprints, policy hashes, and validation results.

An earlier attempt requested `split="train"` through Hugging Face Datasets, but its builder also materialized an official-test Arrow cache. Project code did not select test rows and no model prediction or aggregate outcome was exposed. This Case A infrastructure exposure is permanently recorded in `FINAL_ACCESS_LEDGER.json` and `artifacts/failures/data_prepare_20260904_001.json`; the cache was not erased. Direct split-specific retrieval replaced that path.

## Deterministic partition

Each ID is SHA-256 over source, immutable revision, and the whitespace-normalized/case-folded question. A second SHA-256 maps the ID uniformly to `[0,1)`:

| Role | Interval | Rows | Authorized use |
|---|---:|---:|---|
| FIT | `[0.0, 0.8)` | 6,046 | Optimization and training |
| VALIDATION | `[0.8, 0.9)` | 723 | Prompt/model candidate selection |
| POLICY | `[0.9, 1.0)` | 704 | SLO/release-policy development and internal checks |

Official GSM8K test is FINAL and MATH-500 is OOD. Neither is authorized for development decisions.

## Schema

Each curated row contains `example_id`, `question`, `raw_solution`, `reference_final_answer`, `normalized_final_answer`, prompt and reference-solution token counts, `split_role`, and `source_revision`. Pydantic rejects extra fields, blanks, invalid roles, and nonpositive token counts. The GSM8K reference parser requires exactly one `####` answer marker and numeric normalization removes separators without changing value.

## Quality results

All 7,473 train rows passed. Null cells, blank questions, normalized duplicate rows, duplicate IDs, cross-role overlap, and malformed final-answer markers were all zero. The curated fingerprint is `88699b5b035e2102ea2523632fe460cd30dd0c5470aa3102ca55c11df4cf4ec8`.

| Distribution | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| Prompt tokens | 56 | 89 | 101 | 128 | 212 |
| Reference-solution tokens | 111 | 195 | 229 | 288 | 453 |
| Combined tokens | 168 | 275 | 312 | 390 | 536 |

## Sequence-length decision

The chosen limit is 640 tokens: `ceil(max_observed * 1.10 / 128) * 128`. The pipeline fails for a required limit above the configured 4,096-token ceiling and never silently truncates. The 640 limit is a data-derived training/preprocessing limit; generation output limits are independently frozen by the model protocol.

## Limitations and ethics

GSM8K is English grade-school mathematics with worked solutions; it does not represent broad reasoning, multilingual use, factual reliability, safety, or real production traffic. Exact-match answer normalization can miss semantically equivalent nonnumeric responses and cannot measure reasoning faithfulness. Source solutions may contain annotation artifacts. Structural slices are descriptive, not causal. Protected data must remain unavailable to tuning even though an infrastructure cache incident technically materialized the test shard. Subsequent protected qualification covered all 1,319 GSM8K test and 500 MATH-500 rows; the access history and evaluator-recovery identities are preserved in `FINAL_ACCESS_LEDGER.json`.

Canonical machine-readable evidence is in `artifacts/data/data_quality.json` and `artifacts/data/dataset_manifest.json`; row-level data is local-only in `artifacts/data/split_manifest.parquet`.
