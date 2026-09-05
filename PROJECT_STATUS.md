# Project Status

- Current stage: MODEL_DEVELOPMENT / fixed batch-size-4 full M0/M1 validation ready
- Current Git identity: unborn `main` branch; commit SHA unavailable; local identity `Jinwoo Bae <bjw-0907@hanmail.net>`
- Active environment: Windows PowerShell control plane; WSL `.venv` TRAIN/MODEL/ANALYSIS on Python 3.12.14, PyTorch 2.14.0+cu130; serving environment not yet created
- Completed stages: persistence controls; preflight; locked WSL training environment; typed foundation and quality gate; package/CI; GSM8K train-only curation and validation (7,473 rows); identical two-run fingerprint verification (88699b...4ec8)
- Protected-data access status: GSM8K test technically materialized once by dataset-builder infrastructure while `train` was requested; no test row was selected and no scientific outcome was exposed (Case A, recorded in FINAL_ACCESS_LEDGER.json). MATH-500 remains unaccessed.
- Important artifacts: control documents; uv.lock; environment/quality manifests; DATA_CARD.md; artifacts/data/data_quality.json; artifacts/data/dataset_manifest.json; local-only artifacts/data/split_manifest.parquet
- Active failures: historical data attempt 001 is preserved; model pilot 001 was interrupted before results because decoder-only batching used right padding. Left-padding control added for retry.
- Long-run estimate: batch-size-4 pilot measured 35.57 s for 4 examples/config and projects 6,429.98 s (1.79 h) for 723 examples/config, excluding cold load. Full validation materially answers R007; the evaluator is resumable and flushes every batch.
- Next action: launch resumable `dev-baselines-sf-model-v1-b4` over all 723 VALIDATION examples for M0 and M1; preserve every completed batch and monitor thermals/failures
