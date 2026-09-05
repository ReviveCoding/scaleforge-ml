# Project Status

- Current stage: MODEL_DEVELOPMENT / compact LoRA pilot implementation
- Current Git identity: unborn `main` branch; commit SHA unavailable; local identity `Jinwoo Bae <bjw-0907@hanmail.net>`
- Active environment: Windows PowerShell control plane; WSL `.venv` TRAIN/MODEL/ANALYSIS on Python 3.12.14, PyTorch 2.14.0+cu130; serving environment not yet created
- Completed stages: foundation/data; deterministic M0/M1 full validation; canonical paired analysis; M0 selected as strongest development baseline (development evidence only)
- Protected-data access status: GSM8K test technically materialized once by dataset-builder infrastructure while `train` was requested; no test row was selected and no scientific outcome was exposed (Case A, recorded in FINAL_ACCESS_LEDGER.json). MATH-500 remains unaccessed.
- Important artifacts: control documents; uv.lock; environment/quality manifests; DATA_CARD.md; artifacts/data/data_quality.json; artifacts/data/dataset_manifest.json; local-only artifacts/data/split_manifest.parquet
- Active failures: historical data attempt 001 is preserved; model pilot 001 was interrupted before results because decoder-only batching used right padding. Left-padding control added for retry.
- Long-run estimate: batch-size-16 measured 55.16 s for 16 examples/config and projects 2,492.71 s (41.55 min) for 723 examples/config, excluding cold load. It completed without OOM at 4,602 MiB sampled VRAM; no >1-hour launch is now planned.
- Next action: implement response-only BF16 LoRA training with fail-closed 640-token handling, measure a short pilot, and estimate compact candidate costs before any >1-hour training run
