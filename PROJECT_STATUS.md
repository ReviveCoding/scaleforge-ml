# Project Status

- Current stage: CANDIDATE_SELECTION / one-epoch rank-32 attention+MLP LoRA finalist
- Current Git identity: unborn `main` branch; commit SHA unavailable; local identity `Jinwoo Bae <bjw-0907@hanmail.net>`
- Active environment: Windows PowerShell control plane; WSL `.venv` TRAIN/MODEL/ANALYSIS on Python 3.12.14, PyTorch 2.14.0+cu130; serving environment not yet created
- Completed stages: foundation/data; deterministic M0/M1 full validation; canonical paired analysis; M0 selected as strongest development baseline (development evidence only)
- Protected-data access status: GSM8K test technically materialized once by dataset-builder infrastructure while `train` was requested; no test row was selected and no scientific outcome was exposed (Case A, recorded in FINAL_ACCESS_LEDGER.json). MATH-500 remains unaccessed.
- Important artifacts: control documents; uv.lock; environment/quality manifests; DATA_CARD.md; artifacts/data/data_quality.json; artifacts/data/dataset_manifest.json; local-only artifacts/data/split_manifest.parquet
- Active failures: historical failures preserved; LR2e-4 attention LoRA ranks 8/16 regressed on matched 64-example diagnostics (73.44% M0 to 35.94%/32.81%) and are eliminated.
- Long-run estimate: rank-32 attention+MLP measured 6.54 s/step; 378 steps (approximately one epoch at global batch 16) project 2,473 s (41.2 min) plus preprocessing/load. The run materially determines whether the least-regressive pilot can recover; no >1-hour run is planned.
- Next action: checkpoint the four-cell pilot selection, run the sole finalist for 378 steps on all 6,046 FIT examples, then apply the same 64-example diagnostic before authorizing full validation
