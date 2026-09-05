# Project Status

- Current stage: CANDIDATE_SELECTION / corrected one-epoch LoRA finalist ready
- Current Git identity: `main` at `2d848f7`; local identity `Jinwoo Bae <bjw-0907@hanmail.net>`
- Active environment: Windows PowerShell control plane; WSL `.venv` TRAIN/MODEL/ANALYSIS on Python 3.12.14, PyTorch 2.14.0+cu130; serving environment not yet created
- Completed stages: foundation/data; deterministic M0/M1 full validation; canonical paired analysis; M0 selected as strongest development baseline (development evidence only)
- Protected-data access status: GSM8K test technically materialized once by dataset-builder infrastructure while `train` was requested; no test row was selected and no scientific outcome was exposed (Case A, recorded in FINAL_ACCESS_LEDGER.json). MATH-500 remains unaccessed.
- Important artifacts: control documents; uv.lock; environment/quality manifests; DATA_CARD.md; artifacts/data/data_quality.json; artifacts/data/dataset_manifest.json; local-only artifacts/data/split_manifest.parquet
- Active failures: historical failures preserved; KV-cache memory hypothesis rejected (35 MiB difference, throughput thermally/order confounded). Dynamic padding was adopted after the controlled pilot; no unresolved blocker prevents finalist training.
- Long-run estimate: dynamic-padding pilot measured 5.86 s/step, projecting 2,214.5 s (36.9 min) for 378 steps plus setup. It removes only trailing padding and materially answers the sole-finalist quality question; no >1-hour run is planned.
- Next action: checkpoint validated dynamic runtime, then train `finalist-lora-r32-dynamic-378s` from step zero on all 6,046 FIT examples and evaluate the fixed 64-example diagnostic
