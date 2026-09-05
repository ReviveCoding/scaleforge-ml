# Project Status

- Current stage: FROZEN preparation / SF-MODEL-v2 consumed by parser canonicalization defect; SF-MODEL-v3 exact-output rescore required
- Current Git identity: frozen v2 source `3d3e927`; local identity `Jinwoo Bae <bjw-0907@hanmail.net>`
- Active environment: Windows PowerShell control plane; WSL `.venv` TRAIN/MODEL/ANALYSIS on Python 3.12.14, PyTorch 2.14.0+cu130; serving environment not yet created
- Completed stages: foundation/data; deterministic M0/M1 full validation; compact LoRA search; sole finalist training and full paired development validation; complete SF-MODEL-v2 four-cell protected generation matrix; M0 retained and rank-32 LoRA rejected
- Protected-data access status: SF-MODEL-v1 is consumed after an M* MATH-500 parser crash at 208 predictions. SF-MODEL-v2 completed all 3,638 predictions, then read-only analysis found that overflow-safe formatting failed to remove trailing decimal zeros. V2 is consumed; its immutable prediction text will be frozen and re-scored under SF-MODEL-v3. All accesses remain recorded.
- Important artifacts: control documents; uv.lock; environment/quality manifests; DATA_CARD.md; artifacts/data/data_quality.json; artifacts/data/dataset_manifest.json; local-only artifacts/data/split_manifest.parquet
- Active failures: SF-MODEL-v1 overflow crash is preserved. SF-MODEL-v2 is complete but non-final because `format(Decimal("28.00"), "f")` preserved `.00`, changing 19 M0 GSM8K outcomes. Exact generation text reproduced across v1/v2; this is parser drift, not GPU nondeterminism.
- Long-run estimate: SF-MODEL-v3 is a CPU rescore of already complete immutable prediction text and is expected to finish in minutes; no model regeneration is scientifically necessary.
- Next action: checkpoint v2 evidence and corrected parser, freeze v3 source prediction hashes/reference revisions/evaluator, execute protected v3 rescore, and close model qualification
