# Project Status

- Current stage: QUALIFICATION / SF-MODEL-v1 consumed by protected parser failure; SF-MODEL-v2 required
- Current Git identity: frozen v1 source `ca42a6f`; local identity `Jinwoo Bae <bjw-0907@hanmail.net>`
- Active environment: Windows PowerShell control plane; WSL `.venv` TRAIN/MODEL/ANALYSIS on Python 3.12.14, PyTorch 2.14.0+cu130; serving environment not yet created
- Completed stages: foundation/data; deterministic M0/M1 full validation; compact LoRA search; sole finalist training and full paired development validation; M0 retained and rank-32 LoRA frozen as M* challenger (development evidence only)
- Protected-data access status: SF-MODEL-v1 completed paired GSM8K qualification and M0 MATH-500; M* MATH-500 exposed 208 predictions before a parser crash. V1 is consumed and cannot be patched/resumed. All accesses are recorded in FINAL_ACCESS_LEDGER.json.
- Important artifacts: control documents; uv.lock; environment/quality manifests; DATA_CARD.md; artifacts/data/data_quality.json; artifacts/data/dataset_manifest.json; local-only artifacts/data/split_manifest.parquet
- Active failures: `qual-sf-model-v1-math500-mstar` crashed at 208/500 after `Decimal.quantize(1)` raised `InvalidOperation` on an extreme integer; partial protected evidence is preserved and non-claimable.
- Long-run estimate: each v2 protected run is below one hour based on v1 measurements (GSM8K 46–56 minutes; MATH-500 27 minutes for M0, with M* projected below one hour despite long outputs).
- Next action: checkpoint v1 evidence, implement/test overflow-safe integral normalization under SF-MODEL-v2, create a new freeze, and rerun the complete four-cell protected matrix
