# Project Status

- Current stage: DEVELOPMENT / training systems baseline and profile-led interventions
- Current Git identity: frozen v3 source `553796b`; local identity `Jinwoo Bae <bjw-0907@hanmail.net>`
- Active environment: Windows PowerShell control plane; WSL `.venv` TRAIN/MODEL/ANALYSIS on Python 3.12.14, PyTorch 2.14.0+cu130; serving environment not yet created
- Completed stages: foundation/data; deterministic M0/M1 validation; compact LoRA selection; complete protected generation; SF-MODEL-v3 hash-frozen evaluator-only rescore and paired analysis; M0 retained and rank-32 LoRA rejected; model figures F01-F03
- Protected-data access status: CLOSED for scientific model work. V1 and V2 failures remain preserved. V3 re-scored complete immutable V2 outputs under a corrected frozen parser and passed cardinality/pairing gates. No further model/prompt/parser tuning is permitted under these identities.
- Important artifacts: control documents; uv.lock; environment/quality manifests; DATA_CARD.md; artifacts/data/data_quality.json; artifacts/data/dataset_manifest.json; local-only artifacts/data/split_manifest.parquet
- Active failures: historical V1 overflow crash and V2 trailing-zero canonicalization defect are preserved and non-final. No active model qualification failure.
- Long-run estimate: the six-run balanced T0/T* qualification matrix is projected at approximately 40 minutes from measured pilots; every individual run is below one hour and directly resolves R014.
- Next action: checkpoint and freeze the six-run T0 versus dynamic-padding+compile matrix, then execute three fresh-cache replicates per configuration in balanced order
