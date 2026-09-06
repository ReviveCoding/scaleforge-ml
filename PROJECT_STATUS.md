# Project Status

- Current stage: CLOSED after final independent audit
- Project status: COMPLETE_WITH_EXTERNAL_DISTRIBUTED_PENDING
- Current Git identity: branch `main`; local identity `Jinwoo Bae <bjw-0907@hanmail.net>`; immutable scientific source SHAs are recorded separately below
- Frozen source identities: model `553796b`; training `0008038`; serving `04f740b`; serving qualification `9a02578`
- Active environment: Windows PowerShell control plane; WSL `.venv` TRAIN/MODEL/ANALYSIS on Python 3.12.14 and PyTorch 2.14.0+cu130; isolated WSL `.venv-serve` on Python 3.12.14, PyTorch 2.13.0+cu132, and vLLM 0.28.0
- Completed stages: foundation, data, baseline selection, compact LoRA selection, protected model qualification, profile-led training qualification, HF/vLLM serving qualification, local distributed launch validation, canonical warehouse, release gates, figures, UI, cards/report, claim evidence, interview guide, package build, and final audit
- Protected-data access status: CLOSED. V1/V2 failures and v3 evaluator-only recovery are preserved. No further tuning or scientific change is permitted under these identities.
- Important artifacts: `artifacts/analysis/model/model_qualification_sf_model_v3.json`; `artifacts/analysis/training/training_qualification_sf_train_v1.json`; `artifacts/analysis/serving/serving_qualification_sf_serve_v2.json`; `artifacts/analysis/warehouse_integrity.json`; `artifacts/analysis/release_decision.json`; `artifacts/audit/final_audit.json`
- Active failures: no open local implementation failure. Historical parser/profiler failures remain evidence. vLLM V2 WSL UVA incompatibility and repeatable V1 EngineCore force-kill/leaked-semaphore shutdown remain unresolved environmental/lifecycle limitations.
- MODEL decision: PASS, select M0 and reject rank-32 attention+MLP LoRA
- TRAINING SYSTEM decision: PASS, select T4 dynamic padding plus `torch.compile`, with n=3 thermal/order qualifier and cold-cost break-even
- SERVING decision: REVIEW, select experimental S2 vLLM V1-runner at concurrency 2; withhold production-readiness claim
- DISTRIBUTED status: BLOCKED_EXTERNAL on one physical CUDA GPU; CPU/Gloo launch passes; FSDP2 NOT_APPLICABLE; no numeric scaling claim
- REPRODUCIBILITY decision: REVIEW because protected evaluator recovery was required, despite passing locks, hashes, canonicalization, tests, and CI
- Local quality status: Ruff PASS; strict mypy PASS; pytest 60 PASS with 81.56% coverage (80% gate); wheel/sdist build PASS; Streamlit 1.63.0 smoke PASS
- Warehouse status: PASS; 47 unique runs, 3,638 protected predictions, 6 training runs, 36 serving runs, 2,304 measured serving requests, 29,441 telemetry rows, 21 preserved failure events, and 5 independent gates
- Next action: optional authorized execution of `DISTRIBUTED_QUALIFICATION_PENDING.md` on a real two-GPU CUDA host; otherwise no local work remains
