# Requirements Matrix

Statuses: `NOT_STARTED`, `IN_PROGRESS`, `PASS`, `PASS_WITH_LIMITATIONS`, `BLOCKED_EXTERNAL`, `NOT_APPLICABLE`.

| requirement_id | description | implementation_path | evidence_artifact | status |
|---|---|---|---|---|
| R001 | Persist specification and project controls | root control documents | PROJECT_SPEC.md; REQUIREMENTS_MATRIX.md; PROJECT_STATUS.md; DECISIONS.md; EXPERIMENT_PROTOCOL.md; RUNBOOK.md; CLAIM_LEDGER.json | PASS |
| R002 | Record Windows/WSL/GPU preflight before dependency installation | scripts/preflight.ps1; scripts/preflight.sh | artifacts/manifests/environment_preflight.json | PASS |
| R003 | Reproducible separated training and serving environments | pyproject.toml; uv.lock; requirements-serve.lock | artifacts/manifests/train_environment.json; serve_environment.txt | IN_PROGRESS |
| R004 | Official model/data provenance, revisions, licenses, hashes | src/scaleforge/data | artifacts/data/dataset_manifest.json; model manifest pending | IN_PROGRESS |
| R005 | Deterministic FIT/VALIDATION/POLICY partition and leakage controls | src/scaleforge/data/splits.py | artifacts/data/split_manifest.parquet | PASS |
| R006 | Schema, parser, quality, duplicate, token-length, sequence validation | src/scaleforge/data; src/scaleforge/evaluation | artifacts/data/data_quality.json; DATA_CARD.md | PASS |
| R007 | Strong deterministic M0/M1 baselines and development-only selection | src/scaleforge/modeling; configs/model | artifacts/model/baseline_selection.json | PASS |
| R008 | Compact controlled LoRA selection and exactly one M* | src/scaleforge/training; configs/model | artifacts/model/candidate_selection.json | IN_PROGRESS |
| R009 | Freeze manifest and protected-access ledger | src/scaleforge/protocol | FREEZE_MANIFEST.json; FINAL_ACCESS_LEDGER.json | NOT_STARTED |
| R010 | Protected GSM8K/MATH-500 qualification without tuning | scripts/qualify_model.py | artifacts/raw/model; artifacts/analysis/model_quality.parquet | NOT_STARTED |
| R011 | Paired quality statistics, transitions, slices, OOD check | src/scaleforge/analysis/quality.py | artifacts/analysis/model_statistics.json; reports/figures/F01-F03* | NOT_STARTED |
| R012 | Competent BF16 eager T0 and fixed systems workload | src/scaleforge/benchmarks/training.py | artifacts/raw/training | NOT_STARTED |
| R013 | Profile-led controlled training interventions | src/scaleforge/profiling | artifacts/profiles; reports/figures/F04-F06* | NOT_STARTED |
| R014 | Replicated synchronized training qualification with telemetry | src/scaleforge/benchmarks | artifacts/warehouse/training_runs.parquet; gpu_telemetry.parquet | NOT_STARTED |
| R015 | Separate WSL vLLM installation and inventory | requirements-serve.lock; scripts/setup_serve.sh | artifacts/manifests/serve_environment.txt | NOT_STARTED |
| R016 | HF/vLLM candidates with apples-to-apples serving corpus | src/scaleforge/serving | artifacts/raw/serving | NOT_STARTED |
| R017 | FastAPI health/generate/metrics and structured logging | api/ | tests/test_api.py; artifacts/logs | IN_PROGRESS |
| R018 | Development SLO proposal frozen before qualification | configs/serving/slo.yaml | artifacts/serving/slo_freeze.json | NOT_STARTED |
| R019 | Load test request telemetry, failures, tails, Pareto and knee | src/scaleforge/loadtest; analysis | artifacts/warehouse/serving_requests.parquet; reports/figures/F07-F10* | NOT_STARTED |
| R020 | Detect GPU count; real distributed metrics or honest external block | src/scaleforge/distributed | artifacts/manifests/gpu_topology.json; DISTRIBUTED_QUALIFICATION_PENDING.md | IN_PROGRESS |
| R021 | DDP launch/test; justified FSDP2 only | scripts/distributed; tests | artifacts/raw/distributed; distributed runbook | NOT_STARTED |
| R022 | Validated Parquet/DuckDB canonical result warehouse | src/scaleforge/warehouse | artifacts/warehouse/*.parquet; artifacts/warehouse/results.duckdb | NOT_STARTED |
| R023 | Correct uncertainty, failure, thermal, and order analysis | src/scaleforge/analysis | artifacts/analysis; reports/figures/F13* | NOT_STARTED |
| R024 | Independent critical release gates | src/scaleforge/release | artifacts/warehouse/release_gate_results.parquet; F14* | NOT_STARTED |
| R025 | Professional typed package and configuration | pyproject.toml; src/scaleforge; configs | dist/*; mypy/ruff evidence | IN_PROGRESS |
| R026 | Tests and meaningful non-GPU coverage gate | tests/ | artifacts/quality/foundation_quality.json; coverage.xml | PASS |
| R027 | CPU CI, build/smoke, optional manual GPU workflow | .github/workflows | workflow YAML; local validation | PASS |
| R028 | Minimal end-to-end Streamlit demonstration | ui/ | ui smoke evidence | NOT_STARTED |
| R029 | Data/model/system/evaluation cards and final report | *.md; reports/ | FINAL_TECHNICAL_REPORT.md | NOT_STARTED |
| R030 | Claim ledger, resume evidence, interview explanations | CLAIM_LEDGER.json; reports/ | reports/RESUME_EVIDENCE.md; artifacts/analysis/resume_claims.json; INTERVIEW_GUIDE.md | NOT_STARTED |
| R031 | Data-supported figures and underlying tables | src/scaleforge/reporting | reports/figures; artifacts/analysis/figure_tables | NOT_STARTED |
| R032 | Full independent audit and cross-check | audit tooling; root docs | FINAL_AUDIT.md | NOT_STARTED |
| R033 | Phase checkpoint review and protected-boundary verification | scripts/checkpoint_review.py | artifacts/audit/checkpoints | IN_PROGRESS |
