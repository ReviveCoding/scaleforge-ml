# Requirements Matrix

Statuses: `NOT_STARTED`, `IN_PROGRESS`, `PASS`, `PASS_WITH_LIMITATIONS`, `BLOCKED_EXTERNAL`, `NOT_APPLICABLE`.

| requirement_id | description | implementation_path | evidence_artifact | status |
|---|---|---|---|---|
| R001 | Persist specification and project controls | root control documents | PROJECT_SPEC.md; REQUIREMENTS_MATRIX.md; PROJECT_STATUS.md; DECISIONS.md; EXPERIMENT_PROTOCOL.md; RUNBOOK.md; CLAIM_LEDGER.json | PASS |
| R002 | Record Windows/WSL/GPU preflight before dependency installation | scripts/preflight.ps1; scripts/preflight.sh | artifacts/manifests/environment_preflight.json | PASS |
| R003 | Reproducible separated training and serving environments | pyproject.toml; uv.lock; requirements-serve.lock | artifacts/manifests/train_environment.json; artifacts/manifests/serve_environment.json | PASS |
| R004 | Official model/data provenance, revisions, licenses, hashes | src/scaleforge/data | artifacts/data/dataset_manifest.json; FREEZE_MANIFEST.json | PASS |
| R005 | Deterministic FIT/VALIDATION/POLICY partition and leakage controls | src/scaleforge/data/splits.py | artifacts/data/split_manifest.parquet | PASS |
| R006 | Schema, parser, quality, duplicate, token-length, sequence validation | src/scaleforge/data; src/scaleforge/evaluation | artifacts/data/data_quality.json; DATA_CARD.md | PASS |
| R007 | Strong deterministic M0/M1 baselines and development-only selection | src/scaleforge/modeling; configs/model | artifacts/model/baseline_selection.json | PASS |
| R008 | Compact controlled LoRA selection and exactly one M* | src/scaleforge/training; configs/model | artifacts/model/candidate_selection.json; artifacts/warehouse/model_predictions_candidate_selection.parquet | PASS |
| R009 | Freeze manifest and protected-access ledger | src/scaleforge/protocol | FREEZE_MANIFEST.json; FINAL_ACCESS_LEDGER.json | PASS_WITH_LIMITATIONS |
| R010 | Protected GSM8K/MATH-500 qualification without tuning | scripts/qualify_model.py; scripts/rescore_model_v3.py | artifacts/raw/model; artifacts/analysis/model/model_qualification_sf_model_v3.json | PASS_WITH_LIMITATIONS |
| R011 | Paired quality statistics, transitions, slices, OOD check | src/scaleforge/analysis/quality.py | artifacts/analysis/model/model_qualification_sf_model_v3.json; reports/figures/F01-F03* | PASS |
| R012 | Competent BF16 eager T0 and fixed systems workload | src/scaleforge/benchmarks/training.py | artifacts/raw/training; artifacts/manifests/training_freeze.json | PASS |
| R013 | Profile-led controlled training interventions | src/scaleforge/profiling | artifacts/profiles; reports/figures/F04-F06* | PASS |
| R014 | Replicated synchronized training qualification with telemetry | src/scaleforge/benchmarks | artifacts/analysis/training/training_qualification_sf_train_v1.json; artifacts/warehouse/training_runs.parquet; gpu_telemetry.parquet | PASS_WITH_LIMITATIONS |
| R015 | Separate WSL vLLM installation and inventory | requirements-serve.lock; scripts/setup_serve.sh | artifacts/manifests/serve_environment.txt; artifacts/manifests/serve_environment.json | PASS_WITH_LIMITATIONS |
| R016 | HF/vLLM candidates with apples-to-apples serving corpus | api/hf_server.py; scripts/load_test.py; scripts/run_serving_grid.py | artifacts/raw/serving; artifacts/manifests/serving_corpus.json; artifacts/analysis/serving/serving_qualification_sf_serve_v2.json | PASS_WITH_LIMITATIONS |
| R017 | FastAPI health/generate/metrics and structured logging | api/ | tests/test_api.py; tests/test_hf_server_api.py | PASS |
| R018 | Development SLO proposal frozen before qualification | configs/serving/slo.yaml; scripts/propose_serving_slo.py | artifacts/analysis/serving/hf_baseline_slo_pilot.parquet; configs/serving/slo.yaml | PASS |
| R019 | Load test request telemetry, failures, tails, Pareto and knee | scripts/load_test.py; scripts/run_serving_grid.py; scripts/analyze_serving_qualification.py | artifacts/warehouse/serving_requests.parquet; artifacts/warehouse/serving_runs.parquet; reports/figures/F07-F10* | PASS |
| R020 | Detect GPU count; real distributed metrics or honest external block | scripts/detect_gpu_topology.py; src/scaleforge/distributed | artifacts/manifests/gpu_topology.json; artifacts/analysis/distributed/distributed_status.json; DISTRIBUTED_QUALIFICATION_PENDING.md | BLOCKED_EXTERNAL |
| R021 | DDP launch/test; justified FSDP2 only | scripts/benchmark_ddp_training.py; scripts/distributed_smoke.py; tests/test_distributed_workload.py | artifacts/raw/distributed; DISTRIBUTED_QUALIFICATION_PENDING.md | PASS_WITH_LIMITATIONS |
| R022 | Validated Parquet/DuckDB canonical result warehouse | src/scaleforge/warehouse; scripts/build_warehouse.py | artifacts/analysis/warehouse_integrity.json; artifacts/warehouse/*.parquet; artifacts/warehouse/results.duckdb | PASS |
| R023 | Correct uncertainty, failure, thermal, and order analysis | src/scaleforge/analysis; scripts/analyze_serving_qualification.py | artifacts/analysis/serving/serving_qualification_sf_serve_v2.json; artifacts/failures; reports/figures/F13* | PASS_WITH_LIMITATIONS |
| R024 | Independent critical release gates | src/scaleforge/release; scripts/finalize_release.py | artifacts/warehouse/release_gate_results.parquet; artifacts/analysis/release_decision.json; F14* | PASS |
| R025 | Professional typed package and configuration | pyproject.toml; src/scaleforge; configs | dist/*; mypy/ruff evidence | PASS |
| R026 | Tests and meaningful non-GPU coverage gate | tests/ | artifacts/quality/final_quality.json; coverage.xml | PASS |
| R027 | CPU CI, build/smoke, optional manual GPU workflow | .github/workflows | workflow YAML; local validation | PASS |
| R028 | Minimal end-to-end Streamlit demonstration | ui/ | tests/test_ui_dashboard.py | PASS |
| R029 | Data/model/system/evaluation cards and final report | *.md; reports/ | FINAL_TECHNICAL_REPORT.md | PASS |
| R030 | Claim ledger, resume evidence, interview explanations | CLAIM_LEDGER.json; reports/ | reports/RESUME_EVIDENCE.md; artifacts/analysis/resume_claims.json; INTERVIEW_GUIDE.md | PASS |
| R031 | Data-supported figures and underlying tables | scripts/plot_model_results.py; scripts/plot_training_results.py; scripts/plot_serving_results.py; scripts/finalize_release.py | reports/figures; artifacts/analysis/figure_tables | PASS |
| R032 | Full independent audit and cross-check | scripts/audit_release.py; root docs | FINAL_AUDIT.md; artifacts/audit/final_audit.json | PASS_WITH_LIMITATIONS |
| R033 | Phase checkpoint review and protected-boundary verification | control documents and stage review artifacts | artifacts/audit/checkpoints | PASS |
