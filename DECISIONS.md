# Decision Log

| ID | Date | Decision | Rationale | Status |
|---|---|---|---|---|
| D001 | 2026-09-04 | Use `main` as the initial branch | Required for a repository without meaningful history | ADOPTED |
| D002 | 2026-09-04 | Keep protected GSM8K test and MATH-500 inaccessible until a complete freeze manifest exists | Prevents test leakage and post-outcome tuning | ADOPTED |
| D003 | 2026-09-04 | Permit no numeric public claims until canonical qualification artifacts support them | Prevents aspirational or pilot results from becoming claims | ADOPTED |
| D004 | 2026-09-04 | Use WSL2 for all CUDA/vLLM work and separate train/serve environments | Matches platform requirements and isolates vLLM dependencies | ADOPTED |
| D005 | 2026-09-04 | Treat unavailable second physical GPU as an external qualification block, never as permission to synthesize scaling results | Protects distributed-claim validity | ADOPTED |
| D006 | 2026-09-04 | Use `uv` with separate WSL virtual environments | `uv 0.12.7` is already available and the serving stack needs isolation | ADOPTED |
| D007 | 2026-09-04 | Mark local numerical distributed qualification `BLOCKED_EXTERNAL` after launch-code validation | Preflight found exactly one physical CUDA GPU | ADOPTED |
| D008 | 2026-09-04 | Lock the training environment on Python 3.12.14 and PyTorch 2.14.0+cu130 | The resolved stack passes CUDA visibility and foundation quality gates | ADOPTED |
| D009 | 2026-09-04 | Retain the project-local WSL environment despite drvfs copy overhead | It preserves the default workspace-local contract; timed workloads can use an approved ext4 cache only if pilots prove necessity | ADOPTED |
