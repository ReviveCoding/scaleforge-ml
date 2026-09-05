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
| D010 | 2026-09-04 | Replace `datasets.load_dataset(split="train")` acquisition with direct immutable train-parquet retrieval | The Datasets builder materialized GSM8K test despite the requested split, violating the intended protected-data boundary at infrastructure level | ADOPTED |
| D011 | 2026-09-05 | Require left padding for decoder-only batched baseline generation | Pilot 001 exposed Transformers' correctness warning under the tokenizer's right-padding default | ADOPTED |
| D012 | 2026-09-05 | Treat the first four-example M0/M1 result as instrumentation-only | Four paired examples cannot support prompt selection; the measured 3.33-hour projection is used only for compute planning | ADOPTED |
| D013 | 2026-09-05 | Record `/home/bjw-0/.cache/huggingface/xet` as an incidental non-authoritative transport cache from the first model download | `cache_dir` did not fully redirect Xet metadata/chunks; subsequent commands explicitly set project-local HF/Xet paths | ADOPTED |
| D014 | 2026-09-05 | Freeze baseline evaluation batch size at 4 | It reduced projected runtime from 3.33 h to 1.79 h, fits memory, and will be applied equally; pilot comparisons showed BF16 greedy outputs are batch-shape sensitive, so execution shape must remain fixed | ADOPTED |
