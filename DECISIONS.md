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
| D014 | 2026-09-05 | Reject batch size 4 as the final baseline evaluation runtime | Although it projected 1.79 h, full-run telemetry showed only 23-37% sampled utilization and 3,444 MiB VRAM; 84 partial M0 rows were preserved and not used for quality selection | SUPERSEDED |
| D015 | 2026-09-05 | Evaluate batch size 16 under the same scientific workload | The intervention targets observed GPU underutilization and applies equally to M0 and M1; acceptance depends on reliability and runtime, not quality | ADOPTED |
| D016 | 2026-09-05 | Adopt batch size 16 for full M0/M1 development validation | The 16-example/config pilot completed reliably at 4,602 MiB VRAM and reduced projected generation time from 6,430 s to 2,493 s; pilot quality was excluded from the runtime decision | ADOPTED |
| D017 | 2026-09-05 | Retain sequence length 640 after measuring actual chat-formatted FIT sequences | Across 6,046 examples, formatted max was 574; 640 supplies documented headroom without truncation | ADOPTED |
| D018 | 2026-09-05 | Reject LoRA microbatch 4 and test microbatch 2 with accumulation 8 | The 20-step run completed but allocated 15,984 MiB and sampled 16,038 MiB used, leaving inadequate reliability headroom; global batch remains 16 | ADOPTED |
| D019 | 2026-09-05 | Adopt LoRA microbatch 2 / accumulation 8 | At the same global batch 16 it improved pilot throughput 28.6% and reduced peak allocated memory 41.1% versus microbatch 4 | ADOPTED |
| D020 | 2026-09-05 | Advance only rank-32 attention+MLP LR 1e-4 to one-epoch-equivalent training | All 20-step adapters regressed on the matched 64-example diagnostic; rank 32 had the least regression and lowest final pilot loss, so it alone merits more compute before rejecting LoRA | ADOPTED |
