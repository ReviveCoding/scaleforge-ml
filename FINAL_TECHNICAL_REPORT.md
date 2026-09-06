# ScaleForge-ML Final Technical Report

## Executive result

ScaleForge-ML answers its central question with a qualified yes for systems performance and a no for the tested adaptation: a frozen pretrained Qwen2.5-1.5B-Instruct baseline was retained because LoRA materially reduced GSM8K and MATH-500 exact match; profile-led dynamic padding plus compilation improved the fixed training workload; vLLM improved SLO-compliant serving throughput but exposed an unresolved shutdown defect; and multi-GPU numbers were withheld because the machine has one physical GPU.

The project status is `COMPLETE_WITH_EXTERNAL_DISTRIBUTED_PENDING`. Independent gates are MODEL PASS, TRAINING SYSTEM PASS, SERVING REVIEW, DISTRIBUTED BLOCKED_EXTERNAL, and REPRODUCIBILITY REVIEW.

## Problem and requirements

The work tests whether a Transformer workload can improve model quality and systems performance while preserving leakage controls, reproducibility, reliability, and defensible release decisions. Requirements cover official data provenance, strong deterministic baselines, compact LoRA selection, protected qualification, profile-led GPU optimization, production-style inference, request-level load evidence, paired statistics, a canonical result warehouse, tests/CI, communication artifacts, and real rather than simulated distributed claims. `REQUIREMENTS_MATRIX.md` maps each requirement to code and evidence.

## Environment

Preflight ran before dependency installation. The controller is Windows 10.0.26200 with PowerShell 7.6.5 and Git 2.54.0. WSL2 is Ubuntu 22.04.5, kernel 6.6.114.1, on an i9-13900HX with 32 logical CPUs, 15 GiB visible RAM, and 4 GiB swap. One NVIDIA RTX 4090 Laptop GPU exposes 16,376 MiB VRAM with driver 610.62. Training/model/analysis use Python 3.12.14 and PyTorch 2.14.0+cu130 in `.venv`; serving uses Python 3.12.14, PyTorch 2.13.0+cu132, and vLLM 0.28.0 in `.venv-serve`. Exact inventories and locks are under `artifacts/manifests`, `uv.lock`, and `requirements-serve.lock`.

## Data engineering and leakage control

Official immutable sources are GSM8K `main` revision `740312add88f781978c0658806c59bc2815b9866` and MATH-500 for protected OOD evaluation. The 7,473 GSM8K training examples are assigned by deterministic SHA-256 intervals to FIT 6,046, VALIDATION 723, and POLICY 704. IDs incorporate source, revision, and normalized question. Nulls, blanks, malformed markers, duplicates, normalized duplicates, and cross-role overlap are fail-closed checks.

Prompt/reference combined token length has p50 168, p90 275, p95 312, p99 390, and max 536. A 640-token training limit provides 10% headroom rounded to 128 and truncates no example. Protected official test and MATH-500 were excluded from prompt, parser, LoRA, and SLO selection. One early library acquisition materialized a test cache despite a requested train split; no model outcome was exposed. The incident remains logged and acquisition changed to immutable split-specific Parquet retrieval.

## Baselines and modeling

M0 is the frozen model with deterministic zero-shot reasoning. M1 uses the same model and a fixed few-shot prompt selected on development data. M0 won the predeclared full-validation rule and became the strongest baseline. Generation explicitly freezes sampling off, the chat template, EOS/pad handling, token cap, and parser.

LoRA search was deliberately compact: ranks 8/16/32, approximately 1e-4/2e-4 learning rates, and attention-only versus attention+MLP targets. BF16 response-only training used pilots to remove poor candidates. One rank-32 attention+MLP finalist advanced. On 723 validation examples it scored 58.51% against M0's 78.70%, a -20.19 pp paired difference. It was frozen as M* rather than replaced with a favorable post hoc candidate.

## Protected model results

All 1,319 GSM8K test and 500 MATH-500 examples were evaluated for both frozen configurations. GSM8K M0 exact match was 66.64%; M* was 49.43%. The -17.21 pp paired difference has a 95% bootstrap interval of -20.24 to -14.18 and exact McNemar p=4.23e-27. Transitions were 116 candidate-only, 343 baseline-only, 536 both-correct, and 324 both-wrong. On MATH-500 M0 reached 29.80% and M* 18.80%; delta -11.00 pp, interval -14.80 to -7.00. M* fails both primary and OOD gates and is rejected.

Protected execution exposed two evaluator defects. SF-MODEL-v1 overflowed in Decimal normalization after outcomes were exposed, so it closed and v2 changed only the versioned parser. Complete v2 analysis then found trailing zeros were not canonically normalized. SF-MODEL-v3 froze that correction and re-scored byte-identical v2 generated text. No prompt, checkpoint, threshold, or prediction changed. This produces valid qualified evaluator results with an explicit limitation; it does not justify calling the protected process clean on first pass.

## Training systems engineering

The fixed systems workload keeps checkpoint, data stream, global batch, measured steps, and non-padding token count constant. T0 is BF16 eager PyTorch with a competent DataLoader and fixed width—not an intentionally weak FP32 baseline. PyTorch Profiler showed that fixed padding inflated GEMM/attention work because median formatted length was far below 640. Dynamic per-batch removal of trailing padding preserved every token and reduced work. `torch.compile` was evaluated only after that hypothesis succeeded.

Final qualification used balanced six-run order, warmup, synchronized CUDA, fresh compiler caches, ten measured steps, GPU telemetry, and three paired replicates. Each run processed 35,382 non-padding tokens. T0 median steady throughput was 987.84 tokens/s; T4 dynamic+compile was 4,430.60, a median paired +348.51% with interval +157.26% to +588.82%. Peak allocated VRAM fell 34.29%, from 12,108 to 7,956 MiB.

Compile startup is not hidden: roughly 352-491 seconds, with pairwise break-even 75-270 steps and median 173. For the fixed 378-step target, measured-phase projections including cold cost are 1,360.76 seconds for T0 and 790.98 for T4 (-41.87%). This is a projection, not a directly timed end-to-end 378-step run. With n=3, run order and thermal associations are material, so the broad paired interval and hardware scope accompany the claim.

## Serving system engineering

The common request path uses the selected M0 checkpoint, identical prompt/tokenizer/parser, deterministic decoding, a 512-token cap selected before candidate qualification, and a fixed 64-question POLICY corpus. S0 is HF Transformers BF16; S2 is vLLM continuous batching. The default vLLM V2 runner failed WSL unified virtual addressing before model outcomes. The isolated environment's supported V1 compatibility mode passed deterministic smoke and became the documented candidate.

An HF development pilot set a project-specific interactive SLO before candidate outcomes: p95 TTFT <=500 ms, p95 TPOT <=140 ms, p95 E2E <=33 s, error <=1%, and paired quality/parser non-regression. These are ScaleForge experimental SLOs, not Google SLOs. Qualification covered two runtimes, concurrency 1/2/4/8/16/32, three fresh-server replicates, balanced point order, and 2,304 measured requests. Failures remain in throughput and reliability denominators.

Both runtimes' maximum SLO-compliant point was concurrency 2. S0 delivered 0.1208 requests/s and 32.30 output tokens/s; S2 delivered 0.2070 and 56.05. Paired throughput improved 71.39% (three-replicate bootstrap interval 51.01%-89.52%). S2 p95 TPOT was 39.16 versus 112.51 ms and p95 E2E 16.77 versus 25.70 seconds. Its p95 TTFT worsened from 86.86 to 443.24 ms; the hierarchical interval, 404.74-527.72 ms, crosses the frozen bound. All requests succeeded and serving exact match did not regress descriptively, but the repeated 64 prompts are not 192 independent quality cases.

The throughput/tail-latency frontier prevents selecting raw maximum throughput. Higher concurrencies become dominated or violate TTFT; saturation knees are concurrency 8 for HF and 16 for vLLM. Concurrency 2 is selected. SERVING remains REVIEW because each post-measurement vLLM shutdown force-killed an idle EngineCore and leaked a semaphore. Request-path success cannot be promoted into an unqualified lifecycle-reliability claim.

## Distributed status

Topology detection confirms one physical CUDA GPU. The DDP implementation preserves a fixed global workload and example indices across 1/2 GPUs. A two-rank CPU/Gloo smoke passed initialization and all-reduce sum 3.0 on both ranks. It proves launch plumbing, not CUDA scaling. Consequently speedup, scaling efficiency, NCCL overhead, and memory-per-GPU are absent and prohibited. FSDP2 is NOT_APPLICABLE because the 1.5B workload fits and no real sharding question exists. Replica serving, rather than artificial tensor parallelism, is the appropriate future capacity experiment. `DISTRIBUTED_QUALIFICATION_PENDING.md` supplies exact authorized two-GPU steps.

## Warehouse, statistics, and failure analysis

Raw scripts never author public claims directly. Validation produces canonical Parquet and DuckDB tables, then analysis and independent gates feed the claim ledger. The warehouse contains 47 unique run identities, 3,638 protected predictions, 6 training runs/72 steps, 36 serving runs/2,592 request rows, 29,441 telemetry rows, 2 distributed smoke rows, 21 failure events, and 5 gates. Hashes and counts are in `warehouse_integrity.json`. It rejects incomplete matrices, duplicate identities, inconsistent failure taxonomy, mismatched tokens, and unsafe telemetry joins.

Failures include protected parser defects, profiler/instrumentation failures, the WSL vLLM UVA incompatibility, a pre-request config-ID rejection, and the vLLM shutdown defect. They are preserved even when unfavorable. Model inference uses paired bootstrap and exact McNemar. Training uses paired replicate bootstrap. Serving uses replicate medians and hierarchical request-within-replicate bootstrap, not a naive test over correlated request rows. Thermal, clock, power, and order associations are descriptive diagnostics, never post hoc corrections.

## Release decisions and Pareto judgment

| Gate | Decision | Rationale |
|---|---|---|
| MODEL | PASS | Retain M0; M* is decisively worse on primary and OOD data |
| TRAINING SYSTEM | PASS | T4 clears useful gain, memory, reliability, and target-duration gates |
| SERVING | REVIEW | S2 c2 wins SLO-compliant throughput, but shutdown reliability is unresolved |
| DISTRIBUTED | BLOCKED_EXTERNAL | One physical GPU; implementation evidence is not numerical qualification |
| REPRODUCIBILITY | REVIEW | Locks/hashes/CI pass; protected evaluator required versioned recovery |

No weighted score hides a failed critical dimension. Simplicity wins when complexity lacks evidence: pretrained M0 over LoRA; no unjustified S1/S3; no FSDP2; no concurrency beyond the SLO-compliant point.

## Reproduction and claim boundary

The repository contains exact environment locks, manifests, frozen YAML, source hashes, raw failure evidence, canonical tables, figure source tables, tests, CPU CI, build workflows, FastAPI service, and a read-only Streamlit dashboard. The protected model protocol is CLOSED and must not be tuned or regenerated under its identity. Rebuild validation with `scripts/build_warehouse.py` and release outputs with `scripts/finalize_release.py`; execute the full CPU quality gate before citing results.

Only `CLAIM_LEDGER.json` and `artifacts/analysis/resume_claims.json` authorize numeric claims. They require the single-laptop, n=3, cold-cost, repeated-corpus, parser-history, shutdown, and no-multi-GPU qualifications described above.
