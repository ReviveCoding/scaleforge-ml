# ScaleForge-ML Project Specification

## Purpose

ScaleForge-ML is an evidence-first portfolio project asking whether a Transformer workload can be improved in model quality and systems performance while preserving reproducibility, quality non-regression, reliability, and defensible release decisions. The required workload is `Qwen/Qwen2.5-1.5B-Instruct`; the primary dataset is `openai/gsm8k` (`main`) and the protected OOD set is `HuggingFaceH4/MATH-500`.

## Scope and deliverables

The project covers public-data acquisition and provenance, deterministic leakage-safe preprocessing, strong zero/few-shot baselines, compact LoRA selection, protected evaluation, BF16 training profiling and optimization, HF/vLLM serving and load testing, distributed launch readiness and honest hardware qualification, a Parquet/DuckDB result warehouse, paired statistics, Pareto/release analysis, typed Python packaging, CPU CI, API, a minimal demonstration UI, technical documentation, traceable resume claims, and an independent final audit.

The repository uses `configs/`, `src/scaleforge/`, `scripts/`, `tests/`, `api/`, `ui/`, `artifacts/`, `reports/`, and `.github/workflows/`. Raw model weights and raw large datasets are never committed.

## Operating constraints

- Windows repository: `C:\Users\bjw-0\Downloads\scaleforge-ml`; WSL view: `/mnt/c/Users/bjw-0/Downloads/scaleforge-ml`.
- Linux-native CUDA, training, and vLLM work runs under WSL2. Native-Windows vLLM is forbidden.
- No driver changes, global WSL changes, paid APIs/cloud GPUs, remote pushes, external releases, or unrelated personal-file access.
- Use separate `.venv-train` and `.venv-serve` WSL environments unless compatibility is demonstrated. vLLM gets its compatible fresh stack. Record exact inventories.
- Use local caches by default. Any external ext4 cache must be approved, recorded, non-authoritative, and sync only evidence back.
- Before an experiment expected to exceed one hour, measure a pilot, estimate time, record it, and confirm its necessity. Never run competing heavy GPU jobs concurrently.

## Scientific protocol

Every experiment follows `PILOT -> DEVELOPMENT -> CANDIDATE_SELECTION -> FROZEN -> QUALIFICATION -> ANALYSIS -> CLOSED`. A qualified identity never moves backward. Any scientific change after protected results creates a new identity.

GSM8K official train is deterministically hash-partitioned into FIT, VALIDATION, and POLICY. GSM8K official test and MATH-500 are protected and cannot influence prompts, parsers, hyperparameters, selection, or thresholds. Every protected access is recorded. Before access, Git/source/config/checkpoint/prompt/parser/tokenizer/generation/data revisions/release criteria are hashed in `FREEZE_MANIFEST.json`.

Infrastructure retries follow the exposure rule: without outcomes, retry the identical scientific configuration while preserving failure evidence; with outcomes, only exact recovery/reproducibility is allowed; any scientific change requires a new identity.

## Data contract

Each GSM8K row contains `example_id`, `question`, `raw_solution`, `reference_final_answer`, `normalized_final_answer`, prompt/reference token counts, `split_role`, and `source_revision`. Validation fails closed on null/blank/malformed rows, duplicate or normalized-duplicate leakage, split overlap, parser failure, sequence-limit violations, and fingerprint mismatch. No example is silently truncated. Length p50/p90/p95/p99/max plus documented headroom determines sequence length. Deliver `DATA_CARD.md`, `data_quality.json`, `dataset_manifest.json`, and `split_manifest.parquet`.

## Model-quality contract

- M0: frozen model, deterministic zero-shot reasoning prompt.
- M1: same model, fixed few-shot prompt selected only on development data.
- The strongest frozen baseline is chosen by a predeclared development rule.
- Deterministic generation explicitly freezes sampling off, max tokens, chat template, EOS/pad handling, parser, and generation config.
- Compact BF16 response-only LoRA development considers ranks 8/16/32, learning rates near 1e-4/2e-4, and attention versus attention+MLP targets through pilots then few full-validation finalists. Exactly one M* challenger is frozen.
- Protected primary comparison is M* versus the strongest frozen pretrained baseline using EM, percentage-point delta, paired transitions, paired bootstrap CI, exact McNemar, structural slices, and MATH-500 non-regression. If LoRA lacks robust benefit, retain the simpler baseline.

## Training-systems contract

Freeze checkpoint/data workload after model selection. T0 is a competent BF16 eager PyTorch/DataLoader baseline with fixed global batch, sequence length, and measured steps. Profile before intervention. Test one controlled intervention at a time (input pipeline, bucketing/packing, batch/accumulation, compile, or justified checkpointing), re-profile, and keep/reject. Capture CPU/CUDA time, waits/transfers, top ops, graph breaks, VRAM, synchronized step time, tokens/s, utilization, and GPU telemetry. Separate compile cold cost from steady state and calculate break-even. Finalists use warmup, correct synchronization, at least three (prefer five) replicates, balanced/randomized order, preserved failures, and thermal/order checks.

## Serving contract

Use identical checkpoint, prompt, tokenizer, deterministic decoding, evaluator, and request corpus for S0 HF BF16, optional stable S1 HF compile, S2 vLLM, and development-selected S3. Provide FastAPI `/health`, `/generate`, and `/metrics` with structured logs. Use development-only GSM8K requests, characterize concurrency from 1/2/4/8/16/32 and refine only around saturation. Preserve request-level latency, token, status/failure, queue, TTFT, TPOT/ITL, E2E, run/replicate/config, and GPU state.

Freeze project-specific interactive SLOs from the strong serving baseline pilot before candidate qualification, explicitly noting they are not Google SLOs. Report successful requests/s, output tokens/s, tail metrics, failures/OOM/timeouts, VRAM/utilization, max SLO-compliant throughput, Pareto frontier, dominated points, and saturation knee. Select an operating point, not raw maximum throughput.

## Distributed contract

Detect physical CUDA GPU count. With fewer than two, fabricate no metrics: implement/test launch code, mark numerical qualification `BLOCKED_EXTERNAL`, and provide `DISTRIBUTED_QUALIFICATION_PENDING.md`. If authorized hardware exists, primary scaling is fixed-global-workload 1-vs-2 GPU DDP with speedup, efficiency, communication, sync, throughput, and memory. FSDP2 runs only for a real memory/scaling question; serving prefers replicas for this fitting model.

## Evidence, statistics, and release

Evidence flows `RAW -> VALIDATION -> CANONICALIZATION -> ANALYSIS TABLES -> STATISTICS -> RELEASE DECISION -> PRESENTATION`. Canonical Parquet tables and DuckDB checks verify identities/artifacts, reject duplicate/incomplete runs, normalize units/failures, separate warmup, preserve failures, and safely join telemetry.

Primary analyses use paired methods for model outcomes, replicate uncertainty for training, hierarchical/replicate-aware bootstrap for skewed serving latency, Pareto/knee analysis, and speedup/efficiency for distributed. Naive row-level t-tests are forbidden. Thermal/order effects and failure taxonomies are explicit. Explanatory regressions are diagnostic only.

Independent gates decide MODEL, TRAINING SYSTEM, SERVING, DISTRIBUTED, and REPRODUCIBILITY as PASS/REVIEW/BLOCK (plus permitted distributed external states). No weighted vanity score. Complexity is adopted only when evidence supports it.

## Engineering and reporting

Typed/config-driven code is tested for schemas, leakage, parsing, configs, fixtures, telemetry, canonicalization, statistics, API, artifacts, and gates. Ruff, mypy, pytest/coverage, package build, CPU CI, fixture smoke, and optional manual GPU CI are required. A Streamlit demo follows the evidence core.

Required documentation: README, data/model/system/evaluation cards, final report, resume evidence, interview guide, final audit, runbook, decisions, status, matrices, manifests, ledgers, tables, and data-supported figures F01-F14 as applicable. Every numeric public claim is registered in `CLAIM_LEDGER.json` and checked against canonical artifacts. Negative results are retained and prominent.

## Definition of done

Completion requires all locally feasible implementation, reproducible environments, manifests, preprocessing, strong baselines, selection and protected evidence, training/serving qualification, request telemetry, post-processing, statistics, failure/release analyses, tests/CI, documentation, claims, interview material, and independent audit. If only genuine multi-GPU execution remains, status may be `COMPLETE_WITH_EXTERNAL_DISTRIBUTED_PENDING` with exact commands and hardware requirements.
