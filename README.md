# ScaleForge-ML

ScaleForge-ML is an evidence-first Transformer training and serving project built around one question: can model quality and systems performance improve together without sacrificing reproducibility, reliability, or honest release decisions? It exercises the complete local lifecycle for `Qwen/Qwen2.5-1.5B-Instruct` on GSM8K and MATH-500: provenance-controlled data, deterministic baselines, LoRA adaptation, protected evaluation, PyTorch profiling and compilation, FastAPI/vLLM serving, load testing, statistical analysis, and fail-closed evidence release.

## Strongest verified results

- Model: retain frozen M0. On all 1,319 GSM8K test examples M0 reached **66.64% exact match**; rank-32 attention+MLP LoRA reached 49.43% (**-17.21 pp**, paired bootstrap 95% CI -20.24 to -14.18). MATH-500 also regressed from 29.80% to 18.80% (-11.00 pp). The LoRA challenger is rejected.
- Training: dynamic padding plus `torch.compile` reached median **4,430.60 non-padding tokens/s** versus 987.84 for BF16 eager fixed-width (**+348.51%**, paired n=3 bootstrap interval +157.26% to +588.82%) and reduced median peak allocated VRAM from 12,108 to 7,956 MiB (-34.29%). Cold compile cost is charged; projected 378-step wall time improved 41.87%, with median break-even at 173 steps.
- Serving: at the frozen ScaleForge SLO-compliant concurrency-2 point, vLLM reached **0.2070 successful requests/s** versus 0.1208 for HF (**+71.39%**, paired n=3 bootstrap interval +51.01% to +89.52%). It reduced p95 TPOT 65.19% and p95 E2E 34.75%, but increased p95 TTFT from 86.86 to 443.24 ms. All 2,304 measured requests succeeded.
- Distributed: numerical qualification is **BLOCKED_EXTERNAL** because this host exposes one physical GPU. Two-rank CPU/Gloo launch and fixed-global-workload DDP code pass, but no multi-GPU speedup is claimed.

These are workload-specific results from one RTX 4090 Laptop GPU. The training sample is three paired runs and has material thermal/order associations. Serving remains REVIEW because every vLLM process required an unclean EngineCore shutdown. Full claim boundaries are in [CLAIM_LEDGER.json](CLAIM_LEDGER.json).

## Architecture

```text
official immutable sources
  -> schema/provenance checks -> hash FIT | VALIDATION | POLICY split
  -> development-only baseline/LoRA selection
  -> hash freeze -> protected GSM8K + MATH-500 qualification
  -> fixed workload -> profiler-led training interventions
  -> frozen POLICY corpus/SLO -> HF and vLLM load qualification
  -> raw artifacts + failures -> Parquet/DuckDB validation
  -> paired/hierarchical statistics -> independent release gates
  -> cards, report, claim ledger, resume/interview evidence
```

The code uses a typed `src/scaleforge` package, config-controlled scripts, separate WSL train and serving environments, FastAPI endpoints (`/health`, `/generate`, `/metrics`), request-level and GPU telemetry, CPU CI, and a read-only Streamlit evidence dashboard.

## Final selected configuration

| Layer | Selection | Decision |
|---|---|---|
| Model | M0 frozen Qwen2.5-1.5B-Instruct, deterministic zero-shot reasoning | PASS; LoRA blocked |
| Training | T4 dynamic per-batch padding + `torch.compile`, BF16 | PASS with claim qualifiers |
| Serving | vLLM 0.28 V1 compatibility runner, BF16, concurrency 2 | REVIEW |
| Distributed | Future 1-GPU vs 2-GPU DDP fixed-global-workload study | BLOCKED_EXTERNAL |
| Reproducibility | Exact locks, hashes, canonical warehouse, CPU quality gates | REVIEW due protected evaluator history |

The service objectives are ScaleForge experimental SLOs, not Google SLOs: p95 TTFT <=500 ms, p95 TPOT <=140 ms, p95 E2E <=33 s, error rate <=1%, and quality/parser non-regression.

## Major limitations

- A protected-evaluation parser overflow and later trailing-zero bug required new immutable protocol identities. SF-MODEL-v3 is a frozen evaluator-only rescore of complete SF-MODEL-v2 generation text, not a fresh generation replicate.
- Exact match is conservative and is not symbolic equivalence or reasoning-faithfulness evaluation.
- The laptop training qualification has n=3 paired replicates and observable order/thermal confounding; the full 378-step timing is projected from measured cold and steady phases.
- The 64-request serving corpus repeats within each fresh-server replicate, so prefix-cache state carries across concurrency points as frozen. vLLM V2 failed WSL UVA and V1 compatibility mode was used.
- All serving requests succeeded, but vLLM shutdown force-killed an idle EngineCore and leaked a semaphore. This prevents a production-readiness claim.
- One physical GPU prevents real local DDP scaling measurements. FSDP2 is not applicable to a 1.5B model that fits comfortably.

## Repository map

- `configs/`: frozen model, training, serving, SLO, and distributed configurations
- `src/scaleforge/`: data, evaluation, training, profiling, statistics, warehouse, and release logic
- `scripts/`: acquisition, freezes, experiments, post-processing, plotting, and qualification tools
- `api/`: FastAPI schema/service plus the qualified HF runtime
- `ui/`: read-only Streamlit evidence dashboard
- `tests/`: CPU-safe unit, integration-fixture, API, statistics, artifact, and release tests
- `artifacts/warehouse/`: canonical Parquet tables and DuckDB database
- `artifacts/analysis/`: qualification decisions, figure tables, and resume-safe claim export
- `reports/figures/`: figures F01-F10, F13, and F14

## Reproduction

Use Windows for repository control and WSL2 Ubuntu for all Linux-native CUDA/vLLM work. The training environment is `.venv`; the isolated serving environment is `.venv-serve`. Do not install native-Windows vLLM.

```bash
cd /mnt/c/Users/bjw-0/Downloads/scaleforge-ml
.venv/bin/python -m ruff check src tests scripts api ui
.venv/bin/python -m mypy src
.venv/bin/python -m pytest
.venv/bin/python scripts/build_warehouse.py
.venv/bin/python scripts/finalize_release.py
streamlit run ui/app.py
```

Protected model generation is CLOSED and must not be rerun for tuning. Exact stage commands, environment setup, guardrails, and future two-GPU commands are in [RUNBOOK.md](RUNBOOK.md) and [DISTRIBUTED_QUALIFICATION_PENDING.md](DISTRIBUTED_QUALIFICATION_PENDING.md).

## Evidence entry points

- [Final technical report](FINAL_TECHNICAL_REPORT.md)
- [Data card](DATA_CARD.md), [model card](MODEL_CARD.md), [system card](SYSTEM_CARD.md), [evaluation card](EVALUATION_CARD.md)
- [Resume evidence](reports/RESUME_EVIDENCE.md) and [interview guide](INTERVIEW_GUIDE.md)
- [Final audit](FINAL_AUDIT.md) and [requirements matrix](REQUIREMENTS_MATRIX.md)
- `artifacts/analysis/release_decision.json` and `artifacts/analysis/warehouse_integrity.json`

No raw model weights or large raw datasets are committed. No cloud GPU, paid API, remote push, or fabricated distributed result was used.
