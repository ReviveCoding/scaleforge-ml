# ScaleForge-ML Runbook

Run repository/control work from Windows and Linux-native ML/CUDA/vLLM work from WSL2 Ubuntu. Never install native-Windows vLLM, change drivers, run simultaneous heavy GPU jobs, or tune against protected GSM8K test/MATH-500 outcomes. Existing protected protocols are CLOSED.

## Paths and environments

```text
Windows: C:\Users\bjw-0\Downloads\scaleforge-ml
WSL:     /mnt/c/Users/bjw-0/Downloads/scaleforge-ml
Train:   .venv       (Python 3.12.14, PyTorch 2.14.0+cu130)
Serve:   .venv-serve (Python 3.12.14, PyTorch 2.13.0+cu132, vLLM 0.28.0)
```

Environment inventories are `artifacts/manifests/train_environment.json`, `serve_environment.json`, and `serve_environment.txt`; dependency locks are `uv.lock` and `requirements-serve.lock`. Project-local caches are preferred. No external cache is a source of truth.

## CPU quality and package gates

```bash
cd /mnt/c/Users/bjw-0/Downloads/scaleforge-ml
.venv/bin/python -m ruff format --check src tests scripts api ui
.venv/bin/python -m ruff check src tests scripts api ui
.venv/bin/python -m mypy src
.venv/bin/python -m pytest
.venv/bin/python -m build
```

`pytest` enforces 80% coverage for non-GPU core logic. Hosted CI runs equivalent CPU-safe checks without downloading model weights. The manual GPU workflow is optional/self-hosted.

## Safe evidence rebuild

These commands only validate already completed qualification artifacts and regenerate canonical tables/figures/release summaries:

```bash
.venv/bin/python scripts/analyze_model_qualification.py
.venv/bin/python scripts/analyze_training_qualification.py
.venv/bin/python scripts/analyze_serving_qualification.py
.venv/bin/python scripts/build_warehouse.py
.venv/bin/python scripts/plot_model_results.py
.venv/bin/python scripts/plot_training_results.py
.venv/bin/python scripts/plot_serving_results.py
.venv/bin/python scripts/finalize_release.py
```

`build_warehouse.py` fails closed on missing/incomplete matrices, duplicate IDs, inconsistent failures, unequal training workloads, or telemetry identity mismatch. It writes canonical Parquet tables, `results.duckdb`, and `warehouse_integrity.json`.

## Services

HF baseline in the train environment:

```bash
.venv/bin/python scripts/serve_hf.py --host 127.0.0.1 --port 8001
```

vLLM must run only in `.venv-serve`. Under this WSL host the default V2 runner fails unified virtual addressing; the qualified fallback sets `VLLM_USE_V2_MODEL_RUNNER=0` and serves the frozen M0 checkpoint with the OpenAI-compatible API. Consult `configs/serving/qualification.yaml` before reproducing. Do not describe the fallback as vLLM V2.

Read-only Streamlit evidence dashboard:

```bash
.venv/bin/streamlit run ui/app.py
```

The dashboard reads canonical analysis files; it does not launch inference or edit evidence.

## Data and model state

The deterministic development pipeline is implemented by `scripts/prepare_data.py`, but official-test acquisition and protected generation must not be rerun for development. `FREEZE_MANIFEST.json`, `FINAL_ACCESS_LEDGER.json`, and model protocol v3 govern the closed result. A scientific change requires a new experiment identity, fresh freeze, and explicit acknowledgement that earlier protected evidence was consumed.

The historical model progression is immutable:

1. v1 closed after parser overflow with outcomes exposed.
2. v2 completed with an overflow-safe parser, then closed after a trailing-zero defect was observed.
3. v3 froze a corrected evaluator and re-scored byte-identical v2 text; no generation/model/prompt changed.

## Training and serving reproduction discipline

Before any new run longer than one hour, measure a pilot, record total estimated time in `PROJECT_STATUS.md`, and confirm it resolves an open requirement. Training uses fresh compile caches, synchronized CUDA, warmup, equal token workload, three or more replicates, balanced order, and telemetry. Serving uses fresh server processes, the frozen 64-request POLICY corpus, the six-level concurrency grid, request-level failures, and frozen SLOs. Never rerun because a result is unfavorable.

## Distributed qualification

Current numerical status is `BLOCKED_EXTERNAL`: one physical CUDA GPU. The two-process CPU/Gloo command used to validate launch plumbing is:

```bash
torchrun --standalone --nproc-per-node=2 scripts/distributed_smoke.py \
  --run-id distributed-launch-smoke-20260906 --backend gloo
```

It is not scaling evidence. Exact freeze, 1-GPU reference, 2-GPU DDP, order, hardware, validation, and analysis commands for a later authorized host are in `DISTRIBUTED_QUALIFICATION_PENDING.md`. Do not run FSDP2 unless a new measured memory question justifies a new protocol.

## Failure and checkpoint handling

Preserve the command, environment, timestamps, logs, partial artifacts, taxonomy, and protected-outcome exposure. If outcomes exist, retry only the identical frozen configuration for reproducibility/recovery; a scientific change creates a new identity. Do not erase failed attempts.

At each stage boundary inspect the diff, run applicable checks, validate hashes/evidence, verify the protected ledger, update `REQUIREMENTS_MATRIX.md`, `PROJECT_STATUS.md`, `DECISIONS.md`, and `CLAIM_LEDGER.json`, then write a local checkpoint. Never push without an explicit request.
