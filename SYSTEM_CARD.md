# ScaleForge-ML System Card

## Scope and environment

ScaleForge separates WSL environments: `.venv` for training/model/analysis and `.venv-serve` for vLLM. The qualified host is Windows 10.0.26200 with WSL2 Ubuntu 22.04, kernel 6.6.114.1, Intel i9-13900HX, 15 GiB WSL RAM, and one RTX 4090 Laptop GPU with 16,376 MiB VRAM and driver 610.62. Training used PyTorch 2.14.0+cu130; serving used isolated PyTorch 2.13.0+cu132 and vLLM 0.28.0. The CUDA compatibility value shown by `nvidia-smi` is not represented as an installed toolkit.

## Training system

T0 is a competent BF16 eager baseline with fixed global batch, fixed 640-token storage width, synchronized CUDA timing, and standard DataLoader behavior. Profiling identified padding-driven excess matrix/attention work. T1 removed trailing padding per batch without truncation. T4 added `torch.compile` only after measuring steady-state benefit and cold cost.

Across three paired, fresh-process qualification replicates and 35,382 identical non-padding tokens per run, T4 produced a 4,430.60 tokens/s median versus 987.84 for T0 (+348.51%, paired bootstrap interval +157.26% to +588.82%). Median peak allocated VRAM fell from 12,108 to 7,956 MiB. Cold cost ranged roughly 352-491 seconds; break-even ranged 75-270 steps (median 173). The 378-step wall-time improvement of 41.87% is projected from measured cold and steady phases, not directly timed end to end. TRAINING SYSTEM is PASS with strong n=3, laptop thermal, and order qualifiers.

## Serving system

Both runtimes expose equivalent health/generate/metrics behavior and use the same M0 checkpoint, prompt, tokenizer, deterministic decoder, 512-token cap, and 64-request POLICY corpus. S0 is HF Transformers BF16. S2 is vLLM continuous batching in its V1 compatibility runner because vLLM V2 failed WSL unified virtual addressing before outcomes.

The frozen ScaleForge experimental SLO—not a Google SLO—is p95 TTFT <=500 ms, p95 TPOT <=140 ms, p95 E2E <=33 seconds, error rate <=1%, and paired quality/parser non-regression. Both runtimes' maximum compliant point was concurrency 2. S2 improved median throughput 71.39%, output-token throughput 73.53%, p95 TPOT 65.19%, and p95 E2E 34.75%; p95 TTFT worsened 410.30%, from 86.86 to 443.24 ms. Its hierarchical 95% TTFT interval reaches 527.72 ms, beyond the point-estimate SLO. All 2,304 requests succeeded, but every vLLM shutdown force-killed an idle EngineCore and leaked a semaphore. SERVING is REVIEW, not production-ready.

## Distributed and operations

Only one physical CUDA GPU is available. Two-process CPU/Gloo initialization and all-reduce pass, and the DDP harness fixes the global example stream and global batch across 1/2 GPUs. Numerical DDP qualification is BLOCKED_EXTERNAL; no speedup, efficiency, communication, or memory claim is allowed. FSDP2 and tensor-parallel serving are not justified for this 1.5B workload. The exact future procedure is in `DISTRIBUTED_QUALIFICATION_PENDING.md`.

Canonical operational evidence is in `artifacts/warehouse`, `artifacts/profiles`, `artifacts/analysis/training`, `artifacts/analysis/serving`, and `artifacts/failures`.
