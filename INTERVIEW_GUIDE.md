# ScaleForge-ML Interview Guide

## 30-second explanation

I built an evidence-first ML system around Qwen2.5-1.5B and GSM8K. I controlled data leakage, compared deterministic prompting with a compact LoRA search, froze the experiment, and rejected LoRA when protected GSM8K and MATH-500 both regressed. Then I profiled the fixed training workload: removing wasted padding and compiling improved median steady throughput from 988 to 4,431 tokens/s, with cold cost and laptop thermal effects disclosed. For serving, vLLM raised SLO-compliant throughput 71%, but I kept deployment at REVIEW because shutdown was unreliable. The repository makes every claim traceable through Parquet/DuckDB artifacts, statistics, tests, and release gates.

## 2-minute explanation

The central question was not simply whether I could fine-tune and serve a Transformer; it was whether I could improve it under controls strong enough to support an engineering decision. I used immutable GSM8K and MATH-500 revisions, split the official GSM8K training set by deterministic hashes into FIT, VALIDATION, and POLICY, and protected the official test and OOD set from tuning.

I built deterministic zero- and few-shot pretrained baselines, then ran a compact LoRA search. The sole finalist lost about 20 points on validation. I still froze one challenger and followed the protocol through protected qualification. It fell 17.21 points on GSM8K and 11 points on MATH-500, so I retained the simpler pretrained model. Two parser defects were handled by closing protocol identities and freezing an evaluator-only rescore rather than erasing or tuning against failures.

For training, PyTorch Profiler showed excess work from padding to 640 tokens when typical examples were much shorter. Dynamic padding plus `torch.compile` raised median synchronized steady throughput 348.51% and reduced allocated VRAM 34.29% in three paired runs. I separately measured 352-491 seconds of cold cost and calculated a 173-step median break-even, so compile is recommended only for the 378-step target, not tiny jobs.

For serving, I froze a development-derived ScaleForge SLO, tested HF and vLLM across six concurrency levels and three fresh-server replicates, preserved failures, and used tail latency and a Pareto frontier. vLLM at concurrency 2 raised compliant throughput 71.39% and reduced TPOT and E2E tails, but worsened TTFT and shut down uncleanly. That makes it a REVIEW candidate rather than a production claim. A one-GPU topology means distributed numbers are honestly blocked, with an exact two-GPU DDP runbook left for later.

## 5-minute technical walkthrough

1. **Contract and provenance.** The repository persists requirements, experiment states, decisions, protected-access logs, freeze hashes, and a claim ledger. GSM8K train is retrieved by immutable split-specific artifact; each example gets a stable hash ID and split. Schema and duplicate checks fail closed. Length analysis chose 640 without truncation.
2. **Model development.** M0/M1 share the same checkpoint and deterministic decoder, so prompt strategy is the only baseline variable. A small LoRA matrix explores rank, LR, and target modules; pilots gate full validation. The chosen M* is not a cherry-picked winner—it is the only finalist and then fails both paired protected gates.
3. **Protected recovery.** V1 parser overflow happened after partial outcomes, so I did not patch in place. V2 ran a frozen overflow-safe parser; analysis exposed trailing-zero normalization. V3 freezes that evaluator correction and reuses byte-identical V2 predictions. This preserves scientific lineage and keeps reproducibility at REVIEW.
4. **Training systems.** T0 is BF16 eager, not FP32 theater. Profile data led to one intervention at a time: dynamic padding, then compilation. Finalists use equal non-padding tokens, fixed batches and steps, CUDA synchronization, warmup, fresh caches, three paired replicates, and GPU telemetry. Cold cost and target-duration break-even determine adoption.
5. **Serving systems.** The HF and vLLM services share request corpus and semantics. Every request records token counts, queue, TTFT, TPOT, E2E, status, failure type, and GPU state. Candidate qualification uses fresh server processes and balanced order. The operating point maximizes throughput subject to frozen tail, error, and quality gates—not raw throughput.
6. **Evidence system.** Raw outputs are immutable inputs to validators. Canonical Parquet and DuckDB tables reject duplicates and incomplete matrices. Model statistics are example-paired; training is replicate-paired; serving bootstraps requests within replicate. Release gates remain independent. Recruiter-facing numbers are exported only from the claim ledger.
7. **Limits.** One laptop GPU, n=3 performance replicates, thermal/order correlations, a repeated serving corpus, conservative MATH parsing, vLLM V1 WSL fallback, and an unclean shutdown constrain the claims. Distributed performance is absent by design.

## Common questions

### What problem did you solve?

I built and qualified the decision process around model adaptation, training efficiency, and serving capacity. The useful outcome is knowing what to adopt, what to reject, and exactly how strong the evidence is.

### Why this model? Why not a larger model?

Qwen2.5-1.5B-Instruct is a real instruction-tuned Transformer that fits a 16 GB laptop GPU, allowing repeated controlled experiments without fallback or paid compute. A larger model would reduce replication and make systems comparisons confounded by memory failure.

### Why GSM8K?

It provides train/test structure, exact final answers, worked solutions for response-only adaptation, and enough examples for pairing and structural slices. MATH-500 adds a harder protected OOD check.

### Why this baseline?

M0/M1 use deterministic decoding and identical checkpoint/runtime controls; full development validation chooses the stronger one. T0 is competent BF16 eager, and S0 is a functioning HF BF16 service, so gains are not against strawmen.

### Why LoRA, and why did you reject it?

LoRA tests parameter-efficient adaptation within one-GPU constraints. The finalist decisively regressed on validation, GSM8K test, and MATH-500; retaining it would add memory, artifacts, and operational complexity without benefit.

### How did you prevent leakage?

Stable hash splits isolate FIT, VALIDATION, and POLICY. Protected test/OOD data are opened only after source, parser, prompt, checkpoint, generation, and gates are hashed. Every access is logged, and no result-driven retuning occurs under the same identity.

### What did the profiler identify? Which optimization mattered?

The profiler tied fixed-width padding to excess matrix and attention work. Dynamic trimming removed trailing padding without losing tokens; adding compile then improved steady execution enough to amortize cold cost for the target duration.

### What failed, and why was a candidate rejected?

LoRA failed quality; protected parsers exposed overflow and normalization defects; vLLM V2 failed WSL UVA; vLLM shutdown was unclean. Each failure is retained. Candidates are rejected by predeclared independent gates, not a weighted score.

### How did you measure latency, and why tails?

Request-level monotonic timestamps separate queue, TTFT, TPOT, and E2E. p95/p99 reveal contention and user-visible stalls hidden by the mean. Hierarchical bootstrap respects request nesting inside fresh-server replicates.

### How did you choose the serving point?

Maximum SLO-compliant throughput is the highest throughput among points passing frozen TTFT, TPOT, E2E, error, and quality gates. The Pareto frontier removes dominated points, and the knee shows where concurrency yields diminishing returns. Both runtimes select concurrency 2.

### How did you control laptop thermal effects?

I used balanced run order, fresh processes, warmups, no competing GPU workloads, and telemetry for temperature, clocks, power, utilization, and memory. Remaining associations are reported as confounds rather than statistically adjusted away at n=3.

### What is DDP scaling efficiency? Why no FSDP2?

For N GPUs, speedup is one-GPU time divided by N-GPU time for the same global workload; efficiency is speedup/N. I did not calculate it because only one GPU exists. FSDP2 is unjustified because the 1.5B workload fits; sharding would answer no current memory question.

### What would you change with 8 GPUs?

First execute the frozen 1/2-GPU DDP strong-scaling matrix, then extend to 4/8 with identical global work and randomized order. I would measure NCCL/compute overlap and per-GPU memory, test replica serving for capacity, and introduce FSDP2 only if a larger model or optimizer state creates a real memory constraint.

### What are the main limitations?

Single laptop hardware, three performance replicates, protected evaluator recovery, conservative exact match, repeated serving prompts and prefix-cache carryover, vLLM V1 compatibility mode, unclean shutdown, and no physical multi-GPU evidence.
