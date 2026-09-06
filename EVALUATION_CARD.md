# ScaleForge-ML Evaluation Card

## Experimental contract

Experiments move through PILOT, DEVELOPMENT, CANDIDATE_SELECTION, FROZEN, QUALIFICATION, ANALYSIS, and CLOSED without moving backward under the same identity. GSM8K official test and MATH-500 are protected. Prompt, parser, LoRA, threshold, and release decisions use only FIT/VALIDATION/POLICY data. Every protected access is retained in `FINAL_ACCESS_LEDGER.json`.

## Statistical units

- Model quality: paired example is the unit. Report exact match, candidate-only/baseline-only transitions, 20,000-sample paired bootstrap intervals, and exact McNemar tests.
- Training: paired fresh-process replicate is the unit. Report non-padding tokens/s, step quantiles, peak memory, and paired-replicate bootstrap uncertainty.
- Serving: fresh-server replicate is the performance unit; request rows remain nested. Report p50/p95/p99 latency, hierarchical request-within-replicate intervals, successful throughput with failures in denominators, Pareto dominance, and saturation knees. No naive request-row t-test is used.
- Distributed: speedup and efficiency require physical 1/2-GPU execution. CPU launch smoke is not performance evidence.

## Integrity and failure accounting

The canonical warehouse rejects missing schemas, incomplete frozen matrices, duplicate run/request/example identities, inconsistent failure labels, unequal training token workloads, and telemetry identity mismatch. It contains 47 run identities, 3,638 protected predictions, 72 training steps, 6 training runs, 2,592 serving request rows (2,304 measured), 36 serving runs, 29,441 GPU samples, 2 distributed-smoke rank rows, 21 historical failure events, and 5 independent release gates. All failed attempts are preserved and normalized rather than dropped.

## Bias and limitations

The single laptop GPU, three performance replicates, shared thermal envelope, and observational clock/power correlations limit external validity. Serving repeats a fixed corpus and carries prefix-cache state within a replicate. MATH-500 exact matching is deliberately conservative. Structural slices are descriptive associations. The corrected SF-MODEL-v3 evaluator reuses immutable v2 text; the protected first pass was not clean, so REPRODUCIBILITY remains REVIEW despite locks, hashes, CI, and passing artifact validation.

Canonical evidence: `artifacts/analysis/warehouse_integrity.json`, `artifacts/warehouse/results.duckdb`, subsystem qualification JSON files, figure tables, and `CLAIM_LEDGER.json`.
