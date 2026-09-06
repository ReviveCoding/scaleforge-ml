# ScaleForge-ML Final Independent Audit

## Audit conclusion

**Overall: COMPLETE_WITH_EXTERNAL_DISTRIBUTED_PENDING.** All locally feasible model, training, serving, warehouse, statistics, software, documentation, and claim-evidence paths are present and validated. Numerical distributed qualification is correctly blocked by the single-GPU topology. Serving and reproducibility retain explicit REVIEW gates; the project is not represented as production-ready or clean on first protected pass.

The automated evidence audit passed on 2026-09-06 (`artifacts/audit/final_audit.json`). The full CPU suite passed 60 tests with 81.56% line coverage against an 80% gate; Ruff and strict mypy passed. Local sdist/wheel generation passed, and Streamlit 1.63.0 was importable through its CLI.

## Subsystem results

| Subsystem | Audit result | Evidence and boundary |
|---|---|---|
| Environment reproducibility | PASS | Pre-install Windows/WSL/GPU manifest, separate exact train/serve inventories, `uv.lock`, `requirements-serve.lock` |
| Data engineering | PASS | Official revisions, 7,473 validated rows, stable 6,046/723/704 split, duplicate/overlap/parser/length checks, no truncation |
| Protected-data control | PASS_WITH_LIMITATIONS | Access ledger complete; Case A builder exposure and v1/v2 evaluator defects retained; no outcome-driven tuning under an identity |
| Model selection and qualification | PASS | Strong M0/M1 development rule, compact LoRA program, one frozen M*, all protected rows paired, M0 retained |
| Training systems | PASS_WITH_LIMITATIONS | BF16 T0, profile-led interventions, fixed tokens, synchronization, fresh compile caches, balanced n=3 runs; thermal/order confound disclosed |
| Serving systems | PASS_WITH_LIMITATIONS | 36/36 points, 2,304 measured requests, tails/quality/failures/Pareto/knee; V1 fallback and unclean shutdown force REVIEW |
| Distributed | BLOCKED_EXTERNAL | One physical GPU; two-rank Gloo and workload invariants pass; no fabricated speedup; exact pending runbook |
| Warehouse/statistics | PASS | Ten nonempty canonical tables, 47 unique runs, normalized 21-event failure history, hashes, paired/hierarchical inference |
| Software engineering | PASS | Typed/config-driven src layout, API, read-only UI, CPU CI, manual GPU workflow, 60 tests, 81.56% coverage, package build |
| Documentation/claims | PASS | README, four cards, report, resume evidence, interview guide, figure source tables, machine-readable claim export |

## Audit checks

### Environment and provenance

The preflight distinguishes driver CUDA compatibility from a toolkit installation. All CUDA/vLLM work was executed through WSL; vLLM was isolated from the training stack. Model, tokenizer, GSM8K, and MATH-500 revisions are immutable in manifests. Large source data and weights are not versioned as project source.

### Split and protected integrity

The hash split is deterministic and role-exclusive. FIT/VALIDATION/POLICY authorization is reflected in scripts and tests. Protected access entries preserve the unintended builder cache, all v1/v2 generations, and v3 evaluator rescores. No backward transition from QUALIFICATION to DEVELOPMENT occurred under one identity. V3 changes evaluator behavior only and hashes its complete source prediction files.

### Fairness and measurement

Model comparisons share checkpoint family, decoder, parser identity, and paired examples. Training T0 is BF16 eager; T4 uses the same global workload and 35,382 measured non-padding tokens. CUDA synchronization, warmup, cold compile cost, peak allocated/reserved memory, and telemetry are present. Serving runtimes share model, corpus, token cap, deterministic settings, and evaluator. Zero request failures remain in denominators, while the shutdown defect remains a separate lifecycle failure. High-concurrency points were retained even when unfavorable.

### Statistical validity

Model inference pairs examples and uses exact McNemar plus paired bootstrap. Training inference pairs only three runs and publishes the wide interval. Serving uses replicate medians and request-within-replicate hierarchical bootstrap rather than treating 2,304 rows as independent. Tail latency, SLO-compliant throughput, non-dominated frontier, and saturation knee drive selection. Temperature, clock, power, and order are reported as observational confounds, not causal adjustments.

### Artifact and claim traceability

The warehouse audit found zero duplicate run IDs, duplicate protected predictions, or duplicate serving request IDs. It verified frozen/qualification input hashes and exact gate decisions. Every ledger artifact path exists; `resume_claims.json` exactly exports supported ledger entries; no distributed numeric claim is supported. Key README, report, resume, and interview values were checked against the canonical ledger/qualification artifacts.

## Cross-checked headline numbers

| Number | Canonical source | Cross-document result |
|---|---|---|
| GSM8K 66.64% M0 / 49.43% M* / -17.21 pp | model v3 qualification | MATCH |
| MATH-500 29.80% / 18.80% / -11.00 pp | model v3 qualification | MATCH |
| Training +348.51%; 12,108 to 7,956 MiB | training v1 qualification | MATCH |
| 378-step projected -41.87%; break-even 173 | training v1 qualification | MATCH and labeled projection |
| Serving 0.1208 to 0.2070 requests/s; +71.39% | serving v2 qualification | MATCH |
| Serving p95 TTFT 86.86 to 443.24 ms | serving v2 qualification | MATCH and negative direction disclosed |
| 2,304 measured requests, zero request failures | serving v2/warehouse | MATCH; shutdown defect separately retained |

## Findings that prevent an unqualified full PASS

1. Protected evaluation required versioned evaluator recovery; reproducibility is REVIEW, not PASS.
2. Performance samples are n=3 on one laptop GPU with thermal/order observations.
3. vLLM V2 is incompatible with this WSL UVA path; V1 compatibility mode is the measured runtime.
4. vLLM request execution succeeded, but shutdown force-killed EngineCore and leaked a semaphore; serving is REVIEW.
5. The repeated serving corpus can benefit vLLM prefix caching within a replicate.
6. Real two-GPU DDP execution is unavailable, so distributed is BLOCKED_EXTERNAL.

## Prohibited claims

Do not claim that LoRA improved quality, training speedup generalizes across hardware, the 378-step runtime was directly measured, every serving latency improved, vLLM is production-ready, ScaleForge SLOs are Google SLOs, MATH exact match is symbolic correctness, or any multi-GPU performance was executed.

## Final disposition

No implementation defect found by the final local gates remains open. The only required external evidence is physical multi-GPU qualification. The next optional action is to execute `DISTRIBUTED_QUALIFICATION_PENDING.md` unchanged on an authorized two-GPU CUDA host, then add a new frozen distributed analysis without modifying closed model/training/serving results.
