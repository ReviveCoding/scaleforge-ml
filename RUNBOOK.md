# ScaleForge-ML Runbook

## Safety and order of operations

Run commands from the repository root. Windows performs repository/control tasks; WSL2 performs Linux-native PyTorch CUDA and vLLM tasks. Never install native-Windows vLLM. Never access protected datasets before the frozen-manifest validator passes. Do not run two heavy GPU jobs concurrently.

## Planned stages

1. `FOUNDATION`: persist controls; record preflight; initialize `main`; checkpoint.
2. `ENGINEERING`: package/config schemas, fixtures, tests, CI, artifact contracts.
3. `DATA_DEVELOPMENT`: acquire only GSM8K train first; validate provenance/schema/parser/splits/token lengths; create data card/manifests.
4. `MODEL_DEVELOPMENT`: deterministic M0/M1 and compact LoRA pilots/finalists using FIT/VALIDATION only.
5. `MODEL_FREEZE`: select M*, freeze all scientific inputs and gates, validate hashes.
6. `MODEL_QUALIFICATION`: ledger protected accesses; execute GSM8K test and MATH-500 once under frozen identity; canonicalize; paired analysis; close.
7. `TRAINING_SYSTEMS`: freeze workload; T0 profile; controlled interventions; replicated finalists; thermal/order analysis.
8. `SERVING_DEVELOPMENT`: isolated vLLM environment; HF baseline pilot; freeze experimental SLO; tune only on development corpus.
9. `SERVING_QUALIFICATION`: balanced replicated load tests; preserve request/failure telemetry; Pareto/knee analysis.
10. `DISTRIBUTED`: detect topology; run real DDP only if >=2 physical GPUs and authorized; otherwise validate launch code and publish pending runbook.
11. `WAREHOUSE_RELEASE`: fail-closed canonicalization, statistics, independent gates, figures, claim ledger.
12. `DOCUMENTATION_AUDIT`: cards/report/resume/interview/UI, stop features, independent cross-check and final audit.

## Phase checkpoint

At every stage boundary: inspect `git diff` and status; run applicable tests/linters/types/build; validate new evidence and hashes; verify the protected-access ledger; update requirements, status, decisions, and claims; write a checkpoint artifact; make a local commit when useful.

## Failure handling

Record command, environment, timestamps, logs, partial artifacts, and taxonomy. Do not delete unfavorable or failed evidence. On two materially identical sandbox/network escalations, stop retrying, diagnose, use a safer workspace-local alternative, continue unaffected work, and ask only if essential.

Exact environment setup, data, training, serving, qualification, and distributed commands will be added only after their scripts exist and have been smoke-tested; unverified commands are not presented as operational.
