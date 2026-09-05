# Experiment Protocol

## Identities and state

Every run has a unique `run_id`, immutable `protocol_identity`, Git SHA/source hash, config hash, dataset fingerprint, environment inventory, start/end timestamps, state, and artifact inventory. Legal states are `PILOT`, `DEVELOPMENT`, `CANDIDATE_SELECTION`, `FROZEN`, `QUALIFICATION`, `ANALYSIS`, and `CLOSED`.

Transitions are forward-only. `PILOT` is instrumentation-only. Selection uses authorized development data. `FROZEN` hashes code, checkpoint, tokenizer, prompt, parser, generation, datasets, and gates. `QUALIFICATION` is protected execution. `ANALYSIS` is read-only. A scientific change after protected exposure creates a new protocol identity.

## Data authorization

- GSM8K official train: deterministically hash-split into FIT, VALIDATION, POLICY.
- FIT: optimization; VALIDATION: model and prompt candidate selection; POLICY: SLO/release-policy development and unbiased internal checks.
- GSM8K official test: protected FINAL.
- MATH-500: protected OOD/non-regression.
- Protected rows and outcomes cannot influence prompt/parser/model/threshold decisions.

Every protected access appends a timestamped record to `FINAL_ACCESS_LEDGER.json`, including actor/process, protocol, dataset/revision, purpose, access type, outcome exposure, artifacts, and failure disposition. Entries are never deleted.

## Freeze and retry

Before protected access, `FREEZE_MANIFEST.json` records SHA-256 hashes for Git/source tree, checkpoint, configs, prompt, parser, tokenizer/generation settings, dataset revisions/fingerprints, and predeclared gates. The manifest is validated before execution.

- Infrastructure failure with no meaningful prediction/aggregate exposure: identical scientific retry allowed; preserve failure.
- Outcomes exposed: only identical recovery/reproducibility retry allowed.
- Desired scientific change after exposure: new identity; prior evidence remains consumed.

## Model protocol

M0 and M1 share the frozen Qwen2.5-1.5B-Instruct revision and deterministic decode. M1's fixed examples and format are selected only on development data. A predeclared rule selects the stronger baseline using validation EM, then parse failure, then simpler M0 as tie-breaker. LoRA uses pilots to eliminate invalid/clearly poor settings and full validation for few finalists; no exhaustive sweep. One M* is selected, then compared against the strongest baseline under protected qualification.

Primary metric is exact normalized answer match. Report paired transitions, paired bootstrap 95% CI with fixed seed, and exact two-sided McNemar. OOD is non-regression only and never tuning. Slices use observable structural variables; interpretations are associative.

## Systems protocol

After model selection, model/data/request workloads are invariant across systems candidates. Baselines are competent, not strawmen. Each intervention follows baseline, profile, explicit bottleneck hypothesis, one change, re-profile, keep/reject. Finalists receive equal measured work, warmup, synchronization, at least three replicates, balanced/randomized order, and identical instrumentation. Failed runs and requests count.

Compile cold cost and steady state are separate, with break-even relative to the target workload. Serving SLOs are proposed from the strong baseline pilot and intended interactive workload, then frozen before candidate qualification. These are ScaleForge experimental SLOs, not Google SLOs. Selection uses SLO-compliant throughput, reliability, quality non-regression, and Pareto efficiency rather than raw maximum throughput.

## Statistical protocol

The unit matches assignment: paired example for quality, replicate/run for training, and replicate-aware/hierarchical resampling for serving requests. Report distributions and confidence intervals; do not use naive row-level t-tests on correlated requests. Check run order, temperature, clock, and power. Preserve and classify OOM, timeout, service, parser, invalid generation, worker, quality, SLO, integrity, and distributed-sync failures.

## Release and claims

Independent gates issue MODEL, TRAINING SYSTEM, SERVING, DISTRIBUTED, and REPRODUCIBILITY decisions. Simpler candidates win when added complexity lacks evidence. Only canonical validated artifacts may feed presentation. Every numeric claim must be `SUPPORTED`, `SUPPORTED_WITH_QUALIFIER`, or `NOT_SUPPORTED` in `CLAIM_LEDGER.json` with a prohibited stronger formulation.

## Closed training identity

`SF-TRAIN-v1` was frozen at Git `00080388a59f9d3ab8ef12d7ba8c62a6443cc7e1`. Its balanced order was T0, T4, T4, T0, T0, T4 with three fresh-process replicates per configuration, isolated compiler caches, two warmup steps, and ten measured synchronized steps. All six runs completed and the identity is closed to further measurement. Analysis uses replicate as the statistical unit, charges first-step/compiler overhead to 378-step projections, and reports order/temperature associations descriptively because three replicates cannot support reliable adjustment.
