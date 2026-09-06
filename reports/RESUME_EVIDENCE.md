# ScaleForge-ML Resume Evidence

## Resume-safe project entry

**ScaleForge-ML | Transformer Training & ML Systems Performance Engineering**

- Built an end-to-end Qwen2.5-1.5B ML systems platform spanning leakage-controlled data, deterministic evaluation, LoRA adaptation, BF16 PyTorch profiling/compilation, FastAPI and vLLM serving, load testing, paired statistics, Parquet/DuckDB evidence, tests, and CI.
- Rejected a rank-32 LoRA after locked evaluation reduced GSM8K exact match from 66.64% to 49.43% (-17.21 pp; paired 95% CI -20.24 to -14.18) and regressed MATH-500 by 11.00 pp, retaining the stronger frozen pretrained baseline.
- Increased median steady training throughput from 988 to 4,431 non-padding tokens/s (+348.51%; paired n=3 interval +157.26% to +588.82%) and reduced peak allocated VRAM 34.29% through profile-led dynamic padding and `torch.compile`; explicitly charged 352-491 s cold cost and qualified laptop order/thermal effects.
- Increased median max SLO-compliant serving throughput 71.39% with vLLM at concurrency 2 while reducing p95 TPOT 65.19% and p95 E2E 34.75% across 2,304 successful measured requests; held deployment at REVIEW after reproducing an unclean EngineCore shutdown.

Do not add a numeric distributed bullet. Only one physical GPU was available, so DDP launch code is tested but scaling performance is not qualified.

## Evidence map

| Statement | Protocol | Canonical evidence | Required qualifier |
|---|---|---|---|
| GSM8K baseline and LoRA rejection | SF-MODEL-v3 | `artifacts/analysis/model/model_qualification_sf_model_v3.json` | evaluator-only rescore of immutable v2 text |
| MATH-500 non-regression failure | SF-MODEL-v3 | same model qualification artifact | conservative exact match, not symbolic equivalence |
| Training throughput and VRAM | SF-TRAIN-v1 | `training_qualification_sf_train_v1.json`, `training_runs.parquet` | paired n=3, one laptop GPU, thermal/order associations |
| Serving throughput and tails | SF-SERVE-v2 | `serving_qualification_sf_serve_v2.json`, request/run Parquet | paired n=3; fixed repeated corpus; V1 runner |
| Request success | SF-SERVE-v2 | `serving_requests.parquet`, shutdown failure artifact | request path succeeded; lifecycle was not clean |

## Prohibited embellishments

- Do not claim LoRA improved quality, a 4.5x hardware-independent training acceleration, a directly timed 378-step speedup, or production vLLM readiness.
- Do not call MATH-500 exact match symbolic accuracy or imply protected v3 generation was rerun.
- Do not claim multi-GPU execution, DDP speedup, scaling efficiency, NCCL overhead, FSDP2 results, or tensor-parallel serving.
- Do not call the frozen service objectives Google SLOs; they are ScaleForge experimental SLOs.

The machine-readable source is `artifacts/analysis/resume_claims.json`; `CLAIM_LEDGER.json` is authoritative if language conflicts.
