# ScaleForge-ML: Experiment Decisions

The LoRA model was rejected because the protected GSM8K exact-match score fell from 66.64% to 49.43%, a **17.21**-point decline. MATH-500 performance declined as well.

Dynamic padding and compilation increased measured throughput by **348.51%**, but cold compilation time and laptop thermal effects must be included when evaluating end-to-end benefit.

vLLM increased SLO-compliant throughput by **71.39%** at concurrency two, but an unreliable EngineCore shutdown prevented promotion.

The machine had **only one GPU**. Two-rank CPU/Gloo qualification does not establish NCCL scaling. The `FINAL_TECHNICAL_REPORT.md` and frozen evidence ledger document the comparison conditions and uncertainties.
