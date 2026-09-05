# ScaleForge-ML

ScaleForge-ML is an evidence-first study of Transformer model quality and GPU systems performance using Qwen2.5-1.5B-Instruct on GSM8K, with protected MATH-500 non-regression evaluation. It is currently under active qualification; no model-quality or performance result is claimed before canonical artifacts and the independent audit support it.

## Verified results

No scientific or performance result is verified yet. The pre-dependency environment preflight is complete and found one RTX 4090 Laptop GPU (16,376 MiB VRAM) visible in WSL2. Distributed numerical qualification therefore requires external multi-GPU hardware and will not be fabricated.

## Architecture

Data provenance and leakage-safe splitting feed deterministic baselines and compact LoRA selection. A frozen protected evaluation precedes profile-led training and serving experiments. Raw run artifacts pass fail-closed validation into Parquet/DuckDB tables, paired statistics, independent release gates, and only then recruiter-facing evidence.

## Selected configuration

Not selected. Qwen/Qwen2.5-1.5B-Instruct is the required primary workload; model, training, and serving candidates remain in development.

## Major limitations

Protected evaluation has not been accessed, GPU qualification has not begun, and only one physical GPU is locally available. This README intentionally contains no aspirational benchmark numbers.

See [PROJECT_SPEC.md](PROJECT_SPEC.md), [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), and [PROJECT_STATUS.md](PROJECT_STATUS.md).
