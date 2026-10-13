# ScaleForge-ML: Measured Results

## Model comparison
A Qwen2.5-1.5B pretrained baseline achieved **66.64%** exact match over 1,319 GSM8K test items. A rank-32 LoRA challenger achieved **49.43%** (−17.21 percentage points, paired 95% bootstrap CI −20.24 to −14.18). MATH-500 also regressed, so the frozen pretrained model was retained.

## Training
Dynamic padding and `torch.compile` raised median non-padding throughput from 987.84 to 4,430.60 tokens/s (**348.51%**) while reducing peak allocated VRAM from 12,108 to 7,956 MiB. Three paired measurements and nontrivial cold-compilation overhead limit broader inference.

## Serving
At the evaluated concurrency-2 SLO operating point, vLLM improved successful throughput by **71.39%**. A recurring EngineCore shutdown issue kept the serving verdict at REVIEW.

## Scope
All measurements use one RTX 4090 Laptop GPU. This is not multi-node or multi-GPU scaling evidence. Reported results, failures, and qualification boundaries are supported by the repository's structured artifacts and claim ledger.
