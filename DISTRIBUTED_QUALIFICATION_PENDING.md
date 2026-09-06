# Distributed Qualification Pending

## Status

`BLOCKED_EXTERNAL`: local preflight exposes one physical CUDA GPU. No DDP speedup,
scaling-efficiency, or multi-GPU memory number has been manufactured. FSDP2 is not
justified for the 1.5B model because it fits comfortably on one GPU; the primary
external experiment is fixed-global-workload DDP strong scaling.

The CPU/Gloo launch smoke validates two-process rank initialization and all-reduce.
It is not performance evidence and sets `numerical_scaling_claim_allowed=false`.

## Required hardware

- Linux host or WSL-visible environment with two distinct NVIDIA CUDA GPUs.
- Each GPU must provide at least 16 GiB usable VRAM.
- Working NCCL peer communication and the locked `.venv` training environment.
- No competing GPU workload; collect temperature, clock, power, utilization, and
  per-GPU VRAM during every run.

## Qualification preparation

Run from the repository root after checking out the audited commit:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
.venv/bin/python scripts/detect_gpu_topology.py
```

Confirm `physical_cuda_gpu_count >= 2`. Review
`configs/distributed/qualification.yaml`, change only `state: DEVELOPMENT` to
`state: FROZEN`, commit that single protocol change, then create the immutable
manifest:

```bash
git add configs/distributed/qualification.yaml
git commit -m "freeze SF-DIST-v1 on two-GPU host"
.venv/bin/python scripts/freeze_distributed.py
git add artifacts/manifests/distributed_freeze.json
git commit -m "record SF-DIST-v1 freeze manifest"
```

Do not change source, config, model, data, or environment after the freeze.

## Exact balanced run matrix

Use fresh processes and run no competing GPU jobs. The global batch is always 16;
one GPU uses microbatch 2 x accumulation 8, while two GPUs use microbatch 2 x
accumulation 4 per rank. Both execute identical global example indices per step.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --standalone --nproc-per-node=1 scripts/benchmark_ddp_training.py --run-id qual-sf-dist-v1-g1-r1-o1 --replicate 1 --run-order 1 --state QUALIFICATION
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc-per-node=2 scripts/benchmark_ddp_training.py --run-id qual-sf-dist-v1-g2-r1-o2 --replicate 1 --run-order 2 --state QUALIFICATION
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc-per-node=2 scripts/benchmark_ddp_training.py --run-id qual-sf-dist-v1-g2-r2-o3 --replicate 2 --run-order 3 --state QUALIFICATION
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --standalone --nproc-per-node=1 scripts/benchmark_ddp_training.py --run-id qual-sf-dist-v1-g1-r2-o4 --replicate 2 --run-order 4 --state QUALIFICATION
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --standalone --nproc-per-node=1 scripts/benchmark_ddp_training.py --run-id qual-sf-dist-v1-g1-r3-o5 --replicate 3 --run-order 5 --state QUALIFICATION
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc-per-node=2 scripts/benchmark_ddp_training.py --run-id qual-sf-dist-v1-g2-r3-o6 --replicate 3 --run-order 6 --state QUALIFICATION
```

Before starting, use one measured development pilot for each GPU count to estimate
runtime and record it in `PROJECT_STATUS.md` if the matrix will exceed one hour.
Preserve every partial run and failure. Analyze only after all six frozen runs pass
identity, cardinality, identical-token, and telemetry checks. Compute speedup as
two-GPU throughput divided by paired one-GPU throughput and efficiency as speedup/2.

## Explicit claim boundary

Until this runbook is executed on real two-GPU hardware and the resulting artifacts
pass the independent audit, distributed performance claims are `NOT_SUPPORTED` and
must be omitted from resumes and recruiter-facing summaries.
