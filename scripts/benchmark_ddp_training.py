from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scaleforge.distributed.workload import (
    gradient_accumulation_steps,
    rank_batch_indices,
)
from scaleforge.protocol import canonical_sha256, sha256_file
from scaleforge.training.cache import tensor_cache
from scaleforge.training.examples import (
    right_padded_batch_width,
    validate_right_padded_attention_mask,
)

CONFIG = Path("configs/distributed/qualification.yaml")
FREEZE = Path("artifacts/manifests/distributed_freeze.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replicate", required=True, type=int)
    parser.add_argument("--run-order", required=True, type=int)
    parser.add_argument("--state", choices=("DEVELOPMENT", "QUALIFICATION"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def start_telemetry(path: Path) -> subprocess.Popen[str]:
    fields = "timestamp,index,name,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used"
    return subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            "--loop-ms=500",
            f"--filename={path}",
        ],
        text=True,
    )


def verify_freeze(config: dict[str, Any], config_path: Path) -> str:
    if not FREEZE.is_file():
        raise ValueError("distributed qualification freeze is absent")
    manifest = json.loads(FREEZE.read_text(encoding="utf-8"))
    if (
        config_path.resolve() != Path(manifest["configuration_path"]).resolve()
        or config["state"] != "FROZEN"
        or canonical_sha256(config) != manifest["configuration_sha256"]
        or sha256_file(Path(config["data"]["path"])) != manifest["data_sha256"]
    ):
        raise ValueError("distributed qualification identity differs from freeze")
    for path_text, expected_hash in manifest["file_sha256"].items():
        if sha256_file(Path(path_text)) != expected_hash:
            raise ValueError(f"distributed frozen source differs: {path_text}")
    git_sha = subprocess_git_sha()
    if git_sha != manifest["git_sha"]:
        raise ValueError("current Git SHA differs from distributed freeze")
    return sha256_file(FREEZE)


def subprocess_git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (1, 2) or torch.cuda.device_count() < world_size:
        raise RuntimeError("run requires one or two distinct visible physical CUDA GPUs")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.state != config["state"]:
        raise ValueError("requested state differs from distributed configuration")
    freeze_sha = verify_freeze(config, args.config) if args.state == "QUALIFICATION" else None
    expected_order = config["qualification"]["balanced_order"]
    if args.run_order < 1 or int(expected_order[args.run_order - 1]) != world_size:
        raise ValueError("world size differs from the predeclared run order")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    seed_everything(int(config["workload"]["seed"]))
    output_dir = Path("artifacts/raw/distributed")
    analysis_dir = Path("artifacts/analysis/distributed")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    steps_path = output_dir / f"{args.run_id}-steps.jsonl"
    summary_path = analysis_dir / f"{args.run_id}.json"
    failure_path = Path("artifacts/failures") / f"{args.run_id}.json"
    telemetry_path = output_dir / f"{args.run_id}-gpu.csv"
    if rank == 0 and any(path.exists() for path in (steps_path, summary_path, failure_path)):
        raise FileExistsError(f"distributed run identity exists: {args.run_id}")
    dist.barrier()
    started_at = datetime.now(UTC).isoformat()
    telemetry: subprocess.Popen[str] | None = None
    try:
        data = pd.read_parquet(config["data"]["path"])
        fit = (
            data.loc[data["split_role"].eq("FIT")]
            .sort_values("example_id")
            .head(int(config["workload"]["examples"]))
        )
        model_id = str(config["model"]["repository"])
        revision = str(config["model"]["revision"])
        cache_dir = Path("artifacts/cache/huggingface")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        baseline = yaml.safe_load(Path("configs/model/baselines.yaml").read_text())
        tensors = tensor_cache(
            fit,
            tokenizer,
            system=baseline["prompts"]["m0"]["system"],
            sequence_length=int(config["data"]["sequence_length"]),
            fingerprint=config["data"]["fingerprint"],
        )
        validate_right_padded_attention_mask(tensors[1])
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, dtype=torch.bfloat16
        ).to(local_rank)
        model.config.use_cache = False
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(config["workload"]["lora_rank"]),
            lora_alpha=int(config["workload"]["lora_alpha"]),
            lora_dropout=float(config["workload"]["lora_dropout"]),
            target_modules=list(config["workload"]["target_modules"]),
            bias="none",
        )
        model = get_peft_model(model, lora)
        model.train()
        ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in ddp.parameters() if parameter.requires_grad],
            lr=float(config["workload"]["learning_rate"]),
            weight_decay=float(config["workload"]["weight_decay"]),
        )
        micro_batch = int(config["workload"]["micro_batch_size"])
        global_batch = int(config["workload"]["global_batch_size"])
        accumulation = gradient_accumulation_steps(
            micro_batch_size=micro_batch,
            global_batch_size=global_batch,
            world_size=world_size,
        )
        warmup_steps = int(config["workload"]["warmup_steps"])
        total_steps = warmup_steps + int(config["workload"]["measured_steps"])
        records: list[dict[str, Any]] = []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            telemetry = start_telemetry(telemetry_path)
        for step in range(total_steps):
            indices = rank_batch_indices(
                example_count=len(fit),
                step=step,
                global_batch_size=global_batch,
                world_size=world_size,
                rank=rank,
            )
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            local_tokens = 0
            local_losses: list[float] = []
            for micro_step in range(accumulation):
                batch_indices = indices[micro_step * micro_batch : (micro_step + 1) * micro_batch]
                input_ids = tensors[0][batch_indices].to(local_rank, non_blocking=True)
                attention_mask = tensors[1][batch_indices].to(local_rank, non_blocking=True)
                labels = tensors[2][batch_indices].to(local_rank, non_blocking=True)
                width = right_padded_batch_width(attention_mask)
                input_ids = input_ids[:, :width]
                attention_mask = attention_mask[:, :width]
                labels = labels[:, :width]
                sync_context = ddp.no_sync() if micro_step < accumulation - 1 else nullcontext()
                with sync_context:
                    output = ddp(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    (output.loss / accumulation).backward()
                local_losses.append(float(output.loss.detach().cpu()))
                local_tokens += int(attention_mask.sum().item())
            optimizer.step()
            torch.cuda.synchronize()
            local_elapsed = time.perf_counter() - started
            token_tensor = torch.tensor(local_tokens, device=local_rank, dtype=torch.long)
            elapsed_tensor = torch.tensor(local_elapsed, device=local_rank, dtype=torch.float64)
            loss_tensor = torch.tensor(np.mean(local_losses), device=local_rank)
            dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_tokens = int(token_tensor.item())
            global_elapsed = float(elapsed_tensor.item())
            record = {
                "run_id": args.run_id,
                "protocol_identity": config["protocol_identity"],
                "state": args.state,
                "replicate": args.replicate,
                "run_order": args.run_order,
                "world_size": world_size,
                "step": step + 1,
                "measured": step >= warmup_steps,
                "global_batch_size": global_batch,
                "global_tokens": global_tokens,
                "step_time_s": global_elapsed,
                "global_tokens_per_s": global_tokens / global_elapsed,
                "mean_rank_loss": float(loss_tensor.item() / world_size),
            }
            records.append(record)
            if rank == 0:
                with steps_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
        peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**20, device=local_rank)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            measured = [record for record in records if record["measured"]]
            summary = {
                "schema_version": "1.0.0",
                "created_at": datetime.now(UTC).isoformat(),
                "started_at": started_at,
                "run_id": args.run_id,
                "protocol_identity": config["protocol_identity"],
                "state": args.state,
                "status": "COMPLETED",
                "replicate": args.replicate,
                "run_order": args.run_order,
                "world_size": world_size,
                "physical_cuda_gpu_count": torch.cuda.device_count(),
                "global_batch_size": global_batch,
                "micro_batch_size_per_rank": micro_batch,
                "gradient_accumulation_steps_per_rank": accumulation,
                "measured_steps": len(measured),
                "global_tokens": sum(row["global_tokens"] for row in measured),
                "median_global_tokens_per_s": float(
                    np.median([row["global_tokens_per_s"] for row in measured])
                ),
                "step_p95_s": float(np.quantile([row["step_time_s"] for row in measured], 0.95)),
                "peak_allocated_memory_max_rank_mib": float(peak.item()),
                "telemetry_path": str(telemetry_path),
                "freeze_manifest_sha256": freeze_sha,
                "numerical_scaling_claim_allowed": world_size == 2
                and args.state == "QUALIFICATION",
            }
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2))
    except Exception as error:
        if rank == 0:
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "created_at": datetime.now(UTC).isoformat(),
                        "run_id": args.run_id,
                        "protocol_identity": config["protocol_identity"],
                        "state": args.state,
                        "failure_type": "distributed_synchronization_failure",
                        "exception_type": type(error).__name__,
                        "failure_message": str(error),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
