from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scaleforge.protocol import canonical_sha256
from scaleforge.training.cache import tensor_cache
from scaleforge.training.examples import (
    right_padded_batch_width,
    validate_right_padded_attention_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--config", type=Path, default=Path("configs/model/lora_candidates.yaml"))
    parser.add_argument(
        "--baseline-config", type=Path, default=Path("configs/model/baselines.yaml")
    )
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/split_manifest.parquet"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--train-examples", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--state", choices=("PILOT", "CANDIDATE_SELECTION"), default="PILOT")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    baseline = yaml.safe_load(args.baseline_config.read_text(encoding="utf-8"))
    if args.candidate not in config["candidates"]:
        raise ValueError(f"unknown candidate {args.candidate}")
    if config["state"] != "DEVELOPMENT" or config["data"]["split_role"] != "FIT":
        raise ValueError("LoRA pilots are restricted to DEVELOPMENT/FIT")
    runtime = config["runtime"]
    if runtime["global_batch_size"] != (
        runtime["micro_batch_size"] * runtime["gradient_accumulation_steps"]
    ):
        raise ValueError("global batch contract is inconsistent")
    if runtime["gradient_checkpointing"]:
        raise ValueError("checkpointing is not justified for the initial memory pilot")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA support is required")
    seed = int(runtime["seed"])
    seed_everything(seed)

    candidate = config["candidates"][args.candidate]
    steps = int(args.steps or runtime["pilot_steps"])
    train_examples = int(args.train_examples or runtime["pilot_train_examples"])
    frame = pd.read_parquet(args.data)
    fit = frame.loc[frame["split_role"].eq("FIT")].sort_values("example_id")
    if train_examples > 0:
        fit = fit.head(train_examples)
    model_id = config["model"]["repository"]
    revision = config["model"]["revision"]
    cache_dir = Path("artifacts/cache/huggingface")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tensors = tensor_cache(
        fit,
        tokenizer,
        system=baseline["prompts"]["m0"]["system"],
        sequence_length=int(config["data"]["sequence_length"]),
        fingerprint=config["data"]["fingerprint"],
    )
    validate_right_padded_attention_mask(tensors[1])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=int(runtime["micro_batch_size"]),
        shuffle=True,
        generator=generator,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.config.use_cache = False
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(candidate["rank"]),
        lora_alpha=int(candidate["rank"]) * 2,
        lora_dropout=0.05,
        target_modules=list(candidate["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(candidate["learning_rate"]),
        weight_decay=float(runtime["weight_decay"]),
    )
    warmup_steps = max(1, math.ceil(steps * float(runtime["warmup_ratio"])))

    def learning_rate(step: int) -> float:
        base = float(candidate["learning_rate"])
        if step < warmup_steps:
            return base * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return base * max(0.0, 1.0 - progress)

    run_id = args.run_id or f"{args.candidate}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    raw_dir = Path("artifacts/raw/training")
    raw_dir.mkdir(parents=True, exist_ok=True)
    steps_path = raw_dir / f"{run_id}-steps.jsonl"
    if steps_path.exists():
        raise FileExistsError(f"run ID already exists: {run_id}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader)
    execution_started = time.perf_counter()
    total_tokens = 0
    step_records: list[dict[str, Any]] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        losses: list[float] = []
        step_tokens = 0
        for _ in range(int(runtime["gradient_accumulation_steps"])):
            try:
                input_ids, attention_mask, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                input_ids, attention_mask, labels = next(iterator)
            input_ids = input_ids.to("cuda", dtype=torch.long, non_blocking=True)
            attention_mask = attention_mask.to("cuda", non_blocking=True)
            labels = labels.to("cuda", dtype=torch.long, non_blocking=True)
            if runtime["dynamic_batch_padding"]:
                batch_width = right_padded_batch_width(attention_mask)
                input_ids = input_ids[:, :batch_width]
                attention_mask = attention_mask[:, :batch_width]
                labels = labels[:, :batch_width]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / int(runtime["gradient_accumulation_steps"])
            loss.backward()
            losses.append(float(outputs.loss.detach().cpu()))
            step_tokens += int(attention_mask.sum().item())
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()
        step_time = time.perf_counter() - step_started
        total_tokens += step_tokens
        record = {
            "run_id": run_id,
            "protocol_identity": config["protocol_identity"],
            "state": args.state,
            "candidate": args.candidate,
            "step": step + 1,
            "loss": float(np.mean(losses)),
            "learning_rate": lr,
            "step_time_s": step_time,
            "tokens": step_tokens,
            "tokens_per_s": step_tokens / step_time,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        step_records.append(record)
        with steps_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
    elapsed = time.perf_counter() - execution_started
    adapter_path = Path("artifacts/models/lora") / run_id
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    summary = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "protocol_identity": config["protocol_identity"],
        "state": args.state,
        "candidate": args.candidate,
        "config_hash": canonical_sha256(config),
        "steps": steps,
        "fit_examples_available": len(fit),
        "global_batch_size": runtime["global_batch_size"],
        "sequence_length": config["data"]["sequence_length"],
        "training_use_cache": bool(model.config.use_cache),
        "dynamic_batch_padding": bool(runtime["dynamic_batch_padding"]),
        "elapsed_s": elapsed,
        "mean_step_time_s": float(np.mean([row["step_time_s"] for row in step_records])),
        "p95_step_time_s": float(np.quantile([row["step_time_s"] for row in step_records], 0.95)),
        "mean_tokens_per_s": total_tokens / sum(row["step_time_s"] for row in step_records),
        "initial_loss": step_records[0]["loss"],
        "final_loss": step_records[-1]["loss"],
        "peak_allocated_mib": max(row["peak_allocated_mib"] for row in step_records),
        "peak_reserved_mib": max(row["peak_reserved_mib"] for row in step_records),
        "adapter_path": str(adapter_path),
        "protected_data_used": False,
    }
    summary_path = Path("artifacts/analysis/lora") / f"{run_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
