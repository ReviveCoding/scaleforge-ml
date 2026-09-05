from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scaleforge.benchmarks.training import (
    summarize_training_steps,
    validate_training_system_config,
)
from scaleforge.protocol import canonical_sha256, sha256_file
from scaleforge.training.cache import tensor_cache
from scaleforge.training.examples import (
    right_padded_batch_width,
    validate_right_padded_attention_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--run-order", type=int, required=True)
    parser.add_argument("--state", choices=("DEVELOPMENT", "QUALIFICATION"), required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/training/systems.yaml"))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def start_telemetry(path: Path) -> subprocess.Popen[str] | None:
    fields = "timestamp,index,name,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used"
    try:
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
    except OSError:
        return None


def verify_qualification_freeze(config: dict[str, Any], config_path: Path) -> tuple[str, str]:
    manifest_path = Path("artifacts/manifests/training_freeze.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config_path.resolve() != Path(manifest["configuration_path"]).resolve():
        raise ValueError("qualification config path differs from the frozen path")
    if config["protocol_identity"] != manifest["protocol_identity"]:
        raise ValueError("qualification protocol identity differs from the freeze manifest")
    if canonical_sha256(config) != manifest["configuration_sha256"]:
        raise ValueError("qualification config hash differs from the freeze manifest")
    for path_text, expected_hash in manifest["file_sha256"].items():
        if sha256_file(Path(path_text)) != expected_hash:
            raise ValueError(f"frozen source hash mismatch: {path_text}")
    if sha256_file(Path(config["data"]["path"])) != manifest["data_sha256"]:
        raise ValueError("frozen training data hash mismatch")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if git_sha != manifest["git_sha"]:
        raise ValueError("current Git SHA differs from the training freeze manifest")
    return str(manifest_path), sha256_file(manifest_path)


def main() -> None:
    args = parse_args()
    raw_dir = Path("artifacts/raw/training")
    analysis_dir = Path("artifacts/analysis/training")
    profile_dir = Path("artifacts/profiles/training")
    for directory in (raw_dir, analysis_dir, profile_dir):
        directory.mkdir(parents=True, exist_ok=True)
    steps_path = raw_dir / f"{args.run_id}-steps.jsonl"
    summary_path = analysis_dir / f"{args.run_id}.json"
    failure_path = Path("artifacts/failures") / f"{args.run_id}.json"
    if any(path.exists() for path in (steps_path, summary_path, failure_path)):
        raise FileExistsError(f"run ID already exists: {args.run_id}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    workload = validate_training_system_config(config)
    freeze_path = None
    freeze_sha256 = None
    if args.state == "QUALIFICATION":
        if config["state"] != "FROZEN":
            raise ValueError("qualification requires a FROZEN configuration")
        freeze_path, freeze_sha256 = verify_qualification_freeze(config, args.config)
    elif args.state != config["state"]:
        raise ValueError("requested state differs from the configuration state")
    if args.config_id not in config["configs"]:
        raise ValueError(f"unknown training config: {args.config_id}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA support is required")
    baseline = yaml.safe_load(Path("configs/model/baselines.yaml").read_text(encoding="utf-8"))
    selected = config["configs"][args.config_id]
    telemetry_path = raw_dir / f"{args.run_id}-gpu.csv"
    telemetry: subprocess.Popen[str] | None = None
    started_at = datetime.now(UTC).isoformat()
    try:
        seed = int(config["workload"]["seed"])
        seed_everything(seed)
        frame = pd.read_parquet(config["data"]["path"])
        fit = (
            frame.loc[frame["split_role"].eq("FIT")]
            .sort_values("example_id")
            .head(int(config["workload"]["examples"]))
        )
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
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=workload.micro_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            drop_last=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, dtype=torch.bfloat16
        ).to("cuda")
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
        compile_prepare_started = time.perf_counter()
        compile_cache_path = None
        if selected["compile"]:
            compile_cache_path = Path("artifacts/cache/torchinductor") / args.run_id
            compile_cache_path.mkdir(parents=True, exist_ok=False)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache_path.resolve())
            model = torch.compile(model, dynamic=bool(selected["dynamic_batch_padding"]))
        compile_prepare_s = time.perf_counter() - compile_prepare_started
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["workload"]["learning_rate"]),
            weight_decay=float(config["workload"]["weight_decay"]),
        )
        total_steps = workload.warmup_steps + workload.measured_steps
        profiler = None
        if args.profile:
            active = min(
                int(config["qualification"]["profiler_active_steps"]), workload.measured_steps
            )
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=0, warmup=workload.warmup_steps, active=active, repeat=1),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            profiler.start()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        telemetry = start_telemetry(telemetry_path)
        iterator = iter(loader)
        records: list[dict[str, Any]] = []
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            step_tokens = 0
            losses = []
            for _ in range(workload.gradient_accumulation_steps):
                try:
                    input_ids, attention_mask, labels = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    input_ids, attention_mask, labels = next(iterator)
                input_ids = input_ids.to("cuda", dtype=torch.long, non_blocking=True)
                attention_mask = attention_mask.to("cuda", non_blocking=True)
                labels = labels.to("cuda", dtype=torch.long, non_blocking=True)
                if selected["dynamic_batch_padding"]:
                    width = right_padded_batch_width(attention_mask)
                    input_ids = input_ids[:, :width]
                    attention_mask = attention_mask[:, :width]
                    labels = labels[:, :width]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / workload.gradient_accumulation_steps
                loss.backward()
                losses.append(float(outputs.loss.detach().cpu()))
                step_tokens += int(attention_mask.sum().item())
            optimizer.step()
            torch.cuda.synchronize()
            step_time = time.perf_counter() - step_started
            record = {
                "run_id": args.run_id,
                "protocol_identity": config["protocol_identity"],
                "state": args.state,
                "config_id": args.config_id,
                "replicate": args.replicate,
                "run_order": args.run_order,
                "step": step + 1,
                "measured": step >= workload.warmup_steps,
                "loss": float(np.mean(losses)),
                "step_time_s": step_time,
                "tokens": step_tokens,
                "tokens_per_s": step_tokens / step_time,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            }
            records.append(record)
            with steps_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            if profiler is not None:
                profiler.step()
        if profiler is not None:
            profiler.stop()
            profiler.export_chrome_trace(str(profile_dir / f"{args.run_id}-trace.json"))
            table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
            (profile_dir / f"{args.run_id}-top-operators.txt").write_text(
                table + "\n", encoding="utf-8"
            )
        summary = {
            "schema_version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "started_at": started_at,
            "run_id": args.run_id,
            "protocol_identity": config["protocol_identity"],
            "state": args.state,
            "status": "COMPLETED",
            "config_id": args.config_id,
            "replicate": args.replicate,
            "run_order": args.run_order,
            "config_hash": canonical_sha256(config),
            "freeze_manifest_path": freeze_path,
            "freeze_manifest_sha256": freeze_sha256,
            "hypothesis": selected["hypothesis"],
            "dynamic_batch_padding": bool(selected["dynamic_batch_padding"]),
            "compile": bool(selected["compile"]),
            "compile_cache_path": str(compile_cache_path) if compile_cache_path else None,
            "compile_prepare_s": compile_prepare_s,
            "warmup_step_times_s": [row["step_time_s"] for row in records if not row["measured"]],
            **summarize_training_steps(records),
            "telemetry_path": str(telemetry_path) if telemetry is not None else None,
            "profile_trace": (
                str(profile_dir / f"{args.run_id}-trace.json") if args.profile else None
            ),
            "protected_data_used": False,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    except Exception as error:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "created_at": datetime.now(UTC).isoformat(),
                    "started_at": started_at,
                    "run_id": args.run_id,
                    "state": args.state,
                    "config_id": args.config_id,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "partial_steps_path": str(steps_path) if steps_path.exists() else None,
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


if __name__ == "__main__":
    main()
