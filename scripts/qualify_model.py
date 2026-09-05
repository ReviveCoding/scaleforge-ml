from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scaleforge.evaluation.answers import (
    normalize_numeric_answer,
    parse_generated_answer,
    parse_reference_gsm8k,
)
from scaleforge.modeling.prompts import PromptContract, build_messages
from scaleforge.protocol import canonical_sha256, sha256_file

FREEZE_PATH = Path("FREEZE_MANIFEST.json")
CONFIG_PATH = Path("configs/model/qualification.yaml")
LEDGER_PATH = Path("FINAL_ACCESS_LEDGER.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("gsm8k", "math500"))
    parser.add_argument("config_id", choices=("m0", "mstar_lora_r32_attn_mlp"))
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def verify_freeze(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    if manifest["state"] != "FROZEN" or manifest["protocol_identity"] != "SF-MODEL-v1":
        raise ValueError("invalid frozen protocol identity")
    if canonical_sha256(config) != manifest["configuration_sha256"]:
        raise ValueError("qualification configuration differs from freeze")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if git_sha != manifest["git_sha"]:
        raise ValueError("Git HEAD differs from frozen source commit")
    for path_text, expected_hash in manifest["file_sha256"].items():
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"frozen artifact mismatch: {path}")


def update_ledger(entry: dict[str, Any]) -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    entries = ledger["entries"]
    existing = [
        index for index, value in enumerate(entries) if value["access_id"] == entry["access_id"]
    ]
    if len(existing) > 1:
        raise ValueError("duplicate protected access IDs")
    if existing:
        entries[existing[0]] = entry
    else:
        entries.append(entry)
    ledger["updated_at"] = datetime.now(UTC).isoformat()
    temporary = LEDGER_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LEDGER_PATH)


def acquire(dataset_id: str, dataset: dict[str, Any], run_id: str) -> tuple[pd.DataFrame, Path]:
    suffix = ".parquet" if dataset_id == "gsm8k" else ".jsonl"
    raw_path = Path("artifacts/data/protected") / f"{dataset_id}-{dataset['revision']}{suffix}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        urllib.request.urlretrieve(dataset["source_url"], raw_path)
    if dataset_id == "gsm8k":
        frame = pd.read_parquet(raw_path)
    else:
        frame = pd.read_json(raw_path, lines=True)
    if len(frame) != int(dataset["expected_rows"]):
        raise ValueError(f"protected dataset cardinality mismatch: {len(frame)}")
    question_column = "question" if dataset_id == "gsm8k" else "problem"
    if (
        frame[question_column].isna().any()
        or frame[question_column].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("protected dataset contains null or blank questions")
    if frame[question_column].duplicated().any():
        raise ValueError("protected dataset contains duplicate questions")
    manifest_path = Path("artifacts/manifests") / f"protected_{dataset_id}_{run_id}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset": dataset,
                "rows": len(frame),
                "raw_sha256": sha256_file(raw_path),
                "retrieved_or_verified_at": datetime.now(UTC).isoformat(),
                "raw_data_committed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return frame, raw_path


def reference_answer(dataset_id: str, row: Any) -> tuple[str, str]:
    if dataset_id == "gsm8k":
        parsed = parse_reference_gsm8k(str(row.answer))
        if parsed.failure or parsed.normalized is None:
            raise ValueError(f"invalid GSM8K protected reference: {parsed.failure}")
        return parsed.normalized, str(row.answer)
    raw = str(row.answer)
    return normalize_numeric_answer(raw), str(row.solution)


def main() -> None:
    args = parse_args()
    output_path = Path("artifacts/raw/model") / f"{args.run_id}.jsonl"
    summary_path = Path("artifacts/analysis/model") / f"{args.run_id}.json"
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(f"protected run ID already exists: {args.run_id}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    verify_freeze(config, freeze)
    dataset = config["datasets"][args.dataset]
    access_id = f"access-{args.run_id}"
    entry: dict[str, Any] = {
        "access_id": access_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "protocol_identity": "SF-MODEL-v1",
        "dataset": dataset["repository"],
        "revision": dataset["revision"],
        "requested_split": dataset["split"],
        "purpose": f"Protected {args.dataset} qualification for {args.config_id}",
        "run_id": args.run_id,
        "outcome_exposed": False,
        "scientific_outcomes_exposed": [],
        "artifacts": [],
        "disposition": "QUALIFICATION_STARTED",
    }
    update_ledger(entry)
    records: list[dict[str, Any]] = []
    try:
        frame, raw_path = acquire(args.dataset, dataset, args.run_id)
        model_id = config["model"]["repository"]
        revision = config["model"]["revision"]
        cache_dir = Path("artifacts/cache/huggingface")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, dtype=torch.bfloat16
        ).to("cuda")
        if args.config_id != "m0":
            model = PeftModel.from_pretrained(model, config["challenger"]["adapter_path"])
        model.eval()
        baseline_config = yaml.safe_load(Path("configs/model/baselines.yaml").read_text("utf-8"))
        prompt = PromptContract("m0", baseline_config["prompts"]["m0"]["system"], 0)
        question_column = "question" if args.dataset == "gsm8k" else "problem"
        questions = frame[question_column].astype(str).tolist()
        rendered = [
            tokenizer.apply_chat_template(
                build_messages(question, prompt, []), tokenize=False, add_generation_prompt=True
            )
            for question in questions
        ]
        batch_size = int(config["generation"]["batch_size"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started_run = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for offset in range(0, len(frame), batch_size):
            rows = frame.iloc[offset : offset + batch_size]
            prompts = rendered[offset : offset + batch_size]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            torch.cuda.synchronize()
            started_batch = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=int(config["generation"]["max_new_tokens"]),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - started_batch
            input_width = int(inputs["input_ids"].shape[1])
            batch_records = []
            for local_index, sequence in enumerate(generated):
                row = rows.iloc[local_index]
                text = tokenizer.decode(sequence[input_width:], skip_special_tokens=True)
                parsed = parse_generated_answer(text)
                reference, raw_solution = reference_answer(args.dataset, row)
                example_id = (
                    f"gsm8k-test-{offset + local_index:04d}"
                    if args.dataset == "gsm8k"
                    else str(row.unique_id)
                )
                batch_records.append(
                    {
                        "run_id": args.run_id,
                        "protocol_identity": "SF-MODEL-v1",
                        "state": "QUALIFICATION",
                        "dataset_id": args.dataset,
                        "split_role": dataset["split_role"],
                        "config_id": args.config_id,
                        "example_id": example_id,
                        "question": questions[offset + local_index],
                        "reference_solution": raw_solution,
                        "reference_answer": reference,
                        "prediction": text,
                        "parsed_answer": parsed.normalized,
                        "parse_failure": parsed.failure,
                        "correct": parsed.normalized == reference,
                        "prompt_tokens": int(inputs["attention_mask"][local_index].sum().item()),
                        "output_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                        "batch_id": f"{offset // batch_size:04d}",
                        "batch_elapsed_s": batch_elapsed,
                    }
                )
            records.extend(batch_records)
            with output_path.open("a", encoding="utf-8") as stream:
                for record in batch_records:
                    stream.write(json.dumps(record) + "\n")
                stream.flush()
        if len(records) != int(dataset["expected_rows"]):
            raise ValueError("qualification result cardinality mismatch")
        summary = {
            "schema_version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": args.run_id,
            "protocol_identity": "SF-MODEL-v1",
            "state": "QUALIFICATION",
            "dataset_id": args.dataset,
            "config_id": args.config_id,
            "examples": len(records),
            "exact_match": sum(bool(row["correct"]) for row in records) / len(records),
            "parse_failures": sum(row["parse_failure"] is not None for row in records),
            "elapsed_s": time.perf_counter() - started_run,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "freeze_manifest_sha256": sha256_file(FREEZE_PATH),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        entry["outcome_exposed"] = True
        entry["scientific_outcomes_exposed"] = ["predictions", "aggregate_exact_match"]
        entry["artifacts"] = [str(output_path), str(summary_path), str(raw_path)]
        entry["disposition"] = "QUALIFICATION_COMPLETED"
    finally:
        if output_path.exists() and output_path.stat().st_size > 0:
            entry["outcome_exposed"] = True
            if not entry["scientific_outcomes_exposed"]:
                entry["scientific_outcomes_exposed"] = ["partial_predictions"]
            if str(output_path) not in entry["artifacts"]:
                entry["artifacts"].append(str(output_path))
            if entry["disposition"] == "QUALIFICATION_STARTED":
                entry["disposition"] = "QUALIFICATION_INCOMPLETE_OUTCOMES_EXPOSED"
        update_ledger(entry)


if __name__ == "__main__":
    main()
