from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scaleforge.evaluation.answers import parse_generated_answer
from scaleforge.modeling.prompts import PromptContract, build_messages
from scaleforge.protocol import canonical_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state", choices=("PILOT", "CANDIDATE_SELECTION"), default="PILOT")
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/split_manifest.parquet"))
    parser.add_argument("--config", type=Path, default=Path("configs/model/baselines.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.data)
    validation = (
        frame.loc[frame["split_role"].eq("VALIDATION")].sort_values("example_id").head(args.limit)
    )
    if len(validation) != args.limit:
        raise ValueError("validation diagnostic cardinality mismatch")
    model_id = config["model"]["repository"]
    revision = config["model"]["revision"]
    cache_dir = Path("artifacts/cache/huggingface")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir, dtype=torch.bfloat16
    ).to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    contract = PromptContract("m0", config["prompts"]["m0"]["system"], 0)
    rendered = [
        tokenizer.apply_chat_template(
            build_messages(str(row.question), contract, []),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in validation.itertuples()
    ]
    output_dir = Path("artifacts/raw/model")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run_id}.jsonl"
    if output_path.exists():
        raise FileExistsError(args.run_id)
    records: list[dict[str, object]] = []
    batch_size = int(config["pilot"]["batch_size"])
    torch.cuda.reset_peak_memory_stats()
    started_run = time.perf_counter()
    for offset in range(0, len(rendered), batch_size):
        prompts = rendered[offset : offset + batch_size]
        rows = validation.iloc[offset : offset + batch_size]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        batch_elapsed = time.perf_counter() - started
        input_width = int(inputs["input_ids"].shape[1])
        batch_records: list[dict[str, object]] = []
        for index, sequence in enumerate(generated_ids):
            text = tokenizer.decode(sequence[input_width:], skip_special_tokens=True)
            parsed = parse_generated_answer(text)
            reference = str(rows.iloc[index]["normalized_final_answer"])
            batch_records.append(
                {
                    "run_id": args.run_id,
                    "protocol_identity": "SF-MODEL-v1",
                    "state": args.state,
                    "config_id": args.candidate,
                    "example_id": str(rows.iloc[index]["example_id"]),
                    "prediction": text,
                    "parsed_answer": parsed.normalized,
                    "parse_failure": parsed.failure,
                    "reference_answer": reference,
                    "correct": parsed.normalized == reference,
                    "batch_id": f"{offset // batch_size:04d}",
                    "batch_elapsed_s": batch_elapsed,
                    "output_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                }
            )
        records.extend(batch_records)
        with output_path.open("a", encoding="utf-8") as stream:
            for record in batch_records:
                stream.write(json.dumps(record) + "\n")
            stream.flush()
    summary = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "protocol_identity": "SF-MODEL-v1",
        "state": args.state,
        "candidate": args.candidate,
        "adapter_path": str(args.adapter),
        "adapter_config_hash": canonical_sha256(
            json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        ),
        "examples": len(records),
        "exact_match": sum(bool(row["correct"]) for row in records) / len(records),
        "parse_failures": sum(row["parse_failure"] is not None for row in records),
        "elapsed_s": time.perf_counter() - started_run,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "protected_data_used": False,
        "claimable_result": False,
    }
    summary_path = Path("artifacts/analysis/lora") / f"{args.run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
