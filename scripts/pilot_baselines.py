from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from scaleforge.evaluation.answers import parse_generated_answer
from scaleforge.modeling.prompts import PromptContract, build_messages, select_few_shots
from scaleforge.protocol import canonical_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/model/baselines.yaml"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/split_manifest.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/raw/model"))
    parser.add_argument("--limit", type=int, default=None, help="0 means all validation rows")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--state", choices=("PILOT", "DEVELOPMENT"), default="PILOT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["state"] != "DEVELOPMENT":
        raise ValueError("baseline pilot requires DEVELOPMENT state")
    if config["generation"]["do_sample"] is not False:
        raise ValueError("quality baseline must be deterministic")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("this pilot requires exactly one visible CUDA GPU")

    frame = pd.read_parquet(args.data)
    validation = frame.loc[frame["split_role"].eq("VALIDATION")].sort_values("example_id")
    limit = int(config["pilot"]["validation_examples"]) if args.limit is None else args.limit
    if limit > 0:
        validation = validation.head(limit)
    if validation.empty:
        raise ValueError("no authorized validation rows")
    model_id = config["model"]["repository"]
    revision = config["model"]["revision"]
    cache_dir = Path("artifacts/cache/huggingface")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    tokenizer.padding_side = str(config["generation"]["padding_side"])
    if tokenizer.padding_side != "left":
        raise ValueError("decoder-only batched generation requires left padding")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    run_id = args.run_id or (
        f"pilot-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{run_id}.jsonl"
    metadata_path = args.output_dir / f"{run_id}.meta.json"
    config_hash = canonical_sha256(config)
    metadata = {
        "run_id": run_id,
        "protocol_identity": config["protocol_identity"],
        "state": args.state,
        "config_hash": config_hash,
        "target_examples_per_config": len(validation),
    }
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata != metadata:
            raise ValueError("resume metadata differs from the frozen run contract")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    records: list[dict[str, Any]] = []
    if output_path.exists():
        records = [
            json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
        ]
    completed = {(row["config_id"], row["example_id"]) for row in records}
    run_started = time.perf_counter()
    for config_id in ("m0", "m1"):
        prompt_config = config["prompts"][config_id]
        contract = PromptContract(
            config_id, prompt_config["system"], prompt_config["few_shot_count"]
        )
        demonstrations = select_few_shots(frame, contract.few_shot_count)
        messages = [
            build_messages(row.question, contract, demonstrations)
            for row in validation.itertuples()
        ]
        rendered = [
            tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
            for item in messages
        ]
        batch_size = int(config["pilot"]["batch_size"])
        for offset in range(0, len(rendered), batch_size):
            rows = validation.iloc[offset : offset + batch_size]
            positions = [
                index
                for index in range(len(rows))
                if (config_id, str(rows.iloc[index]["example_id"])) not in completed
            ]
            if not positions:
                continue
            prompts = [rendered[offset + index] for index in positions]
            rows = rows.iloc[positions]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=int(config["generation"]["max_new_tokens"]),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            for index, sequence in enumerate(output):
                input_width = int(inputs["input_ids"].shape[1])
                generated = tokenizer.decode(sequence[input_width:], skip_special_tokens=True)
                parsed = parse_generated_answer(generated)
                reference = str(rows.iloc[index]["normalized_final_answer"])
                records.append(
                    {
                        "run_id": run_id,
                        "protocol_identity": config["protocol_identity"],
                        "state": args.state,
                        "batch_id": f"{config_id}-{offset // batch_size:04d}",
                        "config_id": config_id,
                        "example_id": str(rows.iloc[index]["example_id"]),
                        "prediction": generated,
                        "parsed_answer": parsed.normalized,
                        "parse_failure": parsed.failure,
                        "reference_answer": reference,
                        "correct": parsed.normalized == reference,
                        "batch_elapsed_s": elapsed,
                        "output_tokens": len(tokenizer.encode(generated, add_special_tokens=False)),
                    }
                )
            new_records = records[-len(rows) :]
            with output_path.open("a", encoding="utf-8") as stream:
                for record in new_records:
                    stream.write(json.dumps(record) + "\n")
                stream.flush()
    execution_elapsed = time.perf_counter() - run_started
    unique_batches = {
        row["batch_id"]: float(row["batch_elapsed_s"]) for row in records if "batch_id" in row
    }
    measured_generation_s = sum(unique_batches.values())
    summary = {
        "run_id": run_id,
        "protocol_identity": config["protocol_identity"],
        "state": args.state,
        "config_hash": config_hash,
        "model": model_id,
        "revision": revision,
        "examples_per_config": len(validation),
        "execution_elapsed_s": execution_elapsed,
        "measured_generation_s": measured_generation_s,
        "estimated_full_validation_s": measured_generation_s
        * len(frame.loc[frame["split_role"].eq("VALIDATION")])
        / len(validation),
        "results": {
            key: {
                "exact_match": sum(row["correct"] for row in records if row["config_id"] == key)
                / len(validation),
                "parse_failures": sum(
                    bool(row["parse_failure"]) for row in records if row["config_id"] == key
                ),
            }
            for key in ("m0", "m1")
        },
        "raw_predictions": str(output_path),
    }
    summary_path = Path("artifacts/analysis") / f"{run_id}-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
