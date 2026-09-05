from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from scaleforge.data.lengths import choose_sequence_limit, quantiles
from scaleforge.data.schema import MathExample, SplitFractions
from scaleforge.data.splits import (
    assert_no_normalized_overlap,
    assign_train_role,
    normalize_question,
    stable_example_id,
)
from scaleforge.evaluation.answers import parse_reference_gsm8k
from scaleforge.protocol import canonical_sha256
from scaleforge.types import SplitRole


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare authorized GSM8K development data")
    parser.add_argument("--config", type=Path, default=Path("configs/data/gsm8k.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    return parser.parse_args()


def build_rows(
    dataset: list[dict[str, Any]],
    *,
    source_revision: str,
    tokenizer: PreTrainedTokenizerBase,
    fractions: SplitFractions,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, item in enumerate(dataset):
        question = str(item.get("question", ""))
        solution = str(item.get("answer", ""))
        parsed = parse_reference_gsm8k(solution)
        if parsed.failure or parsed.raw is None or parsed.normalized is None:
            failures.append({"row_index": index, "failure": parsed.failure})
            continue
        example_id = stable_example_id("openai/gsm8k:main", source_revision, question)
        role = assign_train_role(example_id, fractions)
        prompt_tokens = len(tokenizer.encode(question, add_special_tokens=False))
        solution_tokens = len(tokenizer.encode(solution, add_special_tokens=False))
        record = MathExample(
            example_id=example_id,
            question=question,
            raw_solution=solution,
            reference_final_answer=parsed.raw,
            normalized_final_answer=parsed.normalized,
            prompt_token_count=prompt_tokens,
            reference_solution_token_count=solution_tokens,
            split_role=role,
            source_revision=source_revision,
        )
        rows.append(record.model_dump(mode="json"))
    return rows, failures


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["authorized_split"] != "train":
        raise ValueError("this development script is hard-limited to the official train split")

    api = HfApi()
    dataset_info = api.dataset_info(config["dataset"], revision="main")
    model_info = api.model_info(config["model_tokenizer"], revision="main")
    dataset_revision = dataset_info.sha
    tokenizer_revision = model_info.sha
    if not dataset_revision or not tokenizer_revision:
        raise RuntimeError("immutable source revisions could not be resolved")

    repository_files = api.list_repo_files(
        config["dataset"],
        repo_type="dataset",
        revision=dataset_revision,
    )
    train_files = [
        name
        for name in repository_files
        if name.startswith(f"{config['configuration']}/train-") and name.endswith(".parquet")
    ]
    if len(train_files) != 1:
        raise RuntimeError(
            f"expected exactly one split-specific train parquet, found {train_files}"
        )
    train_path = hf_hub_download(
        repo_id=config["dataset"],
        filename=train_files[0],
        repo_type="dataset",
        revision=dataset_revision,
        local_dir=args.output_dir / "source",
    )
    dataset = pd.read_parquet(train_path).to_dict(orient="records")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_tokenizer"], revision=tokenizer_revision, use_fast=True
    )
    fractions = SplitFractions(**config["fractions"])
    rows, parse_failures = build_rows(
        dataset,
        source_revision=dataset_revision,
        tokenizer=tokenizer,
        fractions=fractions,
    )
    if parse_failures:
        raise ValueError(f"reference parser failures: {parse_failures[:5]}")
    if len(rows) != len(dataset):
        raise ValueError("row count changed during validation")

    frame = pd.DataFrame(rows)
    normalized = frame["question"].map(normalize_question)
    duplicate_count = int(normalized.duplicated(keep=False).sum())
    assert_no_normalized_overlap(
        {
            role: frame.loc[frame["split_role"].eq(role.value), "question"].tolist()
            for role in (SplitRole.FIT, SplitRole.VALIDATION, SplitRole.POLICY)
        }
    )
    if frame["example_id"].duplicated().any():
        raise ValueError("duplicate example IDs are forbidden")
    if frame.isna().any().any():
        raise ValueError("null curated fields are forbidden")

    prompt_lengths = frame["prompt_token_count"].astype(int).tolist()
    solution_lengths = frame["reference_solution_token_count"].astype(int).tolist()
    combined_lengths = (
        (frame["prompt_token_count"] + frame["reference_solution_token_count"]).astype(int).tolist()
    )
    combined_summary = quantiles(combined_lengths)
    sequence_limit = choose_sequence_limit(
        combined_summary["max"],
        float(config["sequence_headroom_ratio"]),
        int(config["sequence_rounding_multiple"]),
        int(config["maximum_allowed_sequence_length"]),
    )

    frame = frame.sort_values("example_id").reset_index(drop=True)
    fingerprint_columns = frame.astype(str).agg("\0".join, axis=1).tolist()
    data_fingerprint = hashlib.sha256("\n".join(fingerprint_columns).encode()).hexdigest()
    expected_fingerprint = config.get("expected_curated_fingerprint")
    if expected_fingerprint and data_fingerprint != expected_fingerprint:
        raise ValueError(
            f"curated fingerprint drift: expected {expected_fingerprint}, got {data_fingerprint}"
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "split_manifest.parquet", index=False)
    captured_at = datetime.now(UTC).isoformat()
    quality = {
        "schema_version": "1.0.0",
        "captured_at": captured_at,
        "authorized_source_split": "train",
        "row_count": len(frame),
        "role_counts": dict(Counter(frame["split_role"])),
        "null_cells": 0,
        "blank_questions": int(frame["question"].str.strip().eq("").sum()),
        "duplicate_normalized_question_rows": duplicate_count,
        "duplicate_example_ids": 0,
        "cross_role_overlap": 0,
        "malformed_reference_answers": 0,
        "prompt_token_lengths": quantiles(prompt_lengths),
        "reference_solution_token_lengths": quantiles(solution_lengths),
        "combined_token_lengths": combined_summary,
        "selected_sequence_length": sequence_limit,
        "selection_policy": "ceil(max observed * 1.10 / 128) * 128; fail above 4096",
        "silent_truncation_allowed": False,
        "data_fingerprint_sha256": data_fingerprint,
        "fingerprint_matches_frozen_expectation": bool(
            expected_fingerprint and data_fingerprint == expected_fingerprint
        ),
    }
    manifest = {
        "schema_version": "1.0.0",
        "captured_at": captured_at,
        "dataset": config["dataset"],
        "configuration": config["configuration"],
        "accessed_splits": ["train"],
        "protected_splits_accessed": [],
        "dataset_revision": dataset_revision,
        "dataset_license": str(getattr(dataset_info.card_data, "license", "unknown")),
        "dataset_rows": len(frame),
        "dataset_fingerprint_sha256": data_fingerprint,
        "tokenizer": config["model_tokenizer"],
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_artifact_hash": hashlib.sha256(
            tokenizer.backend_tokenizer.to_str().encode()
        ).hexdigest(),
        "split_config_hash": canonical_sha256(config["fractions"]),
        "raw_data_committed": False,
    }
    (output_dir / "data_quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"quality": quality, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
