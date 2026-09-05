from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scaleforge.analysis.quality import paired_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/raw/model/dev-baselines-sf-model-v1-b16.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/model/baseline_selection.json")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/warehouse/model_predictions_development.parquet"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    frame = pd.DataFrame(records)
    required = {
        "run_id",
        "protocol_identity",
        "state",
        "config_id",
        "example_id",
        "parsed_answer",
        "reference_answer",
        "correct",
        "parse_failure",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if frame.duplicated(["config_id", "example_id"]).any():
        raise ValueError("duplicate config/example predictions")
    counts = frame.groupby("config_id")["example_id"].nunique().to_dict()
    if counts != {"m0": 723, "m1": 723}:
        raise ValueError(f"incomplete baseline run: {counts}")
    pivot = frame.pivot(index="example_id", columns="config_id", values="correct")
    if pivot.isna().any().any() or len(pivot) != 723:
        raise ValueError("M0/M1 example sets are not perfectly paired")
    stats = paired_quality(
        pivot["m0"].to_numpy(dtype=bool),
        pivot["m1"].to_numpy(dtype=bool),
        bootstrap_samples=20_000,
        seed=20260905,
    )
    parse_failures = frame.groupby("config_id")["parse_failure"].apply(
        lambda values: int(values.notna().sum())
    )
    winner = "m0"
    if stats.candidate_em > stats.baseline_em or (
        stats.candidate_em == stats.baseline_em and parse_failures["m1"] < parse_failures["m0"]
    ):
        winner = "m1"
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": "SF-MODEL-v1",
        "state": "CANDIDATE_SELECTION",
        "run_id": str(frame["run_id"].iloc[0]),
        "expected_examples_per_config": 723,
        "counts": counts,
        "parse_failures": {key: int(value) for key, value in parse_failures.items()},
        "paired_statistics_m0_vs_m1": asdict(stats),
        "selection_rule": [
            "higher validation exact match",
            "lower parse-failure rate",
            "M0 simpler prompt tie-breaker",
        ],
        "selected_strongest_pretrained_baseline": winner,
        "protected_data_used": False,
        "claimable_as_protected_result": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["config_id", "example_id"]).to_parquet(args.predictions, index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
