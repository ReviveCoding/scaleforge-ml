from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scaleforge.analysis.quality import paired_quality

BASELINE_PATH = Path("artifacts/raw/model/dev-baselines-sf-model-v1-b16.jsonl")
CANDIDATE_PATH = Path("artifacts/raw/model/candidate-finalist-lora-r32-validation723.jsonl")
OUTPUT_PATH = Path("artifacts/model/candidate_selection.json")
PREDICTIONS_PATH = Path("artifacts/warehouse/model_predictions_candidate_selection.parquet")
EXPECTED = 723


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def main() -> None:
    baseline = read_jsonl(BASELINE_PATH)
    baseline = baseline.loc[baseline["config_id"].eq("m0")].copy()
    candidate = read_jsonl(CANDIDATE_PATH).copy()
    candidate["config_id"] = "mstar_lora_r32_attn_mlp"
    frame = pd.concat([baseline, candidate], ignore_index=True)
    required = {
        "run_id",
        "protocol_identity",
        "state",
        "config_id",
        "example_id",
        "correct",
        "parse_failure",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if set(frame["protocol_identity"]) != {"SF-MODEL-v1"}:
        raise ValueError("protocol identity mismatch")
    if frame.duplicated(["config_id", "example_id"]).any():
        raise ValueError("duplicate config/example predictions")
    counts = frame.groupby("config_id")["example_id"].nunique().to_dict()
    expected_counts = {"m0": EXPECTED, "mstar_lora_r32_attn_mlp": EXPECTED}
    if counts != expected_counts:
        raise ValueError(f"incomplete candidate-selection data: {counts}")
    pivot = frame.pivot(index="example_id", columns="config_id", values="correct")
    if len(pivot) != EXPECTED or pivot.isna().any().any():
        raise ValueError("baseline and candidate example sets are not perfectly paired")
    stats = paired_quality(
        pivot["m0"].to_numpy(dtype=bool),
        pivot["mstar_lora_r32_attn_mlp"].to_numpy(dtype=bool),
        bootstrap_samples=20_000,
        seed=20260905,
    )
    parse_failures = frame.groupby("config_id")["parse_failure"].apply(
        lambda values: int(values.notna().sum())
    )
    robust_improvement = stats.delta_pp > 0 and stats.ci_low_pp > 0
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": "SF-MODEL-v1",
        "state": "CANDIDATE_SELECTION",
        "expected_examples_per_config": EXPECTED,
        "counts": counts,
        "parse_failures": {key: int(value) for key, value in parse_failures.items()},
        "paired_statistics_m0_vs_mstar": asdict(stats),
        "selection_rule": (
            "Adopt LoRA only when delta is positive and the paired 95% bootstrap CI "
            "excludes zero; otherwise retain M0."
        ),
        "robust_improvement": robust_improvement,
        "frozen_mstar_challenger": "lora_r32_lr1e4_attn_mlp",
        "challenger_adapter": "artifacts/models/lora/finalist-lora-r32-dynamic-378s-retry1",
        "model_release_recommendation": ("mstar_lora_r32_attn_mlp" if robust_improvement else "m0"),
        "challenger_decision": "ADVANCE" if robust_improvement else "REJECT_RETAIN_M0",
        "protected_data_used": False,
        "claimable_as_protected_result": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["config_id", "example_id"]).to_parquet(PREDICTIONS_PATH, index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
