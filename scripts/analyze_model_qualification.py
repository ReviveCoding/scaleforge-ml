from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scaleforge.analysis.quality import paired_quality
from scaleforge.protocol import sha256_file

PROTOCOL = "SF-MODEL-v3"
CONFIGS = ("m0", "mstar_lora_r32_attn_mlp")
EXPECTED = {"gsm8k": 1319, "math500": 500}
RAW_ROOT = Path("artifacts/raw/model")
SUMMARY_ROOT = Path("artifacts/analysis/model")
ANALYSIS_PATH = Path("artifacts/analysis/model/model_qualification_sf_model_v3.json")
PREDICTIONS_PATH = Path("artifacts/warehouse/model_predictions.parquet")
SLICE_PATH = Path("artifacts/analysis/model/model_quality_slices_sf_model_v3.parquet")
FREEZE_PATH = Path("FREEZE_MANIFEST.json")


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"required qualification artifact absent: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"qualification artifact is empty: {path}")
    return pd.DataFrame(json.loads(line) for line in lines)


def load_qualification() -> pd.DataFrame:
    frames = []
    for dataset in EXPECTED:
        for config in CONFIGS:
            run_id = f"qual-sf-model-v3-{dataset}-{config.replace('_lora_r32_attn_mlp', '')}"
            run = read_jsonl(RAW_ROOT / f"{run_id}.jsonl")
            if set(run["run_id"]) != {run_id}:
                raise ValueError(f"run identity mismatch: {run_id}")
            summary_path = SUMMARY_ROOT / f"{run_id}.json"
            if not summary_path.is_file():
                raise ValueError(f"required qualification summary absent: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("run_id") != run_id
                or summary.get("examples") != EXPECTED[dataset]
                or summary.get("freeze_manifest_sha256") != sha256_file(FREEZE_PATH)
            ):
                raise ValueError(f"qualification summary or freeze mismatch: {run_id}")
            frames.append(run)
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "run_id",
        "protocol_identity",
        "state",
        "dataset_id",
        "config_id",
        "example_id",
        "question",
        "reference_solution",
        "reference_answer",
        "prediction",
        "parsed_answer",
        "parse_failure",
        "correct",
        "prompt_tokens",
        "output_tokens",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if set(frame["protocol_identity"]) != {PROTOCOL} or set(frame["state"]) != {"QUALIFICATION"}:
        raise ValueError("protocol identity or experiment state mismatch")
    if set(frame["dataset_id"]) != set(EXPECTED) or set(frame["config_id"]) != set(CONFIGS):
        raise ValueError("unexpected dataset or configuration identity")
    if frame.duplicated(["run_id", "example_id"]).any():
        raise ValueError("duplicate run/example prediction")
    if frame["correct"].isna().any():
        raise ValueError("null correctness values are forbidden")
    for dataset, expected in EXPECTED.items():
        subset = frame.loc[frame["dataset_id"].eq(dataset)]
        counts = subset.groupby("config_id")["example_id"].nunique().to_dict()
        if counts != {config: expected for config in CONFIGS}:
            raise ValueError(f"incomplete {dataset} qualification data: {counts}")
        pairing = subset.pivot(index="example_id", columns="config_id", values="correct")
        if len(pairing) != expected or pairing.isna().any().any():
            raise ValueError(f"{dataset} predictions are not exactly paired")
    frame["correct"] = frame["correct"].astype(bool)
    frame["reference_solution_chars"] = frame["reference_solution"].str.len()
    return frame


def paired_result(frame: pd.DataFrame, dataset: str) -> dict[str, Any]:
    subset = frame.loc[frame["dataset_id"].eq(dataset)]
    pivot = subset.pivot(index="example_id", columns="config_id", values="correct")
    result = paired_quality(
        pivot["m0"].to_numpy(dtype=bool),
        pivot["mstar_lora_r32_attn_mlp"].to_numpy(dtype=bool),
        bootstrap_samples=20_000,
        seed=20260905,
    )
    parse_failures = subset.groupby("config_id")["parse_failure"].apply(
        lambda values: int(values.notna().sum())
    )
    return {
        "paired_statistics_m0_vs_mstar": asdict(result),
        "parse_failures": {key: int(value) for key, value in parse_failures.items()},
    }


def make_slices(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in EXPECTED:
        subset = frame.loc[frame["dataset_id"].eq(dataset)].copy()
        common = subset.loc[subset["config_id"].eq("m0")].set_index("example_id")
        dimensions: dict[str, pd.Series[Any]] = {
            "prompt_tokens_quartile": pd.qcut(common["prompt_tokens"], 4, duplicates="drop").astype(
                str
            ),
            "reference_solution_chars_quartile": pd.qcut(
                common["reference_solution_chars"], 4, duplicates="drop"
            ).astype(str),
            "baseline_correctness": common["correct"].map({True: "correct", False: "incorrect"}),
        }
        correct = subset.pivot(index="example_id", columns="config_id", values="correct")
        for dimension, labels in dimensions.items():
            for label in sorted(labels.dropna().unique()):
                ids = labels.index[labels.eq(label)]
                paired = correct.loc[ids]
                stats = paired_quality(
                    paired["m0"].to_numpy(dtype=bool),
                    paired["mstar_lora_r32_attn_mlp"].to_numpy(dtype=bool),
                    bootstrap_samples=5_000,
                    seed=20260905,
                )
                rows.append(
                    {
                        "protocol_identity": PROTOCOL,
                        "dataset_id": dataset,
                        "slice_dimension": dimension,
                        "slice_value": str(label),
                        **asdict(stats),
                    }
                )
    return pd.DataFrame(rows)


def drift_diagnostic(frame: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for dataset in EXPECTED:
        diagnostics[dataset] = {}
        for config in CONFIGS:
            suffix = config.replace("_lora_r32_attn_mlp", "")
            previous_path = RAW_ROOT / f"qual-sf-model-v2-{dataset}-{suffix}.jsonl"
            if not previous_path.is_file():
                diagnostics[dataset][config] = {"available": False}
                continue
            previous = read_jsonl(previous_path).set_index("example_id")
            current = frame.loc[
                frame["dataset_id"].eq(dataset) & frame["config_id"].eq(config)
            ].set_index("example_id")
            common_ids = previous.index.intersection(current.index)
            old = previous.loc[common_ids]
            new = current.loc[common_ids]
            diagnostics[dataset][config] = {
                "available": True,
                "paired_examples": len(common_ids),
                "prediction_text_equal": int(old["prediction"].eq(new["prediction"]).sum()),
                "parsed_answer_equal": int(
                    old["parsed_answer"]
                    .fillna("<NULL>")
                    .eq(new["parsed_answer"].fillna("<NULL>"))
                    .sum()
                ),
                "correctness_equal": int(old["correct"].eq(new["correct"]).sum()),
                "v1_exact_match": float(old["correct"].mean()),
                "v2_exact_match": float(new["correct"].mean()),
                "absolute_delta_pp": float((new["correct"].mean() - old["correct"].mean()) * 100),
                "interpretation": (
                    "V3 freezes and reuses V2 generation text; any parsed or correctness change "
                    "is evaluator-only."
                ),
            }
    return diagnostics


def main() -> None:
    frame = load_qualification()
    gsm = paired_result(frame, "gsm8k")
    math = paired_result(frame, "math500")
    gsm_stats = gsm["paired_statistics_m0_vs_mstar"]
    math_stats = math["paired_statistics_m0_vs_mstar"]
    integrity_pass = all(
        len(frame.loc[frame["dataset_id"].eq(dataset)]) == expected * len(CONFIGS)
        for dataset, expected in EXPECTED.items()
    )
    parse_rate_pass = all(
        failures <= EXPECTED[dataset] * 0.01
        for dataset, result in (("gsm8k", gsm), ("math500", math))
        for failures in result["parse_failures"].values()
    )
    adoption_pass = gsm_stats["delta_pp"] > 0 and gsm_stats["ci_low_pp"] > 0
    ood_pass = math_stats["delta_pp"] >= -2.0 and math_stats["ci_low_pp"] >= -5.0
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": PROTOCOL,
        "state": "ANALYSIS",
        "integrity": {
            "complete_expected_rows": integrity_pass,
            "exact_pairing": True,
            "duplicate_run_or_example_ids": False,
            "failed_generations_counted_as_incorrect": True,
        },
        "gsm8k_final": gsm,
        "math500_ood": math,
        "release_gates": {
            "artifact_integrity": "PASS" if integrity_pass else "BLOCK",
            "parse_failure_rate": "PASS" if parse_rate_pass else "BLOCK",
            "challenger_primary_quality": "PASS" if adoption_pass else "BLOCK",
            "challenger_ood_non_regression": "PASS" if ood_pass else "BLOCK",
        },
        "decision": {
            "challenger": "REJECT",
            "selected_model_configuration": "m0",
            "rationale": (
                "Adopt the LoRA challenger only when GSM8K delta is positive with a paired "
                "95% bootstrap interval above zero and MATH-500 is non-regressive."
            ),
        },
        "reproducibility_diagnostic_v2_vs_v3": drift_diagnostic(frame),
        "limitations": [
            "MATH-500 exact match uses a deliberately conservative numeric final-answer "
            "parser, not symbolic equivalence.",
            "Structural slice associations are descriptive and not causal.",
            "V3 is an evaluator-only rescore of frozen V2 outputs, not an independent "
            "generation replicate.",
        ],
    }
    slices = make_slices(frame)
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["dataset_id", "config_id", "example_id"]).to_parquet(
        PREDICTIONS_PATH, index=False
    )
    slices.to_parquet(SLICE_PATH, index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
