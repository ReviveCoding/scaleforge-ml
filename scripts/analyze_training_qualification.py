from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from scaleforge.analysis.training import (
    compile_break_even_steps,
    paired_percent_change,
    projected_wall_time_s,
)
from scaleforge.protocol import canonical_sha256, sha256_file
from scaleforge.telemetry import read_nvidia_smi_csv

PROTOCOL = "SF-TRAIN-v1"
CONFIG_PATH = Path("configs/training/qualification.yaml")
FREEZE_PATH = Path("artifacts/manifests/training_freeze.json")
ANALYSIS_DIR = Path("artifacts/analysis/training")
RAW_DIR = Path("artifacts/raw/training")
WAREHOUSE_DIR = Path("artifacts/warehouse")
OUTPUT = ANALYSIS_DIR / "training_qualification_sf_train_v1.json"
RELEASE_OUTPUT = ANALYSIS_DIR / "training_release_decision_sf_train_v1.json"

EXPECTED = [
    ("qual-sf-train-v1-t0-r1-o1", "t0_eager_fixed", 1, 1),
    ("qual-sf-train-v1-t4-r1-o2", "t4_dynamic_compile", 1, 2),
    ("qual-sf-train-v1-t4-r2-o3", "t4_dynamic_compile", 2, 3),
    ("qual-sf-train-v1-t0-r2-o4", "t0_eager_fixed", 2, 4),
    ("qual-sf-train-v1-t0-r3-o5", "t0_eager_fixed", 3, 5),
    ("qual-sf-train-v1-t4-r3-o6", "t4_dynamic_compile", 3, 6),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def descriptive_spearman(frame: pd.DataFrame, x_column: str, y_column: str) -> dict[str, Any]:
    coefficient, p_value = spearmanr(frame[x_column], frame[y_column])
    return {
        "x_column": x_column,
        "y_column": y_column,
        "coefficient": float(coefficient),
        "p_value_descriptive_only": float(p_value),
        "n": len(frame),
        "material_association_threshold_abs_rho_ge_0_5": bool(abs(coefficient) >= 0.5),
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    freeze = load_json(FREEZE_PATH)
    freeze_sha256 = sha256_file(FREEZE_PATH)
    if config["protocol_identity"] != PROTOCOL or config["state"] != "FROZEN":
        raise ValueError("training protocol is not frozen")
    if canonical_sha256(config) != freeze["configuration_sha256"]:
        raise ValueError("configuration does not match the freeze manifest")
    if config["qualification"]["balanced_order"] != [row[1] for row in EXPECTED]:
        raise ValueError("expected run order differs from the frozen balanced order")

    summaries: list[dict[str, Any]] = []
    step_frames: list[pd.DataFrame] = []
    telemetry_frames: list[pd.DataFrame] = []
    input_hashes: dict[str, str] = {}
    for run_id, config_id, replicate, run_order in EXPECTED:
        summary_path = ANALYSIS_DIR / f"{run_id}.json"
        steps_path = RAW_DIR / f"{run_id}-steps.jsonl"
        telemetry_path = RAW_DIR / f"{run_id}-gpu.csv"
        failure_path = Path("artifacts/failures") / f"{run_id}.json"
        if failure_path.exists():
            raise ValueError(f"qualification failure artifact exists: {failure_path}")
        for path in (summary_path, steps_path, telemetry_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty qualification artifact: {path}")
            input_hashes[str(path)] = sha256_file(path)
        summary = load_json(summary_path)
        identity = (
            summary["run_id"],
            summary["config_id"],
            int(summary["replicate"]),
            int(summary["run_order"]),
        )
        if identity != (run_id, config_id, replicate, run_order):
            raise ValueError(f"run identity mismatch: {identity}")
        if (
            summary["protocol_identity"] != PROTOCOL
            or summary["state"] != "QUALIFICATION"
            or summary["status"] != "COMPLETED"
            or summary["freeze_manifest_sha256"] != freeze_sha256
            or summary["config_hash"] != freeze["configuration_sha256"]
            or summary["protected_data_used"] is not False
        ):
            raise ValueError(f"qualification control mismatch: {run_id}")
        steps = pd.read_json(steps_path, lines=True)
        if (
            len(steps) != 12
            or int(steps["measured"].sum()) != 10
            or set(steps["run_id"]) != {run_id}
            or set(steps["state"]) != {"QUALIFICATION"}
            or steps["step"].duplicated().any()
        ):
            raise ValueError(f"invalid step artifact: {run_id}")
        step_frames.append(steps)
        telemetry = read_nvidia_smi_csv(telemetry_path, run_id=run_id)
        measurement_start = telemetry["timestamp"].max() - pd.to_timedelta(
            float(summary["elapsed_s"]), unit="s"
        )
        telemetry["measurement_phase"] = np.where(
            telemetry["timestamp"].ge(measurement_start),
            "MEASURED_APPROX",
            "WARMUP_OR_COLD",
        )
        telemetry_frames.append(telemetry)
        steady_step_s = float(summary["elapsed_s"]) / int(summary["measured_steps"])
        first_step_s = float(summary["warmup_step_times_s"][0])
        cold_overhead_s = float(summary["compile_prepare_s"]) + max(
            0.0, first_step_s - steady_step_s
        )
        summary["steady_step_mean_s"] = steady_step_s
        summary["first_step_plus_prepare_s"] = first_step_s + float(summary["compile_prepare_s"])
        summary["cold_overhead_s"] = cold_overhead_s
        summary["projected_378_step_wall_s"] = projected_wall_time_s(
            cold_s=cold_overhead_s,
            step_s=steady_step_s,
            target_steps=int(config["workload"]["target_training_steps"]),
        )
        summaries.append(summary)

    runs = pd.DataFrame(summaries).sort_values("run_order").reset_index(drop=True)
    steps = pd.concat(step_frames, ignore_index=True).sort_values(["run_order", "step"])
    telemetry = pd.concat(telemetry_frames, ignore_index=True).sort_values(["run_id", "timestamp"])
    if runs["run_id"].duplicated().any() or len(runs) != 6:
        raise ValueError("qualification run matrix is incomplete or duplicated")
    token_counts = set(runs["tokens"].astype(int))
    if len(token_counts) != 1:
        raise ValueError("candidate and baseline did not process identical measured tokens")

    measured_telemetry = telemetry.loc[telemetry["measurement_phase"].eq("MEASURED_APPROX")]
    telemetry_summary = measured_telemetry.groupby("run_id", as_index=False).agg(
        temperature_c_mean=("temperature_c", "mean"),
        temperature_c_max=("temperature_c", "max"),
        sm_clock_mhz_mean=("sm_clock_mhz", "mean"),
        power_w_mean=("power_w", "mean"),
        utilization_pct_mean=("utilization_pct", "mean"),
        telemetry_samples=("timestamp", "size"),
    )
    runs = runs.merge(telemetry_summary, on="run_id", validate="one_to_one")
    baseline = runs.loc[runs["config_id"].eq("t0_eager_fixed")].sort_values("replicate")
    candidate = runs.loc[runs["config_id"].eq("t4_dynamic_compile")].sort_values("replicate")
    if list(baseline["replicate"]) != [1, 2, 3] or list(candidate["replicate"]) != [1, 2, 3]:
        raise ValueError("qualification replicate matrix is incomplete")

    throughput_change = paired_percent_change(
        baseline["tokens_per_s"].to_numpy(), candidate["tokens_per_s"].to_numpy()
    )
    projected_change = paired_percent_change(
        baseline["projected_378_step_wall_s"].to_numpy(),
        candidate["projected_378_step_wall_s"].to_numpy(),
    )
    break_even_steps = []
    for (_, baseline_row), (_, candidate_row) in zip(
        baseline.iterrows(), candidate.iterrows(), strict=True
    ):
        excess_cold = max(
            0.0, float(candidate_row["cold_overhead_s"] - baseline_row["cold_overhead_s"])
        )
        break_even_steps.append(
            compile_break_even_steps(
                baseline_step_s=float(baseline_row["steady_step_mean_s"]),
                candidate_step_s=float(candidate_row["steady_step_mean_s"]),
                candidate_cold_s=excess_cold,
            )
        )
    finite_break_even = [value for value in break_even_steps if value is not None]
    memory_reduction_pct = (
        1.0
        - float(candidate["peak_allocated_mib"].median())
        / float(baseline["peak_allocated_mib"].median())
    ) * 100.0
    thermal_diagnostics = {
        config_id: {
            "run_order_vs_tokens_per_s": descriptive_spearman(group, "run_order", "tokens_per_s"),
            "temperature_vs_tokens_per_s": descriptive_spearman(
                group, "temperature_c_mean", "tokens_per_s"
            ),
        }
        for config_id, group in runs.groupby("config_id", sort=True)
    }
    thermal_confound = any(
        diagnostics["run_order_vs_tokens_per_s"]["material_association_threshold_abs_rho_ge_0_5"]
        or diagnostics["temperature_vs_tokens_per_s"][
            "material_association_threshold_abs_rho_ge_0_5"
        ]
        for diagnostics in thermal_diagnostics.values()
    )

    gates = {
        "artifact_integrity": True,
        "useful_gain": throughput_change.median_paired_change_pct > 0,
        "memory_non_regression": float(candidate["peak_allocated_mib"].max())
        <= float(baseline["peak_allocated_mib"].min()),
        "reliability": True,
        "adoption_target": float(candidate["projected_378_step_wall_s"].median())
        < float(baseline["projected_378_step_wall_s"].median()),
    }
    decision = "PASS" if all(gates.values()) else "REVIEW"
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": PROTOCOL,
        "state": "ANALYSIS",
        "freeze_manifest_sha256": freeze_sha256,
        "input_artifact_sha256": input_hashes,
        "matrix": {"runs": 6, "replicates_per_config": 3, "measured_steps_per_run": 10},
        "equal_measured_tokens_per_run": next(iter(token_counts)),
        "steady_throughput": asdict(throughput_change),
        "projected_378_step_wall_time_change": asdict(projected_change),
        "break_even_steps_by_pair": break_even_steps,
        "break_even_steps_median": float(np.median(finite_break_even)),
        "peak_allocated_mib": {
            "baseline_median": float(baseline["peak_allocated_mib"].median()),
            "candidate_median": float(candidate["peak_allocated_mib"].median()),
            "candidate_reduction_pct": memory_reduction_pct,
        },
        "thermal_and_order_diagnostics": thermal_diagnostics,
        "thermal_or_order_confound_material": thermal_confound,
        "telemetry_phase_note": (
            "MEASURED_APPROX is the final elapsed_s window of each telemetry trace; sampling "
            "at 500 ms prevents exact step-boundary alignment."
        ),
        "release_gates": gates,
        "decision": decision,
        "selected_configuration": (
            "t4_dynamic_compile" if decision == "PASS" else "t0_eager_fixed"
        ),
        "claim_qualification": (
            "Three paired replicates on one laptop GPU; strong run-order/thermal associations "
            "are possible with n=3 per config, so report the paired interval and cold cost."
        ),
    }
    release = {
        "schema_version": "1.0.0",
        "created_at": result["created_at"],
        "protocol_identity": PROTOCOL,
        "subsystem": "TRAINING_SYSTEM",
        "decision": decision,
        "selected_configuration": result["selected_configuration"],
        "gates": gates,
        "limitations": [
            "Single RTX 4090 Laptop GPU and three replicates per configuration.",
            "Material run-order/thermal association is reported descriptively, not adjusted away.",
            "Compile break-even and 378-step wall time are projections from measured "
            "cold and steady phases.",
        ],
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_parquet(WAREHOUSE_DIR / "training_runs.parquet", index=False)
    steps.to_parquet(WAREHOUSE_DIR / "training_steps.parquet", index=False)
    telemetry.to_parquet(WAREHOUSE_DIR / "gpu_telemetry.parquet", index=False)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    RELEASE_OUTPUT.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
