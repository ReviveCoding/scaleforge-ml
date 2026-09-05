from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from scaleforge.telemetry import read_nvidia_smi_csv
from scaleforge.warehouse import validate_predictions, validate_serving_requests

WAREHOUSE = Path("artifacts/warehouse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-training", action="store_true")
    parser.add_argument("--require-serving", action="store_true")
    return parser.parse_args()


def read_jsonl_files(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return pd.DataFrame(rows)


def collect_failures() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path("artifacts/failures").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "artifact_path": str(path),
                "run_id": payload.get("run_id"),
                "protocol_identity": payload.get("protocol_identity"),
                "state": payload.get("state"),
                "failure_type": payload.get("failure_type", "quality_regression"),
                "disposition": payload.get("disposition", payload.get("decision")),
                "protected_data_used": bool(payload.get("protected_data_used", False)),
                "payload_json": json.dumps(payload, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(WAREHOUSE / "model_predictions.parquet")
    validate_predictions(
        predictions,
        expected_counts={
            ("gsm8k", "m0"): 1319,
            ("gsm8k", "mstar_lora_r32_attn_mlp"): 1319,
            ("math500", "m0"): 500,
            ("math500", "mstar_lora_r32_attn_mlp"): 500,
        },
        protocol_identity="SF-MODEL-v2",
    )
    training_paths = sorted(Path("artifacts/raw/training").glob("*-steps.jsonl"))
    training_steps = read_jsonl_files(training_paths)
    if args.require_training and training_steps.empty:
        raise ValueError("required training runs are absent")
    serving_paths = sorted(Path("artifacts/raw/serving").glob("*-requests.jsonl"))
    serving_requests = read_jsonl_files(serving_paths)
    if args.require_serving and serving_requests.empty:
        raise ValueError("required serving runs are absent")
    if not serving_requests.empty:
        validate_serving_requests(serving_requests)
    telemetry_frames = []
    for path in sorted(Path("artifacts/raw/training").glob("*-gpu.csv")):
        run_id = path.name.removesuffix("-gpu.csv")
        telemetry_frames.append(read_nvidia_smi_csv(path, run_id=run_id))
    for path in sorted(Path("artifacts/raw/serving").glob("*-gpu.csv")):
        run_id = path.name.removesuffix("-gpu.csv")
        telemetry_frames.append(read_nvidia_smi_csv(path, run_id=run_id))
    gpu_telemetry = (
        pd.concat(telemetry_frames, ignore_index=True) if telemetry_frames else pd.DataFrame()
    )
    failures = collect_failures()
    tables = {
        "model_predictions": predictions,
        "training_steps": training_steps,
        "serving_requests": serving_requests,
        "gpu_telemetry": gpu_telemetry,
        "failure_events": failures,
    }
    for name, frame in tables.items():
        frame.to_parquet(WAREHOUSE / f"{name}.parquet", index=False)
    database = duckdb.connect(str(WAREHOUSE / "results.duckdb"))
    try:
        for name, frame in tables.items():
            if frame.empty:
                continue
            database.register(f"_{name}", frame)
            database.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _{name}")
            database.unregister(f"_{name}")
        integrity = {
            "schema_version": "1.0.0",
            "model_prediction_rows": len(predictions),
            "training_step_rows": len(training_steps),
            "serving_request_rows": len(serving_requests),
            "gpu_telemetry_rows": len(gpu_telemetry),
            "failure_event_rows": len(failures),
            "duplicate_model_predictions": int(
                predictions.duplicated(["run_id", "example_id"]).sum()
            ),
            "duplicate_serving_requests": (
                int(serving_requests.duplicated(["run_id", "request_id"]).sum())
                if not serving_requests.empty
                else 0
            ),
            "status": "PASS",
        }
    finally:
        database.close()
    Path("artifacts/analysis/warehouse_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
