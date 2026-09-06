from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from scaleforge.protocol import sha256_file
from scaleforge.warehouse import (
    normalize_failure_type,
    validate_predictions,
    validate_run_manifest,
    validate_serving_requests,
    validate_training_results,
)

WAREHOUSE = Path("artifacts/warehouse")
ANALYSIS = Path("artifacts/analysis")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"required warehouse input is absent or empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def collect_failures() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path("artifacts/failures").glob("*.json")):
        payload = read_json(path)
        original = payload.get("failure_type", payload.get("exception_type", "worker_crash"))
        rows.append(
            {
                "failure_event_id": path.stem,
                "artifact_path": str(path),
                "run_id": payload.get("run_id", payload.get("attempt")),
                "protocol_identity": payload.get("protocol_identity"),
                "state": payload.get("state"),
                "failure_type_original": str(original),
                "failure_type": normalize_failure_type(original),
                "disposition": payload.get("disposition", payload.get("decision")),
                "protected_data_used": bool(payload.get("protected_data_used", False)),
                "protected_outcomes_exposed": payload.get("protected_outcomes_exposed"),
                "payload_json": json.dumps(payload, sort_keys=True),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["failure_event_id"].duplicated().any():
        raise ValueError("failure-event warehouse input is empty or duplicated")
    return frame


def collect_distributed_runs() -> pd.DataFrame:
    paths = sorted(
        Path("artifacts/raw/distributed").glob("distributed-launch-smoke-20260906-rank*.json")
    )
    rows = [{**read_json(path), "artifact_path": str(path)} for path in paths]
    frame = pd.DataFrame(rows)
    if (
        len(frame) != 2
        or set(frame["rank"]) != {0, 1}
        or set(frame["all_reduce_sum"]) != {3.0}
        or frame["numerical_scaling_claim_allowed"].astype(bool).any()
    ):
        raise ValueError("distributed launch smoke is incomplete or claim-unsafe")
    return frame


def release_gates() -> pd.DataFrame:
    model_path = ANALYSIS / "model/model_release_decision_sf_model_v3.json"
    training_path = ANALYSIS / "training/training_release_decision_sf_train_v1.json"
    serving_path = ANALYSIS / "serving/serving_release_decision_sf_serve_v2.json"
    distributed_path = ANALYSIS / "distributed/distributed_status.json"
    model = read_json(model_path)
    training = read_json(training_path)
    serving = read_json(serving_path)
    distributed = read_json(distributed_path)
    rows = [
        {
            "subsystem": "MODEL",
            "protocol_identity": model["protocol_identity"],
            "decision": model["decision"],
            "selected_configuration": model["selected_configuration"],
            "gates_json": json.dumps(model["gates"], sort_keys=True),
            "evidence_path": str(model_path),
        },
        {
            "subsystem": "TRAINING_SYSTEM",
            "protocol_identity": training["protocol_identity"],
            "decision": training["decision"],
            "selected_configuration": training["selected_configuration"],
            "gates_json": json.dumps(training["gates"], sort_keys=True),
            "evidence_path": str(training_path),
        },
        {
            "subsystem": "SERVING",
            "protocol_identity": serving["protocol_identity"],
            "decision": serving["decision"],
            "selected_configuration": json.dumps(serving["selected_configuration"], sort_keys=True),
            "gates_json": json.dumps(serving["release_gates"], sort_keys=True),
            "evidence_path": str(serving_path),
        },
        {
            "subsystem": "DISTRIBUTED",
            "protocol_identity": distributed["protocol_identity"],
            "decision": "BLOCKED_EXTERNAL",
            "selected_configuration": None,
            "gates_json": json.dumps(
                {
                    "physical_cuda_gpu_count_gte_2": False,
                    "cpu_gloo_launch_smoke": True,
                    "numeric_scaling_claim_allowed": False,
                },
                sort_keys=True,
            ),
            "evidence_path": str(distributed_path),
        },
        {
            "subsystem": "REPRODUCIBILITY",
            "protocol_identity": "SF-RELEASE-v1",
            "decision": "REVIEW",
            "selected_configuration": None,
            "gates_json": json.dumps(
                {
                    "environment_locks": True,
                    "artifact_hashes": True,
                    "cpu_ci": True,
                    "protected_evaluation_first_pass_clean": False,
                },
                sort_keys=True,
            ),
            "evidence_path": "FINAL_ACCESS_LEDGER.json",
        },
    ]
    frame = pd.DataFrame(rows)
    allowed = {"PASS", "REVIEW", "BLOCK", "BLOCKED_EXTERNAL", "NOT_APPLICABLE"}
    if set(frame["decision"]) - allowed or frame["subsystem"].duplicated().any():
        raise ValueError("release decisions are invalid or duplicated")
    return frame


def build_run_manifest(
    predictions: pd.DataFrame,
    training_runs: pd.DataFrame,
    serving_runs: pd.DataFrame,
    distributed_runs: pd.DataFrame,
) -> pd.DataFrame:
    model_completed = read_json(ANALYSIS / "model/model_qualification_sf_model_v3.json")[
        "created_at"
    ]
    rows: list[dict[str, Any]] = []
    for run_id, group in predictions.groupby("run_id", sort=True):
        rows.append(
            {
                "run_id": run_id,
                "protocol_identity": group["protocol_identity"].iloc[0],
                "config_id": group["config_id"].iloc[0],
                "state": group["state"].iloc[0],
                "started_at": None,
                "completed_at": model_completed,
                "status": "COMPLETED",
                "subsystem": "MODEL",
                "artifact_paths": "artifacts/warehouse/model_predictions.parquet",
            }
        )
    for row in training_runs.itertuples():
        rows.append(
            {
                "run_id": row.run_id,
                "protocol_identity": row.protocol_identity,
                "config_id": row.config_id,
                "state": row.state,
                "started_at": row.started_at,
                "completed_at": row.created_at,
                "status": row.status,
                "subsystem": "TRAINING_SYSTEM",
                "artifact_paths": f"artifacts/raw/training/{row.run_id}-steps.jsonl",
            }
        )
    for row in serving_runs.itertuples():
        completed = pd.Timestamp(row.completed_at)
        rows.append(
            {
                "run_id": row.run_id,
                "protocol_identity": row.protocol_identity,
                "config_id": row.config_id,
                "state": row.state,
                "started_at": (completed - timedelta(seconds=float(row.elapsed_s))).isoformat(),
                "completed_at": completed.isoformat(),
                "status": row.status,
                "subsystem": "SERVING",
                "artifact_paths": row.request_artifact_path,
            }
        )
    rows.append(
        {
            "run_id": distributed_runs["run_id"].iloc[0],
            "protocol_identity": "SF-DIST-v1",
            "config_id": "cpu_gloo_launch_smoke",
            "state": "PILOT",
            "started_at": None,
            "completed_at": distributed_runs["created_at"].max(),
            "status": "COMPLETED",
            "subsystem": "DISTRIBUTED",
            "artifact_paths": ";".join(distributed_runs["artifact_path"]),
        }
    )
    frame = pd.DataFrame(rows)
    validate_run_manifest(frame)
    return frame


def main() -> None:
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
        protocol_identity="SF-MODEL-v3",
    )
    training_steps = pd.read_parquet(WAREHOUSE / "training_steps.parquet")
    training_runs = pd.read_parquet(WAREHOUSE / "training_runs.parquet")
    validate_training_results(training_steps, training_runs)
    if len(training_runs) != 6 or len(training_steps) != 72:
        raise ValueError("frozen training matrix cardinality mismatch")
    serving_requests = pd.read_parquet(WAREHOUSE / "serving_requests.parquet")
    serving_runs = pd.read_parquet(WAREHOUSE / "serving_runs.parquet")
    validate_serving_requests(serving_requests)
    if (
        len(serving_requests) != 2592
        or int(serving_requests["measured"].sum()) != 2304
        or len(serving_runs) != 36
        or serving_runs["run_id"].duplicated().any()
        or set(serving_requests["protocol_identity"]) != {"SF-SERVE-v2"}
    ):
        raise ValueError("frozen serving matrix cardinality or identity mismatch")

    training_telemetry_path = WAREHOUSE / "training_gpu_telemetry.parquet"
    if training_telemetry_path.is_file():
        training_telemetry = pd.read_parquet(training_telemetry_path)
    else:
        training_telemetry = pd.read_parquet(WAREHOUSE / "gpu_telemetry.parquet")
        training_telemetry = training_telemetry.loc[
            training_telemetry["run_id"].isin(training_runs["run_id"])
        ].copy()
        training_telemetry.to_parquet(training_telemetry_path, index=False)
    serving_telemetry = pd.read_parquet(WAREHOUSE / "serving_gpu_telemetry.parquet")
    if set(training_telemetry["run_id"]) != set(training_runs["run_id"]):
        raise ValueError("training telemetry run identity mismatch")
    if set(serving_telemetry["run_id"]) != set(serving_runs["run_id"]):
        raise ValueError("serving telemetry run identity mismatch")
    training_telemetry["subsystem"] = "TRAINING_SYSTEM"
    serving_telemetry["subsystem"] = "SERVING"
    gpu_telemetry = pd.concat([training_telemetry, serving_telemetry], ignore_index=True)

    distributed_runs = collect_distributed_runs()
    failures = collect_failures()
    gates = release_gates()
    run_manifest = build_run_manifest(predictions, training_runs, serving_runs, distributed_runs)
    tables = {
        "run_manifest": run_manifest,
        "model_predictions": predictions,
        "training_steps": training_steps,
        "training_runs": training_runs,
        "serving_requests": serving_requests,
        "serving_runs": serving_runs,
        "gpu_telemetry": gpu_telemetry,
        "distributed_runs": distributed_runs,
        "failure_events": failures,
        "release_gate_results": gates,
    }
    for name, frame in tables.items():
        if frame.empty:
            raise ValueError(f"canonical warehouse table is empty: {name}")
        frame.to_parquet(WAREHOUSE / f"{name}.parquet", index=False)

    database_path = WAREHOUSE / "results.duckdb"
    database = duckdb.connect(str(database_path))
    try:
        for name, frame in tables.items():
            database.register(f"_{name}", frame)
            database.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _{name}")
            database.unregister(f"_{name}")
        expected = {name: len(frame) for name, frame in tables.items()}
        observed = {
            name: int(database.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for name in tables
        }
        if observed != expected:
            raise ValueError(f"DuckDB/Parquet row-count mismatch: {observed} != {expected}")
    finally:
        database.close()

    integrity = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "table_rows": {name: len(frame) for name, frame in tables.items()},
        "duplicate_run_ids": int(run_manifest["run_id"].duplicated().sum()),
        "duplicate_model_predictions": int(predictions.duplicated(["run_id", "example_id"]).sum()),
        "duplicate_serving_requests": int(
            serving_requests.duplicated(["run_id", "request_id"]).sum()
        ),
        "measured_serving_failures": int(
            (~serving_requests.loc[serving_requests["measured"], "success"]).sum()
        ),
        "parquet_sha256": {name: sha256_file(WAREHOUSE / f"{name}.parquet") for name in tables},
        "duckdb_sha256": sha256_file(database_path),
    }
    (ANALYSIS / "warehouse_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
