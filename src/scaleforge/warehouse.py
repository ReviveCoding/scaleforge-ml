from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REQUIRED_RUN_COLUMNS = {
    "run_id",
    "protocol_identity",
    "config_id",
    "state",
    "started_at",
    "completed_at",
    "status",
}

ALLOWED_FAILURE_TYPES = {
    "OOM",
    "timeout",
    "service_failure",
    "parser_failure",
    "invalid_generation",
    "worker_crash",
    "quality_regression",
    "SLO_violation",
    "artifact_integrity_failure",
    "distributed_synchronization_failure",
    "invalid_service_response",
}


def validate_run_manifest(frame: pd.DataFrame, expected_artifacts: Iterable[str] = ()) -> None:
    missing = REQUIRED_RUN_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing run manifest columns: {sorted(missing)}")
    if frame["run_id"].isna().any() or frame["run_id"].duplicated().any():
        raise ValueError("run IDs must be non-null and unique")
    completed = frame["status"].eq("COMPLETED")
    if frame.loc[completed, "completed_at"].isna().any():
        raise ValueError("completed runs require completed_at")
    if expected_artifacts and "artifact_paths" not in frame.columns:
        raise ValueError("expected artifact validation requires artifact_paths")
    for artifact in expected_artifacts:
        if not frame["artifact_paths"].fillna("").str.contains(artifact, regex=False).all():
            raise ValueError(f"expected artifact absent: {artifact}")


def validate_predictions(
    frame: pd.DataFrame,
    *,
    expected_counts: dict[tuple[str, str], int],
    protocol_identity: str,
) -> None:
    required = {
        "run_id",
        "protocol_identity",
        "state",
        "dataset_id",
        "config_id",
        "example_id",
        "correct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if frame.empty or set(frame["protocol_identity"]) != {protocol_identity}:
        raise ValueError("prediction protocol identity mismatch")
    if set(frame["state"]) != {"QUALIFICATION"}:
        raise ValueError("only qualification predictions enter the canonical protected table")
    if frame.duplicated(["run_id", "example_id"]).any() or frame["correct"].isna().any():
        raise ValueError("duplicate predictions or null correctness values")
    observed = frame.groupby(["dataset_id", "config_id"])["example_id"].nunique().to_dict()
    if observed != expected_counts:
        raise ValueError(f"prediction cardinality mismatch: {observed}")


def validate_serving_requests(frame: pd.DataFrame) -> None:
    required = {
        "run_id",
        "request_id",
        "config_id",
        "replicate",
        "concurrency",
        "measured",
        "success",
        "failure_type",
        "e2e_ms",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing serving columns: {sorted(missing)}")
    if frame.empty or frame.duplicated(["run_id", "request_id"]).any():
        raise ValueError("serving requests must be non-empty and uniquely identified")
    successful_with_failure = frame["success"].astype(bool) & frame["failure_type"].notna()
    failed_without_failure = ~frame["success"].astype(bool) & frame["failure_type"].isna()
    if successful_with_failure.any() or failed_without_failure.any():
        raise ValueError("serving success/failure taxonomy is inconsistent")
    observed = set(frame.loc[frame["failure_type"].notna(), "failure_type"])
    if not observed <= ALLOWED_FAILURE_TYPES:
        raise ValueError(
            f"unknown serving failure types: {sorted(observed - ALLOWED_FAILURE_TYPES)}"
        )
