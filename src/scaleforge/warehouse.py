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
