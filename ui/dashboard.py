from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_dashboard_data(root: Path) -> dict[str, Any]:
    release_path = root / "artifacts/analysis/release_decision.json"
    points_path = root / "artifacts/analysis/serving/serving_qualification_points.parquet"
    training_path = root / "artifacts/warehouse/training_runs.parquet"
    if not release_path.is_file() or not points_path.is_file() or not training_path.is_file():
        raise FileNotFoundError("qualified dashboard artifacts are incomplete")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    points = pd.read_parquet(points_path)
    training = pd.read_parquet(training_path)
    selected = points.loc[
        points["concurrency"].eq(2),
        ["config_id", "successful_requests_per_s", "ttft_p95_ms", "tpot_p95_ms"],
    ].copy()
    if len(selected) != 2 or len(training) != 6:
        raise ValueError("dashboard inputs do not match frozen qualification matrix")
    return {
        "project_status": release["project_status"],
        "gates": pd.DataFrame(release["gates"]),
        "selected_serving": selected,
        "training": training,
    }
