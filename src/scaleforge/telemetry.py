from __future__ import annotations

from pathlib import Path

import pandas as pd

NVIDIA_COLUMNS = [
    "timestamp",
    "gpu_index",
    "gpu_name",
    "temperature_c",
    "sm_clock_mhz",
    "power_w",
    "utilization_pct",
    "memory_used_mib",
]


def read_nvidia_smi_csv(path: Path, *, run_id: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty GPU telemetry: {path}")
    frame = pd.read_csv(path, names=NVIDIA_COLUMNS, dtype=str, skipinitialspace=True)
    if frame.empty or frame["timestamp"].isna().any():
        raise ValueError("GPU telemetry is empty or has null timestamps")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    required_numeric = [
        "gpu_index",
        "temperature_c",
        "sm_clock_mhz",
        "utilization_pct",
        "memory_used_mib",
    ]
    for column in [*required_numeric, "power_w"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["timestamp"].isna().any() or frame[required_numeric].isna().any().any():
        raise ValueError("required GPU telemetry fields are malformed")
    frame["power_valid"] = frame["power_w"].notna() & frame["power_w"].between(0, 300)
    frame.loc[~frame["power_valid"], "power_w"] = pd.NA
    frame["run_id"] = run_id
    return frame
