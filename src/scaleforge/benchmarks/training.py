from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrainingWorkload:
    global_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    measured_steps: int


def validate_training_system_config(config: Mapping[str, Any]) -> TrainingWorkload:
    if config.get("state") not in {"DEVELOPMENT", "FROZEN"}:
        raise ValueError("training systems config must be DEVELOPMENT or FROZEN")
    data = config.get("data", {})
    if data.get("split_role") != "FIT":
        raise ValueError("training systems development is restricted to FIT")
    raw = config.get("workload", {})
    workload = TrainingWorkload(
        global_batch_size=int(raw["global_batch_size"]),
        micro_batch_size=int(raw["micro_batch_size"]),
        gradient_accumulation_steps=int(raw["gradient_accumulation_steps"]),
        warmup_steps=int(raw["warmup_steps"]),
        measured_steps=int(raw["measured_steps"]),
    )
    if workload.global_batch_size != (
        workload.micro_batch_size * workload.gradient_accumulation_steps
    ):
        raise ValueError("global batch does not match microbatch times accumulation")
    if workload.warmup_steps < 1 or workload.measured_steps < 3:
        raise ValueError("benchmark requires warmup and at least three measured steps")
    configs = config.get("configs", {})
    if "t0_eager_fixed" not in configs:
        raise ValueError("competent T0 baseline is required")
    return workload


def summarize_training_steps(records: list[Mapping[str, Any]]) -> dict[str, float]:
    measured = [record for record in records if bool(record["measured"])]
    if not measured:
        raise ValueError("no measured training steps")
    times = np.asarray([float(record["step_time_s"]) for record in measured])
    tokens = np.asarray([int(record["tokens"]) for record in measured])
    return {
        "measured_steps": float(len(measured)),
        "tokens": float(tokens.sum()),
        "elapsed_s": float(times.sum()),
        "tokens_per_s": float(tokens.sum() / times.sum()),
        "step_p50_s": float(np.quantile(times, 0.50)),
        "step_p95_s": float(np.quantile(times, 0.95)),
        "peak_allocated_mib": float(max(float(row["peak_allocated_mib"]) for row in measured)),
        "peak_reserved_mib": float(max(float(row["peak_reserved_mib"]) for row in measured)),
    }
