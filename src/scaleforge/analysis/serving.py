from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    low: float
    high: float
    samples: int


def hierarchical_quantile_interval(
    frame: pd.DataFrame,
    column: str,
    *,
    quantile: float,
    bootstrap_samples: int = 5_000,
    seed: int = 20260905,
) -> BootstrapInterval:
    required = {"replicate", column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing serving bootstrap columns: {sorted(missing)}")
    if frame.empty or frame[column].isna().any():
        raise ValueError("serving bootstrap data must be complete and non-empty")
    if not 0 < quantile < 1 or bootstrap_samples < 1:
        raise ValueError("invalid quantile or bootstrap sample count")
    groups = {
        replicate: values[column].to_numpy(dtype=float)
        for replicate, values in frame.groupby("replicate", sort=True)
    }
    keys = np.asarray(list(groups))
    rng = np.random.default_rng(seed)
    estimates = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        rows = []
        for replicate in chosen:
            values = groups[replicate]
            rows.append(rng.choice(values, size=len(values), replace=True))
        estimates[index] = np.quantile(np.concatenate(rows), quantile)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return BootstrapInterval(
        estimate=float(frame[column].quantile(quantile)),
        low=float(low),
        high=float(high),
        samples=bootstrap_samples,
    )


def mark_pareto_frontier(
    frame: pd.DataFrame,
    *,
    throughput_column: str = "successful_requests_per_s",
    latency_column: str = "ttft_p95_ms",
) -> pd.Series:
    if frame.empty or frame[[throughput_column, latency_column]].isna().any().any():
        raise ValueError("Pareto inputs must be complete and non-empty")
    values = frame[[throughput_column, latency_column]].to_numpy(dtype=float)
    frontier = []
    for index, (throughput, latency) in enumerate(values):
        dominated = False
        for other_index, (other_throughput, other_latency) in enumerate(values):
            if index == other_index:
                continue
            no_worse = other_throughput >= throughput and other_latency <= latency
            strictly_better = other_throughput > throughput or other_latency < latency
            if no_worse and strictly_better:
                dominated = True
                break
        frontier.append(not dominated)
    return pd.Series(frontier, index=frame.index, dtype=bool)


def slo_compliant_throughput(
    frame: pd.DataFrame,
    *,
    ttft_p95_max_ms: float,
    e2e_p95_max_ms: float,
    error_rate_max: float,
) -> tuple[float, str | None]:
    required = {
        "config_id",
        "successful_requests_per_s",
        "ttft_p95_ms",
        "e2e_p95_ms",
        "error_rate",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing SLO columns: {sorted(missing)}")
    compliant = frame.loc[
        frame["ttft_p95_ms"].le(ttft_p95_max_ms)
        & frame["e2e_p95_ms"].le(e2e_p95_max_ms)
        & frame["error_rate"].le(error_rate_max)
    ]
    if compliant.empty:
        return 0.0, None
    winner = compliant.sort_values(
        ["successful_requests_per_s", "config_id"], ascending=[False, True]
    ).iloc[0]
    return float(winner["successful_requests_per_s"]), str(winner["config_id"])
