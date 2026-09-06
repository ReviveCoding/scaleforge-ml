from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedPercentChange:
    baseline_median: float
    candidate_median: float
    estimate_percent: float
    low_percent: float
    high_percent: float
    pairs: int
    samples: int


def paired_replicate_percent_change(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 20260905,
) -> PairedPercentChange:
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.ndim != 1 or candidate.ndim != 1 or baseline.shape != candidate.shape:
        raise ValueError("paired serving vectors must be one-dimensional and equal length")
    if baseline.size == 0 or bootstrap_samples < 1:
        raise ValueError("paired serving vectors and bootstrap must be non-empty")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired serving vectors must be finite")
    if np.any(baseline <= 0):
        raise ValueError("baseline serving values must be positive")
    changes = (candidate / baseline - 1.0) * 100.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(bootstrap_samples, baseline.size))
    estimates = np.median(changes[indices], axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return PairedPercentChange(
        baseline_median=float(np.median(baseline)),
        candidate_median=float(np.median(candidate)),
        estimate_percent=float(np.median(changes)),
        low_percent=float(low),
        high_percent=float(high),
        pairs=int(baseline.size),
        samples=bootstrap_samples,
    )


def saturation_knee(
    frame: pd.DataFrame,
    *,
    concurrency_column: str = "concurrency",
    throughput_column: str = "successful_requests_per_s",
) -> int:
    """Return the geometric knee on a concurrency-throughput curve."""
    ordered = frame.sort_values(concurrency_column)
    concurrency = ordered[concurrency_column].to_numpy(dtype=float)
    throughput = ordered[throughput_column].to_numpy(dtype=float)
    if len(ordered) < 3 or not np.isfinite(concurrency).all() or not np.isfinite(throughput).all():
        raise ValueError("knee inputs must contain at least three complete points")
    if np.any(concurrency <= 0) or np.any(np.diff(concurrency) <= 0):
        raise ValueError("knee concurrency must be positive and strictly increasing")
    x = concurrency.copy()
    if x[-1] == x[0] or throughput.max() == throughput.min():
        return int(concurrency[0])
    x = (x - x[0]) / (x[-1] - x[0])
    y = (throughput - throughput.min()) / (throughput.max() - throughput.min())
    distance = y - (y[0] + (y[-1] - y[0]) * x)
    return int(concurrency[int(np.argmax(distance))])
