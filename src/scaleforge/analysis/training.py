from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedTrainingChange:
    baseline_median: float
    candidate_median: float
    median_paired_change_pct: float
    ci_low_pct: float
    ci_high_pct: float
    pairs: int
    bootstrap_samples: int


def paired_percent_change(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 20260905,
) -> PairedTrainingChange:
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.ndim != 1 or candidate.ndim != 1 or baseline.shape != candidate.shape:
        raise ValueError("paired training vectors must be one-dimensional and equal length")
    if baseline.size == 0 or bootstrap_samples < 1:
        raise ValueError("paired training vectors and bootstrap must be non-empty")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired training vectors must be finite")
    if np.any(baseline <= 0):
        raise ValueError("baseline values must be positive")
    paired_changes = (candidate / baseline - 1.0) * 100.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(bootstrap_samples, baseline.size))
    samples = np.median(paired_changes[indices], axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return PairedTrainingChange(
        baseline_median=float(np.median(baseline)),
        candidate_median=float(np.median(candidate)),
        median_paired_change_pct=float(np.median(paired_changes)),
        ci_low_pct=float(low),
        ci_high_pct=float(high),
        pairs=int(baseline.size),
        bootstrap_samples=bootstrap_samples,
    )


def compile_break_even_steps(
    *, baseline_step_s: float, candidate_step_s: float, candidate_cold_s: float
) -> float | None:
    values = (baseline_step_s, candidate_step_s, candidate_cold_s)
    if not all(np.isfinite(value) and value >= 0 for value in values):
        raise ValueError("break-even inputs must be finite and non-negative")
    per_step_saving = baseline_step_s - candidate_step_s
    if per_step_saving <= 0:
        return None
    return candidate_cold_s / per_step_saving


def projected_wall_time_s(*, cold_s: float, step_s: float, target_steps: int) -> float:
    if not np.isfinite(cold_s) or not np.isfinite(step_s) or cold_s < 0 or step_s <= 0:
        raise ValueError("projection timing inputs are invalid")
    if target_steps < 1:
        raise ValueError("target steps must be positive")
    return cold_s + step_s * target_steps
