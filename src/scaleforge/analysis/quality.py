from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True)
class PairedQuality:
    n: int
    baseline_em: float
    candidate_em: float
    delta_pp: float
    candidate_only: int
    baseline_only: int
    both_correct: int
    both_wrong: int
    ci_low_pp: float
    ci_high_pp: float
    mcnemar_exact_p: float


def paired_quality(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260904,
) -> PairedQuality:
    baseline = np.asarray(baseline, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if baseline.ndim != 1 or candidate.ndim != 1 or len(baseline) != len(candidate):
        raise ValueError("paired vectors must be one-dimensional and equal length")
    if len(baseline) == 0:
        raise ValueError("paired vectors cannot be empty")
    candidate_only = int(np.sum(candidate & ~baseline))
    baseline_only = int(np.sum(baseline & ~candidate))
    discordant = candidate_only + baseline_only
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(candidate_only, discordant, 0.5, alternative="two-sided").pvalue)
    )
    rng = np.random.default_rng(seed)
    deltas = candidate.astype(float) - baseline.astype(float)
    indices = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
    samples = deltas[indices].mean(axis=1) * 100
    low, high = np.quantile(samples, [0.025, 0.975])
    return PairedQuality(
        n=len(baseline),
        baseline_em=float(baseline.mean()),
        candidate_em=float(candidate.mean()),
        delta_pp=float(deltas.mean() * 100),
        candidate_only=candidate_only,
        baseline_only=baseline_only,
        both_correct=int(np.sum(candidate & baseline)),
        both_wrong=int(np.sum(~candidate & ~baseline)),
        ci_low_pp=float(low),
        ci_high_pp=float(high),
        mcnemar_exact_p=p_value,
    )
