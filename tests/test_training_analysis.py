from __future__ import annotations

import numpy as np
import pytest

from scaleforge.analysis.training import (
    compile_break_even_steps,
    paired_percent_change,
    projected_wall_time_s,
)


def test_paired_percent_change_is_seeded_and_pair_aware() -> None:
    baseline = np.array([100.0, 80.0, 120.0])
    candidate = np.array([150.0, 100.0, 180.0])
    first = paired_percent_change(baseline, candidate, bootstrap_samples=500, seed=7)
    second = paired_percent_change(baseline, candidate, bootstrap_samples=500, seed=7)
    assert first == second
    assert first.median_paired_change_pct == pytest.approx(50.0)
    assert first.pairs == 3


def test_training_projection_and_break_even() -> None:
    assert compile_break_even_steps(
        baseline_step_s=2.0, candidate_step_s=1.0, candidate_cold_s=100.0
    ) == pytest.approx(100.0)
    assert (
        compile_break_even_steps(baseline_step_s=1.0, candidate_step_s=2.0, candidate_cold_s=100.0)
        is None
    )
    assert projected_wall_time_s(cold_s=10.0, step_s=2.0, target_steps=5) == 20.0


def test_training_analysis_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        paired_percent_change(np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        projected_wall_time_s(cold_s=0.0, step_s=0.0, target_steps=1)
