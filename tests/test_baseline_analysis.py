import numpy as np

from scaleforge.analysis.quality import paired_quality


def test_direction_is_m0_vs_m1() -> None:
    stats = paired_quality(
        np.array([True, True, False]),
        np.array([True, False, False]),
        bootstrap_samples=100,
    )
    assert stats.baseline_em > stats.candidate_em
    assert stats.baseline_only == 1
    assert stats.candidate_only == 0
