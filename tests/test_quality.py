import numpy as np
import pytest

from scaleforge.analysis.quality import paired_quality


def test_paired_quality_transitions() -> None:
    result = paired_quality(
        np.array([True, False, True, False]),
        np.array([True, True, False, False]),
        bootstrap_samples=500,
    )
    assert result.candidate_only == 1
    assert result.baseline_only == 1
    assert result.both_correct == 1
    assert result.both_wrong == 1
    assert result.delta_pp == 0
    assert result.mcnemar_exact_p == 1


def test_paired_quality_rejects_bad_units() -> None:
    with pytest.raises(ValueError):
        paired_quality(np.array([]), np.array([]))
