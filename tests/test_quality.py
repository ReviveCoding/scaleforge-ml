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


@pytest.mark.parametrize("invalid", [0, -3, True, 2.5])
def test_paired_quality_requires_positive_integer_bootstrap_count(invalid) -> None:
    with pytest.raises(ValueError, match="bootstrap_samples"):
        paired_quality(
            np.array([True, False]),
            np.array([True, True]),
            bootstrap_samples=invalid,
        )


def test_paired_quality_identical_arms_have_zero_delta_and_exact_p_one() -> None:
    values = np.array([True, False, True, False])
    result = paired_quality(values, values, bootstrap_samples=32)
    assert result.delta_pp == 0.0
    assert result.mcnemar_exact_p == 1.0
    assert result.ci_low_pp == result.ci_high_pp == 0.0
