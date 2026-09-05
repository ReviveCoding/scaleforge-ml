import pytest
from pydantic import ValidationError

from scaleforge.data.schema import SplitFractions
from scaleforge.data.splits import (
    assert_no_normalized_overlap,
    assign_train_role,
    stable_example_id,
)
from scaleforge.types import SplitRole


def test_split_is_deterministic() -> None:
    fractions = SplitFractions(fit=0.8, validation=0.1, policy=0.1)
    example_id = stable_example_id("gsm8k", "revision", " What is 2 + 2? ")
    assert example_id == stable_example_id("gsm8k", "revision", "what is 2 + 2?")
    assert assign_train_role(example_id, fractions) == assign_train_role(example_id, fractions)


def test_split_fractions_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        SplitFractions(fit=0.8, validation=0.15, policy=0.1)


def test_normalized_overlap_fails() -> None:
    with pytest.raises(ValueError, match="occurs"):
        assert_no_normalized_overlap(
            {SplitRole.FIT: ["Same  question"], SplitRole.VALIDATION: [" same question "]}
        )
