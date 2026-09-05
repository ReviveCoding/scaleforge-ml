import pytest

from scaleforge.data.lengths import choose_sequence_limit, quantiles


def test_length_summary_and_headroom() -> None:
    assert quantiles([1, 2, 3, 4])["max"] == 4
    assert choose_sequence_limit(100, 1.1, 128, 4096) == 128


def test_sequence_policy_never_truncates() -> None:
    with pytest.raises(ValueError, match="never truncate"):
        choose_sequence_limit(4096, 1.1, 128, 4096)
